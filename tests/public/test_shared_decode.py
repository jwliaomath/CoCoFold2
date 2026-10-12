"""Shared decoder VJP, actual trainer integration and restart compatibility."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from shared_decode import DecodeBatch, configure_decode, restore_decode_mode
from randomness import capture_rng_state, restore_rng_state
from test_b2b_randomness import sampler, same_rng


@pytest.mark.parametrize('count,mini', [(5, 2), (4, 1), (4, 4), (1, 4)])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_gradients_and_optimizer_match_repeated_decode(count, mini, checkpointed):
    def run(mode):
        z = torch.nn.Parameter(torch.linspace(-.5, .5, 12, dtype=torch.float64))
        width = torch.nn.Parameter(torch.tensor(.8, dtype=torch.float64))
        amplitude = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))
        opt = torch.optim.AdamW([z, width, amplitude], lr=.003)
        before = capture_rng_state()
        batch = DecodeBatch(mode, count, mini)
        calls = []
        def decode():
            calls.append(torch.is_grad_enabled())
            # Nonlinear decoder, with activation checkpointing optionally active.
            def net(x):
                return torch.sin(x).square().reshape(4, 3)
            return checkpoint(net, z, use_reentrant=False) if checkpointed and torch.is_grad_enabled() else net(z)
        for start in range(0, count, mini):
            xyz = batch.coordinates(decode)
            samples = torch.arange(start, min(start + mini, count), dtype=z.dtype) + 1
            image = amplitude * torch.exp(-xyz.square().sum(-1)[None] / width.square()) * samples[:, None]
            # Preserve the existing per-microbatch reduction and regularizer.
            loss = image.square().mean() / count + .01 * (width - .4).square()
            loss.backward()
            batch.micro_done()
        del xyz, image, loss
        batch.finish()
        grads = [p.grad.clone() for p in (z, width, amplitude)]
        opt.step()
        same_rng(before, capture_rng_state())
        return grads, [p.detach() for p in (z, width, amplitude)], calls, batch.metadata()
    old, new = run('legacy'), run('shared')
    for a, b in zip(old[:2], new[:2]):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    assert new[2] == ([False, True] if count > mini else [True])
    assert new[3]['decoder_calls'] == (2 if count > mini else 1)


def test_replay_uses_first_randomness_and_preserves_caller_rng():
    torch.manual_seed(9); np.random.seed(9); random.seed(9)
    z = torch.nn.Parameter(torch.ones(3))
    outputs = []
    def decode():
        noise = torch.rand(3) + np.random.rand() + random.random()
        out = torch.sin(z * noise)
        outputs.append(out.detach().clone())
        return out
    batch = DecodeBatch('shared', 3, 2)
    for _ in range(2):
        batch.coordinates(decode).sum().backward()
        torch.rand(2); random.random(); np.random.rand()
        batch.micro_done()
    caller = capture_rng_state()
    batch.finish()
    same_rng(caller, capture_rng_state())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert batch.leaf is None and batch.fn is None


def test_modes_and_old_resume_policy():
    for fixed, expected in [(True, 'shared'), (False, 'legacy')]:
        args = SimpleNamespace(decode_mode='auto', train_deterministic=fixed)
        assert configure_decode(args) == expected
    args = SimpleNamespace(decode_mode='shared', train_deterministic=False)
    with pytest.raises(ValueError, match='fixed stochasticity'):
        configure_decode(args)
    args = SimpleNamespace(decode_mode='auto', train_deterministic=True, _explicit_options=[])
    assert restore_decode_mode(args, {})['decode_mode'] == 'legacy'
    args.decode_mode = 'shared'; args._explicit_options = ['decode_mode']
    with pytest.raises(ValueError, match='Resume cannot change'):
        restore_decode_mode(args, {})
    args.decode_mode = 'auto'
    assert restore_decode_mode(args, {'decode_mode': 'shared'})['decode_mode'] == 'shared'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA replay check requires a GPU')
def test_cuda_replay_only_uses_selected_device(monkeypatch):
    device = torch.device('cuda:0')
    z = torch.nn.Parameter(torch.ones(9, device=device))
    outputs = []
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all',
                        lambda: pytest.fail('Shared decode must not touch all visible GPUs'))
    def decode():
        value = torch.sin(z * torch.rand_like(z))
        outputs.append(value.detach().clone())
        return value
    batch = DecodeBatch('shared', 4, 2, device)
    for _ in range(2):
        batch.coordinates(decode).sum().backward()
        batch.micro_done()
    state = torch.cuda.get_rng_state(device).clone()
    batch.finish()
    assert torch.equal(state, torch.cuda.get_rng_state(device))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_actual_single_trainer_shared_matches_legacy(tmp_path, monkeypatch, sampler, target):
    import train
    from test_epoch_restart import install, fixture
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path, target)
    args.epochs = 1
    args.rng_mode = 'isolated'
    args.update_affine_mat = True
    state = capture_rng_state()
    results = []
    for mode in ('legacy', 'shared'):
        restore_rng_state(state)
        case = copy.deepcopy(args)
        case.decode_mode = mode
        case.output_trained_model_dir = str(tmp_path / (mode + '_'))
        train.main(case)
        results.append(torch.load(tmp_path / (mode + '_1.pth'), weights_only=False))
    for key in ('z_bias', 'rotation', 'translation'):
        torch.testing.assert_close(results[0][key], results[1][key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(results[0]['pred_dict']['coordinate'], results[1]['pred_dict']['coordinate'], atol=2e-6, rtol=2e-5)
    assert results[1]['training_resume']['science_args']['decode_mode'] == 'shared'
    steps = [json.loads(line) for path in (tmp_path / 'shared_records').rglob('metrics.jsonl') for line in path.read_text().splitlines() if json.loads(line)['event'] == 'train_step']
    assert [r['decoder_calls'] for r in steps] == [2, 1]
    assert [r['microbatches'] for r in steps] == [2, 1]


def test_old_checkpoint_resumes_legacy_and_can_warm_start_shared(tmp_path, monkeypatch, sampler):
    import train
    from test_epoch_restart import install, fixture
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path)
    args.epochs = 1; args.decode_mode = 'legacy'
    args.output_trained_model_dir = str(tmp_path / 'old_')
    train.main(args)
    saved = torch.load(tmp_path / 'old_1.pth', weights_only=False)
    del saved['training_resume']['science_args']['decode_mode']
    torch.save(saved, tmp_path / 'historical.pth')
    for warm in (False, True):
        case = copy.deepcopy(args)
        case.resume, case.warm_start = not warm, warm
        case.decode_mode = 'auto'
        case.diffusion_data_dir = str(tmp_path / 'historical.pth')
        case.output_trained_model_dir = str(tmp_path / ('warm_' if warm else 'resume_'))
        case.epochs = 2
        train.main(case)
        output = torch.load(tmp_path / ('warm_2.pth' if warm else 'resume_2.pth'), weights_only=False)
        assert output['training_resume']['science_args']['decode_mode'] == ('shared' if warm else 'legacy')


def test_old_parallel_checkpoint_resumes_legacy(tmp_path, monkeypatch, sampler):
    import hashlib
    from pathlib import Path
    from test_foundations import fake_protenix
    from test_b7b_parallel import setup_case, install_sampler
    from chain_parallel import train_chain_parallel_2d as trainer
    fake_protenix.__wrapped__(monkeypatch)
    install_sampler(monkeypatch, sampler)
    args, _ = setup_case(tmp_path)
    args.decode_mode = 'legacy'; args.epochs = 1
    trainer.train(args)
    index_path = tmp_path/'out/model_epoch_1.json'
    index = json.loads(index_path.read_text())
    del index['science_args']['decode_mode']
    for row in index['components']:
        path = index_path.parent/row['path']
        state = torch.load(path, weights_only=False)
        del state['parallel_resume']['decode_mode']
        torch.save(state, path)
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    historical = tmp_path/'out/historical.json'
    historical.write_text(json.dumps(index))
    args.resume = str(historical); args.epochs = 2; args.decode_mode = 'auto'
    args.output_trained_model_dir = str(tmp_path/'resumed/model_')
    trainer.train(args)
    actual = json.loads((tmp_path/'resumed/model_epoch_2.json').read_text())
    assert actual['science_args']['decode_mode'] == 'legacy'
