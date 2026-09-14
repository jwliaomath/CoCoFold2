"""Seed/RNG tests use the real public sampler with a small analytic denoiser.

Only Protenix's augmentation dependency is substituted on machines without it.
The server runner additionally tests the installed Protenix implementation.
"""
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from randomness import (DiffusionRNGStream, apply_seed_settings, capture_rng_state,
                        data_loader_options, resolve_seeds, restore_rng_state,
                        seed_legacy, seed_value)
from test_foundations import fake_protenix


def same_rng(a, b):
    assert a['python'] == b['python']
    assert a['numpy'][0] == b['numpy'][0]
    np.testing.assert_array_equal(a['numpy'][1], b['numpy'][1])
    assert a['numpy'][2:] == b['numpy'][2:]
    assert torch.equal(a['torch'], b['torch'])
    if a['cuda'] is not None:
        assert len(a['cuda']) == len(b['cuda'])
        for x, y in zip(a['cuda'], b['cuda']):
            assert torch.equal(x, y)


@pytest.fixture
def sampler(monkeypatch):
    if not os.environ.get('COCOFOLD2_REAL_AUGMENTATION'):
        mock = ModuleType('protenix.model.utils')
        def augmentation(x_input_coords, N_sample):
            return (x_input_coords + torch.rand_like(x_input_coords) +
                    float(np.random.rand()) + random.random()).unsqueeze(-3)
        mock.centre_random_augmentation = augmentation
        monkeypatch.setitem(sys.modules, 'protenix.model.utils', mock)
    path = Path(__file__).resolve().parents[2] / 'src/model/generator.py'
    spec = importlib.util.spec_from_file_location('b2b_sampler_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_diffusion


def inputs(mode='legacy', seed=42, fixed=True, device='cpu', chunks=None):
    pair = torch.arange(12, device=device, dtype=torch.float32).reshape(1, 2, 2, 3) / 20
    pair.requires_grad_(True)
    def denoise(x_noisy, pair_z, **kwargs):
        return .25 * x_noisy + pair_z.square().mean()
    return dict(configs=SimpleNamespace(train_deterministic=fixed, rng_mode=mode, diffusion_seed=seed),
                denoise_net=denoise, input_feature_dict={'atom_to_token_idx': torch.tensor([0, 0, 1, 1], device=device)},
                s_inputs=torch.zeros(1, 2, 3, device=device), s_trunk=None, z_trunk=None,
                pair_z=pair, p_lm=None, c_l=None, noise_schedule=torch.tensor([2., 1., 0.], device=device),
                N_sample=3 if chunks else 1, diffusion_chunk_size=chunks)


@pytest.mark.parametrize('mode', ['legacy', 'isolated'])
@pytest.mark.parametrize('chunks', [None, 2])
def test_fixed_seed_changes_noise_and_preserves_gradients(sampler, mode, chunks):
    kw = inputs(mode=mode, chunks=chunks)
    first, second = sampler(**kw), sampler(**kw)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.square().mean().backward()
    assert torch.isfinite(kw['pair_z'].grad).all() and kw['pair_z'].grad.abs().sum() > 0
    kw['configs'].diffusion_seed = 7
    assert not torch.equal(first, sampler(**kw))


@pytest.mark.parametrize('fixed', [True, False])
def test_isolated_sampler_preserves_all_caller_rngs(sampler, fixed):
    before = capture_rng_state()
    sampler(**inputs(mode='isolated', fixed=fixed))
    same_rng(before, capture_rng_state())


def test_isolated_exception_restores_caller_rngs(sampler):
    kw = inputs(mode='isolated')
    def fail(**kwargs):
        raise RuntimeError('intentional denoiser failure')
    kw['denoise_net'] = fail
    before = capture_rng_state()
    with pytest.raises(RuntimeError, match='intentional'):
        sampler(**kw)
    same_rng(before, capture_rng_state())


def test_resampled_stream_advances_and_replays(sampler):
    kw = inputs(mode='isolated', fixed=False)
    sequence = [sampler(**kw).detach() for _ in range(3)]
    assert not torch.equal(sequence[0], sequence[1])
    replay = inputs(mode='isolated', fixed=False)
    for expected in sequence:
        torch.testing.assert_close(expected, sampler(**replay), rtol=0, atol=0)
    snapshot = copy.deepcopy(kw['configs'].diffusion_rng_stream)
    expected = sampler(**kw)
    kw['configs'].diffusion_rng_stream = snapshot
    torch.testing.assert_close(expected, sampler(**kw), rtol=0, atol=0)


def test_stream_serialization_replays_next_sample(tmp_path, sampler):
    kw = inputs(mode='isolated', fixed=False)
    sampler(**kw)
    path = tmp_path / 'stream.pt'
    torch.save(kw['configs'], path)
    expected = sampler(**kw)
    kw['configs'] = torch.load(path, map_location='cpu', weights_only=False)
    torch.testing.assert_close(expected, sampler(**kw), rtol=0, atol=0)


def test_global_export_uses_recorded_noise_seed(tmp_path, monkeypatch, sampler):
    import get_pdb
    from test_foundations import make_cache, write_fixture
    from structure_io import read_template
    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.25))
        def forward(self, x_noisy, pair_z, **kwargs):
            return x_noisy * self.scale + pair_z.square().mean()
    mock = ModuleType('protenix.model.modules.diffusion')
    mock.DiffusionModule = Denoiser
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', mock)
    cache = make_cache()
    cache.update(pred_dict={}, p_lm=None, c_l=None, N_sample=1, inplace_safe=False,
                 enable_efficient_fusion=False, refinement_seed_settings={'diffusion_seed': 19},
                 z_mul=None, z_bias=None)  # Initial cache; refined export is covered by B5 tests.
    path = tmp_path / 'cache.pt'
    torch.save(cache, path)
    captured = []
    def sample(configs, training=False, **kw):
        assert configs.diffusion_seed == 19
        value = sampler(configs=configs, **configs.sample_diffusion, **kw)
        captured.append(value.detach())
        return value
    monkeypatch.setattr(get_pdb, '_sample_diffusion', sample)
    args = SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / 'out'),
                           cif_path=str(write_fixture(tmp_path)), pdbid='test', output_format='cif')
    get_pdb.main(args)
    exported, _, _ = read_template(tmp_path / 'out/test_initial_prediction.cif')
    torch.testing.assert_close(exported, captured[0][0], atol=1e-6, rtol=0)


def test_isolation_keeps_legacy_fixed_coordinates(sampler):
    legacy, isolated = inputs(), inputs(mode='isolated')
    a, b = sampler(**legacy), sampler(**isolated)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(legacy['pair_z'].grad, isolated['pair_z'].grad, rtol=0, atol=0)


def test_data_order_independent_of_sampling_and_diffusion_seed(sampler):
    def order(data_seed, diffusion_seed, calls):
        settings = resolve_seeds(SimpleNamespace(seed=42, rng_mode='isolated', data_seed=data_seed))
        loader = DataLoader(torch.arange(21), batch_size=3, shuffle=True, **data_loader_options(settings))
        rows = []
        for _ in range(2):
            for batch in loader:
                rows.extend(batch.tolist())
                for _ in range(calls):
                    sampler(**inputs(mode='isolated', seed=diffusion_seed))
        return rows
    assert order(19, 42, 0) == order(19, 7, 2)
    assert order(19, 42, 0) != order(20, 42, 0)
    assert data_loader_options(resolve_seeds(SimpleNamespace())) == {}


@pytest.mark.parametrize('bad', [-1, 2**32, True, 1.5, 'abc'])
def test_seed_range(bad):
    with pytest.raises(ValueError, match='Seed'):
        seed_value(bad)


def test_seed_resolution_and_legacy_cache_fallback():
    from get_pdb import build_parser
    args = build_parser().parse_args(['--pdbid', 'x', '--diffusion_data_dir', 'x', '--out_dir', 'x'])
    cache = {'configs': SimpleNamespace(train_seed=101, seeds=[101])}
    assert resolve_seeds(args, 'export', cache)['diffusion_seed'] == 42
    cache['refinement_seed_settings'] = {'diffusion_seed': 19}
    assert resolve_seeds(args, 'export', cache)['diffusion_seed'] == 19
    args.seed = 7
    assert resolve_seeds(args, 'export', cache)['diffusion_seed'] == 7
    train = resolve_seeds(SimpleNamespace(seed=19, diffusion_seed=7, rng_mode='isolated', data_seed=2))
    assert (train['seed'], train['diffusion_seed'], train['data_seed']) == (19, 7, 2)
    cfg = SimpleNamespace(seeds=[101], train_seed=101)
    apply_seed_settings(cfg, train)
    assert cfg.seeds == [101] and cfg.train_seed == 101  # preserve original cache provenance


def test_seed_conflicts_fail_before_file_or_model_loading(monkeypatch):
    import train
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('cache was loaded'))
    args = SimpleNamespace(data_seed=4, rng_mode='legacy')
    with pytest.raises(ValueError, match='data-seed'):
        train.main(args)
    args = SimpleNamespace(diffusion_seed=4, train_deterministic=False, rng_mode='legacy')
    with pytest.raises(ValueError, match='resampled noise'):
        train.main(args)


def test_inference_prediction_seeds_remain_separate():
    from inference import build_parser
    args, rest = build_parser().parse_known_args(['--input_json_path', 'x', '--diffusion-seed', '7', '--seeds', '101'])
    assert rest == ['--seeds', '101']
    assert resolve_seeds(args, 'inference')['diffusion_seed'] == 7
    assert resolve_seeds(args, 'inference')['data_rng'] == 'prediction_pipeline_unchanged'


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_export_inherits_recorded_seed_and_rejects_wrong_block_handoff(tmp_path, fake_protenix, target):
    from test_foundations import make_cache, write_fixture
    from single_structure_decoder import load_protenix
    from coordinate_transform import CoordinateTransform, template_atom_keys
    from structure_io import read_template
    from get_pdb import main
    template = write_fixture(tmp_path)
    cache = make_cache(target)
    cache['refinement_seed_settings'] = {'diffusion_seed': 19}
    path = tmp_path / 'cache.pt'
    torch.save(cache, path)
    decoder, pair, _ = load_protenix(path, 'cpu', seed=19)
    raw = decoder(pair[None])[0].detach()
    transform = CoordinateTransform(torch.eye(3)[None], torch.tensor([[1., 2., 3.]]),
                                    torch.zeros(6, dtype=torch.long), template_atom_keys(template), ['all'], raw)
    cache['coordinate_transform'] = transform.export_checkpoint(raw)
    torch.save(cache, path)
    args = SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / 'out'),
                           cif_path=str(template), pdbid='test', output_format=None)
    main(args)
    exported, _, _ = read_template(tmp_path / 'out/test_block_prediction.cif')
    torch.testing.assert_close(exported, transform(raw), atol=1e-6, rtol=0)
    args.seed = 7
    args.out_dir = str(tmp_path / 'wrong_seed')
    with pytest.raises(ValueError, match='handoff|coordinates|mismatch|drift'):
        main(args)
    # B3 records the failed invocation, but must not export an invalid structure.
    assert not (Path(args.out_dir) / "test_block_prediction.cif").exists()


def test_block_train_passes_seed_to_decoder_and_checkpoint(tmp_path, fake_protenix):
    import json
    import train
    from test_b2_inputs import train_fixture
    from single_structure_decoder import load_protenix
    from coordinate_transform import template_atom_keys
    args, cache = train_fixture(tmp_path)
    args.seed, args.diffusion_seed, args.data_seed, args.rng_mode = 19, 7, 3, 'isolated'
    decoder, pair, _ = load_protenix(args.diffusion_data_dir, 'cpu', seed=7)
    raw = decoder(pair[None])[0].detach()
    manifest = dict(format='cocofold2-block-manifest-v1', body_names=['all'], atoms=[
        dict(key=key, body_id=0, target=(raw[i] + 2).tolist(), fit_core=True)
        for i, key in enumerate(template_atom_keys(args.cif_path))])
    alignment = tmp_path / 'alignment.json'
    alignment.write_text(json.dumps(manifest), encoding='utf-8')
    args.coordinate_mode, args.block_alignment = 'blocks', str(alignment)
    args.epochs, args.max_steps = 1, 2
    train.main(args)
    saved = torch.load(tmp_path / 'run_1.pth', weights_only=False)
    assert saved['refinement_seed_settings'] == resolve_seeds(args)
    assert saved['coordinate_sampler'].endswith('seed7')
    assert torch.isfinite(saved['z_bias']).all() and saved['z_bias'].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA-specific RNG isolation')
def test_cuda_rng_isolation(sampler):
    kw = inputs(mode='isolated', device='cuda')
    before = capture_rng_state()
    sampler(**kw).sum().backward()
    same_rng(before, capture_rng_state())
