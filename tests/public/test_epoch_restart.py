"""Completed-epoch resume against uninterrupted real optimizer/GMM training."""
import copy
import json
import sys
from types import ModuleType

import pytest
import torch

from randomness import capture_rng_state, restore_rng_state
from test_b2b_randomness import sampler, same_rng
from test_b2_inputs import train_fixture
from test_foundations import FakeDenoiser
from structure_io import read_template


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def install(monkeypatch, sampler):
    import train
    class Analytic(FakeDenoiser):
        def forward(self, x_noisy, pair_z, z_trunk=None, **kwargs):
            pair = pair_z if z_trunk is None else z_trunk
            return x_noisy * (.25 + .05 * pair.mean()) + pair.square().mean()
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = Analytic
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)
    monkeypatch.setattr(train, '_sample_diffusion', lambda configs, training=False, **kw:
                        sampler(configs=configs, **configs.sample_diffusion, **kw))


def fixture(tmp_path, target='pair_z', geometry='global'):
    args, cache = train_fixture(tmp_path)
    cache['z_mul'], cache['z_bias'] = None, None
    if target == 'z_trunk':
        cache['z_trunk'], cache['pair_z'] = cache['pair_z'], None
    torch.save(cache, args.diffusion_data_dir)
    if geometry != 'global':
        from prepare_block_alignment import build_chains
        build_chains(args.cif_path, tmp_path / 'chains.json')
        args.coordinate_mode, args.block_alignment = 'blocks', str(tmp_path / 'chains.json')
        args.alignment_sampler = 'global' if geometry == 'chain' else 'block'
    return args, cache


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
@pytest.mark.parametrize('mode', ['legacy', 'isolated'])
@pytest.mark.parametrize('case', ['global_fixed', 'global_resampled', 'chain', 'block'])
def test_completed_epoch_resume_matches_continuous(tmp_path, monkeypatch, sampler, target, mode, case, device='cpu', gmm_kernel='legacy', boundary_step_limit=False, width_mode='legacy'):
    import train
    install(monkeypatch, sampler)
    geometry = case if case in ('chain', 'block') else 'global'
    args, _ = fixture(tmp_path, target, geometry)
    args.device = device
    args.gmm_kernel = gmm_kernel
    args.gmm_sdev_init_mode = width_mode
    args.gmm_molmap_resolution_A = 2.0 if width_mode == 'molmap' else None
    orders = []
    class ObservedLoader(train.DataLoader):
        def __iter__(self):
            order = []
            orders.append(order)
            for batch in super().__iter__():
                order.extend(batch[-1].tolist())
                yield batch
    monkeypatch.setattr(train, 'DataLoader', ObservedLoader)
    args.rng_mode, args.train_deterministic = mode, case != 'global_resampled'
    args.update_affine_mat = True
    args.learn_gmm = mode == 'legacy'  # both optimizer group layouts
    args.seed = 19
    args.epochs = 2
    initial_rng = capture_rng_state()
    args.output_trained_model_dir = str(tmp_path / 'continuous_')
    train.main(copy.deepcopy(args))
    uninterrupted = torch.load(tmp_path / 'continuous_2.pth', weights_only=False)
    end_rng = capture_rng_state()
    restore_rng_state(initial_rng)
    split = copy.deepcopy(args)
    split.epochs, split.output_trained_model_dir = 1, str(tmp_path / 'split_')
    if boundary_step_limit:
        split.max_steps = 2
    train.main(split)
    path = tmp_path / 'split_1.pth'
    first_bytes = path.read_bytes()
    # Disturb RNGs between jobs; restored training must not use this state.
    torch.rand(13)
    resumed = copy.deepcopy(args)
    resumed.resume, resumed.block_alignment = True, None
    resumed.diffusion_data_dir = str(path)
    resumed.output_trained_model_dir = str(tmp_path / 'resumed_')
    train.main(resumed)
    actual = torch.load(tmp_path / 'resumed_2.pth', weights_only=False)
    for key in ('z_trunk', 'pair_z', 'z_bias', 'z_mul', 'gmm', 'opt_state', 'rotation', 'translation', 'pred_dict', 'training_progress'):
        equal_tree(uninterrupted[key], actual[key])
    if geometry != 'global':
        for key in ('rotations', 'translations', 'atom_body_id', 'fit_core', 'fit_targets'):
            if key in uninterrupted['coordinate_transform']:
                equal_tree(uninterrupted['coordinate_transform'][key], actual['coordinate_transform'][key])
    same_rng(end_rng, capture_rng_state())
    same_rng(uninterrupted['training_resume']['rng'], actual['training_resume']['rng'])
    equal_tree(uninterrupted['training_resume']['data_generator_state'], actual['training_resume']['data_generator_state'])
    a, b = uninterrupted['training_resume']['diffusion_rng_stream'], actual['training_resume']['diffusion_rng_stream']
    if a is not None:
        same_rng(a.state, b.state)
    assert path.read_bytes() == first_bytes and not (tmp_path / 'resumed_1.pth').exists()
    assert orders[:2] == orders[2:]  # uninterrupted epochs == first run + resumed epoch
    torch.testing.assert_close(read_template(tmp_path / 'continuous_2.cif')[0], read_template(tmp_path / 'resumed_2.cif')[0], atol=0, rtol=0)
    record = next((tmp_path / 'resumed_records').iterdir())
    events = [json.loads(line) for line in (record / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['global_step'] for row in events if row['event'] == 'train_step'] == [3, 4]
    assert any(row['event'] == 'resume' and row['parent_run_id'] for row in events)


def test_molmap_completed_epoch_resume(tmp_path, monkeypatch, sampler):
    test_completed_epoch_resume_matches_continuous(
        tmp_path, monkeypatch, sampler, 'pair_z', 'isolated', 'global_fixed', width_mode='molmap')


def test_pre_width_extension_checkpoint_resumes(tmp_path, monkeypatch, sampler):
    import train
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path)
    args.epochs = 1
    train.main(args)
    path = tmp_path / 'run_1.pth'
    saved = torch.load(path, weights_only=False)
    saved['gmm']['config'].pop('width_initialization', None)
    for key in ('gmm_sdev_init_mode', 'gmm_molmap_resolution_A'):
        saved['training_resume']['science_args'].pop(key, None)
    torch.save(saved, path)
    args.resume, args.epochs = True, 2
    args.diffusion_data_dir = str(path)
    args.output_trained_model_dir = str(tmp_path / 'old_resumed_')
    train.main(args)
    actual = torch.load(tmp_path / 'old_resumed_2.pth', weights_only=False)
    assert actual['training_resume']['epoch_complete']
    assert actual['training_resume']['next_epoch'] == 2
    assert 'width_initialization' not in actual['gmm']['config']


@pytest.mark.parametrize('problem', ['old', 'partial', 'config', 'data', 'target', 'environment'])
def test_resume_rejects_before_model(tmp_path, monkeypatch, sampler, problem):
    import train
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path)
    args.epochs = 1
    if problem == 'partial':
        args.max_steps = 1
    train.main(args)
    path = tmp_path / 'run_1.pth'
    saved = torch.load(path, weights_only=False)
    if problem == 'old':
        del saved['training_resume']
    if problem == 'environment':
        saved['training_resume']['runtime']['torch'] = 'different-version'
    torch.save(saved, path)
    resumed = copy.deepcopy(args)
    resumed.resume, resumed.epochs, resumed.max_steps = True, 2, None
    resumed.diffusion_data_dir, resumed.output_trained_model_dir = str(path), str(tmp_path / 'bad_')
    if problem == 'config':
        resumed.lr_bias = .3
        resumed._explicit_options = ['lr_bias']
    if problem == 'data':
        with open(args.star_data_dir, 'a', encoding='utf-8') as handle:
            handle.write('\n# changed input\n')
    if problem == 'target':
        resumed.epochs = 1
    module = sys.modules['protenix.model.modules.diffusion']
    monkeypatch.setattr(module, 'DiffusionModule', lambda **kw: pytest.fail('Model allocated before preflight'))
    with pytest.raises(ValueError):
        train.main(resumed)
    if problem == 'partial':
        assert saved['training_progress']['epoch_completed'] == 0
        assert saved['training_resume']['epoch_complete'] is False


@pytest.mark.parametrize('keep_state', [False, True])
def test_warm_start_legacy_inherits_or_initializes(tmp_path, monkeypatch, sampler, keep_state):
    import train
    from checkpoint_sampling import effective_latent
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path)
    args.epochs = 1
    train.main(args)
    saved = torch.load(tmp_path / 'run_1.pth', weights_only=False)
    z, pair = effective_latent(saved, saved['z_trunk'], saved['pair_z'])
    if not keep_state:
        for key in ('gmm', 'atom_weights', 'sdevs', 'rotation', 'translation', 'enable_efficient_fusion', 'training_resume'):
            saved.pop(key, None)
    legacy_path = tmp_path / 'legacy.pth'
    torch.save(saved, legacy_path)
    args.warm_start, args.learn_gmm, args.update_affine_mat = True, False, False
    args.diffusion_data_dir, args.output_trained_model_dir = str(legacy_path), str(tmp_path / 'warm_')
    args.lr_bias, args.seed = .002, 7
    train.main(args)
    actual = torch.load(tmp_path / 'warm_1.pth', weights_only=False)
    equal_tree(actual['pair_z'], pair)
    assert actual['training_progress']['global_step'] == 2
    assert {int(v['step']) for v in actual['opt_state']['state'].values()} == {2}
    assert actual['opt_state']['param_groups'][0]['lr'] == .002
    if keep_state:
        equal_tree(actual['gmm']['state_dict'], saved['gmm']['state_dict'])
        equal_tree(actual['rotation'], saved['rotation'])
        equal_tree(actual['translation'], saved['translation'])
    record = next((tmp_path / 'warm_records').iterdir())
    stages = json.loads((record / 'resolved_config.json').read_text(encoding='utf-8'))['stages']
    restart = next(row['values'] for row in stages if row['stage'] == 'restart')
    assert restart['placement_source'] == ('saved_global_transform' if keep_state else 'fitted_from_current_cif')


@pytest.mark.parametrize('failure_epoch', [1, 2])
def test_oom_does_not_save_partial_checkpoint(tmp_path, monkeypatch, sampler, failure_epoch):
    import train
    install(monkeypatch, sampler)
    args, _ = fixture(tmp_path)
    args.epochs = failure_epoch
    original = train._sample_diffusion
    calls = 0
    def fail_during_training(*a, **kw):
        nonlocal calls
        if torch.is_grad_enabled():
            calls += 1
            if calls > 3 * (failure_epoch - 1):
                raise torch.cuda.OutOfMemoryError('simulated OOM')
        return original(*a, **kw)
    monkeypatch.setattr(train, '_sample_diffusion', fail_during_training)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        train.main(args)
    assert not (tmp_path / f'run_{failure_epoch}.pth').exists()
    if failure_epoch == 2:
        saved = torch.load(tmp_path / 'run_1.pth', weights_only=False)
        assert saved['training_resume']['epoch_complete'] is True


def test_resume_in_fresh_process_inherits_defaults(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    args, _ = fixture(tmp_path)
    public = Path(__file__).parent
    command = [sys.executable, str(public / 'restart_worker.py'),
               '--star_data_dir', args.star_data_dir, '--cif_path', args.cif_path,
               '--device', 'cpu', '--boxsize', '24', '--batch_size', '2', '--mini_batch_size', '1',
               '--projection-frame', 'legacy']
    env = dict(os.environ)
    for label, epochs, checkpoint, flags in (
        ('full', 2, args.diffusion_data_dir, ['--rng-mode', 'isolated', '--seed', '19', '--no-train-deterministic', '--no-learn-gmm', '--lr_bias', '.002']),
        ('split', 1, args.diffusion_data_dir, ['--rng-mode', 'isolated', '--seed', '19', '--no-train-deterministic', '--no-learn-gmm', '--lr_bias', '.002']),
        ('resume', 2, str(tmp_path / 'split_1.pth'), ['--resume']),
    ):
        completed = subprocess.run(command + ['--epochs', str(epochs), '--diffusion_data_dir', checkpoint,
                                    '--output_trained_model_dir', str(tmp_path / (label + '_'))] + flags,
                                   env=env, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    continuous = torch.load(tmp_path / 'full_2.pth', weights_only=False)
    resumed = torch.load(tmp_path / 'resume_2.pth', weights_only=False)
    for key in ('z_bias', 'gmm', 'opt_state', 'rotation', 'translation', 'pred_dict', 'refinement_seed_settings'):
        equal_tree(continuous[key], resumed[key])
    assert resumed['gmm_learning_enabled'] is False
    assert resumed['refinement_seed_settings']['seed'] == 19
    assert resumed['opt_state']['param_groups'][0]['lr'] == .002


@pytest.mark.parametrize('kernel', ['isotropic', 'anisotropic'])
def test_covariance_gmm_resume(tmp_path, monkeypatch, sampler, kernel):
    test_completed_epoch_resume_matches_continuous(tmp_path, monkeypatch, sampler,
        'pair_z', 'legacy', 'global_fixed', gmm_kernel=kernel)


def test_step_limit_at_epoch_boundary_can_resume(tmp_path, monkeypatch, sampler):
    test_completed_epoch_resume_matches_continuous(tmp_path, monkeypatch, sampler,
        'pair_z', 'legacy', 'global_resampled', boundary_step_limit=True)
