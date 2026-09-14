"""Post-step export, old latent/placement decoding and replay without real weights."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from checkpoint_sampling import effective_latent, global_coordinates, is_refinement
from randomness import capture_rng_state
from structure_io import read_template
from test_b2_inputs import train_fixture
from test_b2b_randomness import sampler, same_rng
from test_foundations import FakeDenoiser


@pytest.mark.parametrize('mode', ['legacy', 'isolated'])
@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('target', ['z_trunk', 'pair_z'])
def test_post_step_checkpoint_export_and_sampling_replay(tmp_path, monkeypatch, sampler, target, fixed, mode):
    import train
    import get_pdb
    class Analytic(FakeDenoiser):
        def forward(self, x_noisy, pair_z, z_trunk, **kwargs):
            pair = pair_z if z_trunk is None else z_trunk
            return x_noisy * (.25 + .05 * pair.mean()) + pair.square().mean()
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = Analytic
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)
    def sample(configs, training=False, **kwargs):
        return sampler(configs=configs, **configs.sample_diffusion, **kwargs)
    monkeypatch.setattr(train, '_sample_diffusion', sample)
    monkeypatch.setattr(get_pdb, '_sample_diffusion', sample)
    args, cache = train_fixture(tmp_path)
    if target == 'z_trunk':
        cache['z_trunk'], cache['pair_z'] = cache['pair_z'], None
    torch.save(cache, args.diffusion_data_dir)
    args.epochs, args.max_steps = 1, 2
    args.train_deterministic, args.rng_mode = fixed, mode
    train.main(args)
    path = tmp_path / 'run_1.pth'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert is_refinement(saved) and saved['training_progress'] == dict(epoch_completed=1, global_step=2, coordinate_step=2)
    assert saved['z_bias'].abs().sum() > 0
    settings = dict(seed=42, diffusion_seed=42, rng_mode=mode)
    raw = get_pdb._decode_aligned_global(saved, torch.device('cpu'), settings)
    torch.testing.assert_close(raw, saved['pred_dict']['coordinate'][0], rtol=0, atol=0)
    torch.testing.assert_close(read_template(tmp_path / 'run_1.cif')[0], global_coordinates(saved, raw), atol=1e-6, rtol=0)
    for frame in ('reference', 'raw'):
        get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / frame),
                     cif_path=args.cif_path, pdbid='sample', output_format='cif', coordinate_frame=frame))
        actual = read_template(tmp_path / frame / 'sample_refined_prediction.cif')[0]
        torch.testing.assert_close(actual, global_coordinates(saved, raw, frame), atol=1e-6, rtol=0)
    # Simulated historical file: no new header or sampling snapshot. Nonzero multiplicative/additive bias.
    legacy = copy.deepcopy(cache)
    legacy.update(z_mul=torch.tensor(1.3), z_bias=saved['z_bias'], rotation=saved['rotation'], translation=saved['translation'])
    z, pair = effective_latent(legacy, legacy['z_trunk'], legacy['pair_z'])
    explicit = copy.deepcopy(legacy)
    explicit.update(z_trunk=z, pair_z=pair, z_mul=None, z_bias=None)
    torch.testing.assert_close(get_pdb._decode_aligned_global(legacy, torch.device('cpu'), settings),
                               get_pdb._decode_aligned_global(explicit, torch.device('cpu'), settings), atol=0, rtol=0)
    legacy_path = tmp_path / 'legacy.pth'
    torch.save(legacy, legacy_path)
    get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(legacy_path), out_dir=str(tmp_path / 'legacy'),
                 cif_path=args.cif_path, pdbid='sample', output_format='cif'))
    expected_raw = get_pdb._decode_aligned_global(explicit, torch.device('cpu'), settings)
    torch.testing.assert_close(read_template(tmp_path / 'legacy/sample_refined_prediction.cif')[0],
                               global_coordinates(legacy, expected_raw), atol=1e-6, rtol=0)


def test_checkpoint_classification_and_missing_placement():
    raw = torch.arange(9).reshape(3, 3).float()
    assert not is_refinement(dict(z_bias=None, z_mul=None))
    assert global_coordinates({}, raw) is raw
    with pytest.raises(ValueError, match='schema'):
        is_refinement(dict(checkpoint_schema={'version': 999}))
    cache = dict(z_bias=torch.tensor(.1))
    with pytest.raises(ValueError, match='lacks saved'):
        global_coordinates(cache, raw)
    assert global_coordinates(cache, raw, 'raw') is raw
    cache.update(rotation=torch.eye(3), translation=torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(global_coordinates(cache, raw), raw + torch.tensor([1., 2., 3.]))
    with pytest.raises(ValueError, match='shape'):
        effective_latent(dict(z_bias=torch.ones(2, 3, 3)), None, torch.ones(3, 3))


@pytest.mark.parametrize('mode', ['legacy', 'isolated'])
def test_export_rng_exception_restores_both_streams(mode):
    from checkpoint_sampling import preserve_sampling, sampling_snapshot, replay_sampling
    from randomness import DiffusionRNGStream
    config = SimpleNamespace(train_deterministic=False, rng_mode=mode, diffusion_seed=19,
                             diffusion_rng_stream=DiffusionRNGStream(19))
    with config.diffusion_rng_stream.use():
        torch.rand(3)
    before = capture_rng_state()
    stream_before = copy.deepcopy(config.diffusion_rng_stream.state)
    snapshot = sampling_snapshot(config, 'cpu')
    with pytest.raises(RuntimeError), preserve_sampling(config):
        torch.rand(4)
        with config.diffusion_rng_stream.use():
            torch.rand(4)
        raise RuntimeError('decode failure')
    same_rng(before, capture_rng_state())
    same_rng(stream_before, config.diffusion_rng_stream.state)
    with pytest.raises(RuntimeError), replay_sampling(config, snapshot, 'cpu'):
        torch.rand(3)
        raise RuntimeError('replay failure')
    same_rng(before, capture_rng_state())
    same_rng(stream_before, config.diffusion_rng_stream.state)
