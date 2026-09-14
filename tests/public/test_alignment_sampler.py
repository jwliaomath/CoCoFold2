"""Alignment does not select a different decoder for new chain refinement."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from refinement_runtime import alignment_sampler, apply_saved_global_bias
from coordinate_transform import CoordinateTransform
from prepare_block_alignment import build_chains
from structure_io import read_template, write_coordinates
from test_b2_inputs import train_fixture
from test_b2b_randomness import sampler, same_rng
from test_foundations import FakeDenoiser
from test_b4b_alignment import four_chains, rotations


def test_decoder_selection_and_checkpoint_handoff(tmp_path):
    manifest = tmp_path / 'chains.json'
    manifest.write_text(json.dumps({'grouping': 'auth_asym_id'}), encoding='utf-8')
    args = SimpleNamespace(coordinate_mode='blocks', block_alignment=str(manifest))
    assert alignment_sampler(args, {}) == 'global'
    assert alignment_sampler(args, {'coordinate_transform': {}}) == 'block'
    assert alignment_sampler(args, {'coordinate_transform': {}, 'refinement_sampler': 'global'}) == 'global'
    args.alignment_sampler = 'block'
    with pytest.raises(ValueError, match='cannot switch'):
        alignment_sampler(args, {'coordinate_transform': {}, 'refinement_sampler': 'global'})
    with pytest.raises(ValueError, match='Unknown'):
        alignment_sampler(args, {'coordinate_transform': {}, 'refinement_sampler': 'wrong'})
    assert alignment_sampler(args, {}) == 'block'


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_bias_application_only_for_new_marked_checkpoint(target):
    z = torch.ones(2, 2, 3) if target == 'z_trunk' else None
    pair = torch.ones(2, 2, 3) * 9
    cache = dict(coordinate_transform={}, refinement_sampler='global', z_mul=torch.tensor(2.), z_bias=torch.tensor(.5))
    a, b = apply_saved_global_bias(cache, z, pair)
    torch.testing.assert_close(a if z is not None else b, (z if z is not None else pair) * 2 + .5)
    if z is not None:
        assert b is pair
    del cache['refinement_sampler']
    a, b = apply_saved_global_bias(cache, z, pair)
    assert a is z and b is pair


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
@pytest.mark.parametrize('update', [False, True])
@pytest.mark.parametrize('rng_mode', ['legacy', 'isolated'])
def test_new_chain_preserves_global_inputs_and_exports_saved_bias(tmp_path, monkeypatch, sampler, target, update, rng_mode):
    import train
    import get_pdb
    import single_structure_decoder
    from randomness import capture_rng_state
    created, calls = [], []
    class AnalyticDenoiser(FakeDenoiser):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(self)
        def forward(self, x_noisy, pair_z, z_trunk, p_lm, c_l, **kwargs):
            value = z_trunk if z_trunk is not None else pair_z
            return x_noisy * (.25 + .05 * value.mean()) + value.square().mean() + .01 * (p_lm.mean() + c_l.mean())
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = AnalyticDenoiser
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)
    monkeypatch.setattr(single_structure_decoder, 'load_protenix', lambda *a, **k: pytest.fail('old block decoder was loaded'))
    def sample(configs, training=False, **kwargs):
        out = sampler(configs=configs, **configs.sample_diffusion, **kwargs)
        calls.append(dict(shapes=(kwargs['s_inputs'].shape, kwargs['pair_z'].shape),
                          dtype=kwargs['s_inputs'].dtype, fusion=kwargs['enable_efficient_fusion'],
                          p_lm=kwargs['p_lm'].clone(), c_l=kwargs['c_l'].clone(),
                          raw=out.detach().clone(), rng=capture_rng_state()))
        return out
    monkeypatch.setattr(train, '_sample_diffusion', sample)
    monkeypatch.setattr(get_pdb, '_sample_diffusion', sample)
    args, cache = train_fixture(tmp_path)
    template = four_chains(tmp_path)
    base = torch.arange(48).float().reshape(4, 4, 3) / 80
    cache.update(pair_z=base.clone(), z_trunk=base.clone() if target == 'z_trunk' else None,
                 s_inputs=torch.zeros(4, 3), s_trunk=torch.zeros(4, 3),
                 p_lm=torch.full((2, 3), .7), c_l=torch.full((2, 3), .2),
                 enable_efficient_fusion=True, z_bias=None, z_mul=None)
    cache['input_feature_dict'] = dict(atom_to_token_idx=torch.arange(4).repeat_interleave(4), relp=torch.zeros(4, 4))
    torch.save(cache, args.diffusion_data_dir)
    config = cache['configs']
    config.train_deterministic = True
    raw = sampler(configs=config, **config.sample_diffusion, denoise_net=AnalyticDenoiser(),
                    input_feature_dict=cache['input_feature_dict'], s_inputs=cache['s_inputs'],
                    s_trunk=cache['s_trunk'], z_trunk=cache['z_trunk'], pair_z=cache['pair_z'],
                    p_lm=cache['p_lm'], c_l=cache['c_l'], N_sample=1, noise_schedule=cache['noise_schedule'])[0].detach()
    placed = torch.einsum('ni,nij->nj', raw, rotations()[torch.arange(4).repeat_interleave(4)]) + 3
    write_coordinates(template, template, placed.numpy())
    manifest = tmp_path / 'chains.json'
    build_chains(template, manifest)
    args.cif_path, args.block_alignment, args.coordinate_mode = str(template), str(manifest), 'blocks'
    args.update_affine_mat, args.epochs, args.max_steps, args.learn_gmm = update, 1, 2, False
    args.rng_mode = rng_mode
    created.clear()
    train.main(args)
    assert len(created) == 1
    # The export-only extra forward must leave training RNG untouched.
    same_rng(calls[-2]['rng'], capture_rng_state())
    for call in calls:
        assert call['shapes'] == (torch.Size([4, 3]), torch.Size([4, 4, 3]))
        assert call['dtype'] == cache['s_inputs'].dtype and call['fusion'] is True
        assert torch.equal(call['p_lm'], cache['p_lm']) and torch.equal(call['c_l'], cache['c_l'])
    torch.testing.assert_close(calls[0]['raw'][0], raw, rtol=0, atol=0)
    checkpoint = tmp_path / 'run_1.pth'
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['refinement_sampler'] == 'global'
    assert saved['z_bias'].shape == (4, 4, 3) and saved['z_bias'].abs().sum() > 1e-10
    assert len(saved['opt_state']['param_groups']) == 1
    geometry = CoordinateTransform.from_checkpoint(saved['coordinate_transform'], template, 'cpu')
    actual = read_template(tmp_path / 'run_1.cif')[0]
    torch.testing.assert_close(actual, geometry(saved['pred_dict']['coordinate'][0]), rtol=0, atol=1e-6)
    get_pdb.main(SimpleNamespace(device='cpu', diffusion_data_dir=str(checkpoint), out_dir=str(tmp_path / 'export'),
                                 cif_path=str(template), pdbid='new', output_format='cif'))
    torch.testing.assert_close(read_template(tmp_path / 'export/new_block_prediction.cif')[0], actual, atol=1e-6, rtol=0)
    # Warm initialization restores the saved effective latent and the original decoder.
    args.diffusion_data_dir, args.block_alignment = str(checkpoint), None
    args.output_trained_model_dir = str(tmp_path / 'warm_')
    before = len(calls)
    train.main(args)
    torch.testing.assert_close(calls[before]['raw'], saved['pred_dict']['coordinate'], atol=0, rtol=0)
    assert len(created) == 3  # first train, export, warm initialization
