"""CPU checks for public single-structure decoding, I/O and block export.

No private research modules, trained weights or network access are required.
"""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import gemmi
import numpy as np
import pytest
import torch

from coordinate_transform import CoordinateTransform, template_atom_keys
from single_structure_decoder import SingleStructureDecoder, load_protenix
from structure_io import read_template, write_coordinates


class FakeDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.25))
        self.diffusion_conditioning = SimpleNamespace(
            prepare_cache=lambda relp, z, inplace: z + .125)

    def forward(self, x_noisy, pair_z, **kwargs):
        signal = pair_z.square().mean(dim=(-3, -2, -1)).reshape(1, 1, 1, 1)
        return x_noisy * self.scale + signal


def make_cache(target='pair_z'):
    base = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 20
    return dict(
        configs=SimpleNamespace(model=SimpleNamespace(diffusion_module={}),
                                sample_diffusion=dict(gamma0=.2, gamma_min=.1,
                                                      noise_scale_lambda=1., step_scale_eta=1.),
                                infer_setting=SimpleNamespace(chunk_size=None)),
        model_state=FakeDenoiser().state_dict(),
        input_feature_dict=dict(atom_to_token_idx=torch.tensor([0, 0, 0, 1, 1, 1]), relp=torch.zeros(2, 2)),
        s_inputs=torch.zeros(2, 3), s_trunk=torch.zeros(2, 3),
        noise_schedule=torch.tensor([2., 1., 0.]),
        pair_z=base.clone() if target == 'pair_z' else torch.full_like(base, 99),
        z_trunk=base.clone() if target == 'z_trunk' else None,
        z_mul=torch.tensor(1.5), z_bias=torch.full_like(base, .3))


@pytest.fixture
def fake_protenix(monkeypatch):
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = FakeDenoiser
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)


def write_fixture(folder, suffix='.cif'):
    structure = gemmi.Structure()
    structure.name = 'test'
    model = gemmi.Model('1')
    for chain_index, name in enumerate(('A', 'B')):
        chain = gemmi.Chain(name)
        for index, point in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0))):
            residue = gemmi.Residue()
            residue.name = 'ALA'
            residue.seqid = gemmi.SeqId(index + 1, ' ')
            atom = gemmi.Atom()
            atom.name = 'CA'
            atom.element = gemmi.Element('C')
            atom.pos = gemmi.Position(point[0] + 3 * chain_index, point[1], point[2])
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.assign_label_seq_id()
    path = folder / ('template' + suffix)
    if suffix == '.cif':
        structure.make_mmcif_document().write_file(str(path))
    else:
        structure.write_pdb(str(path))
    return path


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_saved_bias_is_applied_once(tmp_path, fake_protenix, target):
    cache = make_cache(target)
    path = tmp_path / 'cache.pt'
    torch.save(cache, path)
    decoder, effective, meta = load_protenix(path, 'cpu')
    expected = cache[target] * cache['z_mul'] + cache['z_bias']
    if target == 'z_trunk':
        expected = expected + .125
    torch.testing.assert_close(effective, expected, rtol=0, atol=0)
    assert meta['original_pair_target'] == target
    pair = effective[None].clone().requires_grad_(True)
    coordinates = decoder(pair)
    coordinates.square().sum().backward()
    assert coordinates.shape == (1, 6, 3)
    assert torch.isfinite(pair.grad).all() and pair.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in decoder.module.parameters())
    decoder.train()
    assert not decoder.training and not decoder.module.training


def decoder_fixture(seed=42, checkpoint_steps=False):
    cache = make_cache()
    return SingleStructureDecoder(
        FakeDenoiser(), cache['input_feature_dict'], cache['s_inputs'], cache['s_trunk'],
        cache['noise_schedule'], cache['configs'].sample_diffusion,
        seed=seed, checkpoint_steps=checkpoint_steps), cache['pair_z'][None]


def test_fixed_noise_and_global_rng_unchanged():
    decoder, pair = decoder_fixture()
    rng = torch.get_rng_state().clone()
    first = decoder(pair)
    torch.testing.assert_close(first, decoder(pair), rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    different, _ = decoder_fixture(seed=7)
    assert not torch.equal(first, different(pair))


def test_step_checkpoint_preserves_output_and_pair_gradient():
    outputs, grads = [], []
    for enabled in (False, True):
        decoder, pair = decoder_fixture(checkpoint_steps=enabled)
        pair = pair.clone().requires_grad_(True)
        output = decoder(pair)
        outputs.append(output)
        grads.append(torch.autograd.grad(output.square().sum(), pair)[0])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(2, 2, 2, 3), (2, 2, 3), (0, 2, 2, 3)])
def test_decoder_rejects_batch_or_wrong_rank(shape):
    decoder, _ = decoder_fixture()
    with pytest.raises(ValueError, match='Single-structure'):
        decoder(torch.zeros(shape))


@pytest.mark.parametrize('schedule', [[2., 1., .1], [1., 2., 0.], [1., float('nan'), 0.]])
def test_invalid_schedule(schedule):
    decoder, pair = decoder_fixture()
    decoder.schedule = torch.tensor(schedule)
    with pytest.raises(ValueError, match='schedule|Schedule'):
        decoder(pair)


@pytest.mark.parametrize('suffix', ['.pdb', '.cif'])
def test_structure_roundtrip(tmp_path, suffix):
    template = write_fixture(tmp_path, suffix)
    xyz, weights, labels = read_template(template)
    moved = xyz + torch.tensor([1.25, -2., .5])
    output = tmp_path / ('moved' + suffix)
    write_coordinates(template, output, moved.numpy())
    reread, reweights, relabels = read_template(output)
    torch.testing.assert_close(reread, moved, rtol=0, atol=1e-6)
    assert labels == relabels
    assert torch.equal(weights, reweights)
    assert torch.equal(weights, torch.full((6,), 6.))
    assert template_atom_keys(template) == template_atom_keys(output)


def test_structure_rejects_nonfinite_and_wrong_count(tmp_path):
    template = write_fixture(tmp_path)
    with pytest.raises(ValueError, match='finite'):
        write_coordinates(template, tmp_path / 'bad.cif', np.full((6, 3), np.nan))
    with pytest.raises(ValueError, match='count'):
        write_coordinates(template, tmp_path / 'bad.cif', np.zeros((5, 3)))


def test_public_block_fit_gradient_and_checkpoint(tmp_path):
    template = write_fixture(tmp_path)
    raw, _, _ = read_template(template)
    rotations = torch.stack((torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
                             torch.eye(3)))
    translations = torch.tensor([[2., 1., -.5], [-1., 3., 2.]])
    ids = torch.tensor([0, 0, 0, 1, 1, 1])
    expected = torch.einsum('ni,nij->nj', raw, rotations[ids]) + translations[ids]
    keys = template_atom_keys(template)
    manifest = dict(format='cocofold2-block-manifest-v1', body_names=['A', 'B'],
                    atoms=[dict(key=key, body_id=int(ids[i]), target=expected[i].tolist(), fit_core=True)
                           for i, key in enumerate(keys)])
    path = tmp_path / 'alignment.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    transform = CoordinateTransform.fit_manifest(path, raw, template)
    trainable = raw.clone().requires_grad_(True)
    placed = transform(trainable)
    torch.testing.assert_close(placed, expected, rtol=0, atol=1e-6)
    placed.square().sum().backward()
    analytic = 2 * torch.einsum('ni,nji->nj', expected, rotations[ids])
    torch.testing.assert_close(trainable.grad, analytic, rtol=1e-6, atol=1e-6)
    restored = CoordinateTransform.from_checkpoint(transform.export_checkpoint(raw), template, 'cpu')
    assert restored.check_handoff(raw, tolerance=1e-6) == 0
    torch.testing.assert_close(restored(raw), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_real_get_pdb_block_export(tmp_path, fake_protenix, target):
    template = write_fixture(tmp_path)
    cache = make_cache(target)
    path = tmp_path / 'refined.pt'
    torch.save(cache, path)
    decoder, pair, _ = load_protenix(path, 'cpu')
    raw = decoder(pair[None])[0].detach()
    rotations = torch.eye(3).repeat(2, 1, 1)
    translations = torch.tensor([[1., 2., 3.], [-2., 1., .5]])
    transform = CoordinateTransform(rotations, translations, torch.tensor([0, 0, 0, 1, 1, 1]),
                                    template_atom_keys(template), ['A', 'B'], raw)
    cache['coordinate_transform'] = transform.export_checkpoint(raw)
    torch.save(cache, path)
    from get_pdb import main
    main(SimpleNamespace(device='cpu', diffusion_data_dir=str(path), out_dir=str(tmp_path / 'out'),
                         cif_path=str(template), pdbid='test', output_format=None))
    exported, _, _ = read_template(tmp_path / 'out/test_block_prediction.cif')
    torch.testing.assert_close(exported, transform(raw), atol=1e-6, rtol=0)


@pytest.mark.parametrize('script', ['get_pdb.py', 'prepare_block_alignment.py'])
def test_lightweight_cli_help(script):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / 'src' / script), '--help'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_public_modules_import_without_private_modules():
    src = Path(__file__).resolve().parents[2] / 'src'
    code = (f'import sys; sys.path.insert(0, {str(src)!r}); '
            'import structure_io, single_structure_decoder, coordinate_transform, '
            'cache_structure, get_pdb, prepare_block_alignment; '
            'assert not any(n == "hetero" or n.startswith("hetero.") for n in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
