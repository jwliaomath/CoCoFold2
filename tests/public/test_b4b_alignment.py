"""Per-chain geometry and actual single-cache training with a small analytic denoiser."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import gemmi
import pytest
import torch

from coordinate_transform import (CoordinateTransform, template_atom_keys, read_alignment_manifest,
                                  subset_alignment_manifest, validate_update_threshold)
from prepare_block_alignment import build_chains
from structure_io import read_template, write_coordinates
from test_foundations import FakeDenoiser
from test_b2_inputs import train_fixture


def four_chains(folder):
    structure = gemmi.Structure()
    structure.name = 'four'
    model = gemmi.Model('1')
    for ci, name in enumerate(['AA', 'B', 'CC', 'D']):
        chain = gemmi.Chain(name)
        for i, (atom_name, xyz) in enumerate([('CA', (0, 0, 0)), ('CA', (2, 0, 0)),
                                             ('CA', (0, 3, 0)), ('N', (1, 1, 2))]):
            residue = gemmi.Residue()
            residue.name, residue.seqid = 'ALA', gemmi.SeqId(i + 1, ' ')
            atom = gemmi.Atom()
            atom.name, atom.element = atom_name, gemmi.Element('C' if atom_name == 'CA' else 'N')
            atom.pos = gemmi.Position(xyz[0] + ci * 5, xyz[1], xyz[2])
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.assign_label_seq_id()
    path = folder / 'four.cif'
    structure.make_mmcif_document().write_file(str(path))
    return path


def rotations():
    return torch.tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]],
                         [[0, -1., 0], [1., 0, 0], [0, 0, 1.]],
                         [[1., 0, 0], [0, 0, -1.], [0, 1., 0]],
                         [[-1., 0, 0], [0, -1., 0], [0, 0, 1.]]])


def geometry_fixture(tmp_path, fit='ca'):
    template = four_chains(tmp_path)
    raw = read_template(template)[0]
    ids = torch.arange(4).repeat_interleave(4)
    r = rotations()
    t = torch.tensor([[1., 2, 3], [-2., 3, 1], [3., -1, 2], [-1., 4, -2]])
    placed = torch.einsum('ni,nij->nj', raw, r[ids]) + t[ids]
    write_coordinates(template, template, placed.numpy())
    manifest = tmp_path / 'chains.json'
    data = build_chains(template, manifest, fit)
    return template, manifest, data, raw, placed, r, t, ids


@pytest.mark.parametrize('fit', ['ca', 'all'])
def test_four_chain_fit_identity_gradient_checkpoint_and_subset(tmp_path, fit):
    template, path, data, raw, expected, r, t, ids = geometry_fixture(tmp_path, fit)
    assert data['body_names'] == ['AA', 'B', 'CC', 'D']
    assert sum(a['fit_core'] for a in data['atoms']) == (12 if fit == 'ca' else 16)
    transform = CoordinateTransform.fit_manifest(path, raw, template)
    x = raw.clone().requires_grad_(True)
    y = transform(x)
    torch.testing.assert_close(y, expected, rtol=0, atol=3e-6)
    torch.testing.assert_close(transform.rotations, r, rtol=0, atol=1e-6)
    for i in range(4):
        torch.testing.assert_close(torch.pdist(y[ids == i]), torch.pdist(raw[ids == i]), atol=3e-6, rtol=0)
    grad, = torch.autograd.grad(y.square().sum(), x)
    torch.testing.assert_close(grad, 2 * torch.einsum('ni,nji->nj', expected, r[ids]), atol=1e-5, rtol=0)
    saved = transform.export_checkpoint(raw)
    restored = CoordinateTransform.from_checkpoint(saved, template, 'cpu')
    assert restored.check_handoff(raw) == 0
    torch.testing.assert_close(restored(raw), y)
    # Two complete bodies per future component, with explicit global identities.
    for start in (0, 8):
        keys = template_atom_keys(template)[start:start+8]
        subset = subset_alignment_manifest(data, list(reversed(keys)))
        assert len(subset['body_names']) == 2 and len(subset['atoms']) == 8
        assert [a['key'] for a in subset['atoms']] == list(reversed(keys))
        assert {a['body_id'] for a in subset['atoms']} == {0, 1}
        assert 'rotations' not in subset
    with pytest.raises(ValueError, match='every atom'):
        subset_alignment_manifest(data, template_atom_keys(template)[:7])
    with pytest.raises(ValueError, match='unknown'):
        subset_alignment_manifest(data, [['UNKNOWN', 1, '', 'CA', 'C']])


@pytest.mark.parametrize('problem,match', [('missing', 'identities'), ('duplicate', 'identities'),
    ('assignment', 'own body'), ('core', 'three'), ('collinear', 'collinear'),
    ('names', 'duplicate'), ('nan', 'coordinates')])
def test_bad_manifest_rejected_before_cache(tmp_path, monkeypatch, problem, match):
    args, _ = train_fixture(tmp_path)
    template, path, data, *_ = geometry_fixture(tmp_path)
    args.cif_path, args.block_alignment, args.coordinate_mode = str(template), str(path), 'blocks'
    if problem == 'missing': data['atoms'].pop()
    if problem == 'duplicate': data['atoms'].append(copy.deepcopy(data['atoms'][0]))
    if problem == 'assignment': data['atoms'][0]['body_id'] = 1
    if problem == 'core': data['atoms'][0]['fit_core'] = False
    if problem == 'collinear':
        for i in range(3): data['atoms'][i]['target'] = [i, 0., 0.]
    if problem == 'names': data['body_names'][1] = data['body_names'][0]
    if problem == 'nan': data['atoms'][0]['target'][0] = float('nan')
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('large cache loaded'))
    from input_validation import validate_train_inputs
    with pytest.raises(ValueError, match=match):
        validate_train_inputs(args)


def test_independent_updates_shared_threshold_and_legacy_checkpoint(tmp_path):
    template, path, data, raw, target, r, t, ids = geometry_fixture(tmp_path)
    transform = CoordinateTransform.fit_manifest(path, raw, template)
    old = transform.export_checkpoint(raw)
    moved = raw.clone()
    flip = rotations()[1]
    moved[ids == 0] = moved[ids == 0] @ flip + 1
    # Translation alone does not trigger the historical rotation-based rule.
    moved[ids == 1] += 7
    moved[ids == 2] = moved[ids == 2] @ rotations()[3] - 2
    events = transform.update_from_coordinates(moved, 2.5)
    assert [e['body'] for e in events] == ['AA', 'CC']
    assert all(e['threshold'] == 2.5 for e in events)
    for i in (1, 3):
        assert torch.equal(transform.rotations[i], old['rotations'][i])
        assert torch.equal(transform.translations[i], old['translations'][i])
    for i in (0, 2):
        torch.testing.assert_close(transform(moved)[ids == i], target[ids == i], atol=3e-6, rtol=0)
    assert transform.update_from_coordinates(moved, 2.5) == []
    saved = transform.export_checkpoint(moved)
    restored = CoordinateTransform.from_checkpoint(saved, template, 'cpu')
    torch.testing.assert_close(restored(moved), transform(moved))
    assert restored.report['update_count_by_body'] == {'AA': 1, 'CC': 1}
    legacy = copy.deepcopy(old)
    legacy.pop('fit_targets')
    legacy.pop('fit_core')
    fixed = CoordinateTransform.from_checkpoint(legacy, template, 'cpu')
    torch.testing.assert_close(fixed(raw), target, atol=3e-6, rtol=0)
    with pytest.raises(ValueError, match='Old block checkpoint'):
        fixed.update_from_coordinates(raw)
    with pytest.raises(ValueError, match='handoff differs'):
        restored.check_handoff(raw)


@pytest.mark.parametrize('bad', [float('nan'), -1, 3])
def test_update_threshold_validation(bad):
    with pytest.raises(ValueError, match='threshold'):
        validate_update_threshold(bad)


@pytest.mark.parametrize('bad', ['scale', 'shear', 'reflection'])
def test_invalid_rigid_transform_rejected(tmp_path, bad):
    template, path, _, raw, *_ = geometry_fixture(tmp_path)
    saved = CoordinateTransform.fit_manifest(path, raw, template).export_checkpoint(raw)
    if bad == 'scale': saved['rotations'][0] *= 2
    if bad == 'shear': saved['rotations'][0, 0, 1] += .2
    if bad == 'reflection': saved['rotations'][0, :, 0] *= -1
    with pytest.raises(ValueError, match='proper rigid'):
        CoordinateTransform.from_checkpoint(saved, template, 'cpu')


def test_chain_cli_and_no_overwrite(tmp_path):
    template = four_chains(tmp_path)
    script = Path(__file__).resolve().parents[2] / 'src/prepare_block_alignment.py'
    output = tmp_path / 'chains.json'
    command = [sys.executable, str(script), '--target', str(template), '--by-chain', '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding='utf-8'))['provenance']['fit_atoms'] == 'ca'
    before = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and output.read_bytes() == before


def test_partial_core_keeps_noncore_atoms_and_rejects_collinear_source(tmp_path):
    template, path, data, raw, expected, *_ = geometry_fixture(tmp_path)
    # Non-core reference atoms do not affect the fit and are not dropped.
    for atom in data['atoms']:
        if not atom['fit_core']:
            atom['target'] = [100., 100., 100.]
    path.write_text(json.dumps(data), encoding='utf-8')
    transform = CoordinateTransform.fit_manifest(path, raw, template)
    assert len(transform.atom_keys) == 16
    torch.testing.assert_close(transform(raw), expected, rtol=0, atol=3e-6)
    raw[:3] = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    with pytest.raises(ValueError, match='noncollinear'):
        CoordinateTransform.fit_manifest(path, raw, template)


def test_legacy_saved_transform_update_preflight(tmp_path):
    from input_validation import validate_train_inputs
    args, cache = train_fixture(tmp_path)
    coords = read_template(args.cif_path)[0]
    transform = CoordinateTransform(torch.eye(3)[None], torch.zeros(1, 3),
                    torch.zeros(6, dtype=torch.long), template_atom_keys(args.cif_path), ['all'], coords)
    cache['coordinate_transform'] = transform.export_checkpoint(coords)
    torch.save(cache, args.diffusion_data_dir)
    validate_train_inputs(args)  # Old fixed checkpoints still load without new metadata.
    args.update_affine_mat = True
    with pytest.raises(ValueError, match='lacks fitting targets/core'):
        validate_train_inputs(args)


@pytest.mark.parametrize('update', [False, True])
@pytest.mark.parametrize('target', ['pair_z', 'z_trunk'])
def test_four_chains_one_decoder_one_latent_training(tmp_path, monkeypatch, update, target):
    import train
    import single_structure_decoder
    created = []
    class SensitiveDenoiser(FakeDenoiser):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created.append(self)
        def forward(self, x_noisy, pair_z, **kwargs):
            return x_noisy * (self.scale + .1 * pair_z.mean()) + pair_z.square().mean()
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = SensitiveDenoiser
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)
    args, cache = train_fixture(tmp_path)
    template = four_chains(tmp_path)
    base = torch.arange(48).float().reshape(4, 4, 3) / 80
    cache.update(pair_z=base if target == 'pair_z' else torch.ones_like(base),
                 z_trunk=base if target == 'z_trunk' else None,
                 s_inputs=torch.zeros(4, 3), s_trunk=torch.zeros(4, 3),
                 z_bias=torch.zeros_like(base), z_mul=torch.tensor(1.))
    cache['input_feature_dict'] = dict(atom_to_token_idx=torch.arange(4).repeat_interleave(4), relp=torch.zeros(4, 4))
    torch.save(cache, args.diffusion_data_dir)
    decoder, pair, _ = single_structure_decoder.load_protenix(args.diffusion_data_dir, 'cpu')
    raw = decoder(pair[None])[0].detach()
    ids = torch.arange(4).repeat_interleave(4)
    placed = torch.einsum('ni,nij->nj', raw, rotations()[ids]) + 3
    write_coordinates(template, template, placed.numpy())
    manifest = tmp_path / 'chains.json'
    build_chains(template, manifest)
    created.clear()
    original_load = single_structure_decoder.load_protenix
    def loader(*a, **k):
        model, pair, meta = original_load(*a, **k)
        original_forward = model.forward
        calls = [0]
        def forward(value):
            xyz = original_forward(value)
            calls[0] += 1
            # Emulate a known per-chain decoder rotation after initial placement.
            if calls[0] >= 3:
                transforms = torch.eye(3).repeat(4, 1, 1).to(xyz)
                transforms[0] = rotations()[1].to(xyz)
                xyz = torch.einsum('bni,nij->bnj', xyz, transforms[ids])
            return xyz
        model.forward = forward
        return model, pair, meta
    monkeypatch.setattr(single_structure_decoder, 'load_protenix', loader)
    args.cif_path, args.block_alignment, args.coordinate_mode = str(template), str(manifest), 'blocks'
    args.alignment_sampler = 'block'  # Explicit legacy decoder coverage; new auto-chain path has its own tests.
    args.update_affine_mat, args.epochs, args.max_steps, args.learn_gmm = update, 1, 2, False
    train.main(args)
    assert len(created) == 1
    saved = torch.load(tmp_path / 'run_1.pth', map_location='cpu', weights_only=False)
    assert saved['z_bias'].shape == (4, 4, 3)
    assert torch.isfinite(saved['z_bias']).all() and saved['z_bias'].abs().sum() > 1e-10
    assert len(saved['opt_state']['param_groups']) == 1
    assert len(saved['opt_state']['param_groups'][0]['params']) == 1
    geometry = CoordinateTransform.from_checkpoint(saved['coordinate_transform'], template, 'cpu')
    assert geometry.body_names == ['AA', 'B', 'CC', 'D']
    assert geometry.fit_core.sum() == 12
    torch.testing.assert_close(read_template(tmp_path / 'run_1.cif')[0], geometry(saved['pred_dict']['coordinate'][0]), atol=1e-6, rtol=0)
    record = next((tmp_path / 'run_records').iterdir())
    events = [json.loads(s) for s in (record / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    updates = [e for e in events if e['event'] == 'alignment_update']
    assert bool(updates) is update
    if update:
        assert {e['body'] for e in updates} == {'AA'}
        assert saved['alignment_update_settings'] == {'enabled': True, 'block_trace_threshold': 2.5}
        assert geometry.report['update_count_by_body']['AA'] >= 1
    assert json.loads((record / 'run_identity.json').read_text(encoding='utf-8'))['created_at'].endswith('+08:00')
