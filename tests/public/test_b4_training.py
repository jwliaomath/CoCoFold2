"""Template-based CIF/PDB outputs and real GMM freezing with small CPU inputs."""
import json
from pathlib import Path
import sys
from types import ModuleType

import gemmi
import numpy as np
import pytest
import torch

from gmm import GaussianProjector
from structure_io import read_template
from training_output import export_training_structure, validate_output_template
from test_foundations import FakeDenoiser, write_fixture
from test_b2_inputs import train_fixture


@pytest.mark.parametrize('fmt', ['cif', 'pdb', 'both'])
def test_output_formats_identity_coordinates_and_collision(tmp_path, fmt):
    template = write_fixture(tmp_path)
    coords, weights, labels = read_template(template)
    moved = coords.numpy() + np.array([.123456, 2., -1.])
    paths = export_training_structure(template, tmp_path / 'out', moved, fmt)
    assert {Path(p).suffix for p in paths} == ({'.cif', '.pdb'} if fmt == 'both' else {'.' + fmt})
    for path in paths:
        structure = gemmi.read_structure(path)
        assert [c.name for c in structure[0]] == ['A', 'B']
        actual = np.array([[a.pos.x, a.pos.y, a.pos.z] for c in structure[0] for r in c for a in r])
        np.testing.assert_allclose(actual, moved, atol=1e-6 if path.endswith('.cif') else 5e-4, rtol=0)
        if path.endswith('.cif'):
            assert gemmi.cif.read_file(path).sole_block().find_values('_atom_site.Cartn_x')
            assert read_template(path)[2] == labels
        else:
            from utils import replace_cif_coordinates
            old = tmp_path / 'legacy.pdb'
            replace_cif_coordinates(str(template), str(old), moved)
            assert Path(path).read_bytes() == old.read_bytes()
    before = {path: Path(path).read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        export_training_structure(template, tmp_path / 'out', moved, fmt)
    assert all(Path(path).read_bytes() == content for path, content in before.items())


def test_reference_cif_remains_required_and_defaults():
    from train import build_parser
    parser = build_parser()
    assert parser.get_default('output_format') == 'cif'
    assert parser.get_default('learn_gmm') is True
    with pytest.raises(SystemExit):
        parser.parse_args(['--star_data_dir', 'x', '--diffusion_data_dir', 'x', '--output_trained_model_dir', 'x'])


def test_pdb_limits_rejected_before_cache_load(tmp_path, monkeypatch):
    from input_validation import validate_train_inputs
    args, _ = train_fixture(tmp_path)
    structure = gemmi.read_structure(args.cif_path)
    structure[0][0].name = 'LONG_CHAIN'
    structure.make_mmcif_document().write_file(args.cif_path)
    validate_output_template(args.cif_path, 6, 'cif')
    args.output_format = 'pdb'
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('large cache loaded'))
    with pytest.raises(ValueError, match='single-character'):
        validate_train_inputs(args)


@pytest.mark.parametrize('bad', ['nan', 'count', 'overflow'])
def test_invalid_output_has_no_partial_file(tmp_path, bad):
    template = write_fixture(tmp_path)
    xyz = read_template(template)[0].numpy()
    if bad == 'nan':
        xyz[0, 0] = np.nan
    elif bad == 'count':
        xyz = xyz[:-1]
    else:
        xyz[0, 0] = 100000
    with pytest.raises(ValueError):
        export_training_structure(template, tmp_path / 'bad', xyz, 'both')
    assert not (tmp_path / 'bad.cif').exists() and not (tmp_path / 'bad.pdb').exists()


@pytest.mark.parametrize('kernel', ['legacy', 'isotropic', 'anisotropic'])
@pytest.mark.parametrize('learn', [False, True])
def test_training_gmm_freeze_and_latent_gradient(tmp_path, monkeypatch, kernel, learn):
    import train
    from single_structure_decoder import load_protenix
    from coordinate_transform import template_atom_keys
    class SensitiveDenoiser(FakeDenoiser):
        def forward(self, x_noisy, pair_z, **kwargs):
            return x_noisy * (self.scale + .1 * pair_z.mean()) + pair_z.square().mean()
    module = ModuleType('protenix.model.modules.diffusion')
    module.DiffusionModule = SensitiveDenoiser
    monkeypatch.setitem(sys.modules, 'protenix.model.modules.diffusion', module)
    args, cache = train_fixture(tmp_path)
    args.gmm_kernel, args.learn_gmm = kernel, learn
    fmt = "pdb" if learn and kernel == "isotropic" else "both" if learn and kernel == "anisotropic" else "cif"
    args.output_format = fmt
    args.epochs, args.max_steps = 1, 2
    decoder, pair, _ = load_protenix(args.diffusion_data_dir, 'cpu')
    raw = decoder(pair[None])[0].detach()
    alignment = tmp_path / 'alignment.json'
    alignment.write_text(json.dumps(dict(format='cocofold2-block-manifest-v1', body_names=['all'], atoms=[
        dict(key=key, body_id=0, target=(raw[i] + 2).tolist(), fit_core=True)
        for i, key in enumerate(template_atom_keys(args.cif_path))])), encoding='utf-8')
    args.coordinate_mode, args.block_alignment = 'blocks', str(alignment)
    original = train.gmm_from_arguments
    tracked = {}
    def factory(*a, **kw):
        obj = original(*a, **kw)
        tracked['gmm'] = obj
        tracked['before'] = {k: v.clone() for k, v in obj.state_dict().items()}
        for name, parameter in obj.named_parameters():
            parameter.register_hook(lambda grad, key=name: tracked.setdefault(key + '_grads', []).append(float(grad.abs().sum())))
        return obj
    monkeypatch.setattr(train, 'gmm_from_arguments', factory)
    train.main(args)
    saved = torch.load(tmp_path / 'run_1.pth', map_location='cpu', weights_only=False)
    assert saved['gmm_learning_enabled'] is learn and saved['structure_output_format'] == fmt
    assert len(saved['opt_state']['param_groups']) == (3 if learn else 1)
    assert torch.isfinite(saved['z_bias']).all() and saved['z_bias'].abs().sum() > 1e-10
    for name, param in tracked['gmm'].named_parameters():
        assert param.requires_grad is learn
        if learn:
            assert any(value > 0 for value in tracked[name + '_grads'])
            assert not torch.equal(param, tracked['before'][name])
        else:
            assert param.grad is None and name + '_grads' not in tracked
            assert torch.equal(param, tracked['before'][name])
            assert torch.equal(saved['gmm']['state_dict'][name], tracked['before'][name])
    for extension in ('cif', 'pdb'):
        expected = fmt == extension or fmt == 'both'
        for stem in ('run__', 'run_aligned_initial', 'run_1'):
            assert (tmp_path / f'{stem}.{extension}').exists() is expected
    extension = 'pdb' if fmt == 'pdb' else 'cif'
    exported = read_template(tmp_path / ('run_1.' + extension))[0]
    from coordinate_transform import CoordinateTransform
    transform = CoordinateTransform.from_checkpoint(saved['coordinate_transform'], args.cif_path, 'cpu')
    torch.testing.assert_close(exported, transform(saved['pred_dict']['coordinate'][0]), rtol=0, atol=5e-4 if fmt == 'pdb' else 1e-6)
    restored = GaussianProjector.from_checkpoint(saved, 'cpu')
    for name, value in restored.state_dict().items():
        assert torch.equal(value, saved['gmm']['state_dict'][name])
    record_dir = next((tmp_path / 'run_records').iterdir())
    stages = json.loads((record_dir / 'resolved_config.json').read_text(encoding='utf-8'))['stages']
    actual = next(stage['values'] for stage in stages if stage['stage'] == 'optimizer_and_geometry')
    assert actual['gmm_learning'] is learn and actual['output_format'] == fmt


def test_old_gmm_payload_and_refreezing_clear_gradients():
    old = {'atom_weights': torch.tensor([6., 7.]), 'sdevs': torch.full((2, 2), .7)}
    gmm = GaussianProjector.from_checkpoint(old)
    sum(p.sum() for p in gmm.parameters()).backward()
    gmm.set_learning_enabled(False)
    assert all(p.grad is None and not p.requires_grad for p in gmm.parameters())
    gmm.set_learning_enabled(True)
    assert all(p.requires_grad for p in gmm.parameters())
    assert torch.equal(gmm.atom_weights, old['atom_weights'])
    assert torch.equal(gmm.sdevs, old['sdevs'])
