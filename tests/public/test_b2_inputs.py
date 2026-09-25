"""Real tiny STAR/MRCS files and CPU CLI/preflight checks; no model downloads."""
import json
from pathlib import Path
import subprocess
import sys

import mrcfile
import numpy as np
import pandas as pd
import pytest
import torch

from particledataset import ParticleDataset
from input_validation import validate_train_inputs
from test_foundations import make_cache, write_fixture, FakeDenoiser, fake_protenix
from train import build_parser, main


def write_particle_star(data, path):
    # Generate these simple numeric/string loop fixtures independently of the
    # reader version. CSV space-delimited quoting protects filenames with spaces.
    # Real ParticleDataset still parses the file through installed starfile.
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        for name, frame in data.items():
            handle.write(f'data_{name}\n\nloop_\n')
            for index, column in enumerate(frame.columns, 1):
                handle.write(f'_{column} #{index}\n')
            frame.to_csv(handle, sep=' ', header=False, index=False, float_format='%.6f')
            handle.write('\n\n')


def particles(folder, version=3, count=3, box=24):
    stack = folder / 'particle stack.mrcs'
    pixels = np.random.default_rng(9).normal(size=(count, box, box)).astype(np.float32)
    with mrcfile.new(stack, overwrite=True) as handle:
        handle.set_data(pixels)
        handle.voxel_size = 1.
    table = pd.DataFrame(dict(
        rlnImageName=[f'{i+1}@{stack.name}' for i in range(count)],
        rlnAngleRot=11., rlnAngleTilt=22., rlnAnglePsi=33.,
        rlnDefocusU=10000., rlnDefocusV=11000., rlnDefocusAngle=15.,
    ))
    optical = dict(rlnVoltage=300., rlnSphericalAberration=2.7, rlnAmplitudeContrast=.1)
    if version == 2:
        table = table.assign(rlnOriginX=1., rlnOriginY=-2., **optical)
        data = {'images': table}
    else:
        table = table.assign(rlnOriginXAngst=1., rlnOriginYAngst=-2., rlnOpticsGroup=1)
        data = dict(particles=table, optics=pd.DataFrame([
            dict(rlnOpticsGroup=1, rlnImagePixelSize=1., **optical)]))
    path = folder / 'particles.star'
    write_particle_star(data, path)
    return path, stack, pixels, data


@pytest.mark.parametrize('version', [2, 3])
@pytest.mark.parametrize('mode', ['star_parent', 'explicit_root', 'absolute'])
def test_particle_path_matrix(tmp_path, monkeypatch, version, mode):
    star, stack, pixels, data = particles(tmp_path, version)
    key = 'images' if version == 2 else 'particles'
    root = None
    if mode == 'explicit_root':
        root = str(tmp_path)  # Deliberately no trailing slash.
        sub = tmp_path / 'metadata'
        sub.mkdir()
        star = sub / star.name
    elif mode == 'absolute':
        data[key]['rlnImageName'] = [f'{i+1}@{stack}' for i in range(3)]
        root = str(tmp_path / 'unused_root')
    write_particle_star(data, star)
    monkeypatch.chdir(tmp_path.parent)
    dataset = ParticleDataset(str(star), root, 1.)
    report = dataset.validate(24)
    assert report['n_particles'] == 3 and report['n_stacks'] == 1
    image, ctf, shift, rotation, full_rotation, index = dataset[1]
    np.testing.assert_array_equal(image, pixels[1])
    np.testing.assert_allclose(ctf, [300000., 10000., 11000., np.pi/12, 2.7e7, .1, 0., 1.])
    np.testing.assert_array_equal(shift, [1., -2.])
    np.testing.assert_allclose(rotation, full_rotation.T[:2], atol=1e-15)
    assert index == 1


@pytest.mark.parametrize('problem,match', [
    ('index', 'out of range'), ('zero', '>= 1'), ('missing_stack', 'does not exist'),
    ('box', 'box size'), ('nan', 'Nonfinite'), ('group', 'missing optics'),
    ('duplicate_group', 'unique'), ('apix', 'pixel size'), ('missing_column', 'missed'),
    # 0.4.x materializes an empty loop as one all-null row; both errors reject
    # the same invalid input before training, without requiring identical text.
    pytest.param('empty', 'No particles|Nonfinite rlnAngleRot', id='empty-No particles'),
])
def test_particle_input_errors(tmp_path, problem, match):
    star, stack, _, data = particles(tmp_path)
    if problem == 'index': data['particles'].loc[0, 'rlnImageName'] = f'4@{stack.name}'
    if problem == 'zero': data['particles'].loc[0, 'rlnImageName'] = f'0@{stack.name}'
    if problem == 'missing_stack': data['particles'].loc[0, 'rlnImageName'] = '1@absent.mrcs'
    if problem == 'nan': data['particles'].loc[0, 'rlnDefocusU'] = np.inf
    if problem == 'group': data['particles'].loc[0, 'rlnOpticsGroup'] = 2
    if problem == 'duplicate_group': data['optics'] = pd.concat([data['optics'], data['optics']])
    if problem == 'apix': data['optics']['rlnImagePixelSize'] = 2.
    if problem == 'missing_column': del data['particles']['rlnAngleRot']
    if problem == 'empty': data['particles'] = data['particles'].iloc[:0]
    write_particle_star(data, star)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        ParticleDataset(str(star)).validate(12 if problem == 'box' else 24)


@pytest.mark.parametrize('pixel,match', [(np.nan, 'Nonfinite'), (1., 'constant')])
def test_invalid_pixels_when_read(tmp_path, pixel, match):
    star, stack, pixels, _ = particles(tmp_path)
    with mrcfile.open(stack, mode='r+') as handle:
        handle.data[0] = pixel
    dataset = ParticleDataset(str(star), norm=True)
    dataset.validate(24)
    with pytest.raises(ValueError, match=match):
        dataset[0]


def train_fixture(folder):
    star, _, _, _ = particles(folder)
    template = write_fixture(folder)
    cache = make_cache()
    cache.update(pred_dict={}, p_lm=None, c_l=None, N_sample=1, inplace_safe=False,
                 enable_efficient_fusion=False)
    path = folder / 'cache.pt'
    torch.save(cache, path)
    args = build_parser().parse_args([
        '--star_data_dir', str(star), '--cif_path', str(template),
        '--diffusion_data_dir', str(path), '--output_trained_model_dir', str(folder / 'run_'),
        '--device', 'cpu', '--boxsize', '24', '--batch_size', '2', '--mini_batch_size', '1',
        '--projection-frame', 'legacy',
    ])
    return args, cache


def test_new_projection_default_requires_map_frame_origin(tmp_path):
    defaults = build_parser().parse_args([
        '--star_data_dir', 's', '--cif_path', 'c',
        '--diffusion_data_dir', 'd', '--output_trained_model_dir', 'o'])
    assert defaults.projection_frame == 'fixed' and defaults.projection_origin is None
    args, _ = train_fixture(tmp_path)
    args.projection_frame = 'fixed'
    args.projection_origin = None
    with pytest.raises(ValueError, match='--projection-origin'):
        validate_train_inputs(args)
    args.projection_origin = (0., 0., 0.)
    _, _, report = validate_train_inputs(args)
    assert args.projection_origin == (0., 0., 0.)
    assert report['projection_frame'] == 'fixed'
    assert report['projection_origin_A'] == [0., 0., 0.]
    assert report['n_particles'] == 3


def test_valid_preflight_without_protenix(tmp_path, monkeypatch):
    args, cache = train_fixture(tmp_path)
    args.check_inputs = True
    monkeypatch.delitem(sys.modules, 'protenix.model.modules.diffusion', raising=False)
    report = main(args)
    assert report['n_atoms'] == 6 and report['n_tokens'] == 2
    assert report['n_particles'] == 3
    assert 'protenix.model.modules.diffusion' not in sys.modules
    assert not list(tmp_path.glob('run_*'))


@pytest.mark.parametrize('problem,match', [
    ('star', 'does not exist|file'), ('halfmap', 'together'), ('gmm', 'supports amplitude'),
    ('sigma', 'sigma_floor'), ('nan_lr', 'finite'),
])
def test_cheap_errors_before_loading_cache(tmp_path, monkeypatch, problem, match):
    args, _ = train_fixture(tmp_path)
    if problem == 'star': args.star_data_dir = str(tmp_path / 'missing.star')
    if problem == 'halfmap': args.halfmap1 = 'missing.mrc'
    if problem == 'gmm': args.gmm_amplitude = 'peak_3d'
    if problem == 'sigma': args.gmm_sigma_floor = float('nan')
    if problem == 'nan_lr': args.lr_bias = float('nan')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Large cache was loaded'))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        validate_train_inputs(args)


@pytest.mark.parametrize('problem,match', [
    ('field', 'missing fields'), ('count', 'atom count'), ('pair', 'compatible'),
    ('schedule', 'noise schedule'), ('block', 'requires'), ('collision', 'already exists'),
])
def test_cache_schema_and_output_errors(tmp_path, problem, match):
    args, cache = train_fixture(tmp_path)
    if problem == 'field': del cache['s_trunk']
    if problem == 'count': cache['input_feature_dict']['atom_to_token_idx'] = torch.zeros(5, dtype=torch.long)
    if problem == 'pair': cache['pair_z'] = torch.zeros(4, 4, 3)
    if problem == 'schedule': cache['noise_schedule'] = torch.tensor([1., 2., 0.])
    if problem == 'block': args.coordinate_mode = 'blocks'
    if problem == 'collision': (tmp_path / 'run_2.pth').write_bytes(b'keep')
    torch.save(cache, args.diffusion_data_dir)
    with pytest.raises((ValueError, FileExistsError), match=match):
        validate_train_inputs(args)
    if problem == 'collision': assert (tmp_path / 'run_2.pth').read_bytes() == b'keep'


@pytest.mark.parametrize('script', ['train.py', 'inference.py'])
def test_cli_help_no_protenix(script):
    src = Path(__file__).resolve().parents[2] / 'src'
    result = subprocess.run([sys.executable, str(src / script), '--help'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--check-inputs' in result.stdout


@pytest.mark.parametrize('flag,value', [('--epochs', '0'), ('--batch_size', '1.5'), ('--lr_bias', 'nan')])
def test_typed_train_arguments(flag, value):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(['--star_data_dir', 'x', '--cif_path', 'x', '--diffusion_data_dir', 'x',
                                  '--output_trained_model_dir', 'x', flag, value])
    assert error.value.code == 2


def test_real_training_budget_and_lrs(tmp_path, fake_protenix):
    from coordinate_transform import template_atom_keys
    from single_structure_decoder import load_protenix
    args, cache = train_fixture(tmp_path)
    decoder, pair, _ = load_protenix(args.diffusion_data_dir, 'cpu')
    raw = decoder(pair[None])[0].detach()
    manifest = dict(format='cocofold2-block-manifest-v1', body_names=['all'], atoms=[
        dict(key=key, body_id=0, target=(raw[i] + 2).tolist(), fit_core=True)
        for i, key in enumerate(template_atom_keys(args.cif_path))])
    alignment = tmp_path / 'alignment.json'
    alignment.write_text(json.dumps(manifest), encoding='utf-8')
    args.coordinate_mode, args.block_alignment = 'blocks', str(alignment)
    args.epochs, args.max_steps = 4, 3  # Two optimizer batches/epoch; stop within epoch 2.
    args.lr_bias, args.lr_atom_weights, args.lr_sdevs = .007, .008, .009
    main(args)
    first = torch.load(tmp_path / 'run_1.pth', weights_only=False)
    last = torch.load(tmp_path / 'run_2.pth', weights_only=False)
    assert not (tmp_path / 'run_3.pth').exists()
    assert sorted(group['lr'] for group in last['opt_state']['param_groups']) == [.007, .008, .009]
    assert {int(state['step']) for state in first['opt_state']['state'].values()} == {2}
    assert {int(state['step']) for state in last['opt_state']['state'].values()} == {3}
    assert torch.isfinite(last['z_bias']).all() and last['z_bias'].abs().sum() > 0


def test_halfmap_mismatch_before_cache(tmp_path, monkeypatch):
    args, _ = train_fixture(tmp_path)
    for i, size in enumerate((4, 6), 1):
        path = tmp_path / f'half{i}.mrc'
        with mrcfile.new(path) as handle:
            handle.set_data(np.zeros((size, size, size), dtype=np.float32))
            handle.voxel_size = 1.
        setattr(args, f'halfmap{i}', str(path))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Large cache was loaded'))
    with pytest.raises(ValueError, match='Half-map shape'):
        validate_train_inputs(args)
