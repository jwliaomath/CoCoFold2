"""T0-T2 coverage for the independent public CPU gate; no trained model."""
import ast
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    ('train.py', []), ('inference.py', []), ('get_pdb.py', []),
    ('prepare_block_alignment.py', []),
    ('chain_parallel/prepare_contextual_diffusion_caches.py', []),
    ('chain_parallel/prepare_contextual_diffusion_caches.py', ['inspect']),
    ('chain_parallel/prepare_contextual_diffusion_caches.py', ['split']),
    ('chain_parallel/materialize_local_diffusion_cache.py', []),
]


def cli(script, flags):
    return subprocess.run([sys.executable, str(ROOT / 'src' / script), *flags],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_t0_public_inventory_and_compilation():
    """Every allowlisted Python file compiles and has no private hetero import."""
    paths = json.loads((ROOT / 'tests/public_files.json').read_text(encoding='utf-8'))
    assert len(paths) == len(set(paths))
    assert not (ROOT / 'src/hetero').exists()
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and not any(part in p.parts for part in ('__pycache__', '.pytest_cache'))}
    assert actual == set(paths)
    for rel in paths:
        if not rel.endswith('.py'):
            continue
        source = (ROOT / rel).read_text(encoding='utf-8-sig')
        compile(source, rel, 'exec')
        if not rel.startswith('src/'):
            continue
        for node in ast.walk(ast.parse(source)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            assert all('hetero' not in name.split('.') for name in names), rel


def test_t0_imports_without_model_or_private_code():
    """Public lazy entrypoints import without weights, Protenix, or GPU init."""
    modules = ['train', 'inference', 'get_pdb', 'prepare_block_alignment', 'cache_structure',
               'structure_io', 'single_structure_decoder', 'coordinate_transform',
               'checkpoint_io', 'checkpoint_sampling', 'training_restart',
               'input_validation', 'randomness', 'run_recording', 'gmm', 'particledataset']
    code = f"import importlib,sys,torch; [importlib.import_module(n) for n in {modules!r}]; assert not torch.cuda.is_initialized(); assert not any(n.split('.')[0] in ('hetero','protenix') for n in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_t0_cpu_guards_are_active():
    """Check that CPU-only reporting is enforced, including in child processes."""
    assert os.environ.get('COCOFOLD2_CPU_GATE') == '1', 'Run through tests/run_public_tests.py'
    assert getattr(sys, '_cocofold2_cpu_guard', False)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match='blocks CUDA'):
        torch.cuda.init()
    with pytest.raises(ModuleNotFoundError, match='blocks model/private'):
        importlib.import_module('protenix')
    with pytest.raises(ModuleNotFoundError, match='blocks model/private'):
        importlib.import_module('hetero')
    with pytest.raises(RuntimeError, match='blocks network'):
        socket.getaddrinfo('example.invalid', 443)
    result = subprocess.run([sys.executable, '-c', 'import sys; assert sys._cocofold2_cpu_guard'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('script,subcommand', SCRIPTS)
def test_t1_help(script, subcommand):
    """Single-card and parallel preparation CLI help exits without a model."""
    result = cli(script, subcommand + ['--help'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('script,subcommand', SCRIPTS)
def test_t1_invalid_arguments(script, subcommand):
    """Invalid/missing CLI arguments fail with a diagnostic and bounded timeout."""
    result = cli(script, subcommand + ['--this-option-does-not-exist'])
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'error:' in result.stderr.lower()
    assert 'CPU gate blocks' not in result.stderr


@pytest.mark.parametrize('broken', [False, True])
def test_t2_train_preflight_subprocess(tmp_path, broken):
    """Real tiny STAR/MRCS/cache input checks execute without model construction."""
    from test_b2_inputs import train_fixture
    args, _ = train_fixture(tmp_path)
    star = str(tmp_path / 'missing.star') if broken else args.star_data_dir
    result = cli('train.py', ['--star_data_dir', star, '--cif_path', args.cif_path,
        '--diffusion_data_dir', args.diffusion_data_dir, '--output_trained_model_dir', args.output_trained_model_dir,
        '--boxsize', '24', '--device', 'cpu', '--check-inputs'])
    if broken:
        assert result.returncode != 0
        assert 'missing.star' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'n_particles' in result.stdout and 'n_atoms' in result.stdout
    assert 'CPU gate blocks' not in result.stderr
    assert not list(tmp_path.glob('run_*.pth'))
    assert not list(tmp_path.glob('run_*.cif'))
