"""Inference scheduling, partial failure and exclusive cache writes, without Protenix."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import inference
from inference_io import cache_plan, read_inputs, save_diffusion_cache


def config(tmp_path, names=('first', 'second'), seeds=(7, 42)):
    targets = [dict(name=name, sequences=[dict(proteinChain=dict(sequence='AAA', count=1))]) for name in names]
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(targets), encoding='utf-8')
    cfg = SimpleNamespace(input_json_path=str(path), seeds=list(seeds), use_seeds_in_json=False,
                          output_model_dir=str(tmp_path / 'caches'), sample_name='legacy_alias',
                          dump_dir=str(tmp_path / 'outputs'), deterministic=True,
                          skip_amp=SimpleNamespace(confidence_head=True, sample_diffusion=True))
    return cfg, targets


def test_cache_names_and_no_overwrite(tmp_path):
    cfg, targets = config(tmp_path)
    planned = cache_plan(cfg, targets, cfg.seeds)
    assert len(set(planned.values())) == 4
    assert planned[('second', 42)] == tmp_path / 'caches/second/seed_42/diffusion_data.pth'
    cfg.cache_output_path = str(planned[('second', 42)])
    cache = save_diffusion_cache({'value': torch.tensor(3)}, cfg)
    before = cache.read_bytes()
    with pytest.raises(FileExistsError): save_diffusion_cache({}, cfg)
    with pytest.raises(FileExistsError): cache_plan(cfg, targets, cfg.seeds)
    assert cache.read_bytes() == before
    cfg, targets = config(tmp_path, ('only',), (42,))
    assert cache_plan(cfg, targets, cfg.seeds)[('only', 42)].name == 'legacy_alias_diffusion_data.pth'


def test_failed_serialization_removes_only_incomplete_file(tmp_path, monkeypatch):
    cfg, _ = config(tmp_path)
    cfg.cache_output_path = str(tmp_path / 'broken.pt')
    def fail(data, handle):
        handle.write(b'partial')
        raise RuntimeError('simulated disk write failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(RuntimeError): save_diffusion_cache({}, cfg)
    assert not Path(cfg.cache_output_path).exists()
    Path(cfg.cache_output_path).write_bytes(b'existing')
    with pytest.raises(FileExistsError): save_diffusion_cache({}, cfg)
    assert Path(cfg.cache_output_path).read_bytes() == b'existing'


@pytest.mark.parametrize('seeds', [[], [42, 42], [-1], [2**32], [True]])
def test_invalid_seeds(tmp_path, seeds):
    cfg, targets = config(tmp_path)
    with pytest.raises(ValueError): cache_plan(cfg, targets, seeds)


@pytest.mark.parametrize('data', [[], {}, [dict(name='../escape', sequences=[{}])],
                                 [dict(name='test', sequences=[])],
                                 [dict(name='a', sequences=[{}]), dict(name='a', sequences=[{}])]])
def test_invalid_basic_json(tmp_path, data):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError): read_inputs(path)


class Runner:
    def __init__(self, cfg, fail_name=None):
        self.cfg, self.fail_name, self.calls = cfg, fail_name, []
        self.error_dir = str(Path(cfg.dump_dir) / 'ERR')
        self.dumper = SimpleNamespace(dump=lambda **kwargs: None)

    def update_model_configs(self, cfg): self.cfg = cfg

    def predict(self, data):
        self.calls.append((data['sample_name'], self.cfg.cache_output_path))
        if data['sample_name'] == self.fail_name:
            raise RuntimeError('simulated inference failure')
        if self.cfg.output_model_dir:
            save_diffusion_cache(dict(coordinate=torch.zeros(1, 6, 3)), self.cfg)
        return {}


def mock_runtime(monkeypatch, targets, error_name=None):
    monkeypatch.setattr(inference, 'DIST_WRAPPER', SimpleNamespace(rank=0), raising=False)
    monkeypatch.setattr(inference, 'seed_everything', lambda **kwargs: None, raising=False)
    batches = [[(dict(sample_name=target['name'], N_token=torch.tensor(2),
                     entity_poly_type={'1': 'polypeptide(L)'}), None,
                 'simulated feature error' if target['name'] == error_name else '')] for target in targets]
    monkeypatch.setattr(inference, 'get_inference_dataloader', lambda **kwargs: batches, raising=False)


@pytest.mark.parametrize('failure_mode', ['model', 'data', 'none'])
def test_continue_targets_and_seeds_with_summary(tmp_path, monkeypatch, failure_mode):
    cfg, targets = config(tmp_path)
    mock_runtime(monkeypatch, targets, 'first' if failure_mode == 'data' else None)
    runner = Runner(cfg, 'first' if failure_mode == 'model' else None)
    if failure_mode != 'none':
        with pytest.raises(RuntimeError, match='failure'):
            inference.infer_predict(runner, cfg)
    else:
        inference.infer_predict(runner, cfg)
    summary = json.loads((Path(cfg.dump_dir) / 'inference_summary.json').read_text(encoding='utf-8'))
    assert summary['expected_jobs'] == 4
    assert summary['succeeded'] == (4 if failure_mode == 'none' else 2)
    assert summary['failed'] == (0 if failure_mode == 'none' else 2)
    assert sum(name == 'second' for name, _ in runner.calls) == 2
    for row in summary['outcomes']:
        assert row['cache_exists'] == (row['status'] == 'success')


def test_loader_failure_is_not_success(tmp_path, monkeypatch):
    cfg, targets = config(tmp_path)
    mock_runtime(monkeypatch, targets)
    def fail(**kwargs): raise ValueError('cannot construct dataset')
    monkeypatch.setattr(inference, 'get_inference_dataloader', fail)
    with pytest.raises(RuntimeError, match='failure'):
        inference.infer_predict(Runner(cfg), cfg)
    summary = json.loads((Path(cfg.dump_dir) / 'inference_summary.json').read_text(encoding='utf-8'))
    assert summary['succeeded'] == 0
    assert all(row['status'] == 'failed' for row in summary['outcomes'])


def test_json_check_does_not_load_runtime(tmp_path, monkeypatch):
    cfg, _ = config(tmp_path)
    args = inference.build_parser().parse_args(['--input_json_path', cfg.input_json_path, '--check-inputs'])
    monkeypatch.setattr(inference, '_load_runtime_imports', lambda: pytest.fail('loaded Protenix'))
    assert inference.run(args)['targets'] == ['first', 'second']


def test_inference_parser_defaults_and_false():
    parser = inference.build_parser()
    args = parser.parse_args(['--input_json_path', 'input.json'])
    assert args.train_deterministic is True and args.save_pairformer_last_input is False
    assert (args.gamma0, args.gamma_min, args.noise_scale_lambda, args.step_scale_eta) == (0., 0., 1.003, 1.)
    assert (args.N_step, args.N_sample, args.N_step_mini_rollout, args.N_sample_mini_rollout) == (5, 1, 5, 5)
    assert parser.parse_args(['--input_json_path', 'x', '--save_pairformer_last_input', 'false']).save_pairformer_last_input is False


def test_collision_fails_before_runner_construction(tmp_path, monkeypatch):
    cfg, targets = config(tmp_path)
    planned = cache_plan(cfg, targets, cfg.seeds)
    next(iter(planned.values())).write_bytes(b'old result')
    monkeypatch.setattr(inference, 'InferenceRunner', lambda *a: pytest.fail('constructed model'))
    with pytest.raises(FileExistsError): inference.main(cfg)


def test_resource_root_and_explicit_paths(tmp_path, monkeypatch):
    from inference_io import resolve_resource_paths
    class Config(dict):
        pass
    cfg = Config(data=dict(ccd_components_file='keep/components.cif',
                           template=dict(release_dates_path='keep/dates.json')))
    cfg.load_checkpoint_dir = 'keep/checkpoint'
    monkeypatch.chdir(tmp_path)
    resolve_resource_paths(cfg)
    assert cfg.load_checkpoint_dir == str(tmp_path / 'checkpoint')
    assert cfg['data']['ccd_components_file'] == str(tmp_path / 'common/components.cif')
    cfg.load_checkpoint_dir = 'custom/weights'
    cfg['data']['ccd_components_file'] = 'custom/ccd.cif'
    cfg['data']['template']['release_dates_path'] = 'custom/releases.json'
    resolve_resource_paths(cfg, tmp_path / 'resources', [
        '--load_checkpoint_dir=custom/weights', '--data.ccd_components_file', 'custom/ccd.cif',
        '--data.template.release_dates_path=custom/releases.json'])
    assert cfg.load_checkpoint_dir == 'custom/weights'
    assert cfg['data']['ccd_components_file'] == 'custom/ccd.cif'
    assert cfg['data']['template']['release_dates_path'] == 'custom/releases.json'
    assert cfg['data']['pdb_cluster_file'] == str(tmp_path / 'resources/common/clusters-by-entity-40.txt')


def test_partial_failure_has_nonzero_process_exit(tmp_path):
    src = Path(__file__).resolve().parents[2] / 'src'
    tests = Path(__file__).resolve().parent
    script = f'''
import sys
sys.path[:0] = [{str(src)!r}, {str(tests)!r}]
from pathlib import Path
from pytest import MonkeyPatch
from test_b2_inference import config, mock_runtime, Runner
import inference
cfg, targets = config(Path({str(tmp_path)!r}))
with MonkeyPatch.context() as mp:
    mock_runtime(mp, targets)
    inference.infer_predict(Runner(cfg, 'first'), cfg)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'target failure' in result.stderr
    summary = json.loads((tmp_path / 'outputs/inference_summary.json').read_text(encoding='utf-8'))
    assert summary['failed'] == 2 and summary['succeeded'] == 2
