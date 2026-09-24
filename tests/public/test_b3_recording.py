"""Structured experiment records and entry integration without model weights."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from run_recording import RunRecord, RecordingError, recorded, current_record, json_value
from test_foundations import fake_protenix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def args_at(tmp_path, **values):
    return SimpleNamespace(record_dir=str(tmp_path / 'records'), **values)


def test_record_files_argv_script_and_rng_unchanged(tmp_path):
    from randomness import capture_rng_state
    from test_b2b_randomness import same_rng
    script = tmp_path / 'job.slurm'
    script.write_bytes(b'#!/bin/bash\npython train.py "$INPUT"\n')
    args = args_at(tmp_path, original_argv=['train.py', '--input', 'a b.star'], submission_script=str(script))
    @recorded('train')
    def run(args):
        current_record().resolved('test', {'value': 2}, 'test override')
        current_record().event('train_step', global_step=1, frc_loss=-.2, penalty=.1)
        return 7
    before = capture_rng_state()
    assert run(args) == 7
    same_rng(before, capture_rng_state())
    root = Path(args.record_dir)
    assert read(root / 'command.json')['argv'] == args.original_argv
    assert (root / 'submitted_script.sh').read_bytes() == script.read_bytes()
    assert read(root / 'run_summary.json')['status'] == 'success'
    events = [json.loads(line) for line in (root / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [e['event'] for e in events] == ['start', 'train_step', 'success']
    assert all(e['schema_version'] == 1 and e['run_id'] for e in events)
    assert read(root / 'provenance.json')['source_sha256']['run_recording.py']


def test_provenance_ignores_parent_directory_named_hetero(tmp_path):
    """A parent directory name must not hide public source from provenance."""
    source = Path(__file__).resolve().parents[2] / 'src' / 'run_recording.py'
    nested = tmp_path / 'hetero' / 'code' / 'src'
    nested.mkdir(parents=True)
    copy = nested / 'run_recording.py'
    shutil.copy2(source, copy)
    spec = importlib.util.spec_from_file_location('recording_nested_fixture', copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @module.recorded('train')
    def run(args):
        return 7

    records = tmp_path / 'records'
    assert run(SimpleNamespace(record_dir=str(records))) == 7
    hashes = read(records / 'provenance.json')['source_sha256']
    assert 'run_recording.py' in hashes


@pytest.mark.parametrize('exception,status', [(ValueError('bad inputs'), 'failed'), (KeyboardInterrupt(), 'interrupted'), (SystemExit(143), 'interrupted')])
def test_failure_and_interrupt_summary(tmp_path, exception, status):
    @recorded('train')
    def run(args):
        raise exception
    with pytest.raises(type(exception)) as caught:
        run(args_at(tmp_path))
    assert caught.value is exception
    assert read(tmp_path / 'records/run_summary.json')['status'] == status


def test_sigterm_handler_records_and_restores(tmp_path):
    import signal
    prior = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    @recorded('train')
    def run(args):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    try:
        with pytest.raises(SystemExit) as caught:
            run(args_at(tmp_path))
        assert caught.value.code == 128 + signal.SIGTERM
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        assert read(tmp_path / 'records/run_summary.json')['status'] == 'interrupted'
    finally:
        signal.signal(signal.SIGTERM, prior)


def test_existing_record_directory_untouched(tmp_path):
    root = tmp_path / 'records'
    root.mkdir()
    marker = root / 'metrics.jsonl'
    marker.write_text('prior results\n', encoding='utf-8')
    @recorded('train')
    def run(args):
        pytest.fail('calculation started')
    with pytest.raises(FileExistsError):
        run(args_at(tmp_path))
    assert marker.read_text() == 'prior results\n'


def test_record_write_failure_stops_work_and_preserves_original_error(tmp_path, monkeypatch, capsys):
    @recorded('train')
    def run(args):
        current_record().event('train_step', value=float('nan'))
        pytest.fail('continued after invalid metric')
    with pytest.raises(RecordingError, match='JSON'):
        run(args_at(tmp_path))
    assert read(tmp_path / 'records/run_summary.json')['status'] == 'failed'
    @recorded('train')
    def original_error(args):
        monkeypatch.setattr(RunRecord, 'finish', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
        raise ValueError('original failure')
    with pytest.raises(ValueError, match='original failure'):
        original_error(SimpleNamespace(record_dir=str(tmp_path / 'second')))
    assert 'disk full' in capsys.readouterr().err


def test_write_failure_before_model_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(RunRecord, 'write', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    @recorded('train')
    def run(args):
        pytest.fail('calculation started')
    with pytest.raises(OSError, match='disk full'):
        run(args_at(tmp_path))


@pytest.mark.parametrize('entry,field,value,parent', [
    ('train', 'output_trained_model_dir', 'run_', 'run_records'),
    ('train', 'output_trained_model_dir', 'out/', 'out/records'),
    ('inference', 'dump_dir', 'out', 'out/records'),
    ('export', 'out_dir', 'out', 'out/records'),
])
def test_default_record_locations_and_unique_invocations(tmp_path, entry, field, value, parent):
    @recorded(entry)
    def run(args):
        return current_record().directory
    args = SimpleNamespace(**{field: str(tmp_path) + '/' + value})
    first, second = run(args), run(args)
    assert first != second and first.parent == tmp_path / parent
    assert read(first / 'command.json')['argv'] is None


def test_tensor_and_rng_configuration_not_expanded():
    from randomness import DiffusionRNGStream
    value = json_value({'tensor': torch.zeros(10, 10), 'stream': DiffusionRNGStream(7), 'infinite_config': float('inf')})
    text = json.dumps(value, allow_nan=False)
    assert value['tensor']['shape'] == [10, 10] and len(text) < 500


def test_preflight_default_no_records_and_explicit_opt_in(tmp_path):
    import train
    from test_b2_inputs import train_fixture
    args, _ = train_fixture(tmp_path)
    args.check_inputs = True
    train.main(args)
    assert not (tmp_path / 'run_records').exists()
    args.record_dir = str(tmp_path / 'explicit')
    train.main(args)
    assert read(tmp_path / 'explicit/run_summary.json')['status'] == 'success'
    assert read(tmp_path / 'explicit/resolved_config.json')['stages'][0]['values']['n_particles'] == 3


def test_train_metrics_and_checkpoint_records(tmp_path, fake_protenix):
    from test_b2_inputs import test_real_training_budget_and_lrs
    test_real_training_budget_and_lrs(tmp_path, fake_protenix)
    root = next((tmp_path / 'run_records').iterdir())
    summary = read(root / 'run_summary.json')
    assert summary['status'] == 'success'
    events = [json.loads(line) for line in (root / 'metrics.jsonl').read_text(encoding="utf-8").splitlines()]
    steps = [e for e in events if e['event'] == 'train_step']
    assert [e['global_step'] for e in steps] == [1, 2, 3]
    for row in steps:
        assert row['total_loss'] == row['frc_loss'] + row['penalty']
        assert sorted(row['learning_rates']) == [.007, .008, .009]
    assert sum(a['kind'] == 'checkpoint' for a in summary['artifacts']) == 2
    assert all(a['exists'] for a in summary['artifacts'])
    stages = read(root / 'resolved_config.json')['stages']
    assert any(s['stage'] == 'cache_configuration' for s in stages)
    stacks = next(s['values'] for s in stages if s['stage'] == 'particle_stack_identity')
    assert len(stacks) == 1 and stacks[0]['exists'] and not stacks[0]['hash_verified']
    assert read(root / 'requested_config.json')['parser_defaults']['seed'] == 42
    import train
    from run_recording import file_identity
    before = file_identity(tmp_path / 'run_2.pth', checksum=True)
    previous_args = read(root / 'requested_config.json')['arguments']
    with pytest.raises(FileExistsError):
        train.main(SimpleNamespace(**previous_args))
    assert file_identity(tmp_path / 'run_2.pth', checksum=True) == before
    assert len(list((tmp_path / 'run_records').iterdir())) == 2


def test_export_artifact_record(tmp_path, fake_protenix):
    from test_foundations import test_real_get_pdb_block_export
    test_real_get_pdb_block_export(tmp_path, fake_protenix, 'pair_z')
    root = next((tmp_path / 'out/records').iterdir())
    summary = read(root / 'run_summary.json')
    assert summary['status'] == 'success' and len(summary['artifacts']) == 1
    assert summary['artifacts'][0]['path'].endswith('test_block_prediction.cif')


@pytest.mark.parametrize('fail', [False, True])
def test_inference_records_partial_outcomes(tmp_path, monkeypatch, fail):
    import inference
    from test_b2_inference import config, mock_runtime, Runner
    cfg, targets = config(tmp_path)
    mock_runtime(monkeypatch, targets)
    @recorded('inference')
    def run(args):
        return inference.infer_predict(Runner(cfg, 'first' if fail else None), cfg)
    if fail:
        with pytest.raises(RuntimeError, match='failure'):
            run(args_at(tmp_path))
    else:
        run(args_at(tmp_path))
    assert read(tmp_path / 'records/run_summary.json')['status'] == ('failed' if fail else 'success')
    rows = [json.loads(line) for line in (tmp_path / 'records/metrics.jsonl').read_text(encoding="utf-8").splitlines()]
    assert sum(row['event'] == 'prediction_result' for row in rows) == 4


def test_inference_record_failure_is_not_recoverable_target_error(tmp_path, monkeypatch):
    import inference
    from test_b2_inference import config, mock_runtime, Runner
    cfg, targets = config(tmp_path)
    mock_runtime(monkeypatch, targets)
    runner = Runner(cfg)
    @recorded('inference')
    def run(args):
        monkeypatch.setattr(RunRecord, 'resolved', lambda *a, **k: (_ for _ in ()).throw(RecordingError('disk full')))
        inference.infer_predict(runner, cfg)
    with pytest.raises(RecordingError, match='disk full'):
        run(args_at(tmp_path))
    assert runner.calls == []


def test_inference_original_argv_survives_rewrite(tmp_path, monkeypatch):
    import inference
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'name': 'target', 'sequences': [{'proteinChain': {'sequence': 'AA', 'count': 1}}]}]), encoding='utf-8')
    original = ['inference.py', '--input_json_path', str(path), '--seeds', '101', '--record-dir', str(tmp_path / 'records')]
    args, leftovers = inference.build_parser().parse_known_args(original[1:])
    args.original_argv, args.protenix_args = original, leftovers
    monkeypatch.setattr(sys, 'argv', ['inference.py', *leftovers])
    monkeypatch.setattr(inference, '_load_runtime_imports', lambda: (_ for _ in ()).throw(ValueError('runtime unavailable')))
    with pytest.raises(ValueError, match='runtime unavailable'):
        inference.run(args)
    assert read(tmp_path / 'records/command.json')['argv'] == original
    assert read(tmp_path / 'records/requested_config.json')['arguments']['protenix_args'] == ['--seeds', '101']


def test_inference_configuration_override_stages(tmp_path, monkeypatch):
    import inference
    class Config(dict):
        def __init__(self, values):
            super().__init__((k, Config(v) if isinstance(v, dict) else v) for k, v in values.items())
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc
        def __setattr__(self, key, value):
            self[key] = Config(value) if isinstance(value, dict) else value
    model = 'protenix_base_default_v1'
    base = dict(model_name=model, model={'N_cycle': 2}, sample_diffusion={'N_step': 200},
                enable_diffusion_shared_vars_cache=True, dtype='fp32', triangle_multiplicative='torch',
                triangle_attention='torch', enable_tf32=False, enable_efficient_fusion=False)
    replacements = dict(configs_base=base, data_configs={}, inference_configs={},
                        model_configs={model: {'model': {'N_cycle': 4}}},
                        parse_sys_args=lambda: '', parse_configs=lambda configs, **kwargs: Config(configs),
                        _load_runtime_imports=lambda: None, update_gpu_compatible_configs=lambda cfg: cfg,
                        _prepare_inference=lambda cfg: ([{'name': 'target'}], [101], {}),
                        download_inference_cache=lambda cfg: None, main=lambda cfg, prepared: None)
    for name, value in replacements.items():
        monkeypatch.setattr(inference, name, value, raising=False)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps([{'name': 'target', 'sequences': [{'proteinChain': {'sequence': 'AA', 'count': 1}}]}]))
    args = inference.build_parser().parse_args(['--input_json_path', str(source), '--N_step', '7', '--record-dir', str(tmp_path / 'records')])
    args.protenix_args = []
    inference.run(args)
    stages = {s['stage']: s for s in read(tmp_path / 'records/resolved_config.json')['stages']}
    assert stages['base_defaults']['values']['model']['N_cycle'] == 2
    assert stages['protenix_cli']['values']['model']['N_cycle'] == 4
    assert stages['protenix_cli']['values']['sample_diffusion']['N_step'] == 200
    assert stages['cocofold_cli']['values']['configs']['sample_diffusion']['N_step'] == 7
    assert stages['prediction_plan']['values']['prediction_seeds'] == [101]
