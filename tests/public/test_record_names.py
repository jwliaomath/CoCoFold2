"""Beijing record timestamps, portable names and collision-safe creation."""
from datetime import datetime, timezone, timedelta
import json
import re
from types import SimpleNamespace

import pytest

import run_recording as recording


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, 2, 6, 41, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(recording, 'datetime', Clock)
    for name in ('SLURM_JOB_ID', 'RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('job', [None, '1426629', '../unsafe'])
def test_beijing_names_and_full_json_identity(tmp_path, monkeypatch, fixed_clock, job):
    if job is not None:
        monkeypatch.setenv('SLURM_JOB_ID', job)
    args = SimpleNamespace(output_trained_model_dir=str(tmp_path / 'run_'))
    record = recording.RunRecord(args, 'train')
    record.start(args)
    record.finish('success')
    expected = 'job1426629' if job == '1426629' else '[0-9a-f]{8}'
    assert re.fullmatch('20260914_100641_' + expected, record.directory.name)
    identity = json.loads((record.directory / 'run_identity.json').read_text())
    assert re.fullmatch('[0-9a-f]{32}', identity['run_id'])
    assert identity['directory_name'] == record.directory.name
    assert identity['timezone'] == 'Asia/Shanghai'
    assert identity['created_at'] == '2026-09-14T10:06:41+08:00'
    assert identity['slurm_job_id'] == ('1426629' if job == '1426629' else None)
    events = [json.loads(s) for s in (record.directory / 'metrics.jsonl').read_text().splitlines()]
    assert all(e['run_id'] == identity['run_id'] for e in events)
    assert all(datetime.fromisoformat(e['timestamp']).utcoffset() == timedelta(hours=8) for e in events)
    assert json.loads((record.directory / 'run_summary.json').read_text())['run_id'] == identity['run_id']


def test_same_second_same_job_gets_fresh_directory(tmp_path, monkeypatch, fixed_clock):
    monkeypatch.setenv('SLURM_JOB_ID', '1426629')
    args = SimpleNamespace(output_trained_model_dir=str(tmp_path / 'run_'))
    first = recording.RunRecord(args, 'train')
    marker = first.directory / 'keep.txt'
    marker.write_text('original')
    second = recording.RunRecord(args, 'train')
    assert first.directory.name == '20260914_100641_job1426629'
    assert re.fullmatch(first.directory.name + '_[0-9a-f]{8}', second.directory.name)
    assert first.run_id != second.run_id and marker.read_text() == 'original'


def test_rank_label_and_explicit_directory_unchanged(tmp_path, monkeypatch, fixed_clock):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    args = SimpleNamespace(output_trained_model_dir=str(tmp_path / 'run_'))
    record = recording.RunRecord(args, 'train')
    assert record.directory.name.endswith('_rank0')
    args.record_dir = str(tmp_path / 'custom')
    assert recording.RunRecord(args, 'train').directory == tmp_path / 'custom'
    with pytest.raises(FileExistsError):
        recording.RunRecord(args, 'train')
