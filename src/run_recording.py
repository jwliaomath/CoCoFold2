"""Per-invocation experiment records, independent of scientific RNG and artifacts."""
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import sys
import time
import traceback
import uuid

_active = ContextVar('cocofold2_run_record', default=None)
BEIJING = timezone(timedelta(hours=8), name='Asia/Shanghai')


class RecordingError(RuntimeError):
    """A record failure must not be treated as a recoverable inference target error."""


def recording_operation(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RecordingError:
            raise
        except Exception as exc:
            raise RecordingError(f'{function.__name__}: {exc}') from exc
    return call


def timestamp():
    return datetime.now(BEIJING).isoformat()


def json_value(value):
    """Describe tensors/RNG objects without copying weights or traversing RNG state."""
    import torch
    import numpy as np
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'unavailable_reason': 'nonfinite_config_value', 'value': str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return dict(kind='tensor', shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist()) if value.size <= 128 else dict(kind='array', shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if type(value).__name__ == 'DiffusionRNGStream':
        return dict(kind='DiffusionRNGStream', seed=value.seed, state_recorded_in_checkpoint=True)
    if hasattr(value, 'to_dict'):
        return json_value(value.to_dict())
    if type(value).__module__ in ('argparse', 'types') and hasattr(value, '__dict__'):
        return json_value(vars(value))
    return dict(kind=type(value).__module__ + '.' + type(value).__name__, description=str(value)
                if isinstance(value, (torch.dtype, torch.device)) else 'value not serialized')


def file_identity(path, checksum=False):
    path = Path(path).expanduser().resolve()
    result = dict(path=str(path), exists=path.is_file(), sha256=None, hash_verified=False)
    if path.is_file():
        stat = path.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if checksum:
            result.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), hash_verified=True)
    result['identity_method'] = 'sha256' if result['hash_verified'] else 'path_size_mtime_only'
    return result


def environment():
    import torch
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  torch=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_initialized=torch.cuda.is_initialized(),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  process={k: os.environ.get(k) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'SLURM_JOB_ID', 'CUDA_VISIBLE_DEVICES')})
    if torch.cuda.is_initialized():
        result['devices'] = [dict(index=i, name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    for name in ('numpy', 'starfile', 'gemmi', 'protenix'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def add_record_arguments(parser):
    parser.add_argument('--record-dir', default=None, help='New directory for this invocation; existing directories are rejected.')
    parser.add_argument('--submission-script', default=None, help='Copy this submitted shell/Slurm script verbatim into the run record.')


class NullRecord:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def current_record():
    return _active.get() or NullRecord()


class RunRecord:
    def __init__(self, args, entry):
        self.entry = entry
        self.rank = int(os.environ.get('RANK', '0'))
        self.run_id = uuid.uuid4().hex
        self.created_at = timestamp()
        job_id = os.environ.get('SLURM_JOB_ID', '')
        # Scheduler IDs are labels only. Never allow environment text to form a path.
        self.slurm_job_id = job_id if job_id.isascii() and job_id.isdigit() else None
        label = 'job' + self.slurm_job_id if self.slurm_job_id else self.run_id[:8]
        name = datetime.fromisoformat(self.created_at).strftime('%Y%m%d_%H%M%S') + '_' + label
        if self.rank != 0 or int(os.environ.get('WORLD_SIZE', '1')) > 1:
            name += f'_rank{self.rank}'
        explicit = getattr(args, 'record_dir', None)
        if explicit:
            self.directory = Path(explicit).expanduser().resolve()
            self.directory.mkdir(parents=True, exist_ok=False)
        else:
            if entry == 'train':
                root = str(getattr(args, 'output_trained_model_dir', './')) + 'records'
            else:
                root = Path(getattr(args, 'dump_dir' if entry == 'inference' else 'out_dir', '.')) / 'records'
            root = Path(root).expanduser().resolve()
            # Repeated invocations in one Slurm job/second still get independent records.
            for attempt in range(10):
                suffix = '' if attempt == 0 else '_' + uuid.uuid4().hex[:8]
                self.directory = root / (name + suffix)
                try:
                    self.directory.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    if attempt == 9:
                        raise
        self.started = time.monotonic()
        self.artifacts = []
        self.stages = []
        self.metric_path = self.directory / ('metrics.jsonl' if self.rank == 0 else f'metrics_rank{self.rank}.jsonl')

    @recording_operation
    def write(self, name, value):
        text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        temporary = self.directory / (name + '.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        os.replace(temporary, self.directory / name)

    def start(self, args, defaults=None):
        self.write('run_identity.json', dict(schema_version=1, run_id=self.run_id,
                    directory_name=self.directory.name, created_at=self.created_at,
                    timezone='Asia/Shanghai', utc_offset='+08:00',
                    slurm_job_id=self.slurm_job_id, rank=self.rank))
        argv = getattr(args, 'original_argv', None)
        command = dict(argv=argv, argv_source='captured_before_parse' if argv is not None else 'unavailable_python_api_call',
                       executable=sys.executable, cwd=os.getcwd(), started_at=timestamp(),
                       shell_text_reconstructed=True, original_shell_quoting_available=False)
        self.write('command.json', command)
        rendered = shlex.join([sys.executable, *argv]) if argv is not None else '# Python API invocation; original command unavailable'
        (self.directory / 'command.txt').write_text('# POSIX shell reconstruction; see command.json.\n' + rendered + '\n', encoding='utf-8')
        self.write('requested_config.json', dict(schema_version=1, arguments=json_value(vars(args)), parser_defaults=json_value(defaults)))
        self.write('resolved_config.json', dict(schema_version=1, stages=[], status='not_resolved'))
        self.write('environment.json', environment())
        root = Path(__file__).resolve().parent
        sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob('*.py'))
                   if 'hetero' not in p.relative_to(root).parts
                   and p.name not in ('train_hetero.py', 'predict_hetero.py')}
        inputs = {}
        for name in ('diffusion_data_dir', 'star_data_dir', 'cif_path', 'input_json_path', 'block_alignment', 'halfmap1', 'halfmap2'):
            if getattr(args, name, None):
                inputs[name] = file_identity(getattr(args, name))
        self.write('provenance.json', dict(schema_version=1, source_root=str(root), source_sha256=sources,
                                         commit=None, commit_unavailable_reason='source_hash_manifest_used', inputs=inputs,
                                         large_input_hash_policy='metadata_only; no full cache/MRCS hashing'))
        script = getattr(args, 'submission_script', None)
        if script:
            source = Path(script).expanduser().resolve()
            shutil.copyfile(source, self.directory / 'submitted_script.sh')
            self.write('submission_script.json', file_identity(source, checksum=True))
        self.write('run_summary.json', dict(schema_version=1, run_id=self.run_id, status='running', entry=self.entry))
        self.event('start', entry=self.entry, rank=self.rank)
        print(f'Experiment records: {self.directory}', flush=True)

    @recording_operation
    def resolved(self, stage, values, reason):
        self.stages.append(dict(stage=stage, reason=reason, values=json_value(values)))
        self.write('resolved_config.json', dict(schema_version=1, status='resolved', stages=self.stages))

    @recording_operation
    def artifact(self, path, kind):
        self.artifacts.append(dict(kind=kind, **file_identity(path)))
        self.event('artifact', kind=kind, path=str(Path(path).resolve()))

    @recording_operation
    def event(self, event, **values):
        record = dict(schema_version=1, run_id=self.run_id, timestamp=timestamp(), event=event,
                      global_step=None, epoch=None, batch=None)
        record.update(values)
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.metric_path.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(text + '\n')

    def refresh_environment(self):
        self.write('environment.json', environment())

    def finish(self, status, exc=None):
        error = None if exc is None else dict(type=type(exc).__name__, message=str(exc))
        if exc is not None:
            (self.directory / 'failure.txt').write_text(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding='utf-8')
        self.event(status, error=error)
        self.write('run_summary.json', dict(schema_version=1, run_id=self.run_id, entry=self.entry,
                                          status=status, error=error, ended_at=timestamp(),
                                          elapsed_seconds=time.monotonic() - self.started, artifacts=self.artifacts,
                                          termination_limit='SIGKILL/power loss cannot be caught; running means completion unconfirmed'))


def recorded(entry):
    def decorate(function):
        @wraps(function)
        def invoke(args, *positional, **kwargs):
            if getattr(args, 'check_inputs', False) and not getattr(args, 'record_dir', None):
                return function(args, *positional, **kwargs)
            output_key = {'train': 'output_trained_model_dir', 'inference': 'dump_dir', 'export': 'out_dir'}[entry]
            if not hasattr(args, output_key) and not getattr(args, 'record_dir', None):
                # Incomplete Python API calls have no known destination; preserve validation errors.
                return function(args, *positional, **kwargs)
            record = RunRecord(args, entry)
            token = _active.set(record)
            old_handler = None
            try:
                if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                    def terminated(signum, frame):
                        raise SystemExit(128 + signum)
                    try:
                        old_handler = signal.signal(signal.SIGTERM, terminated)
                    except ValueError:  # API invocation in a worker thread.
                        pass
                parser_factory = function.__globals__.get('build_parser')
                defaults = ({a.dest: a.default for a in parser_factory()._actions if a.dest != 'help'}
                            if parser_factory is not None else None)
                record.start(args, defaults)
                result = function(args, *positional, **kwargs)
                record.finish('success')
                return result
            except BaseException as exc:
                try:
                    record.finish('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', exc)
                except BaseException as recording_error:
                    print(f'Recording also failed: {recording_error}; preserving original {type(exc).__name__}: {exc}', file=sys.stderr)
                raise
            finally:
                _active.reset(token)
                if old_handler is not None:
                    signal.signal(signal.SIGTERM, old_handler)
        return invoke
    return decorate
