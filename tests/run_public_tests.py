"""Run the public T0-T3 CPU gate from a fresh allowlisted copy.

Usage: python tests/run_public_tests.py --output /path/to/new/results
No GPU, Protenix installation, weights, or research checkout is needed.
"""
import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


EXCLUDED = [
    'tests/public/test_b2b_randomness.py::test_cuda_rng_isolation',
    'tests/test_gmm.py::GaussianTests::test_cuda_forward_backward_all_modes_and_mixed_legacy_devices',
    'tests/test_gmm_peak3d.py::Peak3DTests::test_peak3d_cuda_backward_including_cpu_amplitudes',
    'tests/test_gmm_peak2d_checkpoint.py::Peak2DCheckpointTests::test_cuda_multiple_adamw_steps_and_cpu_amplitudes',
]
DESCRIPTIONS = {
    'test_alignment_sampler': 'Original sampler, per-chain transforms, saving and export',
    'test_b2_inputs': 'STAR/MRCS, cache and argument preflight',
    'test_b2_inference': 'Inference targets, paths, overwrite protection and failure summaries (mock model)',
    'test_b2b_randomness': 'Seeds, fixed/resampled noise and RNG isolation (analytic model)',
    'test_b3_recording': 'Commands, configuration, JSONL and failure records',
    'test_b4_training': 'CPU training, GMM controls and CIF/PDB output',
    'test_b4b_alignment': 'Per-chain rigid fitting, atom identities and independent updates',
    'test_b5_checkpoint': 'Synchronized export, atomic saving and sampling replay',
    'test_b5_legacy_options': 'Legacy fusion configuration compatibility',
    'test_b5_legacy_raw': 'Raw export of legacy files with missing transforms',
    'test_checkpoint_io': 'CPU snapshots, atomic writes and overwrite protection',
    'test_epoch_restart': 'Epoch resume, warm-start, optimizers and failure saving policy',
    'test_foundations': 'Public decoder, gradients, structure I/O and entrypoints',
    'test_record_names': 'Beijing timestamps, job IDs and random record identifiers',
    'test_gmm': 'Real GMM projections and gradients against the historical implementation',
    'test_gmm_width_init': 'Opt-in physical width mapping, legacy defaults and saved GMM precedence',
    'test_gmm_peak3d': 'GMM kernels and amplitude conventions',
    'test_gmm_peak2d_checkpoint': 'GMM chunk-checkpoint projections and gradients',
    'test_public_gate': 'Independent distribution, model-free imports, CLI and subprocess preflight',
    'test_b6_case': 'Single-state map simulation, STAR readback, reference CIF and output checks',
    'test_b6_cli': 'B6 data-generation and staged-run entrypoints',
    'test_b7b_parallel': 'Parallel parameters, chain fitting, restart and legacy defaults (analytic model)',
    'test_b7b_cli': 'Model-free parallel CLI argument checks',
    'test_b8_case': '6ZBH example scripts, outputs, progress, records and performance reports (no model execution)',
    'test_b9_release': 'Public packaging boundaries, dependencies, documentation links and argument help',
}


def classify(classname, name):
    module = next((key for key in DESCRIPTIONS if key in classname.split('.')), '')
    if name.startswith('test_t0_') or 'modules_import' in name:
        level = 'T0'
    elif name.startswith('test_t1_') or 'cli_help' in name or 'lightweight_cli' in name or module in ('test_b6_cli','test_b7b_cli'):
        level = 'T1'
    elif name.startswith('test_t2_') or (module == 'test_b2_inputs' and 'training_budget' not in name) or 'test_b6_bad_parameters' in name or 'test_b6_invalid_map' in name:
        level = 'T2'
    else:
        level = 'T3'
    return level, DESCRIPTIONS.get(module, module)


def stage_public(repo, destination):
    paths = json.loads((repo / 'tests/public_files.json').read_text(encoding='utf-8'))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Public file manifest must be nonempty and unique')
    hashes = {}
    for rel in paths:
        parts = PurePosixPath(rel).parts
        if not parts or PurePosixPath(rel).is_absolute() or '..' in parts or '\\' in rel or ':' in rel or 'hetero' in parts:
            raise ValueError('Unsafe/private public path: ' + rel)
        source = (repo / rel).resolve()
        source.relative_to(repo)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New results directory; never overwrite an earlier report.')
    parser.add_argument('--timeout', type=int, default=900, help='Whole pytest process timeout in seconds (default: 900).')
    parser.add_argument('--groups', nargs='+', choices=['public','entrypoints','gmm','b6','b6_cli','b7b','b7b_cli','b8','b9'],
                        help='Run only affected groups; omitted means the complete public gate.')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out / 'public_source'
    summary = dict(batch='B6' if args.groups and set(args.groups) <= {'b6','b6_cli'} else 'public_cpu', passed=False, real_weights=False, real_protenix=False,
                   decoder='analytic test substitute', cuda=False, excluded_gpu_tests=EXCLUDED,
                   deferred=['Random/fine-tuning outside current public release scope', 'Multi-process and real-weight validation are separate from this CPU gate'], tests=[])
    summary.update(scope='selected_groups' if args.groups else 'complete_public_gate', selected_groups=args.groups)
    if args.groups and set(args.groups) <= {'b7b','b7b_cli'}:
        summary['batch']='B7b'
    if args.groups == ['b8']:
        summary['batch']='B8'
        summary['decoder']='none; output audit fixtures only'
        summary['deferred']=['B8 full real training and author structural review']
    if args.groups == ['b9']:
        summary['batch']='B9'
        summary['decoder']='none; release tooling and metadata only'
        summary['deferred']=['Fresh full GPU installation and final GitHub review']
    try:
        hashes = stage_public(repo, stage)
        (out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
        guard = out / 'cpu_guard'
        guard.mkdir()
        shutil.copy2(stage / 'tests/public/cpu_guard.py', guard / 'sitecustomize.py')
        env = {k: v for k, v in os.environ.items() if not k.startswith('COCOFOLD2_')}
        env.update(COCOFOLD2_CPU_GATE='1', CUDA_VISIBLE_DEVICES='-1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg',
                   PYTHONPATH=os.pathsep.join([str(guard), str(stage / 'src'), str(stage / 'tests/public')]),
                   TORCH_HOME=str(out / 'unused_torch_cache'), HF_HOME=str(out / 'unused_hf_cache'))
        groups = {
            'public': ['tests/public', '--ignore=tests/public/test_public_gate.py',
                       '--ignore=tests/public/test_b6_case.py','--ignore=tests/public/test_b6_cli.py',
                       '--ignore=tests/public/test_b7b_parallel.py','--ignore=tests/public/test_b7b_cli.py',
                       '--ignore=tests/public/test_b8_case.py','--ignore=tests/public/test_b9_release.py'],
            'entrypoints': ['tests/public/test_public_gate.py'],
            'gmm': ['tests/test_gmm.py', 'tests/test_gmm_peak3d.py', 'tests/test_gmm_peak2d_checkpoint.py', 'tests/test_gmm_width_init.py'],
            'b6': ['tests/public/test_b6_case.py'],
            'b6_cli': ['tests/public/test_b6_cli.py'],
            'b7b': ['tests/public/test_b7b_parallel.py'],
            'b7b_cli': ['tests/public/test_b7b_cli.py'],
            'b8': ['tests/public/test_b8_case.py'],
            'b9': ['tests/public/test_b9_release.py'],
        }
        if args.groups:
            groups = {name: selected for name, selected in groups.items() if name in args.groups}
        commands = {name: [sys.executable, '-m', 'pytest', *selected, '-v', '--tb=short', '-p', 'no:cacheprovider',
                          '--junitxml=' + str(out / (name + '.xml')), '--basetemp=' + str(out / ('fixtures_' + name)),
                          *['--deselect=' + item for item in EXCLUDED]] for name, selected in groups.items()}
        versions = {}
        for package in ('pytest', 'torch', 'numpy', 'scipy', 'pandas', 'gemmi', 'starfile', 'mrcfile', 'biopython'):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        environment = dict(python=sys.version, platform=platform.platform(), packages=versions,
                           commands=commands, cwd=str(stage), argv=sys.argv, timeout=args.timeout,
                           gate_environment={k: env[k] for k in ('COCOFOLD2_CPU_GATE', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'OMP_NUM_THREADS')})
        (out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
        deadline = time.monotonic() + args.timeout
        summary['group_exit_codes'] = {}
        combined = ET.Element('testsuites')
        for name, command in commands.items():
            with (out / (name + '.log')).open('w', encoding='utf-8') as log:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('CPU gate exceeded its total timeout')
                result = subprocess.run(command, cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            summary['group_exit_codes'][name] = result.returncode
            with (out / 'tests.log').open('a', encoding='utf-8') as log:
                log.write('\nGroup: ' + name + '\n')
                log.write((out / (name + '.log')).read_text(encoding='utf-8', errors='replace'))
            report = out / (name + '.xml')
            if report.is_file():
                for suite in ET.parse(report).iter('testsuite'):
                    combined.append(suite)
            else:
                summary.setdefault('missing_reports', []).append(name)
        ET.ElementTree(combined).write(out / 'tests.xml', encoding='utf-8', xml_declaration=True)
        for case in combined.iter('testcase'):
            failure = case.find('failure')
            if failure is None:
                failure = case.find('error')
            skip = case.find('skipped')
            classname, name = case.get('classname', ''), case.get('name', '')
            level, description = classify(classname, name)
            summary['tests'].append(dict(id=classname + '::' + name, description=description + ' — ' + name,
                level=level, status='FAIL' if failure is not None else 'SKIP' if skip is not None else 'PASS',
                seconds=float(case.get('time', 0)), detail=failure.get('message', '') if failure is not None else skip.get('message', '') if skip is not None else ''))
        counts = Counter(row['status'] for row in summary['tests'])
        levels = {level: dict(Counter(row['status'] for row in summary['tests'] if row['level'] == level)) for level in ('T0', 'T1', 'T2', 'T3')}
        summary.update(total=len(summary['tests']), counts={key: counts[key] for key in ('PASS', 'FAIL', 'SKIP')}, levels=levels)
        summary['passed'] = all(code == 0 for code in summary['group_exit_codes'].values()) and not summary.get('missing_reports') and counts['PASS'] > 0 and counts['FAIL'] == counts['SKIP'] == 0 and (bool(args.groups) or all(levels[level].get('PASS', 0) for level in levels))
    except Exception as error:
        summary.update(passed=False, error=type(error).__name__ + ': ' + str(error))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ['# Public CPU checks', '', f"Passed: {summary['passed']}; counts: {summary.get('counts', {})}",
            '', 'Analytic denoiser; real public sampler/GMM/loss/optimizer/I/O. No real Protenix or weights.',
            'Random/fine-tuning outside current public release scope. Multi-process/real parallel: B7b. Four CUDA-only tests excluded.', '',
            '| Level | Test | Result |', '|---|---|---|']
    rows.extend(f"| {r['level']} | {r['description'].replace('|', '/')} | {r['status']} |" for r in summary['tests'])
    if 'error' in summary:
        rows.extend(['', summary['error']])
    (out / 'summary.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'tests'}, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
