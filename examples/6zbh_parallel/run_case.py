"""Run the complete 6ZBH 1+3 example, or check existing outputs on CPU."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def command(args):
    """Explicit author-validated example parameters; trainer defaults are unchanged."""
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(ROOT/'src/chain_parallel/train_chain_parallel_2d.py'),
        '--component_manifest', str(args.manifest.resolve()), '--star_data_dir', str(args.star.resolve()),
        '--mrc_data_dir', str(args.mrc_dir.resolve()), '--output_trained_model_dir', str(args.output.resolve()/'model_'),
        '--record-dir', str(args.output.resolve()/'records'), '--backend', 'nccl',
        '--boxsize', '288', '--apix', '1.073', '--resolution', '3', '--map_resolution', '2.146',
        '--projection-frame', 'fixed', '--projection-origin', '154.512', '154.512', '154.512',
        '--batch_size', '32', '--mini_batch_size', '16', '--particle_sign', '-1', '--transR',
        '--train_deterministic', '--seed', '42', '--rng-mode', 'legacy',
        '--learn-gmm', '--gmm-kernel', 'legacy', '--gmm-amplitude', 'auto',
        '--output-format', 'cif', '--epochs', str(args.epochs),
        *(['--submission-script', str(args.submission_script.resolve())] if args.submission_script else [])]


def run(args):
    launch = command(args)
    if args.dry_run:
        print(json.dumps(dict(command=launch,resource_root=str(args.resource_root.resolve()),
                             particle_subset='entire supplied STAR; no truncation'),indent=2))
        return
    for path in (args.manifest, args.star):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.resource_root, args.mrc_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.submission_script and not args.submission_script.is_file():
        raise FileNotFoundError(args.submission_script)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PROTENIX_ROOT_DIR=str(args.resource_root.resolve()),
        COCOFOLD2_ROOT=str(ROOT/'src'), CHAIN_PARALLEL_DIR=str(ROOT/'src/chain_parallel'))
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT/'src/chain_parallel')])
    started = time.monotonic()
    record = dict(schema_version=1, started_at=timestamp(), argv=sys.argv, command=launch,
                  cwd=str(ROOT), resource_root=env['PROTENIX_ROOT_DIR'], epochs=args.epochs,
                  particle_subset='entire supplied STAR; no truncation', status='running', stages=[])
    (out/'launcher.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    try:
        preflight = [sys.executable, launch[5], *launch[6:], '--check-inputs']
        for name, cmd in [('preflight',preflight),('training',launch)]:
            start = time.monotonic()
            with (out/(name+'.log')).open('x',encoding='utf-8') as log:
                result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['stages'].append(dict(stage=name,command=cmd,returncode=result.returncode,
                                         elapsed_seconds=time.monotonic()-start))
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}; see {out/(name+".log")}')
        record['status'] = 'success'
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/'summary.json').write_text(json.dumps(dict(passed=False,training_complete=False,
            error=record['error'],structure_quality='pending author review'),indent=2)+'\n',encoding='utf-8')
        raise
    finally:
        record.update(ended_at=timestamp(),elapsed_seconds=time.monotonic()-started)
        (out/'launcher.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    from check_run import check_run, write_report
    report = check_run(out,args.epochs)
    write_report(out,report)
    if not report['passed']:
        raise RuntimeError(f'Output validation failed; see {out/"summary.json"}. Use check to recheck without retraining.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage',required=True)
    train = sub.add_parser('run',help='Fresh two-GPU full-STAR refinement; never overwrite a run directory.')
    for flag in ('manifest','star','mrc-dir','resource-root','output'):
        train.add_argument('--'+flag,type=Path,required=True, help={'manifest':'Two-component YAML; embedded paths are manifest-relative.', 'star':'Full STAR file; no particle truncation.', 'mrc-dir':'Root for relative STAR image paths.', 'resource-root':'Directory containing common/ and checkpoint/.', 'output':'New output directory; refuses overwrite.'}[flag])
    train.add_argument('--epochs',type=positive,default=10,help='Example target epochs; default 10.')
    train.add_argument('--submission-script',type=Path, help='Optional submitted shell/Slurm script copied verbatim into run records. Default: %(default)s.')
    train.add_argument('--dry-run',action='store_true',help='Print exact command without imports, files or GPU work.')
    check = sub.add_parser('check',help='Inspect an existing complete example on CPU; never decode or train.')
    check.add_argument('--output',type=Path,required=True, help='Output path for this command; relative paths use the working directory. Existing training output directories are refused. Default: %(default)s.')
    check.add_argument('--epochs',type=positive,default=10, help='Number of epochs to run or audit; partial-prefix acceptance must be explicitly requested. Default: %(default)s.')
    check.add_argument('--completed-prefix',action='store_true',help='Audit only the first N complete epochs; preserve original target/status and write a separate acceptance report.')
    check.add_argument('--reason',help='Required with --completed-prefix; record why the acceptance scope changed.')
    args = parser.parse_args()
    if args.stage == 'run':
        run(args)
    else:
        if args.completed_prefix and not (args.reason and args.reason.strip()):
            parser.error('--completed-prefix requires --reason')
        from check_run import check_run, write_report
        result = check_run(args.output.resolve(),args.epochs,args.completed_prefix,args.reason)
        write_report(args.output.resolve(),result)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
