"""Run T4 in a clean public copy: two CPU Gloo ranks, analytic denoiser only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_public_tests import stage_public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New results directory; refuses overwrite.')
    parser.add_argument('--timeout',type=int,default=300,help='Two-process test timeout in seconds (default: 300).')
    args=parser.parse_args()
    if args.timeout<=0: parser.error('--timeout must be positive')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    stage=out/'public_source'
    hashes=stage_public(Path(__file__).resolve().parents[1],stage)
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not k.startswith(('COCOFOLD2_','RANK','WORLD_SIZE','LOCAL_RANK'))}
    env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',
        PYTHONPATH=str(stage/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(stage/'tests/public/parallel_worker.py'),'--output',str(out/'gloo')]
    started=time.monotonic(); result=dict(passed=False,level='T4',real_weights=False,real_protenix=False,command=command)
    try:
        with (out/'gloo.log').open('w',encoding='utf-8') as log:
            execution=subprocess.run(command,cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        result['returncode']=execution.returncode
        report=out/'gloo/summary.json'
        if execution.returncode==0 and report.is_file():
            result.update(json.loads(report.read_text(encoding='utf-8')))
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    result['elapsed_seconds']=time.monotonic()-started
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'summary.txt').write_text(f"T4 passed: {result['passed']}\nTwo real Gloo processes; analytic denoiser, no real Protenix weights.\nSee gloo.log and gloo/summary.json.\n",encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
