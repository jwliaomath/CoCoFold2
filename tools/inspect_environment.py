"""Record installed dependencies and optionally check model imports in a fresh environment.

No weights are loaded. Does not modify or install any dependency.
"""
import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def inspect(model_imports=False):
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get('Name')}
    result = dict(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=dict(sorted(packages.items(), key=lambda item: item[0].lower())),
                  resource_root=os.environ.get('PROTENIX_ROOT_DIR'), real_weights=False)
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=120)
    result['pip_check'] = dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr)
    result['passed'] = check.returncode == 0
    if model_imports:
        code = (
            "import json,torch,protenix; "
            "from protenix.data.inference.infer_dataloader import get_inference_dataloader; "
            "print(json.dumps(dict(torch=torch.__version__,cuda_build=torch.version.cuda,"
            "protenix_file=protenix.__file__,cuda_available=torch.cuda.is_available(),"
            "devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))"
        )
        check = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=180)
        result['model_import_check'] = dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr)
        result['passed'] = result['passed'] and check.returncode == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report file; refuses overwrite.')
    parser.add_argument('--model-imports', action='store_true',
                        help='Import installed Protenix inference dependencies and inspect visible GPUs; no weights.')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before running checks to protect earlier reports.
    with args.output.open('x', encoding='utf-8') as handle:
        try:
            report = inspect(args.model_imports)
        except Exception as error:
            report = dict(passed=False, error=f'{type(error).__name__}: {error}', real_weights=False)
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'packages'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
