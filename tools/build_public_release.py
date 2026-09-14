"""Copy only the public inventory into a new release directory and ZIP.

Run from any working directory. No Git operation or publishing is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import zipfile


def build(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    # The shared staging routine also validates traversal and resolved source paths.
    stage_public = runpy.run_path(str(repo / 'tests/run_public_tests.py'))['stage_public']
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'CoCoFold2'
    hashes = stage_public(repo, source)
    archive = output / 'CoCoFold2-public.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for rel in sorted(hashes):
            handle.write(source / rel, 'CoCoFold2/' + rel)
    with zipfile.ZipFile(archive) as handle:
        for rel, expected in hashes.items():
            actual = hashlib.sha256(handle.read('CoCoFold2/' + rel)).hexdigest()
            if actual != expected:
                raise ValueError('Archive hash mismatch: ' + rel)
    report = dict(algorithm='sha256', files=hashes, total=len(hashes),
                  archive=archive.name, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  archive_verified=True, published=False)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New delivery directory; existing directories are refused.')
    args = parser.parse_args()
    report = build(Path(__file__).resolve().parents[1], args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
