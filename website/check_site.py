"""Check the generated site's links and code text without network or model access."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
BASE = 'https://website.invalid/CoCoFold2/'


def check(root, repository, report):
    documents = {path.relative_to(root).as_posix(): BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
                 for path in root.rglob('*.html')}
    links = 0
    for name, soup in documents.items():
        ids = [tag['id'] for tag in soup.select('[id]')]
        assert len(ids) == len(set(ids)), f'Duplicate heading/element ID: {name}'
        for tag in soup.select('a[href], link[href], script[src], img[src]'):
            href = tag.get('href', tag.get('src'))
            url = urlsplit(urljoin(BASE + name, href))
            if url.netloc != 'website.invalid':
                continue
            assert url.path.startswith('/CoCoFold2/'), f'Escaped project URL prefix: {name}: {href}'
            target = unquote(url.path[len('/CoCoFold2/'):]) or 'index.html'
            file = (root / target).resolve()
            assert file.is_relative_to(root) and file.is_file(), f'Missing local target: {name}: {href}'
            if url.fragment:
                assert target in documents and documents[target].find(id=unquote(url.fragment)), f'Missing fragment: {name}: {href}'
            links += 1
    code_blocks = 0
    for item in report['pages']:
        source = repository / item['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['source_sha256'], 'Stale site source'
        original = re.findall(r'(?ms)^```[^\n]*\n(.*?)^```[ \t]*$', source.read_text(encoding='utf-8'))
        output = [pre.get_text() for pre in documents[item['output']].find_all('pre')]
        assert original == output, f'Changed command text: {item["source"]}'
        code_blocks += len(output)
    return dict(passed=True, pages=len(documents), internal_links_and_assets=links, unchanged_code_blocks=code_blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / '_site', help='Generated website directory.')
    parser.add_argument('--source-root', type=Path, default=HERE.parent, help='Repository Markdown root.')
    parser.add_argument('--report', type=Path, default=HERE / 'build-report.json', help='Generator provenance JSON.')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    result = check(args.output.resolve(), args.source_root.resolve(), report)
    report['validation'] = result
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
