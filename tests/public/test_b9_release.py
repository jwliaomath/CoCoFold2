"""Release boundaries and documentation checks; no model execution."""
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy
import zipfile

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]


def inventory():
    return json.loads((ROOT/'tests/public_files.json').read_text(encoding='utf-8'))


def test_t0_release_archive_roundtrip_and_private_exclusion(tmp_path):
    """A reviewed allowlist, not the contents of the research checkout, is exported."""
    builder=runpy.run_path(str(ROOT/'tools/build_public_release.py'))['build']
    result=builder(ROOT,tmp_path/'delivery')
    expected=set(inventory())
    assert set(result['files'])==expected
    assert not any('hetero' in Path(p).parts or Path(p).name in ('train_random.py','train_finetune.py','train_hetero.py') for p in expected)
    with zipfile.ZipFile(tmp_path/'delivery/CoCoFold2-public.zip') as archive:
        assert set(archive.namelist())=={'CoCoFold2/'+p for p in expected}
        for rel in expected:
            assert hashlib.sha256(archive.read('CoCoFold2/'+rel)).hexdigest()==result['files'][rel]
    with pytest.raises(FileExistsError):
        builder(ROOT,tmp_path/'delivery')


@pytest.mark.parametrize('bad', ['../outside.py','/absolute.py','src/hetero/private.py', 'src\\private.py'])
def test_t0_release_rejects_unsafe_inventory(tmp_path,bad):
    stage=runpy.run_path(str(ROOT/'tests/run_public_tests.py'))['stage_public']
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/public_files.json').write_text(json.dumps([bad]))
    with pytest.raises(ValueError):
        stage(tmp_path,tmp_path/'out')


def test_t0_ignore_matches_public_inventory():
    lines=(ROOT/'.gitignore').read_text().splitlines()
    assert '/*' in lines
    for rel in inventory():
        assert '!/'+rel in lines,rel
        for parent in Path(rel).parents:
            if parent==Path('.'): continue
            assert '!/'+parent.as_posix()+'/' in lines
            assert '/'+parent.as_posix()+'/*' in lines


def test_t1_all_public_runtime_arguments_have_help():
    for rel in inventory():
        if not rel.endswith('.py') or not rel.startswith(('src/','examples/','tools/')):
            continue
        tree=ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                assert any(k.arg=='help' for k in node.keywords),f'{rel}:{node.lineno}'


def test_t0_local_markdown_links_resolve():
    for rel in inventory():
        if not rel.endswith('.md'): continue
        text=(ROOT/rel).read_text(encoding='utf-8')
        # Public documents use inline links; external URLs and section anchors are excluded.
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith(('#','mailto:')): continue
            target=target.split('#')[0].strip('<>')
            assert (ROOT/rel).parent.joinpath(target).exists(),f'{rel}: {target}'


def test_t0_dependency_files_preserve_full_stack_and_separate_cpu():
    requirements=[s.strip() for s in (ROOT/'requirements.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
    environment=yaml.safe_load((ROOT/'environment.yml').read_text())
    pip=next(item['pip'] for item in environment['dependencies'] if isinstance(item,dict))
    assert requirements==pip
    cpu=(ROOT/'requirements-cpu.txt').read_text()
    assert 'torch==2.7.1' in cpu and 'numpy==2.4.1' in cpu
    assert not any(s.startswith(('protenix==','deepspeed==','triton==')) for s in cpu.splitlines())
    assert 'protenix==1.0.2' in requirements


def test_t0_citation_has_confirmed_version_and_no_placeholder_doi():
    citation=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    assert citation['authors'] and citation['repository-code']
    assert citation['version']=='1.0.0'
    assert 'doi' not in citation
    assert 'doi' not in citation.get('preferred-citation',{})
    assert 'TODO_DOI' not in (ROOT/'CITATION.cff').read_text()
