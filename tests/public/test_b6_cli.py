"""B6 CLI checks in a process isolated from native numerical libraries."""
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('script,args', [('src/simulate_particles.py',[]),
    ('examples/7zdt_7zd5/run_case.py',[]), ('examples/7zdt_7zd5/run_case.py',['predict']),
    ('examples/7zdt_7zd5/run_case.py',['train']), ('examples/7zdt_7zd5/run_case.py',['validate'])])
def test_b6_help_without_protenix(script,args):
    result=subprocess.run([sys.executable,str(ROOT/script),*args,'--help'],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr

