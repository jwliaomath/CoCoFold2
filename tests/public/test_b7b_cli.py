"""Parallel CLI help/invalid options must not import Protenix or load weights."""
from pathlib import Path
import subprocess
import sys
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'src/chain_parallel/train_chain_parallel_2d.py'


@pytest.mark.parametrize('flags,code', [(['--help'],0), ([],2),
    (['--epochs','0'],2),(['--lr_bias','nan'],2),(['--distributed-timeout','0'],2)])
def test_parallel_cli(flags,code):
    if flags and flags != ['--help']:
        flags=['--component_manifest','missing','--star_data_dir','missing','--output_trained_model_dir','unused',*flags]
    result=subprocess.run([sys.executable,str(SCRIPT),*flags],capture_output=True,text=True,timeout=30)
    assert result.returncode==code, result.stdout+result.stderr
    assert 'usage:' in result.stdout+result.stderr
    assert 'No module named' not in result.stderr
