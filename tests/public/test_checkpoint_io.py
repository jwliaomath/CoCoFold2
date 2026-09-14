"""Checkpoint snapshots and failure-safe publication, without model weights."""
from types import SimpleNamespace

import pytest
import torch

import checkpoint_io


def test_cpu_snapshot_aliases_config_and_no_source_mutation():
    source = torch.tensor([1., 2.], requires_grad=True)
    state = dict(tensor=source, again=source, config=SimpleNamespace(nested=[source]))
    state['cycle'] = state
    copied = checkpoint_io.cpu_snapshot(state)
    assert copied['cycle'] is copied
    assert copied['tensor'] is copied['again'] is copied['config'].nested[0]
    assert not copied['tensor'].requires_grad and copied['tensor'].device.type == 'cpu'
    with torch.no_grad():
        source.add_(5)
    assert copied['tensor'].tolist() == [1., 2.]


def test_atomic_save_readback_and_no_overwrite(tmp_path):
    path = tmp_path / 'epoch.pth'
    checkpoint_io.atomic_save_checkpoint({'value': torch.ones(2)}, path)
    assert torch.equal(torch.load(path, weights_only=False)['value'], torch.ones(2))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        checkpoint_io.atomic_save_checkpoint({'value': torch.zeros(2)}, path)
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['serialize', 'publish', 'racing_writer'])
def test_failed_save_leaves_no_partial_checkpoint(tmp_path, monkeypatch, failure):
    path = tmp_path / 'epoch.pth'
    if failure == 'serialize':
        def broken(state, handle):
            handle.write(b'partial')
            raise OSError('simulated write failure')
        monkeypatch.setattr(torch, 'save', broken)
    else:
        original_link = checkpoint_io.os.link
        def publish(source, target):
            if failure == 'racing_writer':
                path.write_bytes(b'other complete checkpoint')
                original_link(source, target)
            raise OSError('simulated publication failure')
        monkeypatch.setattr(checkpoint_io.os, 'link', publish)
    with pytest.raises(OSError):
        checkpoint_io.atomic_save_checkpoint({'x': torch.zeros(2)}, path)
    if failure == 'racing_writer':
        assert path.read_bytes() == b'other complete checkpoint'
    else:
        assert not path.exists()
    assert not list(tmp_path.glob('*.tmp'))
