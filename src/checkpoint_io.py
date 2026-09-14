"""CPU snapshots and atomic, non-overwriting checkpoint publication."""
import copy
import os
from pathlib import Path
import tempfile
from types import ModuleType

import torch


def cpu_snapshot(value):
    """Detach tensors directly to CPU before deepcopy, including config attributes.

    The memo preserves aliases and cycles without cloning model weights on CUDA.
    Other Python configuration types retain their original serialization types.
    """
    memo, seen = {}, set()
    def visit(item):
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            memo[identity] = item.detach().to(device='cpu', copy=True)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)
        elif not isinstance(item, (type, ModuleType)) and hasattr(item, '__dict__'):
            for child in vars(item).values():
                visit(child)
    visit(value)
    return copy.deepcopy(value, memo)


def atomic_save_checkpoint(state, path):
    """Publish a complete file only; neither existing files nor concurrent writers win by overwriting.

    A same-directory hard link atomically publishes the finished temporary file.
    Filesystems without hard-link support fail explicitly; no unsafe overwrite fallback.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = cpu_snapshot(state)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.' + path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(snapshot, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
