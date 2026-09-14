"""Installed as sitecustomize ONLY by run_public_tests.py, including in children.

The CPU gate substitutes the denoiser in test fixtures. It must never obtain
real Protenix code/weights, import private research code, or initialize CUDA.
This is a test guard, not an operating-system security sandbox.
"""
import importlib.abc
import os
import sys

if os.environ.get('COCOFOLD2_CPU_GATE') == '1':
    class BlockModelImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] == 'protenix' or 'hetero' in fullname.split('.'):
                raise ModuleNotFoundError('CPU gate blocks model/private import: ' + fullname)

    sys.meta_path.insert(0, BlockModelImports())

    def no_network(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise RuntimeError('CPU gate blocks network access: ' + event)

    sys.addaudithook(no_network)
    import torch

    def no_cuda(*args, **kwargs):
        raise RuntimeError('CPU gate blocks CUDA initialization')

    torch.cuda._lazy_init = no_cuda
    torch.cuda.init = no_cuda
    sys._cocofold2_cpu_guard = True
