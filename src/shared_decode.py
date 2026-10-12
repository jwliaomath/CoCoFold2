"""Memory-bounded decoder VJP shared across particle microbatches.

Only fixed-stochasticity training shares coordinates. Renderer/GMM backward
still runs for every microbatch; the expensive decoder backward runs once.
"""
import math
import random

import numpy as np
import torch


def _rng_state(device):
    # A component rank must not create CUDA contexts on every visible GPU.
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _restore_rng(state, device):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3], device)


def add_decode_arguments(parser):
    parser.add_argument('--decode-mode', choices=('auto', 'shared', 'legacy'), default='auto',
                        help='auto shares decoding for fixed noise; resampled noise uses legacy. '
                             'shared recomputes once for decoder backward. legacy decodes every '
                             'microbatch. A single microbatch always uses one ordinary decode.')


def resolve_mode(mode, fixed):
    if mode not in ('auto', 'shared', 'legacy'):
        raise ValueError('Invalid decode mode: ' + str(mode))
    if mode == 'auto':
        return 'shared' if fixed else 'legacy'
    if mode == 'shared' and not fixed:
        raise ValueError('--decode-mode shared requires fixed stochasticity; use auto or legacy')
    return mode


def configure_decode(args):
    requested = getattr(args, 'decode_mode', 'auto')
    args.decode_mode = resolve_mode(requested, args.train_deterministic)
    print(f'[decode] requested={requested}, resolved={args.decode_mode}; '
          'one microbatch uses one ordinary decode', flush=True)
    return args.decode_mode


def restore_decode_mode(args, science):
    """Old public checkpoints resume legacy; never infer pilot provenance."""
    saved = dict(science)
    mode = saved.get('decode_mode', 'legacy')
    mode = resolve_mode(mode, saved.get('train_deterministic', args.train_deterministic))
    if 'decode_mode' in getattr(args, '_explicit_options', ()):
        requested = resolve_mode(args.decode_mode, saved.get('train_deterministic', args.train_deterministic))
        if requested != mode:
            raise ValueError('Resume cannot change decode_mode; use --warm-start for a new experiment')
    args.decode_mode = saved['decode_mode'] = mode
    return saved


class DecodeBatch:
    """One optimizer update. The caller owns loss scaling and optimizer.step()."""
    def __init__(self, mode, count, mini, device='cpu'):
        if mode not in ('shared', 'legacy') or count < 1 or mini < 1:
            raise ValueError('Expected resolved decode mode and positive batch sizes')
        self.expected = math.ceil(count / mini)
        self.enabled = mode == 'shared' and self.expected > 1
        self.mode = mode
        self.device = torch.device(device)
        self.calls = self.micro = 0
        self.leaf = self.latest = self.before = self.fn = None

    def coordinates(self, fn):
        self.fn = fn
        if self.enabled:
            if self.leaf is None:
                self.before = _rng_state(self.device)
                with torch.no_grad():
                    self.leaf = self._call(fn).detach().requires_grad_(True)
            self.latest = self.leaf
        else:
            self.latest = self._call(fn)
        return self.latest

    def _call(self, fn):
        self.calls += 1
        result = fn()
        if not isinstance(result, torch.Tensor):
            raise TypeError('Decoder must return a coordinate tensor')
        return result

    def micro_done(self):
        self.micro += 1

    def finish(self):
        if self.micro != self.expected:
            raise RuntimeError('Unexpected microbatch count')
        if self.enabled:
            if self.leaf.grad is None:
                raise RuntimeError('No coordinate gradient reached shared decoder output')
            caller = _rng_state(self.device)
            # Keep replay backward inside the scope, including any activation
            # checkpoint recomputation. It must not advance caller RNGs.
            try:
                _restore_rng(self.before, self.device)
                output = self._call(self.fn)
                output.backward(self.leaf.grad)
            finally:
                _restore_rng(caller, self.device)
            result = output.detach()
        else:
            result = self.latest.detach()
        self.leaf = self.latest = self.before = self.fn = None
        return result

    def metadata(self):
        return dict(decode_mode=self.mode, microbatches=self.micro,
                    decoder_calls=self.calls, shared_decode_active=self.enabled)
