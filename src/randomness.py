"""Explicit seeds with opt-in RNG isolation; legacy sampling stays the default.

Isolation temporarily swaps process RNGs, including those used by Protenix
augmentation. It is intended for sequential sampling in one process, not threads.
RNG snapshots alone are not a complete training resume checkpoint.
"""
import argparse
from contextlib import contextmanager
from functools import wraps
import random

import numpy as np
import torch


def seed_value(value):
    if isinstance(value, bool):
        raise ValueError('Seed must be an integer in [0, 2**32-1]')
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError('Seed must be an integer in [0, 2**32-1]') from exc
    if not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32-1]')
    return int(value)


def seed_argument(value):
    try:
        return seed_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_seed_arguments(parser, entry):
    if entry != 'inference':
        parser.add_argument('--seed', type=seed_argument, default=None if entry == 'export' else 42,
                            help='Run seed (42 for new runs; export inherits recorded diffusion seed, else 42).')
    if entry in ('train', 'inference'):
        parser.add_argument('--diffusion-seed', type=seed_argument, default=None,
                            help='Diffusion noise seed; train defaults to --seed, inference to 42.')
    if entry == 'train':
        parser.add_argument('--data-seed', type=seed_argument, default=None,
                            help='Particle shuffle seed override; single-GPU train requires isolated RNG, parallel uses its dedicated data generator.')
    parser.add_argument('--rng-mode', choices=('legacy', 'isolated'), default='legacy',
                        help='legacy preserves global RNG side effects; isolated keeps sampling RNG separate.')


def resolve_seeds(args, entry='train', cache=None):
    mode = getattr(args, 'rng_mode', 'legacy')
    if mode not in ('legacy', 'isolated'):
        raise ValueError('rng_mode must be legacy or isolated')
    explicit = getattr(args, 'seed', None)
    source = 'explicit' if explicit is not None else 'legacy_fallback_42'
    if entry == 'export':
        saved = (cache or {}).get('refinement_seed_settings')
        if explicit is None and saved is not None:
            explicit = saved['diffusion_seed']
            source = 'refinement_checkpoint'
        seed = seed_value(42 if explicit is None else explicit)
        return dict(schema_version=1, seed=seed, diffusion_seed=seed, rng_mode=mode, source=source)
    seed = seed_value(42 if explicit is None else explicit)
    override = getattr(args, 'diffusion_seed', None)
    diffusion = seed_value(seed if override is None else override)
    data_override = getattr(args, 'data_seed', None)
    if data_override is not None and mode != 'isolated':
        raise ValueError('--data-seed requires --rng-mode isolated')
    if override is not None and mode == 'legacy' and not getattr(args, 'train_deterministic', True):
        raise ValueError('--diffusion-seed with resampled noise requires --rng-mode isolated; '
                         'legacy resampling uses the advancing global RNG')
    return dict(schema_version=1, seed=seed, diffusion_seed=diffusion, rng_mode=mode,
                data_seed=seed_value(seed if data_override is None else data_override) if entry == 'train' and mode == 'isolated' else None,
                data_rng=('independent_generator' if mode == 'isolated' else 'legacy_global')
                if entry == 'train' else 'prediction_pipeline_unchanged',
                fixed_noise=bool(getattr(args, 'train_deterministic', True)))


def apply_seed_settings(configs, settings):
    configs.diffusion_seed = settings['diffusion_seed']
    configs.rng_mode = settings['rng_mode']
    # New runs start a new stream, even when their input is a previous cache.
    configs.diffusion_rng_stream = None


def seed_legacy(seed):
    """Preserve the original seed operations and their order (no Python reset)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class DiffusionRNGStream:
    """An advancing isolated stream for resampled stochasticity."""
    def __init__(self, seed):
        self.seed = seed_value(seed)
        self.state = None

    @contextmanager
    def use(self):
        caller = capture_rng_state()
        try:
            if self.state is None:
                seed_legacy(self.seed)
                random.seed(self.seed)
            else:
                restore_rng_state(self.state)
            yield
        finally:
            self.state = capture_rng_state()
            restore_rng_state(caller)


def isolated_sampling(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        configs = kwargs.get('configs', args[0] if args else None)
        mode = getattr(configs, 'rng_mode', 'legacy')
        if mode == 'legacy':
            return function(*args, **kwargs)
        if mode != 'isolated':
            raise ValueError('rng_mode must be legacy or isolated')
        seed = seed_value(getattr(configs, 'diffusion_seed', 42))
        stream = getattr(configs, 'diffusion_rng_stream', None)
        if configs.train_deterministic:
            stream = DiffusionRNGStream(seed)
        elif stream is None or stream.seed != seed:
            stream = DiffusionRNGStream(seed)
            configs.diffusion_rng_stream = stream
        with stream.use():
            return function(*args, **kwargs)
    return wrapped


def data_loader_options(settings):
    if settings['rng_mode'] == 'legacy':
        return {}  # Even passing a newly seeded generator would change old order.
    return {'generator': torch.Generator().manual_seed(settings['data_seed'])}
