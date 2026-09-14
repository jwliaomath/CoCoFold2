"""Export sampling snapshots; these are not full training-resume states."""
import copy
from contextlib import contextmanager

import torch

from randomness import capture_rng_state, restore_rng_state


def sampling_snapshot(configs, device):
    return dict(version=1, fixed_noise=bool(configs.train_deterministic),
                device=str(torch.device(device)), rng=capture_rng_state(),
                diffusion_rng_stream=copy.deepcopy(getattr(configs, 'diffusion_rng_stream', None)),
                matmul_precision=torch.get_float32_matmul_precision(),
                seeds=dict(diffusion_seed=getattr(configs, 'diffusion_seed', 42),
                           rng_mode=getattr(configs, 'rng_mode', 'legacy')))


@contextmanager
def preserve_sampling(configs):
    caller = capture_rng_state()
    stream = copy.deepcopy(getattr(configs, 'diffusion_rng_stream', None))
    try:
        yield
    finally:
        restore_rng_state(caller)
        configs.diffusion_rng_stream = stream


def validate_sampling_snapshot(snapshot, device):
    if snapshot.get('version') != 1:
        raise ValueError('Unsupported checkpoint export sampling version')
    target, source = torch.device(device), torch.device(snapshot['device'])
    if not snapshot['fixed_noise'] and source.type != target.type:
        raise ValueError('Resampled export replay requires the original device type; use --seed for a new sample')


@contextmanager
def replay_sampling(configs, snapshot, device):
    """Replay on the selected GPU; resampled CPU/CUDA transfers require a new seed."""
    validate_sampling_snapshot(snapshot, device)
    target, source = torch.device(device), torch.device(snapshot['device'])
    previous = dict(train_deterministic=configs.train_deterministic,
                    diffusion_seed=getattr(configs, 'diffusion_seed', 42),
                    rng_mode=getattr(configs, 'rng_mode', 'legacy'))
    precision = torch.get_float32_matmul_precision()
    with preserve_sampling(configs):
        try:
            configs.train_deterministic = snapshot['fixed_noise']
            configs.diffusion_seed = snapshot['seeds']['diffusion_seed']
            configs.rng_mode = snapshot['seeds']['rng_mode']
            torch.set_float32_matmul_precision(snapshot['matmul_precision'])
            if not snapshot['fixed_noise']:
                def remap(state):
                    state = copy.deepcopy(state)
                    if target.type == 'cuda':
                        states = torch.cuda.get_rng_state_all()
                        states[target.index if target.index is not None else torch.cuda.current_device()] = state['cuda'][source.index or 0].cpu()
                        state['cuda'] = states
                    else:
                        state['cuda'] = None
                    return state
                restore_rng_state(remap(snapshot['rng']))
                configs.diffusion_rng_stream = copy.deepcopy(snapshot['diffusion_rng_stream'])
                if configs.diffusion_rng_stream is not None and configs.diffusion_rng_stream.state is not None:
                    configs.diffusion_rng_stream.state = remap(configs.diffusion_rng_stream.state)
            yield
        finally:
            torch.set_float32_matmul_precision(precision)
            for key, value in previous.items():
                setattr(configs, key, value)


def is_refinement(cache):
    header = cache.get('checkpoint_schema')
    if header is not None:
        if header != dict(name='cocofold2_refinement', version=1):
            raise ValueError('Unsupported CoCoFold2 checkpoint schema')
        return True
    return any(cache.get(key) is not None for key in ('z_bias', 'z_mul', 'opt_state', 'coordinate_transform'))


def effective_latent(cache, z, pair):
    value = z if z is not None else pair
    if value is None:
        raise ValueError('Checkpoint contains neither z_trunk nor pair_z')
    for key in ('z_mul', 'z_bias'):
        operand = cache.get(key)
        if operand is not None:
            operand = torch.as_tensor(operand, device=value.device)
            result = value * operand if key == 'z_mul' else value + operand
            if result.shape != value.shape or not torch.isfinite(result).all():
                raise ValueError(f'Invalid checkpoint {key}: shape or finite-value mismatch')
            value = result
    return (value.detach(), pair) if z is not None else (z, value.detach())


def global_coordinates(cache, raw, frame='reference'):
    if frame == 'raw' or not is_refinement(cache):
        return raw
    if cache.get('rotation') is None or cache.get('translation') is None:
        raise ValueError('Refinement checkpoint lacks saved rotation/translation; use --coordinate-frame raw')
    rotation = torch.as_tensor(cache['rotation'], device=raw.device, dtype=raw.dtype)
    translation = torch.as_tensor(cache['translation'], device=raw.device, dtype=raw.dtype)
    if rotation.shape != (3, 3) or translation.shape not in ((3,), (1, 3)):
        raise ValueError('Invalid saved rotation/translation shape')
    if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
        raise ValueError('Saved rotation/translation must be finite')
    return raw @ rotation.T + translation
