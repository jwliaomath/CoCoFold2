"""Frozen cache decoding for one structure, used by block refinement/export.

This preserves the existing block sampler for a leading batch dimension of
one. It does not replace the global train.py sampler. Protenix is imported
only when loading real model weights.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def tree_to(value, device):
    """Detach and place nested cached tensors on the requested device."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    return value


def single(value, rank, name):
    if value is None:
        return None
    while value.ndim > rank and value.shape[0] == 1:
        value = value[0]
    if value.ndim != rank:
        raise ValueError(f'{name}: expected one topology, rank {rank}, got {tuple(value.shape)}')
    return value


def _check_single_pair(pair):
    if pair.ndim != 4 or pair.shape[0] != 1:
        raise ValueError('Single-structure decoder requires pair shape [1,T,T,C]')


def sample_fixed_noise(denoise, pair, schedule, n_atom, seed=42, gamma0=.8,
                       gamma_min=1., noise_scale_lambda=1.003, step_scale_eta=1.5,
                       checkpoint_steps=False):
    """Decode one structure with the historical block fixed-noise Euler sampler."""
    _check_single_pair(pair)
    if schedule.ndim != 1 or len(schedule) < 2 or not torch.isfinite(schedule).all():
        raise ValueError('Invalid diffusion schedule')
    if not torch.all(schedule[:-1] > schedule[1:]) or schedule[-1] != 0:
        raise ValueError('Schedule must strictly decrease to zero')
    generator = torch.Generator(device=pair.device).manual_seed(seed)

    def noise():
        return torch.randn((1, 1, n_atom, 3), device=pair.device,
                           dtype=pair.dtype, generator=generator)

    x = schedule[0] * noise()
    for previous, current in zip(schedule[:-1], schedule[1:]):
        gamma = gamma0 if current > gamma_min else 0.
        level = previous * (1 + gamma)
        noisy = x + noise_scale_lambda * (level.square() - previous.square()).sqrt() * noise()
        t = level.expand(1, 1)
        if checkpoint_steps and torch.is_grad_enabled():
            clean = checkpoint(denoise, noisy, t, pair, use_reentrant=False,
                               preserve_rng_state=True)
        else:
            clean = denoise(noisy, t, pair)
        if clean.shape != noisy.shape:
            raise ValueError(f'Diffusion output {clean.shape} != {noisy.shape}')
        x = noisy + step_scale_eta * (current - level) * (noisy - clean) / level
    return x[:, 0]


class SingleStructureDecoder(nn.Module):
    """One frozen decoder with gradients through the supplied pair representation."""

    def __init__(self, module, features, s_inputs, s_trunk, schedule,
                 sampler_options=None, seed=42, checkpoint_steps=False):
        super().__init__()
        self.module = module.eval().requires_grad_(False)
        self.features = features
        self.register_buffer('s_inputs', single(s_inputs, 2, 's_inputs'))
        self.register_buffer('s_trunk', single(s_trunk, 2, 's_trunk'))
        self.register_buffer('schedule', schedule)
        self.sampler_options = sampler_options or {}
        self.seed = seed
        self.checkpoint_steps = checkpoint_steps

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, pair):
        _check_single_pair(pair)
        si, st = self.s_inputs.unsqueeze(0), self.s_trunk.unsqueeze(0)

        def denoise(x, t, p):
            return self.module(
                x_noisy=x, t_hat_noise_level=t, input_feature_dict=self.features,
                s_inputs=si, s_trunk=st, z_trunk=None, pair_z=p, p_lm=None, c_l=None,
                chunk_size=self.sampler_options.get('attn_chunk_size'),
                inplace_safe=False, enable_efficient_fusion=False)

        options = {k: v for k, v in self.sampler_options.items() if k != 'attn_chunk_size'}
        return sample_fixed_noise(
            denoise, pair, self.schedule, self.features['atom_to_token_idx'].shape[-1],
            self.seed, checkpoint_steps=self.checkpoint_steps, **options)


def load_protenix(cache_path, device, seed=42, checkpoint_steps=False, apply_saved_bias=True):
    """Load an initial cache or single-state checkpoint, applying saved bias once.

    Historical z_trunk takes precedence when present. The returned conditioned
    pair is for decoding; callers retaining z_trunk optimization keep that
    original target. apply_saved_bias=False returns the original base for a
    caller that separately restores its trainable bias and optimizer state.
    This function alone is not a complete optimizer/RNG resume operation.
    """
    from protenix.model.modules.diffusion import DiffusionModule

    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    configs = cache['configs']
    model = DiffusionModule(**configs.model.diffusion_module).to(device).eval().requires_grad_(False)
    model.load_state_dict(cache['model_state'], strict=True)
    features = tree_to(cache['input_feature_dict'], device)
    if features['atom_to_token_idx'].ndim != 1:
        raise ValueError('Expected unbatched topology atom_to_token_idx [A]')
    target_name = 'z_trunk' if cache.get('z_trunk') is not None else 'pair_z'
    base = single(tree_to(cache[target_name], device), 3, target_name)
    if base is None:
        raise ValueError('Cache has neither z_trunk nor pair_z')
    with torch.no_grad():
        if apply_saved_bias and cache.get('z_mul') is not None:
            base = base * tree_to(cache['z_mul'], device)
        if apply_saved_bias and cache.get('z_bias') is not None:
            base = base + tree_to(cache['z_bias'], device)
        base = single(base, 3, 'refined pair')
        if target_name == 'z_trunk':
            base = model.diffusion_conditioning.prepare_cache(features['relp'], base, False)
    base = single(base, 3, 'conditioned pair').detach().float()
    options = {k: configs.sample_diffusion.get(k)
               for k in ('gamma0', 'gamma_min', 'noise_scale_lambda', 'step_scale_eta')}
    options = {k: v for k, v in options.items() if v is not None}
    options['attn_chunk_size'] = configs.infer_setting.chunk_size
    decoder = SingleStructureDecoder(
        model, features, tree_to(cache['s_inputs'], device).float(),
        tree_to(cache['s_trunk'], device).float(), tree_to(cache['noise_schedule'], device).float(),
        options, seed, checkpoint_steps)
    metadata = {k: cache.get(k) for k in (
        'gmm', 'atom_weights', 'sdevs', 'gmm_kernel', 'rotation', 'translation', 'coordinate_transform')}
    metadata['original_pair_target'] = target_name
    return decoder, base, metadata
