"""Decoder identity is independent of reference-coordinate alignment."""
import json
from pathlib import Path


def alignment_sampler(args, cache):
    requested = getattr(args, 'alignment_sampler', 'auto')
    if requested not in ('auto', 'global', 'block'):
        raise ValueError('--alignment-sampler must be auto, global or block')
    aligned = getattr(args, 'coordinate_mode', 'global') == 'blocks' or cache.get('coordinate_transform') is not None
    if not aligned:
        if requested != 'auto':
            raise ValueError('--alignment-sampler applies only to block/chain alignment')
        return 'global'
    if cache.get('coordinate_transform') is not None:
        saved = saved_alignment_sampler(cache)
        if requested not in ('auto', saved):
            raise ValueError(f'Checkpoint uses {saved} sampler; cannot switch decoder at handoff')
        return saved
    if requested != 'auto':
        return requested
    path = getattr(args, 'block_alignment', None)
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else {}
    # Old explicitly mapped block initialization retains its historical sampler.
    return 'global' if manifest.get('grouping') == 'auth_asym_id' else 'block'


def saved_alignment_sampler(cache):
    value = cache.get('refinement_sampler', 'block')
    if value not in ('global', 'block'):
        raise ValueError('Unknown refinement_sampler in aligned checkpoint')
    return value


def apply_saved_global_bias(cache, z, pair):
    """Apply the saved perturbation once only for explicitly marked new checkpoints."""
    if cache.get('coordinate_transform') is None or cache.get('refinement_sampler') != 'global':
        return z, pair
    value = z if z is not None else pair
    for name, operation in (('z_mul', lambda x, v: x * v), ('z_bias', lambda x, v: x + v)):
        if cache.get(name) is not None:
            value = operation(value, cache[name].to(value.device))
    value = value.detach()
    return (value, pair) if z is not None else (z, value)
