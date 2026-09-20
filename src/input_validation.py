"""CPU input checks performed before constructing the diffusion model."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from cli_utils import check_output_directory, require_file
from particledataset import ParticleDataset
from utils import cif_to_tensor


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tree(v, device) for v in value)
    return value


def validate_train_inputs(args, cache=None):
    from randomness import resolve_seeds
    seed_settings = resolve_seeds(args)
    from training_output import output_formats, validate_output_template
    formats = output_formats(getattr(args, "output_format", "cif"))
    if not isinstance(getattr(args, "learn_gmm", True), bool):
        raise ValueError("--learn-gmm must be boolean")
    for name in ('boxsize', 'batch_size', 'mini_batch_size', 'epochs'):
        value = getattr(args, name, 10 if name == 'epochs' else None)
        if value is None or int(value) != float(value) or int(value) < 1:
            raise ValueError(f'--{name} must be a positive integer')
    for name in ('apix', 'resolution', 'map_resolution', 'lr_bias', 'lr_atom_weights', 'lr_sdevs'):
        value = float(getattr(args, name, 5e-3 if name == 'lr_sdevs' else 1e-2))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name} must be positive and finite')
    if getattr(args, 'max_steps', None) is not None and (
            int(args.max_steps) != float(args.max_steps) or args.max_steps < 1):
        raise ValueError('--max_steps must be a positive integer')
    if not math.isfinite(float(args.particle_sign)):
        raise ValueError('--particle_sign must be finite')
    if args.density_center is not None and (
            len(args.density_center) != 2 or not np.isfinite(args.density_center).all()):
        raise ValueError('--density_center requires two finite numbers')
    if bool(args.halfmap1) != bool(args.halfmap2):
        raise ValueError('--halfmap1 and --halfmap2 must be provided together')
    for name in ('star_data_dir', 'diffusion_data_dir', 'cif_path'):
        setattr(args, name, str(require_file(getattr(args, name), '--' + name)))
    if getattr(args, 'mrc_data_dir', None):
        args.mrc_data_dir = os.path.expanduser(str(args.mrc_data_dir))
    for name in ('halfmap1', 'halfmap2', 'block_alignment'):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, str(require_file(value, '--' + name.replace('_', '-'))))
    # STAR/stack and topology errors are cheaper than deserializing a large cache.
    dataset = ParticleDataset(args.star_data_dir, getattr(args, 'mrc_data_dir', None), float(args.apix),
                              transR=np.diag([1., 1., -1.]) if args.transR else None, norm=args.norm)
    report = dataset.validate(box_size=int(args.boxsize))
    coords, weights = cif_to_tensor(args.cif_path)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0 or not torch.isfinite(coords).all():
        raise ValueError('CIF must contain nonempty finite coordinates [N,3]')
    validate_output_template(args.cif_path, len(coords), getattr(args, 'output_format', 'cif'))
    from coordinate_transform import read_alignment_manifest, validate_update_threshold
    validate_update_threshold(getattr(args, 'block_update_trace_threshold', 2.5))
    if getattr(args, 'block_alignment', None):
        # Target identity/core errors do not need a large cache or a GPU model.
        read_alignment_manifest(args.block_alignment, args.cif_path)
    from gmm import gmm_from_arguments
    # CPU construction uses no random sampling; validate kernel combinations early.
    gmm_from_arguments(weights, args, shape_device='cpu', record_initialization=False)
    if args.halfmap1:
        import mrcfile
        headers = []
        for path in (args.halfmap1, args.halfmap2):
            with mrcfile.mmap(path, mode='r') as mrc:
                shape = tuple(mrc.data.shape)
                voxel = np.array([float(mrc.voxel_size[k]) for k in ('x', 'y', 'z')])
            if len(shape) != 3 or min(shape) < 2 or not np.isfinite(voxel).all() or np.any(voxel <= 0):
                raise ValueError(f'Invalid half-map shape or voxel size: {path}')
            headers.append((shape, voxel))
        if headers[0][0] != headers[1][0] or not np.isclose(headers[0][1][0], headers[1][1][0]):
            raise ValueError('Half-map shape or voxel size mismatch')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError(f'CUDA device unavailable: {device}')
    if cache is None:
        cache = torch.load(args.diffusion_data_dir, map_location='cpu', weights_only=False)
    required = ('pred_dict', 'input_feature_dict', 's_inputs', 's_trunk', 'z_trunk', 'pair_z',
                'p_lm', 'c_l', 'N_sample', 'noise_schedule', 'inplace_safe', 'configs',
                'enable_efficient_fusion', 'model_state')
    if not isinstance(cache, dict):
        raise ValueError('Diffusion cache must be a mapping')
    from checkpoint_sampling import is_refinement
    is_refinement(cache)  # Reject unknown versioned formats before model allocation.
    missing = [name for name in required if name not in cache]
    if missing:
        raise ValueError(f'Diffusion cache missing fields: {missing}')
    features = cache['input_feature_dict']
    if not isinstance(features, dict) or not isinstance(cache['pred_dict'], dict):
        raise ValueError('Cached input_feature_dict and pred_dict must be mappings')
    atom_to_token = features.get('atom_to_token_idx')
    if not isinstance(atom_to_token, torch.Tensor) or atom_to_token.numel() != len(coords):
        raise ValueError('CIF/cache atom count mismatch')
    for name in ('s_inputs', 's_trunk'):
        if not isinstance(cache[name], torch.Tensor) or cache[name].ndim < 2:
            raise ValueError(f'Invalid cached {name}')
    n_token = cache['s_inputs'].shape[-2]
    if cache['s_trunk'].shape[-2] != n_token:
        raise ValueError('Cached single features have inconsistent token counts')
    if atom_to_token.dtype not in (torch.int32, torch.int64) or torch.any(atom_to_token < 0) or torch.any(atom_to_token >= n_token):
        raise ValueError('Invalid cached atom_to_token_idx')
    config = cache['configs']
    if (not hasattr(getattr(config, 'model', None), 'diffusion_module')
            or not hasattr(config, 'sample_diffusion') or not hasattr(config, 'infer_setting')
            or not isinstance(cache['model_state'], dict)):
        raise ValueError('Cache lacks diffusion model configuration or state mapping')
    target = cache['z_trunk'] if cache['z_trunk'] is not None else cache['pair_z']
    if not isinstance(target, torch.Tensor) or target.ndim < 3 or target.shape[-3:-1] != (n_token, n_token):
        raise ValueError('Cache must provide a compatible z_trunk or pair_z tensor')
    schedule = cache['noise_schedule']
    if (not isinstance(schedule, torch.Tensor) or schedule.ndim != 1 or len(schedule) < 2
            or not torch.isfinite(schedule).all() or not torch.all(schedule[:-1] > schedule[1:]) or schedule[-1] != 0):
        raise ValueError('Invalid diffusion noise schedule')
    block_mode = getattr(args, 'coordinate_mode', 'global') == 'blocks' or cache.get('coordinate_transform') is not None
    from refinement_runtime import alignment_sampler
    sampler = alignment_sampler(args, cache)
    if block_mode and sampler == 'global' and cache['N_sample'] != 1:
        raise ValueError('Per-chain refinement requires N_sample=1; do not silently change cached sampling settings')
    if getattr(args, 'block_alignment', None) and not block_mode:
        raise ValueError('--block-alignment requires --coordinate-mode blocks')
    if block_mode and not args.train_deterministic:
        raise ValueError('Block mode requires fixed noise (--train_deterministic)')
    if block_mode and cache.get('coordinate_transform') is None and not getattr(args, 'block_alignment', None):
        raise ValueError('First block initialization requires --block-alignment')
    if cache.get('coordinate_transform') is not None and getattr(args, 'block_alignment', None):
        raise ValueError('Saved block transform already exists; omit --block-alignment')
    if cache.get('coordinate_transform') is not None:
        from coordinate_transform import CoordinateTransform
        transform = CoordinateTransform.from_checkpoint(cache['coordinate_transform'], args.cif_path, 'cpu')
        if args.update_affine_mat and transform.fit_targets is None:
            raise ValueError('Old block checkpoint lacks fitting targets/core; omit --update_affine_mat or initialize a new alignment run')
    from training_restart import validate_restart_inputs, restart_gmm
    validate_restart_inputs(args, cache, report)
    if getattr(args, 'resume', False) or getattr(args, 'warm_start', False):
        restart_gmm(cache, args, weights, 'cpu')
    prefix = os.path.expanduser(str(args.output_trained_model_dir))
    if not prefix.strip() or prefix == 'None':
        raise ValueError('--output_trained_model_dir is required')
    suffixes = [f'_.{fmt}' for fmt in formats] + [f'aligned_initial.{fmt}' for fmt in formats] + ['block_alignment_report.json']
    save_epochs = min(getattr(args, 'epochs', 10), getattr(args, 'max_steps', None) or getattr(args, 'epochs', 10))
    first_epoch = cache['training_resume']['next_epoch'] + 1 if getattr(args, 'resume', False) else 1
    suffixes.extend(f'{epoch}.{ext}' for epoch in range(first_epoch, save_epochs + 1) for ext in ('pth', *formats))
    for suffix in suffixes:
        if Path(prefix + suffix).exists():
            raise FileExistsError(f'Output already exists: {prefix + suffix}; choose a fresh prefix')
    check_output_directory(os.path.dirname(prefix) or '.')
    args.output_trained_model_dir = prefix
    report.update(refinement_sampler=sampler, output_format=getattr(args, 'output_format', 'cif'), gmm_learning=getattr(args, 'learn_gmm', True), seed_settings=seed_settings, n_atoms=len(coords), n_tokens=int(n_token), device=str(device), cache_schema='checked',
                  scope='metadata, stack headers, topology count and cache schema; no model execution')
    return cache, dataset, report
