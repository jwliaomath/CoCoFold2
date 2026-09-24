"""Single-process restart at completed epoch boundaries; no particle cursor."""
import argparse
import copy
import hashlib
import importlib.metadata
from pathlib import Path

import torch

SCIENCE_ARGS = (
    'seed', 'diffusion_seed', 'data_seed', 'rng_mode', 'train_deterministic',
    'learn_gmm', 'transR', 'particle_sign', 'boxsize', 'apix', 'norm', 'resolution',
    'density_center', 'batch_size', 'mini_batch_size', 'update_affine_mat',
    'map_resolution', 'fsc_gamma', 'fsc_smooth_win', 'lr_bias', 'lr_atom_weights',
    'lr_sdevs', 'gmm_kernel', 'gmm_amplitude', 'gmm_sigma_floor',
    'gmm_atom_chunk_size', 'gmm_checkpoint_chunks', 'gmm_checkpoint_peak2d',
    'coordinate_mode', 'alignment_sampler', 'coordinate_handoff_tolerance',
    'block_update_trace_threshold', 'gmm_sdev_init_mode', 'gmm_molmap_resolution_A',
    'projection_frame', 'projection_origin',
)


class RestartArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        import sys
        tokens = list(sys.argv[1:] if args is None else args)
        parsed, remaining = super().parse_known_args(tokens, namespace)
        parsed._explicit_options = sorted({self._option_string_actions[t.split('=', 1)[0]].dest
            for t in tokens if t.split('=', 1)[0] in self._option_string_actions})
        return parsed, remaining


def add_restart_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true', help='Continue a completed-epoch checkpoint supplied with --diffusion_data_dir; --epochs is the total target.')
    group.add_argument('--warm-start', action='store_true', help='Start a new run from saved latent/GMM/placement; reset optimizer and progress.')


def runtime_signature(device):
    try:
        protenix = importlib.metadata.version('protenix')
    except importlib.metadata.PackageNotFoundError:
        protenix = None
    device = torch.device(device)
    return dict(torch=torch.__version__, protenix=protenix, device=str(device),
                cpu_threads=torch.get_num_threads(),
                cuda_build=torch.version.cuda, cuda_count=torch.cuda.device_count(),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32)


def input_identity(args, report):
    def identity(path, content=False):
        if not path:
            return None
        path = Path(path).resolve()
        stat = path.stat()
        value = dict(path=str(path), bytes=stat.st_size)
        if content:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            value['sha256'] = digest.hexdigest()
        else:
            value['mtime_ns'] = stat.st_mtime_ns
        return value
    return dict(star=identity(args.star_data_dir, True), cif=identity(args.cif_path, True),
                halfmaps=[identity(getattr(args, k, None)) for k in ('halfmap1', 'halfmap2')],
                stacks=[identity(row['path']) for row in report.get('stack_headers', [])],
                n_particles=report['n_particles'],
                stack_check='path_size_mtime; no full MRCS content hash')


def prepare_restart(args):
    """Resolve saved configuration before dataset/GMM validation. Never load a model."""
    resume, warm = getattr(args, 'resume', False), getattr(args, 'warm_start', False)
    if not resume and not warm:
        return None
    if resume and warm:
        raise ValueError('--resume and --warm-start are mutually exclusive')
    cache = torch.load(args.diffusion_data_dir, map_location='cpu', weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError('Checkpoint must be a mapping')
    from checkpoint_sampling import is_refinement
    if not is_refinement(cache):
        raise ValueError('Restart requires a refinement checkpoint, not an initial prediction cache')
    if resume:
        state = cache.get('training_resume')
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('Checkpoint lacks complete epoch resume state; use --warm-start')
        if not state.get('epoch_complete'):
            raise ValueError('Checkpoint stopped within an epoch; use a completed epoch or --warm-start')
        required = ('science_args', 'next_epoch', 'global_step', 'rng', 'diffusion_rng_stream',
                    'data_generator_state', 'inputs', 'input_paths', 'runtime', 'parent_run_id')
        missing = [key for key in required if key not in state]
        missing += [key for key in ('opt_state', 'gmm', 'z_bias', 'refinement_seed_settings') if cache.get(key) is None]
        if missing:
            raise ValueError('Incomplete resume state: ' + ', '.join(missing))
        # Pre-extension checkpoints do not have fresh-width provenance. Their
        # saved GMM state is authoritative; do not require newly added fields.
        missing_science = [key for key in SCIENCE_ARGS if hasattr(args, key) and key not in state['science_args']
                           and key not in ('gmm_sdev_init_mode', 'gmm_molmap_resolution_A',
                                           'projection_frame', 'projection_origin')]
        if missing_science:
            raise ValueError('Incomplete saved training configuration: ' + ', '.join(missing_science))
        if cache.get('coordinate_transform') is None and (cache.get('rotation') is None or cache.get('translation') is None):
            raise ValueError('Resume requires saved rotation/translation')
        explicit = set(getattr(args, '_explicit_options', ()))
        for key, default in (('projection_frame', 'legacy'), ('projection_origin', (0., 0., 0.))):
            requested = (tuple(getattr(args, key, default)) if key == 'projection_origin'
                         else getattr(args, key, default))
            if key not in state['science_args'] and key in explicit and requested != default:
                raise ValueError('Old checkpoint has legacy projection; use --warm-start to change ' + key)
        conflicts = [key for key, value in state['science_args'].items() if key in explicit
                     and (tuple(getattr(args, key, ())) != tuple(value) if key == 'projection_origin'
                          else getattr(args, key, None) != value)]
        if conflicts:
            raise ValueError('Resume cannot change scientific settings; use --warm-start: ' + ', '.join(conflicts))
        for key, value in state['science_args'].items():
            setattr(args, key, copy.deepcopy(value))
        for key, value in state['input_paths'].items():
            if key not in explicit:
                setattr(args, key, value)
        if not isinstance(state['next_epoch'], int) or state['next_epoch'] < 1 or state['global_step'] < 1:
            raise ValueError('Invalid completed epoch/step in resume checkpoint')
        if cache.get('training_progress', {}).get('epoch_completed') != state['next_epoch']:
            raise ValueError('Resume epoch count disagrees with checkpoint progress')
        if cache['training_progress'].get('global_step') != state['global_step']:
            raise ValueError('Resume step count disagrees with checkpoint progress')
        from randomness import resolve_seeds
        if resolve_seeds(args) != cache['refinement_seed_settings']:
            raise ValueError('Resume seed configuration is inconsistent')
        if state['science_args']['rng_mode'] == 'isolated' and state['data_generator_state'] is None:
            raise ValueError('Resume lacks isolated DataLoader generator state')
        if getattr(args, 'block_alignment', None):
            raise ValueError('Resume inherits saved alignment; omit --block-alignment')
        if args.epochs <= state['next_epoch']:
            raise ValueError('--epochs must exceed the completed epoch count (total target, not additional epochs)')
        if args.max_steps is not None and args.max_steps <= state['global_step']:
            raise ValueError('--max_steps must exceed the saved global step')
        if runtime_signature(args.device) != state['runtime']:
            raise ValueError('Resume requires matching device/software/precision settings; use --warm-start for a different environment')
        if cache.get('z_mul') is not None:
            raise ValueError('This resume version requires an additive latent bias (z_mul=None)')
    else:
        # Both are derived/runtime-only fields absent in some historical checkpoints.
        cache.setdefault('pred_dict', {})
        cache.setdefault('inplace_safe', False)
        # Modern payloads define their representation; current CLI still controls learning and LRs.
        if cache.get('gmm') is not None:
            config = cache['gmm']['config']
            inherited = dict(gmm_kernel=config['kernel'], gmm_amplitude=config['amplitude_convention'],
                             gmm_sigma_floor=config['sigma_floor'])
            explicit = set(getattr(args, '_explicit_options', ()))
            for key, value in inherited.items():
                requested = getattr(args, key)
                if key in explicit and requested != value and not (key == 'gmm_amplitude' and requested == 'auto'):
                    raise ValueError(f'Warm-start retains saved GMM representation; incompatible {key}')
                setattr(args, key, value)
    # Historical optional field follows the confirmed Protenix inference precedence.
    if 'enable_efficient_fusion' not in cache:
        from get_pdb import _resolve_efficient_fusion
        cache['enable_efficient_fusion'] = _resolve_efficient_fusion(cache)[0]
    return cache


def validate_restart_inputs(args, cache, report):
    if getattr(args, 'resume', False):
        if input_identity(args, report) != cache['training_resume']['inputs']:
            raise ValueError('Resume input identity changed; use original data or --warm-start')
        target = cache['z_trunk'] if cache['z_trunk'] is not None else cache['pair_z']
        if cache['z_bias'].shape != target.shape:
            raise ValueError('Resume bias shape differs from the latent being optimized')
        expected_groups = 3 if args.learn_gmm else 1
        if len(cache['opt_state'].get('param_groups', [])) != expected_groups:
            raise ValueError('Resume optimizer groups disagree with GMM learning setting')
        if cache.get('gmm_learning_enabled') is not args.learn_gmm:
            raise ValueError('Resume GMM learning flags disagree')
        if cache.get('pred_dict', {}).get('coordinate') is None:
            raise ValueError('Resume requires saved post-step coordinates')
    if getattr(args, 'warm_start', False) or getattr(args, 'resume', False):
        from checkpoint_sampling import effective_latent, global_coordinates
        effective_latent(cache, cache['z_trunk'], cache['pair_z'])
        if cache.get('coordinate_transform') is None and cache.get('rotation') is not None and cache.get('translation') is not None:
            global_coordinates(cache, torch.zeros(1, 3))


def restart_gmm(cache, args, weights, device):
    from gmm import GaussianProjector, gmm_from_arguments
    if cache.get('gmm') is not None:
        model = GaussianProjector.from_checkpoint(cache['gmm'], device)
        source = 'checkpoint_gmm'
    else:
        saved_weights = cache.get('atom_weights')
        if saved_weights is not None:
            if saved_weights.shape != weights.shape or not torch.isfinite(saved_weights).all():
                raise ValueError('Invalid saved atom_weights')
            weights = saved_weights.to(device)
        widths = cache.get('sdevs')
        # Saved legacy widths bypass fresh-mode validation and initialization.
        options = copy.copy(args)
        if widths is not None:
            options.gmm_sdev_init_mode = 'legacy'
        model = gmm_from_arguments(weights, options, shape_device=device,
                                   record_initialization=widths is None)
        source = 'saved_amplitudes_new_widths' if saved_weights is not None else 'initialized_from_cli_and_cif'
        widths = cache.get('sdevs')
        if widths is not None:
            if model.kernel != 'legacy' or widths.shape != model.sdevs.shape or not torch.isfinite(widths).all():
                raise ValueError('Saved legacy widths are incompatible with requested GMM')
            with torch.no_grad():
                model.sdevs.copy_(widths.to(device))
            model.width_initialization = None  # Historical initialization is unknown.
            source = 'legacy_gmm_fields' if saved_weights is not None else 'saved_widths_cif_amplitudes'
    model.atom_chunk_size = args.gmm_atom_chunk_size
    model.checkpoint_chunks = args.gmm_checkpoint_chunks
    model.checkpoint_peak2d = args.gmm_checkpoint_peak2d
    if model.atom_weights.numel() != weights.numel() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Saved GMM atom count or finite-value check failed')
    return model, source


def capture_epoch_state(args, configs, loader, inputs, epoch, global_step, complete, run_id):
    from randomness import capture_rng_state
    return dict(version=1, epoch_complete=complete, next_epoch=epoch+1 if complete else epoch,
                global_step=global_step, science_args={k: copy.deepcopy(getattr(args, k)) for k in SCIENCE_ARGS if hasattr(args, k)},
                rng=capture_rng_state(), diffusion_rng_stream=copy.deepcopy(getattr(configs, 'diffusion_rng_stream', None)),
                data_generator_state=loader.generator.get_state() if loader.generator is not None else None,
                input_paths={k: getattr(args, k, None) for k in ('mrc_data_dir', 'halfmap1', 'halfmap2')},
                inputs=inputs, runtime=runtime_signature(args.device), parent_run_id=run_id)


def restore_epoch_rng(state, configs, loader):
    from randomness import restore_rng_state
    configs.diffusion_rng_stream = copy.deepcopy(state['diffusion_rng_stream'])
    if loader.generator is not None:
        if state['data_generator_state'] is None:
            raise ValueError('Resume lacks isolated DataLoader generator state')
        loader.generator.set_state(state['data_generator_state'].cpu())
    restore_rng_state(state['rng'])
