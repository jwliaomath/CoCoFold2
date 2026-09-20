"""Component-parallel refinement using the existing 2-D GMM projection sum.

One component cache per torchrun rank; optional independent chain alignment,
structured records and complete-epoch checkpoint sets.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


PROPOSAL_DIR = Path(__file__).resolve().parent


def _resolve_cocofold2_root() -> Path:
    """Find the directory that directly contains CoCoFold2's Python files."""
    configured_root = os.environ.get("COCOFOLD2_ROOT")
    if configured_root:
        candidates = [Path(configured_root).expanduser().resolve()]
    else:
        repository_root = PROPOSAL_DIR.parents[1]
        candidates = [repository_root, repository_root / "src"]

    required_files = (
        "ctf.py",
        "particledataset.py",
        "pts2img.py",
        "utils.py",
        "utils_halfmap.py",
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required_files):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "COCOFOLD2_ROOT must directly contain ctf.py, particledataset.py, "
        f"pts2img.py, utils.py, and utils_halfmap.py; checked: {checked}"
    )


COCOFOLD2_ROOT = _resolve_cocofold2_root()
sys.path.insert(0, str(PROPOSAL_DIR))
sys.path.insert(0, str(COCOFOLD2_ROOT))

from ctf import compute_ctf
from gmm import GaussianProjector, add_gmm_arguments, gmm_from_arguments
from distributed_gmm import (
    detached_sum,
    distributed_active,
    distributed_project_gaussians,
)
from manifest import ComponentEntry, component_for_rank, load_manifest
from particledataset import ParticleDataset
from utils import (
    cif_to_tensor,
    compute_frc,
    deep_clone,
    kabsch_alignment,
    replace_cif_coordinates,
)
from utils_halfmap import build_halfmap_shell_weights
from randomness import add_seed_arguments, seed_legacy, apply_seed_settings
from run_recording import recorded, current_record, add_record_arguments
from training_restart import RestartArgumentParser, restart_gmm
from training_output import export_training_structure
from cli_utils import positive_int, positive_float
from coordinate_transform import CoordinateTransform
from chain_parallel.parallel_runtime import phase, gather, seeds, preflight, validate_partition, fit_chains
from chain_parallel.parallel_checkpoint import resume_index, validate_resume, restore_resume, save_epoch


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _setup_distributed(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), torch.device(args.device)
    requested_world = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world > 1:
        if args.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL execution requires CUDA")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(args.backend, timeout=timedelta(seconds=args.distributed_timeout))
        return dist.get_rank(), dist.get_world_size(), device

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return 0, 1, device


def _sample_diffusion(configs: Any, training: bool = False, **kwargs: Any) -> torch.Tensor:
    from model.generator import sample_diffusion
    from protenix.utils.torch_utils import autocasting_disable_decorator

    sample_args = {
        key: configs.sample_diffusion.get(key)
        for key in ("gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta")
    }
    sample_args.update(
        {
            "attn_chunk_size": (
                configs.infer_setting.chunk_size if not training else None
            ),
            "diffusion_chunk_size": (
                configs.infer_setting.sample_diffusion_chunk_size
                if not training
                else None
            ),
        }
    )
    sampler = autocasting_disable_decorator(configs.skip_amp.sample_diffusion)(
        sample_diffusion
    )
    return sampler(configs=configs, **sample_args, **kwargs)


class LocalComponent:
    """One source-faithful CoCoFold2 diffusion/GMM state owned by one rank."""

    def __init__(
        self,
        entry: ComponentEntry,
        device: torch.device,
        train_deterministic: bool,
        gmm_options: argparse.Namespace | None = None,
        raw=None,
    ) -> None:
        from protenix.model.modules.diffusion import DiffusionModule

        raw = raw if raw is not None else torch.load(entry.diffusion_data_dir, map_location="cpu", weights_only=False)
        self.entry = entry
        self.device = device
        contextual_metadata = raw.get("contextual_split_metadata")
        if contextual_metadata is not None and not isinstance(contextual_metadata, dict):
            raise ValueError("contextual_split_metadata must be a mapping when present")
        self.contextual_split_metadata = contextual_metadata
        self.model_state_sha256 = _state_dict_sha256(raw["model_state"])
        self.pred_dict = _tree_to_device(raw["pred_dict"], device)
        self.input_feature_dict = _tree_to_device(raw["input_feature_dict"], device)
        self.s_inputs = raw["s_inputs"].to(device)
        self.s_trunk = raw["s_trunk"].to(device)
        self.z_trunk = (
            None if raw["z_trunk"] is None else raw["z_trunk"].to(device)
        )
        self.pair_z = None if raw["pair_z"] is None else raw["pair_z"].to(device)
        self.p_lm = None if raw["p_lm"] is None else raw["p_lm"].to(device)
        self.c_l = None if raw["c_l"] is None else raw["c_l"].to(device)
        self.n_sample = int(raw["N_sample"])
        self.noise_schedule = raw["noise_schedule"].to(device)
        self.inplace_safe = bool(raw["inplace_safe"])
        self.configs = raw["configs"]
        self.configs.train_deterministic = bool(train_deterministic)
        # The source trainer used deterministic seed 42.  The workspace's
        # generator now exposes that old constant through this config field.
        self.seed_settings = seeds(gmm_options)
        apply_seed_settings(self.configs, self.seed_settings)
        self.configs.train_seed = self.seed_settings['diffusion_seed']
        self.transform = None
        self.update_threshold = gmm_options.block_update_trace_threshold
        if gmm_options.warm_start:
            from checkpoint_sampling import effective_latent
            self.z_trunk, self.pair_z = effective_latent(raw, self.z_trunk, self.pair_z)
        self.enable_efficient_fusion = bool(raw["enable_efficient_fusion"])

        self.diffusion_module = DiffusionModule(
            **self.configs.model.diffusion_module
        ).to(device)
        self.diffusion_module.load_state_dict(raw["model_state"])
        self.diffusion_module.eval().requires_grad_(False)

        if self.z_trunk is None:
            if self.pair_z is None:
                raise ValueError("cache contains neither z_trunk nor pair_z")
            self.z_bias_target = "pair_z"
            bias_template = self.pair_z
        else:
            self.z_bias_target = "z_trunk"
            bias_template = self.z_trunk
        self.z_bias = torch.nn.Parameter(torch.zeros_like(bias_template))
        if gmm_options.resume:
            self.z_bias.data.copy_(raw['z_bias'].to(device))

        with torch.no_grad():
            initial_samples = self.sample(use_cached_inplace_safe=True)
            initial_coordinates = initial_samples[0]
        reference_coordinates, initial_atom_weights = cif_to_tensor(str(entry.cif_path))
        if reference_coordinates.shape != initial_coordinates.shape:
            raise ValueError(
                f"component {entry.component_id}: CIF coordinates "
                f"{tuple(reference_coordinates.shape)} do not match diffusion "
                f"coordinates {tuple(initial_coordinates.shape)}"
            )
        self.reference_coordinates = reference_coordinates.to(device)
        initial_weights = initial_atom_weights.detach().to(device=device, dtype=torch.float32)
        if gmm_options is not None and (gmm_options.resume or gmm_options.warm_start):
            self.gmm, source = restart_gmm(raw, gmm_options, initial_weights, device)
            current_record().event('gmm_restart', source=source)
        else:
            self.gmm = (GaussianProjector(initial_weights) if gmm_options is None
                        else gmm_from_arguments(initial_weights, gmm_options))
        self.gmm.requires_grad_(gmm_options.learn_gmm)
        self.atom_weights = self.gmm.atom_weights
        with torch.no_grad():
            _, rotation, translation = kabsch_alignment(
                initial_coordinates,
                self.reference_coordinates,
                return_transform=True,
            )
        self.rotation = rotation
        self.translation = translation
        if (gmm_options.resume or gmm_options.warm_start) and raw.get('rotation') is not None and raw.get('translation') is not None:
            from checkpoint_sampling import global_coordinates
            global_coordinates(raw, initial_coordinates)  # Validate dimensions and finite values.
            self.rotation = raw['rotation'].to(device)
            self.translation = raw['translation'].to(device)
        if raw.get('coordinate_transform') is not None:
            self.transform = CoordinateTransform.from_checkpoint(raw['coordinate_transform'], entry.cif_path, device)
        elif entry.alignment_manifest:
            self.transform = CoordinateTransform.fit_manifest(entry.alignment_manifest, initial_coordinates, entry.cif_path)
        elif gmm_options.by_chain:
            self.transform = fit_chains(entry.cif_path, initial_coordinates, gmm_options.fit_atoms)
        self.pred_dict["coordinate"] = initial_samples

    def sample(self, use_cached_inplace_safe: bool = False) -> torch.Tensor:
        z_trunk = self.z_trunk
        pair_z = self.pair_z
        if self.z_bias_target == "pair_z":
            pair_z = pair_z + self.z_bias
        else:
            z_trunk = z_trunk + self.z_bias
        return _sample_diffusion(
            configs=self.configs,
            training=False,
            denoise_net=self.diffusion_module,
            input_feature_dict=self.input_feature_dict,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=self.p_lm,
            c_l=self.c_l,
            N_sample=self.n_sample,
            noise_schedule=self.noise_schedule,
            inplace_safe=(self.inplace_safe if use_cached_inplace_safe else False),
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

    def place_coordinates(
        self, coordinates: torch.Tensor, update_affine_mat: bool
    ) -> torch.Tensor:
        if self.transform is not None:
            if update_affine_mat:
                events = self.transform.update_from_coordinates(coordinates, self.update_threshold)
                for event in events:
                    current_record().event('alignment_update', component_id=self.entry.component_id, **event)
            return self.transform(coordinates)
        if update_affine_mat:
            with torch.no_grad():
                _, current_rotation, current_translation = kabsch_alignment(
                    coordinates,
                    self.reference_coordinates,
                    return_transform=True,
                )
                rotation_diff = torch.trace(current_rotation @ self.rotation.T)
                if rotation_diff < 2.5:
                    print(
                        f"[{self.entry.component_id}] Flip happened; updating affine",
                        flush=True,
                    )
                    self.rotation = current_rotation
                    self.translation = current_translation
                    current_record().event('alignment_update', component_id=self.entry.component_id,
                        rotation=current_rotation.detach().cpu().tolist(), translation=current_translation.detach().cpu().tolist())
        return coordinates @ self.rotation.T + self.translation


def _particle_dataset(args: argparse.Namespace) -> ParticleDataset:
    trans_r = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(3, 3)
        if args.transR
        else None
    )
    return ParticleDataset(
        str(args.star_data_dir),
        args.mrc_data_dir,
        float(args.apix),
        transR=trans_r,
        norm=args.norm,
    )


def _frequency_grid(box_size: int, apix: float) -> torch.Tensor:
    frequencies = (
        np.stack(
            np.meshgrid(
                np.linspace(-0.5, 0.5, box_size, endpoint=False),
                np.linspace(-0.5, 0.5, box_size, endpoint=False),
            ),
            -1,
        )
        / apix
    )
    return torch.from_numpy(frequencies.reshape(-1, 2)).unsqueeze(0).float()


def _ctf_for_batch(
    parameters: torch.Tensor,
    frequencies: torch.Tensor,
    box_size: int,
    device: torch.device,
) -> torch.Tensor:
    parameters = parameters.float()
    voltage, defocus_u, defocus_v, angle, cs, amplitude, phase, _ = parameters.T
    ctf = compute_ctf(
        freqs=frequencies,
        dfu=defocus_u[:, None],
        dfv=defocus_v[:, None],
        dfang=angle[:, None],
        volt=voltage[:, None],
        cs=cs[:, None],
        w=amplitude[:, None],
        phase_shift=phase[:, None],
        bfactor=None,
    )
    return ctf.reshape(-1, 1, box_size, box_size).to(device=device, dtype=torch.float32)


def _assert_same_batch_indices(
    indices: torch.Tensor, world_size: int, device: torch.device
) -> None:
    if world_size == 1:
        return
    value = indices.to(device=device, dtype=torch.long)
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    if any(not torch.equal(value, other) for other in gathered):
        raise RuntimeError("particle DataLoader order differs across ranks")


def _global_parameter_counts(
    component: LocalComponent, world_size: int
) -> tuple[int, int]:
    counts = torch.tensor(
        component.gmm.regularization_counts(),
        device=component.device,
        dtype=torch.long,
    )
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


def _validate_same_model(
    component: LocalComponent, world_size: int
) -> list[dict[str, Any]]:
    local = {
        "rank": dist.get_rank() if distributed_active() else 0,
        "component_id": component.entry.component_id,
        "model_state_sha256": component.model_state_sha256,
        "n_atom": component.atom_weights.numel(),
        "z_bias_shape": list(component.z_bias.shape),
        "contextual_split_metadata": component.contextual_split_metadata,
        "gmm_config": component.gmm.config(),
    }
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    result = [item for item in gathered if item is not None]
    if len({json.dumps(item["gmm_config"], sort_keys=True) for item in result}) != 1:
        raise ValueError("all components must use identical GMM modes and conventions")
    if len({item["model_state_sha256"] for item in result}) != 1:
        raise ValueError("component caches do not contain identical diffusion weights")
    contextual_flags = [item["contextual_split_metadata"] is not None for item in result]
    if any(contextual_flags) and not all(contextual_flags):
        raise ValueError(
            "do not mix independent and full-context diagonal caches in one run"
        )
    if all(contextual_flags):
        source_hashes = {
            item["contextual_split_metadata"].get("source_cache_sha256")
            for item in result
        }
        if None in source_hashes or len(source_hashes) != 1:
            raise ValueError(
                "all contextual component caches must record the same source cache hash"
            )
    return result


def train(args: argparse.Namespace) -> None:
    prefix = os.path.expanduser(str(args.output_trained_model_dir))
    trailing = prefix.endswith(('/', '\\'))
    args.output_trained_model_dir = os.path.abspath(prefix) + (os.sep if trailing else '')
    if args.check_inputs:
        entries, index = resume_index(args, load_manifest(args.component_manifest))
        for rank in range(len(entries)):
            component_for_rank(entries, rank, len(entries))
        reports = []
        for entry in sorted(entries, key=lambda e:e.rank):
            raw, dataset, report = preflight(args, entry)
            reports.append(report)
            del raw, dataset
        validate_partition(reports)
        print(json.dumps(dict(passed=True, components=len(reports), model_loaded=False)))
        return
    rank, world_size, device = _setup_distributed(args)
    try:
        entries, index = phase('manifest/resume', lambda: resume_index(args, load_manifest(args.component_manifest)))
        entry = phase('rank assignment', lambda: component_for_rank(entries, rank, world_size))
        if args.record_dir:
            args.record_dir = str(Path(args.record_dir)/f'rank{rank}')
        _train_initialized(args, entries, entry, device, index)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@recorded('train')
def _train_initialized(args, entries, entry, device, index):
    rank, world_size = entry.rank, len(entries)
    record = current_record()
    raw, dataset, report = phase('input preflight', lambda: preflight(args, entry))
    reports = gather(report)
    phase('partition', lambda: validate_partition(reports))
    phase('resume inputs', lambda: validate_resume(args, index, reports, raw, device))
    settings = seeds(args)
    from chain_parallel.parallel_checkpoint import SCIENCE
    from run_recording import json_value
    scientific = {key:json_value(getattr(args,key)) for key in SCIENCE if hasattr(args,key)}
    configurations = gather(scientific)
    if any(row != configurations[0] for row in configurations):
        raise ValueError('Scientific settings/seeds differ across ranks')
    seed_legacy(settings['seed'])
    record.resolved('parallel_science', scientific, 'Saved configuration inherited for resume; original parallel defaults otherwise')
    record.resolved('parallel_inputs', reports, 'All-rank CPU checks before model construction')
    record.resolved('seeds', settings, 'Shared DataLoader generator; same diffusion seed on every component')
    component = phase('model initialization', lambda: LocalComponent(entry, device, args.train_deterministic, args, raw))
    replica_metadata = _validate_same_model(component, world_size)
    uses_contextual_diagonal_cache = all(item['contextual_split_metadata'] is not None for item in replica_metadata)
    global_atom_count, global_sdev_count = _global_parameter_counts(component, world_size)
    optimizer = torch.optim.AdamW([component.z_bias], lr=args.lr_bias)
    if args.learn_gmm:
        optimizer.add_param_group({'params': component.gmm.amplitude_parameters(), 'lr': args.lr_atom_weights})
        optimizer.add_param_group({'params': component.gmm.shape_parameters(), 'lr': args.lr_sdevs})
    output_prefix = Path(f'{args.output_trained_model_dir}{entry.component_id}_rank{rank}_')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    def write_initial():
        files = export_training_structure(entry.cif_path, f'{output_prefix}_',
            component.pred_dict['coordinate'][0].detach().cpu().numpy(), args.output_format)
        for item in files: record.artifact(item, 'raw_initial_structure')
    phase('initial output', write_initial)
    loader_generator = torch.Generator().manual_seed(settings['data_seed'])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=loader_generator)
    first_epoch, global_step = 0, 0
    if args.resume:
        first_epoch, global_step = phase('restore epoch', lambda: restore_resume(raw, component, optimizer, data_loader))
    del raw
    box_size = int(args.boxsize)
    apix = float(args.apix)
    frequencies = _frequency_grid(box_size, apix)
    density_center = (
        torch.tensor([box_size / 2, box_size / 2], dtype=torch.float32, device=device)
        if args.density_center is None
        else torch.tensor(args.density_center, dtype=torch.float32, device=device).unsqueeze(0)
    )
    target_resolution = float(args.map_resolution)

    shell_weight_freqs, shell_weights = build_halfmap_shell_weights(
        halfmap1_path=args.halfmap1,
        halfmap2_path=args.halfmap2,
        gamma=float(args.fsc_gamma),
        smooth_win=int(args.fsc_smooth_win),
        device=device,
        dtype=torch.float32,
    )
    if rank == 0:
        if shell_weights is None:
            print("[FRC] No half-maps provided. Using source active compute_frc.")
        else:
            print(
                "[FRC] Half-map weights were loaded, but train.py's active "
                "compute_frc does not consume them."
            )
        metadata_path = output_prefix.parent / "chain_parallel_2d_run_metadata.json"
        if metadata_path.exists():
            raise FileExistsError(f"refusing to overwrite {metadata_path}")
        metadata_path.write_text(
            json.dumps(
                {
                    "method": (
                        "full-context-diagonal-component-2d-gmm-projection-sum"
                        if uses_contextual_diagonal_cache
                        else "independent-component-2d-gmm-projection-sum"
                    ),
                    "source_logic": "train.py + pts2img.py + active utils.py::compute_frc",
                    "gmm_config": component.gmm.config(),
                    "exact_full_complex_pairformer_equivalent": False,
                    "uses_full_complex_contextual_diagonal_cache": (
                        uses_contextual_diagonal_cache
                    ),
                    "replicas": replica_metadata,
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "world_size": world_size,
                    "particle_count": len(dataset),
                    "box_size": box_size,
                    "apix": apix,
                    "resolution": float(args.resolution),
                    "map_resolution": target_resolution,
                    "epochs": args.epochs,
                    "batch_size": int(args.batch_size),
                    "mini_batch_size": int(args.mini_batch_size),
                    "learning_rates": {
                        "z_bias": args.lr_bias,
                        "atom_weights": args.lr_atom_weights,
                        "sdevs": args.lr_sdevs,
                    },
                    "penalty_limits": [0.1, 0.8, 1.0, 20.0],
                    "peak_memory": "Not reported and not yet measured in this workspace.",
                    "step_time": "Not reported and not yet measured in this workspace.",
                    "communication_time": "Not reported and not yet measured in this workspace.",
                    "oom_boundary": "Not reported and not yet measured in this workspace.",
                    "per_rank_metrics_pattern": "chain_parallel_2d_metrics_rank{rank}.jsonl",
                    "batch_time_definition": (
                        "Local elapsed time from the pre-batch CUDA synchronization "
                        "through forward, distributed collectives, backward, optimizer, "
                        "and the post-batch CUDA synchronization; metric-file I/O is excluded."
                    ),
                    "distributed_step_time_definition": (
                        "For each (epoch,batch), take the maximum batch_time_seconds "
                        "across rank metric files."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics_path = output_prefix.parent / "chain_parallel_2d_metrics.jsonl"
    rank_metrics_path = (
        output_prefix.parent / f"chain_parallel_2d_metrics_rank{rank}.jsonl"
    )
    if rank_metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {rank_metrics_path}")
    last_placed_coordinates: torch.Tensor | None = None
    stopped = False
    for epoch in range(first_epoch, args.epochs):
        iterator = iter(data_loader)
        completed = False
        for batch_number in range(len(data_loader)):
            batch = phase('read particle batch', lambda: next(iterator))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            data, parameters, translations, rotations, _, indices = batch
            _assert_same_batch_indices(indices, world_size, device)
            data = data.unsqueeze(1).to(device=device, dtype=torch.float32)
            translations = translations.to(device=device, dtype=torch.float32)
            rotations = rotations.to(device=device, dtype=torch.float32)
            ctf = _ctf_for_batch(parameters, frequencies, box_size, device)

            losses = torch.zeros((), dtype=torch.float32)
            penalties = torch.zeros((), dtype=torch.float32)
            for start_index in range(0, data.shape[0], int(args.mini_batch_size)):
                end_index = min(
                    start_index + int(args.mini_batch_size), data.shape[0]
                )
                coordinate_samples = phase("decode", component.sample)
                coordinates = coordinate_samples[0]
                component.pred_dict["coordinate"] = coordinate_samples
                placed_coordinates = phase("placement", lambda: component.place_coordinates(coordinates, args.update_affine_mat))
                last_placed_coordinates = placed_coordinates
                projection = distributed_project_gaussians(
                    projector=component.gmm,
                    atom_coordinates=placed_coordinates,
                    rotations=rotations[start_index:end_index],
                    translations=translations[start_index:end_index],
                    density_center=density_center,
                    resolution=float(args.resolution),
                    box_size=box_size,
                    apix=apix,
                    cutoff_range=5.0,
                    sigma_factor=1.0 / (math.pi * math.sqrt(2.0)),
                )
                projection = projection * float(args.particle_sign)
                loss_frc = -compute_frc(
                    proj=projection.float(),
                    data=data[start_index:end_index].float(),
                    ctf=ctf[start_index:end_index].float(),
                    box_size=box_size,
                    max_freq=(2.0 * apix) / target_resolution,
                ) / data.shape[0]
                local_penalty = component.gmm.regularization(
                    global_atom_count=global_atom_count,
                    global_width_count=global_sdev_count,
                    limits=(0.1, 0.8, 1.0, 20.0),
                )
                # The particle objective is duplicated on all ranks.  Dividing
                # it by P compensates for the autograd SUM backward.  The local
                # penalty is already this rank's contribution to the original
                # full-complex means and therefore is not divided by P.
                loss = loss_frc / world_size + local_penalty
                phase('finite loss', lambda: _require_finite(loss, 'loss'))
                loss.backward()

                losses += loss_frc.detach().cpu()
                penalties += detached_sum(local_penalty).cpu()

            phase('finite gradients', lambda: [_require_finite(p.grad, 'gradient') for group in optimizer.param_groups for p in group['params'] if p.grad is not None])
            optimizer.step()
            phase('finite parameters', lambda: [_require_finite(p, 'parameter') for group in optimizer.param_groups for p in group['params']])
            optimizer.zero_grad()
            global_step += 1
            completed = batch_number + 1 == len(data_loader)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                peak_reserved_memory_mb = (
                    torch.cuda.max_memory_reserved(device) / 1024**2
                )
            else:
                peak_memory_mb = None
                peak_reserved_memory_mb = None
            elapsed = time.perf_counter() - start

            local_row = {
                "rank": rank,
                "component_id": entry.component_id,
                "epoch": epoch,
                "batch": batch_number,
                "frc_loss": float(losses),
                "gmm_penalty": float(penalties),
                "peak_allocated_memory_mb": peak_memory_mb,
                "peak_reserved_memory_mb": peak_reserved_memory_mb,
                "batch_time_seconds": elapsed,
                "communication_time_seconds": "Not reported and not yet measured in this workspace.",
            }
            record.event('train_step', global_step=global_step, epoch=epoch, batch=batch_number,
                component_id=entry.component_id, particle_indices=indices.tolist(),
                total_loss=float(losses+penalties), elapsed_seconds=elapsed,
                gmm_learning=args.learn_gmm, peak_memory_mb=peak_memory_mb)
            with rank_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(local_row) + "\n")

            if rank == 0:
                row = {
                    "epoch": epoch,
                    "batch": batch_number,
                    "frc_loss": float(losses),
                    "gmm_penalty": float(penalties),
                    "rank0_peak_memory_mb": peak_memory_mb,
                    "rank0_peak_reserved_memory_mb": peak_reserved_memory_mb,
                    "rank0_batch_time_seconds": elapsed,
                    "communication_time_seconds": "Not reported and not yet measured in this workspace.",
                    "rank_metrics_file": rank_metrics_path.name,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

            if args.max_steps is not None and global_step >= args.max_steps:
                stopped = True
                break
        if last_placed_coordinates is None:
            raise RuntimeError('Particle DataLoader produced no batches')
        save_epoch(args, component, optimizer, data_loader, reports, epoch, global_step, completed, output_prefix)
        if device.type == 'cuda': torch.cuda.empty_cache()
        if stopped: break


def _require_finite(value, label):
    if not torch.isfinite(value).all():
        raise ValueError(f'Nonfinite {label}; no exception checkpoint will be saved')


def build_parser() -> argparse.ArgumentParser:
    parser = RestartArgumentParser(description="One component cache per rank; component-wide or per-chain rigid placement.")
    parser.add_argument("--component_manifest", required=True, help='YAML assigning one component cache/reference CIF per rank; embedded paths are relative to the manifest. Default: %(default)s.')
    parser.add_argument("--star_data_dir", required=True, help='RELION STAR file containing particle image references, poses and CTF metadata. Default: %(default)s.')
    parser.add_argument("--mrc_data_dir", default=None, help='Root for relative STAR image paths; omitted means the STAR directory. Absolute image paths are used directly. Default: %(default)s.')
    parser.add_argument("--output_trained_model_dir", required=True, help='Output filename prefix; a trailing slash selects a directory. Use a new run location. Default: %(default)s.')
    parser.add_argument("--transR", action="store_true", default=False, help='Use the validated pose-convention matrix diag(1,1,-1); enable only for the matching upstream orientation convention. Default: %(default)s.')
    parser.add_argument("--particle_sign", default=-1.0, type=float, help='Multiplier applied to rendered particle projections; keep the validated data sign convention. Default: %(default)s.')
    parser.add_argument("--boxsize", default=256, type=int, help='Square particle image width/height in pixels; must match STAR/MRCS inputs. Default: %(default)s.')
    parser.add_argument("--apix", default=1.0, type=float, help='Experimental pixel size in Angstrom per pixel; must be positive. Default: %(default)s.')
    parser.add_argument("--norm", action="store_true", default=False, help='Min-max normalize each observed particle to [0,1]; constant images are rejected. Default: %(default)s.')
    parser.add_argument("--resolution", default=3.0, type=float, help='Legacy GMM coordinate/grid scale parameter; not generally a molmap resolution in Angstrom. Default: %(default)s.')
    parser.add_argument("--density_center", default=None, type=float, nargs=2, help='Two image-center coordinates in pixels; omitted uses the box center. Default: %(default)s.')
    parser.add_argument(
        "--train_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True, help='Reuse fixed diffusion stochasticity; disabling resamples noise. Per-chain placement requires fixed stochasticity. Default: %(default)s.')
    parser.add_argument("--device", default="cuda:0", help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl", help='Distributed communication backend: NCCL for CUDA training, Gloo for CPU tests. Default: %(default)s.')
    parser.add_argument("--batch_size", default=32, type=int, help='Particles per optimizer update; all parallel ranks process the same batch. Default: %(default)s.')
    parser.add_argument("--mini_batch_size", default=12, type=int, help='Particles per loss/backward microbatch inside each update; smaller values trade memory for more decoding. Default: %(default)s.')
    parser.add_argument("--update_affine_mat", action="store_true", default=False, help='Enable the existing rigid-transform update safeguard; per-chain mode tests the shared threshold independently per chain. Default: %(default)s.')
    parser.add_argument("--map_resolution", default=5.0, type=float, help='Angstrom resolution cutoff of the active particle FRC objective; not the GMM rendering width. Default: %(default)s.')
    parser.add_argument("--halfmap1", default=None, help='Optional first half-map file, paired with halfmap2. Weights are prepared but not consumed by the current active FRC loss. Default: %(default)s.')
    parser.add_argument("--halfmap2", default=None, help='Optional second half-map file, paired with halfmap1; current active FRC does not consume the prepared weights. Default: %(default)s.')
    parser.add_argument("--fsc_gamma", default=1.0, type=float, help='Exponent for optional half-map weights; these weights currently do not alter the active FRC objective. Default: %(default)s.')
    parser.add_argument("--fsc_smooth_win", default=0, type=int, help='Nonnegative smoothing window for optional half-map weights; zero disables smoothing. Default: %(default)s.')
    parser.add_argument('--epochs', type=positive_int, default=10, help='Total target epochs, including resumed epochs.')
    parser.add_argument('--max_steps', type=positive_int, default=None, help='Stop after this cumulative optimizer step; partial epoch is warm-start only.')
    parser.add_argument('--lr_bias', type=positive_float, default=.01, help='AdamW learning rate for the target-specific latent perturbation. Default: %(default)s.')
    parser.add_argument('--lr_atom_weights', type=positive_float, default=.01, help='AdamW learning rate for GMM amplitudes; unused when GMM learning is disabled. Default: %(default)s.')
    parser.add_argument('--lr_sdevs', type=positive_float, default=.005, help='AdamW learning rate for GMM widths/shape parameters; unused when GMM learning is disabled. Default: %(default)s.')
    parser.add_argument('--learn-gmm', action=argparse.BooleanOptionalAction, default=True, help='Learn both GMM amplitudes and widths; --no-learn-gmm freezes both without blocking coordinate gradients. Default: %(default)s.')
    parser.add_argument('--output-format', choices=('cif','pdb','both'), default='cif', help='CIF always retained for merged output; pdb also writes PDB.')
    parser.add_argument('--by-chain', action='store_true', help='Each local CIF author chain fits independently within the same component decoder.')
    parser.add_argument('--fit-atoms', choices=('ca','all'), default='ca', help='Rigid-fitting core within each local chain: ca uses C-alpha, all uses all atoms; all atoms remain in the output. Default: %(default)s.')
    parser.add_argument('--block-update-trace-threshold', type=float, default=2.5, help='Common rotation trace threshold (-1,3) for independent per-chain affine updates. Default: %(default)s.')
    parser.add_argument('--check-inputs', action='store_true', help='Check all component inputs on CPU without constructing models.')
    parser.add_argument('--distributed-timeout', type=positive_int, default=120, help='Collective timeout in seconds; torchrun terminates peers on worker failure.')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Complete epoch JSON index; fixed grouping/world size and inherited science settings.')
    group.add_argument('--warm-start', action='store_true', help='Component manifest points to old/new refinement checkpoints; reset optimizer/progress.')
    add_seed_arguments(parser, 'train')
    add_record_arguments(parser)
    add_gmm_arguments(parser)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.original_argv = list(sys.argv)
    if parsed_args.distributed_timeout <= 0:
        raise ValueError('--distributed-timeout must be positive')
    print(parsed_args, flush=True)
    train(parsed_args)
