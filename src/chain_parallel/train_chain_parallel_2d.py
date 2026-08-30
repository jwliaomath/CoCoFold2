"""Proposed chain-parallel trainer that preserves CoCoFold2's 2-D GMM logic.

This is a new, standalone proposal.  It does not modify or import the previous
3-D chain-parallel implementation.  Run it from the CoCoFold2 repository root
with one component per torchrun rank.
"""

from __future__ import annotations

import argparse
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
from distributed_gmm import (
    detached_sum,
    distributed_active,
    distributed_pdb2img,
    local_source_equivalent_penalty,
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
        dist.init_process_group(args.backend)
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
    ) -> None:
        from protenix.model.modules.diffusion import DiffusionModule

        raw = torch.load(entry.diffusion_data_dir, map_location="cpu", weights_only=False)
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
        self.configs.train_seed = 42
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
        self.atom_weights = torch.nn.Parameter(
            initial_atom_weights.detach().to(device=device, dtype=torch.float32)
        )
        self.sdevs = torch.nn.Parameter(
            torch.full(
                (self.atom_weights.numel(), 2),
                3.0 / (math.pi * math.sqrt(2.0)),
                device=device,
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            _, rotation, translation = kabsch_alignment(
                initial_coordinates,
                self.reference_coordinates,
                return_transform=True,
            )
        self.rotation = rotation
        self.translation = translation
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
        return coordinates @ self.rotation.T + self.translation


def _particle_dataset(args: argparse.Namespace) -> ParticleDataset:
    trans_r = (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(3, 3)
        if args.transR
        else None
    )
    return ParticleDataset(
        str(args.star_data_dir),
        str(args.mrc_data_dir),
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
        [component.atom_weights.numel(), component.sdevs.numel()],
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
    }
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local)
    result = [item for item in gathered if item is not None]
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
    entries = load_manifest(args.component_manifest)
    rank, world_size, device = _setup_distributed(args)
    entry = component_for_rank(entries, rank, world_size)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    for required in (
        entry.diffusion_data_dir,
        entry.cif_path,
        Path(args.star_data_dir),
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    component = LocalComponent(entry, device, args.train_deterministic)
    replica_metadata = _validate_same_model(component, world_size)
    uses_contextual_diagonal_cache = all(
        item["contextual_split_metadata"] is not None for item in replica_metadata
    )
    global_atom_count, global_sdev_count = _global_parameter_counts(
        component, world_size
    )

    optimizer = torch.optim.AdamW([component.z_bias], lr=1e-2)
    optimizer.add_param_group({"params": component.atom_weights, "lr": 1e-2})
    optimizer.add_param_group({"params": component.sdevs, "lr": 5e-3})

    output_prefix = Path(
        f"{args.output_trained_model_dir}{entry.component_id}_rank{rank}_"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    initial_output = Path(f"{output_prefix}_.pdb")
    if initial_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing proposal run: {initial_output}"
        )
    replace_cif_coordinates(
        input_cif=str(entry.cif_path),
        output_cif=str(initial_output),
        new_coords=component.pred_dict["coordinate"][0].detach().cpu().numpy(),
    )

    dataset = _particle_dataset(args)
    loader_generator = torch.Generator().manual_seed(42)
    data_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=loader_generator,
    )
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
                    "epochs": 10,
                    "batch_size": int(args.batch_size),
                    "mini_batch_size": int(args.mini_batch_size),
                    "learning_rates": {
                        "z_bias": 1e-2,
                        "atom_weights": 1e-2,
                        "sdevs": 5e-3,
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
    for epoch in range(10):
        for batch_number, batch in enumerate(data_loader):
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
                coordinate_samples = component.sample()
                coordinates = coordinate_samples[0]
                component.pred_dict["coordinate"] = coordinate_samples
                placed_coordinates = component.place_coordinates(
                    coordinates, args.update_affine_mat
                )
                last_placed_coordinates = placed_coordinates
                projection = distributed_pdb2img(
                    atom_coordinates=placed_coordinates,
                    atom_weights=component.atom_weights,
                    sdevs=component.sdevs,
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
                local_penalty = local_source_equivalent_penalty(
                    atom_weights=component.atom_weights,
                    sdevs=component.sdevs,
                    global_atom_count=global_atom_count,
                    global_sdev_count=global_sdev_count,
                    limits=(0.1, 0.8, 1.0, 20.0),
                )
                # The particle objective is duplicated on all ranks.  Dividing
                # it by P compensates for the autograd SUM backward.  The local
                # penalty is already this rank's contribution to the original
                # full-complex means and therefore is not divided by P.
                loss = loss_frc / world_size + local_penalty
                loss.backward()

                losses += loss_frc.detach().cpu()
                penalties += detached_sum(local_penalty).cpu()

            optimizer.step()
            optimizer.zero_grad()
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

        if last_placed_coordinates is None:
            raise RuntimeError("particle DataLoader produced no batches")
        model_path = Path(f"{output_prefix}{epoch + 1}.pth")
        model_data = {
            "model_state": component.diffusion_module.state_dict(),
            "opt_state": optimizer.state_dict(),
            "atom_weights": component.atom_weights,
            "sdevs": component.sdevs,
            "pred_dict": component.pred_dict,
            "input_feature_dict": component.input_feature_dict,
            "s_inputs": component.s_inputs,
            "s_trunk": component.s_trunk,
            "z_trunk": component.z_trunk,
            "pair_z": component.pair_z,
            "p_lm": component.p_lm,
            "c_l": component.c_l,
            "N_sample": component.n_sample,
            "noise_schedule": component.noise_schedule,
            "inplace_safe": component.inplace_safe,
            "configs": component.configs,
            "s_inputs_bias": None,
            "s_bias": None,
            "z_bias": component.z_bias,
            "z_mul": None,
            "component_id": entry.component_id,
            "rotation": component.rotation,
            "translation": component.translation,
            "contextual_split_metadata": component.contextual_split_metadata,
        }
        torch.save(deep_clone(model_data), model_path)
        replace_cif_coordinates(
            input_cif=str(entry.cif_path),
            output_cif=f"{output_prefix}{epoch + 1}.pdb",
            new_coords=last_placed_coordinates.detach().cpu().numpy(),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if distributed_active():
            dist.barrier()

    if distributed_active():
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component_manifest", required=True)
    parser.add_argument("--star_data_dir", required=True)
    parser.add_argument("--mrc_data_dir", required=True)
    parser.add_argument("--output_trained_model_dir", required=True)
    parser.add_argument("--transR", action="store_true", default=False)
    parser.add_argument("--particle_sign", default=-1.0, type=float)
    parser.add_argument("--boxsize", default=256, type=int)
    parser.add_argument("--apix", default=1.0, type=float)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--resolution", default=3.0, type=float)
    parser.add_argument("--density_center", default=None, type=float, nargs=2)
    parser.add_argument(
        "--train_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--mini_batch_size", default=12, type=int)
    parser.add_argument("--update_affine_mat", action="store_true", default=False)
    parser.add_argument("--map_resolution", default=5.0, type=float)
    parser.add_argument("--halfmap1", default=None)
    parser.add_argument("--halfmap2", default=None)
    parser.add_argument("--fsc_gamma", default=1.0, type=float)
    parser.add_argument("--fsc_smooth_win", default=0, type=int)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print(parsed_args, flush=True)
    train(parsed_args)
