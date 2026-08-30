"""Materialize a component-local Protenix shared diffusion cache.

The full-complex run may deliberately save ``z_trunk`` instead of constructing
the full ``pair_z`` tensor.  After a contextual diagonal block has been sliced,
this utility applies the original Protenix cache preparation functions to that
smaller block exactly once and saves ``pair_z``, ``p_lm``, and ``c_l``.

The input cache is never edited.  The output cache drops ``z_trunk`` so the
existing CoCoFold2 sampler takes its normal cached-``pair_z`` path.  This is a
performance transformation for repeated sampling; it is not exact full-complex
diffusion because cross-component diffusion interactions remain absent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch


PROPOSAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROPOSAL_DIR))

from contextual_cache import ContextualCacheError, validate_component_cache
from prepare_contextual_diffusion_caches import (
    _build_diffusion_module,
    _configure_import_path,
    _load_cache,
    _sha256,
    _tree_to_cpu,
    _tree_to_device,
)


PrepareDecorator = Callable[[Callable[..., Any]], Callable[..., Any]]


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }


def _assert_finite(
    name: str, value: torch.Tensor, *, chunk_elements: int
) -> dict[str, Any]:
    """Reject NaN/Inf without allocating a full-tensor boolean mask."""
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    descriptor = _tensor_descriptor(value)
    if not (value.is_floating_point() or value.is_complex()):
        descriptor["all_finite"] = True
        return descriptor

    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements]
        if not bool(torch.isfinite(chunk).all().item()):
            raise ContextualCacheError(f"materialized {name} contains NaN or Inf")
    descriptor["all_finite"] = True
    return descriptor


def _require_z_trunk_cache(cache: dict[str, Any]) -> None:
    validate_component_cache(cache, require_atom_local=True)
    if not isinstance(cache.get("z_trunk"), torch.Tensor):
        raise ContextualCacheError(
            "input must be a split component cache containing z_trunk"
        )
    if cache.get("pair_z") is not None:
        raise ContextualCacheError(
            "input already contains pair_z; refusing to materialize it again"
        )
    if cache.get("p_lm") is not None or cache.get("c_l") is not None:
        raise ContextualCacheError(
            "z_trunk input unexpectedly contains p_lm or c_l"
        )


def materialize_local_shared_cache(
    cache: dict[str, Any],
    *,
    diffusion_module: torch.nn.Module,
    device: torch.device,
    prepare_decorator: PrepareDecorator,
    source_cache: str,
    source_cache_sha256: str,
    chunk_elements: int = 1_000_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a new cached-``pair_z`` component and numerical descriptors.

    ``prepare_decorator`` is the source Protenix autocast-control decorator.
    It is injected explicitly so the small unit tests do not need a Protenix
    installation and the real CLI still uses the exact source implementation.
    """
    _require_z_trunk_cache(cache)
    features = _tree_to_device(cache["input_feature_dict"], device)
    z_trunk = cache["z_trunk"].to(device)

    prepare_pair_cache = prepare_decorator(
        diffusion_module.diffusion_conditioning.prepare_cache
    )
    prepare_atom_cache = prepare_decorator(
        diffusion_module.atom_attention_encoder.prepare_cache
    )

    with torch.no_grad():
        # This is the same source call used when
        # enable_diffusion_shared_vars_cache=True, but it is applied only after
        # the full-context z_trunk diagonal has been reduced to one component.
        pair_z = prepare_pair_cache(features["relp"], z_trunk, False)
        if not isinstance(pair_z, torch.Tensor):
            raise ContextualCacheError(
                "diffusion_conditioning.prepare_cache did not return a tensor"
            )
        p_lm, c_l = prepare_atom_cache(
            ref_pos=features["ref_pos"],
            ref_charge=features["ref_charge"],
            ref_mask=features["ref_mask"],
            ref_element=features["ref_element"],
            ref_atom_name_chars=features["ref_atom_name_chars"],
            atom_to_token_idx=features["atom_to_token_idx"],
            d_lm=features["d_lm"],
            v_lm=features["v_lm"],
            pad_info=features["pad_info"],
            r_l=True,
            z=pair_z,
            inplace_safe=False,
        )
        if not isinstance(p_lm, torch.Tensor) or not isinstance(c_l, torch.Tensor):
            raise ContextualCacheError(
                "atom_attention_encoder.prepare_cache must return tensor p_lm and c_l"
            )

    tensor_descriptors = {
        "pair_z": _assert_finite(
            "pair_z", pair_z, chunk_elements=chunk_elements
        ),
        "p_lm": _assert_finite("p_lm", p_lm, chunk_elements=chunk_elements),
        "c_l": _assert_finite("c_l", c_l, chunk_elements=chunk_elements),
    }

    result = dict(cache)
    result["input_feature_dict"] = _tree_to_cpu(features)
    result["z_trunk"] = None
    result["pair_z"] = _tree_to_cpu(pair_z)
    result["p_lm"] = _tree_to_cpu(p_lm)
    result["c_l"] = _tree_to_cpu(c_l)

    materialization = {
        "schema_version": 1,
        "method": "component_local_shared_diffusion_cache_from_z_trunk",
        "source_component_cache": source_cache,
        "source_component_cache_sha256": source_cache_sha256,
        "device": str(device),
        "layernorm_type": os.environ.get("LAYERNORM_TYPE", "fast_layernorm"),
        "enable_efficient_fusion": bool(cache["enable_efficient_fusion"]),
        "tensors": tensor_descriptors,
        "optimization_semantics": (
            "The output uses cached-pair_z refinement semantics. A later z_bias is "
            "therefore applied in pair_z space, as in the existing pair_z cache path, "
            "rather than before diffusion conditioning in z_trunk space."
        ),
        "scientific_status": "Not reported and not yet measured in this workspace.",
    }
    metadata = dict(cache["contextual_split_metadata"])
    metadata.update(
        {
            "conditioning_tensor": "pair_z",
            "materialized_from_conditioning_tensor": "z_trunk",
            "local_shared_diffusion_cache_materialized": True,
            "shared_atom_cache_rebuilt": True,
            "pair_z_origin": (
                "component_local_diffusion_conditioning_from_full_context_"
                "z_trunk_diagonal"
            ),
            "materialization": materialization,
        }
    )
    result["contextual_split_metadata"] = metadata
    result["local_diffusion_cache_materialization"] = materialization

    validate_component_cache(result, require_atom_local=True)
    return result, tensor_descriptors


def command_materialize(args: argparse.Namespace) -> None:
    _configure_import_path()
    from protenix.utils.torch_utils import autocasting_disable_decorator

    input_path = Path(args.input_cache).expanduser().resolve()
    output_path = Path(args.output_cache).expanduser().resolve()
    if input_path == output_path:
        raise ContextualCacheError("output cache must differ from input cache")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite to replace it"
        )

    source_hash = _sha256(input_path)
    cache = _load_cache(input_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)

    diffusion_module = _build_diffusion_module(cache, device)
    prepare_decorator = autocasting_disable_decorator(
        cache["configs"].skip_amp.sample_diffusion
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result, tensor_descriptors = materialize_local_shared_cache(
        cache,
        diffusion_module=diffusion_module,
        device=device,
        prepare_decorator=prepare_decorator,
        source_cache=str(input_path),
        source_cache_sha256=source_hash,
        chunk_elements=args.chunk_elements,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_bytes: int | None = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes: int | None = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated_bytes = None
        peak_reserved_bytes = None
    elapsed = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    output_hash = _sha256(output_path)
    report = {
        "schema_version": 1,
        "method": "component_local_shared_diffusion_cache_from_z_trunk",
        "input_cache": str(input_path),
        "input_cache_sha256": source_hash,
        "output_cache": str(output_path),
        "output_cache_sha256": output_hash,
        "component_id": result["contextual_split_metadata"]["component_id"],
        "asym_ids": result["contextual_split_metadata"]["asym_ids"],
        "n_token": result["contextual_split_metadata"]["component_n_token"],
        "n_atom": result["contextual_split_metadata"]["component_n_atom"],
        "device": str(device),
        "wall_time_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "tensors": tensor_descriptors,
        "scientific_status": "Not reported and not yet measured in this workspace.",
    }
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one split contextual z_trunk cache into a component-local "
            "shared pair_z/p_lm/c_l cache"
        )
    )
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-json")
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_elements < 1:
        raise ValueError("--chunk-elements must be positive")
    command_materialize(args)


if __name__ == "__main__":
    main()
