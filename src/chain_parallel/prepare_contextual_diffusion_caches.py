"""Inspect and split a full-complex cache into contextual diagonal caches.

This is an additive proposal utility.  It imports the same Protenix functions
used by CoCoFold2 to rebuild component-local atom windows and shared atom
conditioning after the full-context token/pair tensors have been sliced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


PROPOSAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROPOSAL_DIR))

from contextual_cache import (
    ContextualCacheError,
    inspect_contextual_cache,
    slice_contextual_cache,
    validate_component_cache,
)


def _resolve_cocofold2_root() -> Path:
    configured = os.environ.get("COCOFOLD2_ROOT")
    candidates = (
        [Path(configured).expanduser().resolve()]
        if configured
        else [PROPOSAL_DIR.parents[1], PROPOSAL_DIR.parents[1] / "src"]
    )
    for candidate in candidates:
        if (candidate / "model" / "protenix.py").is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "COCOFOLD2_ROOT must directly contain model/protenix.py; checked: " + checked
    )


def _configure_import_path() -> Path:
    root = _resolve_cocofold2_root()
    sys.path.insert(0, str(root))
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _load_cache(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ContextualCacheError(f"{path} does not contain a diffusion-cache mapping")
    return value


def _build_diffusion_module(cache: dict[str, Any], device: torch.device):
    from protenix.model.modules.diffusion import DiffusionModule

    module = DiffusionModule(**cache["configs"].model.diffusion_module).to(device)
    module.load_state_dict(cache["model_state"])
    module.eval().requires_grad_(False)
    return module


def _rebuild_atom_local_state(
    component: dict[str, Any],
    diffusion_module: torch.nn.Module,
    device: torch.device,
) -> None:
    """Use the source functions to rebuild windows after atom reindexing."""
    from model.protenix import update_input_feature_dict
    from protenix.utils.torch_utils import autocasting_disable_decorator

    features = dict(component["input_feature_dict"])
    for key in ("d_lm", "v_lm", "pad_info"):
        features.pop(key, None)
    features = _tree_to_device(features, device)
    with torch.no_grad():
        features = update_input_feature_dict(features)

        if component["pair_z"] is not None:
            pair_z = component["pair_z"].to(device)
            prepare_atom_cache = autocasting_disable_decorator(
                component["configs"].skip_amp.sample_diffusion
            )(diffusion_module.atom_attention_encoder.prepare_cache)
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
            component["p_lm"] = _tree_to_cpu(p_lm)
            component["c_l"] = _tree_to_cpu(c_l)
            component["contextual_split_metadata"][
                "shared_atom_cache_rebuilt"
            ] = True
        else:
            # This matches the source non-shared-cache path: DiffusionModule
            # prepares pair_z, p_lm, and c_l during each denoising call.
            component["p_lm"] = None
            component["c_l"] = None
            component["contextual_split_metadata"][
                "shared_atom_cache_rebuilt"
            ] = False

    component["input_feature_dict"] = _tree_to_cpu(features)
    component["contextual_split_metadata"]["atom_local_features_rebuilt"] = True
    validate_component_cache(component, require_atom_local=True)


def _resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_split_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ContextualCacheError("split spec must be a schema_version: 1 mapping")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise ContextualCacheError("split spec must contain a non-empty components list")
    return value


def _validate_partition(
    cache: dict[str, Any], components: list[dict[str, Any]], require_complete: bool
) -> None:
    summary = inspect_contextual_cache(cache)
    available = {
        int(item["asym_id"]) for item in summary["asym_components"]
    }
    used: set[int] = set()
    ids: set[str] = set()
    for item in components:
        component_id = str(item.get("id", ""))
        if component_id in ids:
            raise ContextualCacheError(f"duplicate component id: {component_id}")
        ids.add(component_id)
        selected = {int(value) for value in item.get("asym_ids", [])}
        overlap = used.intersection(selected)
        if overlap:
            raise ContextualCacheError(
                f"asym_ids assigned to more than one component: {sorted(overlap)}"
            )
        unknown = selected.difference(available)
        if unknown:
            raise ContextualCacheError(f"unknown asym_ids: {sorted(unknown)}")
        used.update(selected)
    if require_complete and used != available:
        raise ContextualCacheError(
            f"complete partition required; assigned {sorted(used)}, available {sorted(available)}"
        )


def command_inspect(args: argparse.Namespace) -> None:
    _configure_import_path()
    cache_path = Path(args.cache).expanduser().resolve()
    cache = _load_cache(cache_path)
    report = {
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        **inspect_contextual_cache(cache),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def command_split(args: argparse.Namespace) -> None:
    _configure_import_path()
    spec_path = Path(args.spec).expanduser().resolve()
    spec = _load_split_spec(spec_path)
    base = spec_path.parent
    source_path = _resolve_from(base, str(spec["source_cache"]))
    source_cache = _load_cache(source_path)
    components = spec["components"]
    require_complete = bool(spec.get("require_complete_partition", True))
    _validate_partition(source_cache, components, require_complete)
    source_summary = inspect_contextual_cache(source_cache)
    if "expected_source_n_token" in spec and int(
        spec["expected_source_n_token"]
    ) != int(source_summary["n_token"]):
        raise ContextualCacheError(
            "source token count does not match expected_source_n_token"
        )
    if "expected_source_n_atom" in spec and int(spec["expected_source_n_atom"]) != int(
        source_summary["n_atom"]
    ):
        raise ContextualCacheError(
            "source atom count does not match expected_source_n_atom"
        )

    source_hash = _sha256(source_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)

    diffusion_module = _build_diffusion_module(source_cache, device)
    output_records: list[dict[str, Any]] = []
    source_resolved = source_path.resolve()
    for item in components:
        component_id = str(item["id"])
        output_path = _resolve_from(base, str(item["output_cache"]))
        if output_path.resolve() == source_resolved:
            raise ContextualCacheError("an output cache cannot overwrite the source cache")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output_path}; pass --overwrite to replace it"
            )

        component = slice_contextual_cache(
            source_cache,
            component_id=component_id,
            asym_ids=item["asym_ids"],
        )
        component_metadata = component["contextual_split_metadata"]
        if "expected_n_token" in item and int(item["expected_n_token"]) != int(
            component_metadata["component_n_token"]
        ):
            raise ContextualCacheError(
                f"component {component_id}: selected token count does not match "
                "expected_n_token"
            )
        if "expected_n_atom" in item and int(item["expected_n_atom"]) != int(
            component_metadata["component_n_atom"]
        ):
            raise ContextualCacheError(
                f"component {component_id}: selected atom count does not match "
                "expected_n_atom"
            )
        component["contextual_split_metadata"].update(
            {
                "source_cache": str(source_path),
                "source_cache_sha256": source_hash,
                "split_spec": str(spec_path),
            }
        )
        _rebuild_atom_local_state(component, diffusion_module, device)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(component, output_path)
        output_hash = _sha256(output_path)
        record = {
            "component_id": component_id,
            "asym_ids": component["contextual_split_metadata"]["asym_ids"],
            "n_token": component["contextual_split_metadata"]["component_n_token"],
            "n_atom": component["contextual_split_metadata"]["component_n_atom"],
            "conditioning_tensor": component["contextual_split_metadata"][
                "conditioning_tensor"
            ],
            "output_cache": str(output_path),
            "output_cache_sha256": output_hash,
        }
        output_records.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "schema_version": 1,
        "method": "full_complex_contextual_diagonal_cache",
        "source_cache": str(source_path),
        "source_cache_sha256": source_hash,
        "device_used_for_atom_cache_rebuild": str(device),
        "require_complete_partition": require_complete,
        "components": output_records,
        "scientific_status": (
            "Not reported and not yet measured in this workspace."
        ),
    }
    report_path_value = spec.get("report_json")
    if report_path_value:
        report_path = _resolve_from(base, str(report_path_value))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare full-context diagonal component diffusion caches"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="print asym_id, token, atom, and tensor-shape metadata"
    )
    inspect_parser.add_argument("--cache", required=True)
    inspect_parser.add_argument("--output-json")
    inspect_parser.set_defaults(function=command_inspect)

    split_parser = subparsers.add_parser(
        "split", help="slice contextual token/pair features and rebuild atom caches"
    )
    split_parser.add_argument("--spec", required=True)
    split_parser.add_argument("--device", default="cpu")
    split_parser.add_argument("--overwrite", action="store_true", default=False)
    split_parser.set_defaults(function=command_split)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
