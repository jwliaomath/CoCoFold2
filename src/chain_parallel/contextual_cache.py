"""Strict slicing of a full-complex CoCoFold2/Protenix diffusion cache.

The full Pairformer is evaluated before this module is used.  For each selected
component, the module keeps the corresponding full-context single features and
the diagonal block of the full-context pair features.  It also selects the
component atoms and remaps the global atom-to-token indices to local indices.

The atom-local ``d_lm``, ``v_lm``, ``pad_info``, ``p_lm``, and ``c_l`` caches
must be rebuilt after slicing because Protenix local atom windows depend on the
atom ordering and component length.  ``slice_contextual_cache`` deliberately
removes those fields; the CLI in ``prepare_contextual_diffusion_caches.py``
rebuilds them with the source Protenix functions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


class ContextualCacheError(ValueError):
    """Raised when a cache cannot be sliced without guessing its semantics."""


CORE_TOKEN_FEATURES = (
    "asym_id",
    "residue_index",
    "entity_id",
    "sym_id",
    "token_index",
)
CORE_ATOM_FEATURES = (
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
)
DERIVED_ATOM_FEATURES = ("d_lm", "v_lm", "pad_info")

REQUIRED_CACHE_KEYS = (
    "model_state",
    "input_feature_dict",
    "s_inputs",
    "s_trunk",
    "z_trunk",
    "pair_z",
    "N_sample",
    "noise_schedule",
    "inplace_safe",
    "configs",
    "enable_efficient_fusion",
)


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _require_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ContextualCacheError(f"{key!r} must be a torch.Tensor")
    return value


def _index_select(value: torch.Tensor, axis: int, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(value, axis, indices.to(device=value.device))


def _starts_with(shape: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
    return len(shape) >= len(prefix) and shape[: len(prefix)] == prefix


def _select_after_batch_prefix(
    value: torch.Tensor,
    indices: torch.Tensor,
    batch_shape: tuple[int, ...],
    expected_size: int,
    feature_name: str,
) -> torch.Tensor:
    shape = tuple(value.shape)
    axis = len(batch_shape)
    if not _starts_with(shape, batch_shape) or len(shape) <= axis:
        raise ContextualCacheError(
            f"{feature_name!r} shape {shape} does not start with batch shape {batch_shape}"
        )
    if shape[axis] != expected_size:
        raise ContextualCacheError(
            f"{feature_name!r} shape {shape} has size {shape[axis]} at its primary "
            f"axis; expected {expected_size}"
        )
    return _index_select(value, axis, indices)


def _select_pair_after_batch_prefix(
    value: torch.Tensor,
    indices: torch.Tensor,
    batch_shape: tuple[int, ...],
    n_token: int,
    feature_name: str,
) -> torch.Tensor:
    shape = tuple(value.shape)
    axis = len(batch_shape)
    if (
        not _starts_with(shape, batch_shape)
        or len(shape) <= axis + 1
        or shape[axis : axis + 2] != (n_token, n_token)
    ):
        raise ContextualCacheError(
            f"{feature_name!r} shape {shape} is not {batch_shape} + "
            f"({n_token}, {n_token}, ...)"
        )
    selected = _index_select(value, axis, indices)
    return _index_select(selected, axis + 1, indices)


def _consistent_indices_for_asym_ids(
    asym_id: torch.Tensor, asym_ids: tuple[int, ...]
) -> torch.Tensor:
    if asym_id.ndim < 1:
        raise ContextualCacheError("input_feature_dict['asym_id'] must have a token axis")
    n_token = int(asym_id.shape[-1])
    rows = asym_id.detach().cpu().reshape(-1, n_token)
    wanted = torch.tensor(asym_ids, dtype=rows.dtype)
    masks = torch.stack([torch.isin(row, wanted) for row in rows])
    if not torch.equal(masks, masks[0].expand_as(masks)):
        raise ContextualCacheError(
            "all cache batch entries must select the same token indices for a component"
        )
    indices = torch.nonzero(masks[0], as_tuple=False).flatten()
    if indices.numel() == 0:
        available = sorted(int(value) for value in torch.unique(rows).tolist())
        raise ContextualCacheError(
            f"asym_ids {list(asym_ids)} select no tokens; available asym_ids are {available}"
        )
    return indices


def _consistent_atom_mapping(atom_to_token_idx: torch.Tensor) -> torch.Tensor:
    if atom_to_token_idx.ndim < 1:
        raise ContextualCacheError("atom_to_token_idx must have an atom axis")
    n_atom = int(atom_to_token_idx.shape[-1])
    rows = atom_to_token_idx.detach().cpu().reshape(-1, n_atom).long()
    if not torch.equal(rows, rows[0].expand_as(rows)):
        raise ContextualCacheError(
            "all cache batch entries must have the same atom_to_token_idx mapping"
        )
    return rows[0]


def _selected_atoms_and_local_mapping(
    atom_to_token_idx: torch.Tensor,
    token_indices: torch.Tensor,
    n_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_mapping = _consistent_atom_mapping(atom_to_token_idx)
    if global_mapping.numel() == 0:
        raise ContextualCacheError("cache contains no atoms")
    if int(global_mapping.min()) < 0 or int(global_mapping.max()) >= n_token:
        raise ContextualCacheError(
            "atom_to_token_idx contains values outside the full token range"
        )
    atom_mask = torch.isin(global_mapping, token_indices.long())
    atom_indices = torch.nonzero(atom_mask, as_tuple=False).flatten()
    if atom_indices.numel() == 0:
        raise ContextualCacheError("selected tokens contain no atoms")

    global_to_local = torch.full((n_token,), -1, dtype=torch.long)
    global_to_local[token_indices.long()] = torch.arange(
        token_indices.numel(), dtype=torch.long
    )
    local_mapping = global_to_local[global_mapping[atom_indices]]
    if int(local_mapping.min()) < 0:
        raise AssertionError("internal atom-to-token remapping error")
    return atom_indices, local_mapping


def _broadcast_local_mapping(
    template: torch.Tensor, local_mapping: torch.Tensor
) -> torch.Tensor:
    batch_shape = tuple(template.shape[:-1])
    value = local_mapping.to(device=template.device, dtype=template.dtype)
    if not batch_shape:
        return value
    return value.reshape((1,) * len(batch_shape) + (value.numel(),)).expand(
        *batch_shape, value.numel()
    ).clone()


def _normalise_asym_ids(asym_ids: Iterable[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in asym_ids))
    if not result:
        raise ContextualCacheError("a component must select at least one asym_id")
    return result


def inspect_contextual_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable shape and chain summary without changing the cache."""
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in cache]
    if missing:
        raise ContextualCacheError(f"diffusion cache is missing keys: {missing}")
    features = cache["input_feature_dict"]
    if not isinstance(features, Mapping):
        raise ContextualCacheError("input_feature_dict must be a mapping")

    asym_id = _require_tensor(features, "asym_id")
    atom_to_token = _require_tensor(features, "atom_to_token_idx")
    n_token = int(asym_id.shape[-1])
    n_atom = int(atom_to_token.shape[-1])
    asym_rows = asym_id.detach().cpu().reshape(-1, n_token)
    mapping = _consistent_atom_mapping(atom_to_token)
    if not torch.equal(asym_rows, asym_rows[0].expand_as(asym_rows)):
        raise ContextualCacheError(
            "inspect requires identical asym_id layout in every cache batch entry"
        )

    asym_summary: list[dict[str, int]] = []
    for value in sorted(int(item) for item in torch.unique(asym_rows[0]).tolist()):
        token_indices = torch.nonzero(asym_rows[0] == value, as_tuple=False).flatten()
        atom_count = int(torch.isin(mapping, token_indices).sum())
        asym_summary.append(
            {
                "asym_id": value,
                "n_token": int(token_indices.numel()),
                "n_atom": atom_count,
            }
        )

    conditioning = "pair_z" if cache.get("pair_z") is not None else "z_trunk"
    if cache.get(conditioning) is None:
        raise ContextualCacheError("cache contains neither pair_z nor z_trunk")
    return {
        "n_token": n_token,
        "n_atom": n_atom,
        "batch_shape": list(asym_id.shape[:-1]),
        "conditioning_tensor": conditioning,
        "asym_components": asym_summary,
        "top_level_tensor_shapes": {
            key: _shape(cache.get(key))
            for key in ("s_inputs", "s_trunk", "z_trunk", "pair_z", "p_lm", "c_l")
        },
        "input_feature_tensor_shapes": {
            key: _shape(value)
            for key, value in sorted(features.items())
            if isinstance(value, torch.Tensor)
        },
    }


def slice_contextual_cache(
    cache: Mapping[str, Any],
    *,
    component_id: str,
    asym_ids: Iterable[int],
) -> dict[str, Any]:
    """Build one component cache from full-complex contextual representations.

    This function is intentionally strict and only retains input features whose
    token/atom semantics are explicit.  The returned cache is not ready to save
    until the caller rebuilds the derived atom-local features.  ``p_lm`` and
    ``c_l`` are set to ``None`` for the same reason.
    """
    if not component_id or any(character in component_id for character in "/\\"):
        raise ContextualCacheError(f"invalid component_id: {component_id!r}")
    selected_asym_ids = _normalise_asym_ids(asym_ids)
    summary = inspect_contextual_cache(cache)
    features = cache["input_feature_dict"]

    asym_id = _require_tensor(features, "asym_id")
    atom_to_token = _require_tensor(features, "atom_to_token_idx")
    n_token = int(summary["n_token"])
    n_atom = int(summary["n_atom"])
    token_batch_shape = tuple(asym_id.shape[:-1])
    atom_batch_shape = tuple(atom_to_token.shape[:-1])
    if token_batch_shape != atom_batch_shape:
        raise ContextualCacheError(
            f"token batch shape {token_batch_shape} differs from atom batch shape "
            f"{atom_batch_shape}"
        )

    token_indices = _consistent_indices_for_asym_ids(asym_id, selected_asym_ids)
    atom_indices, local_mapping = _selected_atoms_and_local_mapping(
        atom_to_token, token_indices, n_token
    )

    component_features: dict[str, Any] = {}
    for key in CORE_TOKEN_FEATURES:
        value = _require_tensor(features, key)
        component_features[key] = _select_after_batch_prefix(
            value, token_indices, token_batch_shape, n_token, key
        )
    relp = _require_tensor(features, "relp")
    component_features["relp"] = _select_pair_after_batch_prefix(
        relp, token_indices, token_batch_shape, n_token, "relp"
    )
    component_features["atom_to_token_idx"] = _broadcast_local_mapping(
        atom_to_token, local_mapping
    )
    for key in CORE_ATOM_FEATURES:
        value = _require_tensor(features, key)
        component_features[key] = _select_after_batch_prefix(
            value, atom_indices, atom_batch_shape, n_atom, key
        )
    s_inputs = _require_tensor(cache, "s_inputs")
    s_trunk = _require_tensor(cache, "s_trunk")
    component_s_inputs = _select_after_batch_prefix(
        s_inputs, token_indices, token_batch_shape, n_token, "s_inputs"
    )
    component_s_trunk = _select_after_batch_prefix(
        s_trunk, token_indices, token_batch_shape, n_token, "s_trunk"
    )

    z_trunk = cache.get("z_trunk")
    pair_z = cache.get("pair_z")
    if z_trunk is not None and pair_z is not None:
        raise ContextualCacheError("cache unexpectedly contains both z_trunk and pair_z")
    if z_trunk is not None:
        if not isinstance(z_trunk, torch.Tensor):
            raise ContextualCacheError("z_trunk must be a tensor or None")
        component_z_trunk = _select_pair_after_batch_prefix(
            z_trunk, token_indices, token_batch_shape, n_token, "z_trunk"
        )
        component_pair_z = None
        conditioning = "z_trunk"
    else:
        if not isinstance(pair_z, torch.Tensor):
            raise ContextualCacheError("pair_z must be a tensor when z_trunk is None")
        component_z_trunk = None
        component_pair_z = _select_pair_after_batch_prefix(
            pair_z, token_indices, token_batch_shape, n_token, "pair_z"
        )
        conditioning = "pair_z"

    metadata = {
        "schema_version": 1,
        "method": "full_complex_contextual_diagonal_cache",
        "component_id": component_id,
        "asym_ids": list(selected_asym_ids),
        "source_n_token": n_token,
        "source_n_atom": n_atom,
        "component_n_token": int(token_indices.numel()),
        "component_n_atom": int(atom_indices.numel()),
        "source_token_indices": token_indices.tolist(),
        "conditioning_tensor": conditioning,
        "pairformer_context": "full_complex",
        "diffusion_scope": "component_only",
        "exact_full_complex_pairformer_diagonal": True,
        "exact_full_complex_diffusion_equivalent": False,
        "atom_local_features_rebuilt": False,
        "shared_atom_cache_rebuilt": False,
    }

    result = {
        "pred_dict": {},
        "model_state": cache["model_state"],
        "input_feature_dict": component_features,
        "s_inputs": component_s_inputs,
        "s_trunk": component_s_trunk,
        "z_trunk": component_z_trunk,
        "pair_z": component_pair_z,
        # These depend on component-local atom windows and are rebuilt by the CLI.
        "p_lm": None,
        "c_l": None,
        "N_sample": cache["N_sample"],
        "noise_schedule": cache["noise_schedule"],
        "inplace_safe": cache["inplace_safe"],
        "configs": cache["configs"],
        "enable_efficient_fusion": cache["enable_efficient_fusion"],
        "contextual_split_metadata": metadata,
    }
    validate_component_cache(result)
    return result


def validate_component_cache(cache: Mapping[str, Any], require_atom_local: bool = False) -> None:
    """Validate the component-local dimensions used by the diffusion sampler."""
    metadata = cache.get("contextual_split_metadata")
    if not isinstance(metadata, Mapping):
        raise ContextualCacheError("contextual_split_metadata is missing")
    features = cache.get("input_feature_dict")
    if not isinstance(features, Mapping):
        raise ContextualCacheError("component input_feature_dict is missing")
    asym_id = _require_tensor(features, "asym_id")
    atom_to_token = _require_tensor(features, "atom_to_token_idx")
    n_token = int(asym_id.shape[-1])
    n_atom = int(atom_to_token.shape[-1])
    if n_token != int(metadata["component_n_token"]):
        raise ContextualCacheError("component token count disagrees with metadata")
    if n_atom != int(metadata["component_n_atom"]):
        raise ContextualCacheError("component atom count disagrees with metadata")

    mapping = _consistent_atom_mapping(atom_to_token)
    if int(mapping.min()) < 0 or int(mapping.max()) >= n_token:
        raise ContextualCacheError("component atom_to_token_idx is not locally remapped")

    batch_shape = tuple(asym_id.shape[:-1])
    for key in ("s_inputs", "s_trunk"):
        _select_after_batch_prefix(
            _require_tensor(cache, key),
            torch.arange(n_token),
            batch_shape,
            n_token,
            key,
        )
    pair_name = "pair_z" if cache.get("pair_z") is not None else "z_trunk"
    pair = cache.get(pair_name)
    if not isinstance(pair, torch.Tensor):
        raise ContextualCacheError("component contains neither pair_z nor z_trunk")
    _select_pair_after_batch_prefix(
        pair, torch.arange(n_token), batch_shape, n_token, pair_name
    )
    _select_pair_after_batch_prefix(
        _require_tensor(features, "relp"),
        torch.arange(n_token),
        batch_shape,
        n_token,
        "relp",
    )
    for key in CORE_ATOM_FEATURES:
        _select_after_batch_prefix(
            _require_tensor(features, key),
            torch.arange(n_atom),
            batch_shape,
            n_atom,
            key,
        )
    if require_atom_local:
        for key in DERIVED_ATOM_FEATURES:
            if key not in features:
                raise ContextualCacheError(f"rebuilt component cache is missing {key}")
        if cache.get("pair_z") is not None:
            if not isinstance(cache.get("p_lm"), torch.Tensor) or not isinstance(
                cache.get("c_l"), torch.Tensor
            ):
                raise ContextualCacheError(
                    "pair_z cache requires rebuilt p_lm and c_l tensors"
                )
