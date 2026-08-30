"""Small rank-to-component manifest for the proposed 2-D GMM trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ComponentEntry:
    component_id: str
    rank: int
    diffusion_data_dir: Path
    cif_path: Path


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> tuple[ComponentEntry, ...]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("manifest must be a schema_version: 1 mapping")

    entries: list[ComponentEntry] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for item in raw.get("components", []):
        component_id = str(item["id"])
        rank = int(item["rank"])
        if not component_id or any(character in component_id for character in "/\\"):
            raise ValueError(f"invalid component id: {component_id!r}")
        if component_id in seen_ids:
            raise ValueError(f"duplicate component id: {component_id}")
        if rank in seen_ranks:
            raise ValueError(
                "this first proposal requires exactly one component per rank; "
                f"rank {rank} is repeated"
            )
        seen_ids.add(component_id)
        seen_ranks.add(rank)
        entries.append(
            ComponentEntry(
                component_id=component_id,
                rank=rank,
                diffusion_data_dir=_resolve(source.parent, item["diffusion_data_dir"]),
                cif_path=_resolve(source.parent, item["cif_path"]),
            )
        )
    if not entries:
        raise ValueError("manifest contains no components")
    return tuple(entries)


def component_for_rank(
    entries: tuple[ComponentEntry, ...], rank: int, world_size: int
) -> ComponentEntry:
    if len(entries) != world_size:
        raise ValueError(
            "this first proposal requires component count == world size, got "
            f"{len(entries)} components and {world_size} ranks"
        )
    invalid = [entry.rank for entry in entries if not 0 <= entry.rank < world_size]
    if invalid:
        raise ValueError(f"component ranks outside [0,{world_size - 1}]: {invalid}")
    local = [entry for entry in entries if entry.rank == rank]
    if len(local) != 1:
        raise ValueError(f"rank {rank} owns {len(local)} components; expected exactly one")
    return local[0]
