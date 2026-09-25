"""Distributed factoring of CoCoFold2's 2-D GMM renderer.

This proposal intentionally reuses the renderer primitives in ``src/pts2img.py``.
It does not construct a 3-D density.  The only distributed operations are:

1. for legacy projection, an autograd-aware all-gather of each component's
   projected XY minimum, matching one concatenated atom tensor;
2. an autograd-aware SUM of the unshifted component GMM images;
3. one final translation of the assembled image, using the selected frame.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn

from pts2img import centers_rotation, sum_of_gaussians_2d_torch, translation_2d, translation_2d_fixed_frame


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def differentiable_sum(value: torch.Tensor) -> torch.Tensor:
    if not distributed_active():
        return value
    return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM)


def differentiable_global_min(value: torch.Tensor) -> torch.Tensor:
    """Return an elementwise cross-rank minimum with all-gather autograd."""

    if not distributed_active():
        return value
    gathered = dist_nn.all_gather(value)
    return torch.stack(tuple(gathered), dim=0).amin(dim=0)


def detached_sum(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if distributed_active():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def projected_xy(
    atom_coordinates: torch.Tensor,
    rotations: torch.Tensor,
    apix: float,
) -> torch.Tensor:
    """Apply the exact coordinate scaling and rotation used by ``pdb2img``."""

    if atom_coordinates.ndim != 2 or atom_coordinates.shape[-1] != 3:
        raise ValueError("atom_coordinates must have shape [N_atom,3]")
    if rotations.ndim != 3 or rotations.shape[-2:] != (2, 3):
        raise ValueError("rotations must have shape [B,2,3]")
    atoms = atom_coordinates.reshape(1, -1, 3) / float(apix)
    return centers_rotation(atoms, rotations)


def render_component_untranslated(
    projected_coordinates: torch.Tensor,
    global_origin: torch.Tensor,
    atom_weights: torch.Tensor,
    sdevs: torch.Tensor,
    resolution: float,
    box_size: int,
    cutoff_range: float = 5.0,
    sigma_factor: float = 1.0 / (math.pi * math.sqrt(2.0)),
) -> torch.Tensor:
    """Render one component using the shared origin from the full assembly."""

    if projected_coordinates.ndim != 3 or projected_coordinates.shape[-1] != 2:
        raise ValueError("projected_coordinates must have shape [B,N_atom,2]")
    atom_count = projected_coordinates.shape[1]
    if atom_weights.shape != (atom_count,):
        raise ValueError("atom_weights must have shape [N_atom]")
    if sdevs.shape != (atom_count, 2):
        raise ValueError("sdevs must have shape [N_atom,2]")
    if global_origin.shape != (projected_coordinates.shape[0], 1, 2):
        raise ValueError("global_origin must have shape [B,1,2]")

    pad = 3.0 * float(resolution)
    step = float(resolution) / 3.0
    scalar_sdev = float(resolution) * float(sigma_factor)
    centers = (projected_coordinates - global_origin) / step + pad
    batch_size = centers.shape[0]
    image = sum_of_gaussians_2d_torch(
        centers=centers,
        coef=atom_weights.to(device=centers.device),
        sdev=sdevs.to(device=centers.device),
        maxrange=cutoff_range,
        matrices=torch.zeros(
            batch_size,
            box_size,
            box_size,
            device=centers.device,
            dtype=centers.dtype,
        ),
    )
    normalization = (2.0 * math.pi) ** -1 * scalar_sdev**-2
    image = image * normalization
    image = image / step
    return image.unsqueeze(1)


def finish_assembled_projection(
    assembled_untranslated: torch.Tensor,
    translations: torch.Tensor,
    resolution: float,
    box_size: int,
    apix: float,
    density_center: torch.Tensor,
) -> torch.Tensor:
    """Call CoCoFold2's original centering/translation exactly once."""

    step = float(resolution) / 3.0
    return translation_2d(
        assembled_untranslated,
        translations.clone() / step,
        box_size,
        float(apix),
        density_center.to(assembled_untranslated.device),
    )


def distributed_pdb2img(
    atom_coordinates: torch.Tensor,
    atom_weights: torch.Tensor,
    sdevs: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    density_center: torch.Tensor,
    resolution: float,
    box_size: int,
    apix: float,
    cutoff_range: float = 5.0,
    sigma_factor: float = 1.0 / (math.pi * math.sqrt(2.0)),
) -> torch.Tensor:
    """Distributed equivalent of one ``pdb2img`` call on concatenated atoms."""

    local_projected = projected_xy(atom_coordinates, rotations, apix)
    local_origin = local_projected.amin(dim=1, keepdim=True)
    global_origin = differentiable_global_min(local_origin)
    local_image = render_component_untranslated(
        projected_coordinates=local_projected,
        global_origin=global_origin,
        atom_weights=atom_weights,
        sdevs=sdevs,
        resolution=resolution,
        box_size=box_size,
        cutoff_range=cutoff_range,
        sigma_factor=sigma_factor,
    )
    assembled = differentiable_sum(local_image)
    return finish_assembled_projection(
        assembled_untranslated=assembled,
        translations=translations,
        resolution=resolution,
        box_size=box_size,
        apix=apix,
        density_center=density_center,
    )


def local_source_equivalent_penalty(
    atom_weights: torch.Tensor,
    sdevs: torch.Tensor,
    global_atom_count: int,
    global_sdev_count: int,
    limits: tuple[float, float, float, float] = (0.1, 0.8, 1.0, 20.0),
) -> torch.Tensor:
    """Local contribution to the original full-complex GMM penalty.

    Summing this value over ranks equals the two means computed in ``train.py``.
    This weighting remains correct when components have different atom counts.
    """

    sdev_min, sdev_max, weight_min, weight_max = limits
    sdev_violation = torch.relu(sdevs.float() - sdev_max) + torch.relu(
        sdev_min - sdevs.float()
    )
    weight_violation = torch.relu(atom_weights.float() - weight_max) + torch.relu(
        weight_min - atom_weights.float()
    )
    return (
        sdev_violation.sum() / float(global_sdev_count)
        + weight_violation.sum() / float(global_atom_count)
    )


def distributed_project_gaussians(
    projector, atom_coordinates, rotations, translations, density_center,
    resolution, box_size, apix, cutoff_range=5.0,
    sigma_factor=1.0 / (math.pi * math.sqrt(2.0)),
    projection_frame="legacy", projection_origin=(0., 0., 0.),
):
    """Unified kernel API with source-equivalent component aggregation.

    Do not call projector.forward on each component: frame-dependent image
    processing must happen only AFTER summing the component densities.
    """
    if projection_frame not in ("legacy", "fixed"):
        raise ValueError('projection_frame must be legacy or fixed')
    projected = projector.project_coordinates(atom_coordinates, rotations, apix)
    if projection_frame == "fixed":
        # Match GaussianProjector.forward exactly. Every rank uses the same
        # map-frame pivot; no component-dependent minimum or recentering occurs.
        batch = projected.shape[0]
        center = torch.as_tensor(density_center, device=projected.device,
                                 dtype=projected.dtype).reshape(-1, 2)
        if center.shape[0] not in (1, batch) or not torch.isfinite(center).all():
            raise ValueError('density_center must be finite [2] or [B,2]')
        origin = torch.as_tensor(projection_origin, device=projected.device,
                                 dtype=projected.dtype)
        if origin.shape != (3,) or not torch.isfinite(origin).all():
            raise ValueError('projection_origin must be three finite Angstrom coordinates')
        pivot = centers_rotation((origin / apix).reshape(1, 1, 3), rotations)
        step = float(resolution) / 3.0
        raw_origin = pivot - (center[:, None, :] - 3.0 * float(resolution)) * step
        local_image = projector.render_raw(
            projected, rotations, raw_origin, resolution, box_size, cutoff_range, sigma_factor,
        )
        assembled = differentiable_sum(local_image)
        return translation_2d_fixed_frame(
            assembled, translations.to(assembled.device).clone() / step,
        )
    origin = differentiable_global_min(projected.amin(dim=1, keepdim=True))
    local_image = projector.render_raw(
        projected, rotations, origin, resolution, box_size, cutoff_range, sigma_factor,
    )
    assembled = differentiable_sum(local_image)
    return projector.finalize(assembled, translations, resolution, box_size,
                              apix, density_center)
