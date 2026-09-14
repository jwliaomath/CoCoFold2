"""Shared, trainable Gaussian kernels for single-state and component training.

legacy: original independent 2-D widths, peak amplitudes, renderer and ReLU loss.
isotropic/anisotropic: positive reference-frame covariance and analytic projection.
New kernels default to reference_mass: q = 2*pi*sigma_init**2 * atom_weights.
The latter parameter retains the old amplitude's numerical initialization scale.
All widths are internal-grid lengths, after the coordinate apix/step transform.
"""
from __future__ import annotations

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F

from pts2img import (
    centers_rotation, pdb2img, project_gaussian_covariances,
    sum_of_gaussians_2d_covariance, sum_of_gaussians_2d_torch, translation_2d,
)


class GaussianProjector(nn.Module):
    MODES = ("legacy", "isotropic", "anisotropic")
    VERSION = 1

    def __init__(self, initial_weights, kernel="legacy", sigma_init=None,
                 sigma_floor=1e-4, amplitude_convention="auto",
                 atom_chunk_size=64, checkpoint_chunks=True, shape_device=None):
        super().__init__()
        if kernel not in self.MODES:
            raise ValueError(f"unknown Gaussian kernel: {kernel}")
        if initial_weights.ndim != 1 or initial_weights.numel() == 0:
            raise ValueError("initial_weights must have nonempty shape [N_atom]")
        if not initial_weights.is_floating_point():
            raise ValueError("initial_weights must be floating point")
        sigma_init = 3 / (math.pi * math.sqrt(2)) if sigma_init is None else float(sigma_init)
        if not (0 < sigma_floor < sigma_init and math.isfinite(sigma_init)):
            raise ValueError("require 0 < sigma_floor < finite sigma_init")
        if atom_chunk_size <= 0:
            raise ValueError("atom_chunk_size must be positive")
        if amplitude_convention == "auto":
            amplitude_convention = "peak_2d" if kernel == "legacy" else "reference_mass"
        allowed = (("peak_2d",) if kernel == "legacy" else
                   ("reference_mass",) if kernel == "anisotropic" else
                   ("peak_2d", "reference_mass"))
        if amplitude_convention not in allowed:
            raise ValueError(f"{kernel} supports amplitude conventions {allowed}")
        self.kernel = kernel
        self.sigma_init = sigma_init
        self.sigma_floor = float(sigma_floor)
        self.amplitude_convention = amplitude_convention
        self.atom_chunk_size = int(atom_chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        # Preserve the caller's amplitude device, including the original main
        # trainer's CPU weights + GPU widths. Do not silently change legacy SGD.
        self.atom_weights = nn.Parameter(initial_weights.detach().clone())
        shape_device = initial_weights.device if shape_device is None else shape_device
        dtype = initial_weights.dtype
        count = initial_weights.numel()
        if kernel == "legacy":
            self.sdevs = nn.Parameter(torch.full((count, 2), sigma_init,
                                                device=shape_device, dtype=dtype))
        elif kernel == "isotropic":
            raw = math.log(math.expm1(sigma_init - self.sigma_floor))
            self.raw_sigma = nn.Parameter(torch.full((count, 1), raw,
                                                    device=shape_device, dtype=dtype))
        else:
            diagonal = math.sqrt(sigma_init**2 - self.sigma_floor**2)
            raw = torch.zeros(count, 6, device=shape_device, dtype=dtype)
            raw[:, :3] = math.log(math.expm1(diagonal))
            self.raw_cholesky = nn.Parameter(raw)

    @property
    def n_atom(self):
        return self.atom_weights.numel()

    def amplitude_parameters(self):
        return [self.atom_weights]

    def shape_parameters(self):
        return ([self.sdevs] if self.kernel == "legacy" else
                [self.raw_sigma] if self.kernel == "isotropic" else [self.raw_cholesky])

    def widths(self):
        """Physical widths used by the ReLU penalty, not raw optimizer entries."""
        if self.kernel == "legacy":
            return self.sdevs
        if self.kernel == "isotropic":
            return F.softplus(self.raw_sigma) + self.sigma_floor
        return torch.linalg.eigvalsh(self.covariance()).clamp_min(0).sqrt()

    def covariance(self):
        """Covariance in aligned reference-frame internal-grid units squared."""
        if self.kernel == "legacy":
            raise ValueError("legacy screen-fixed widths do not define a 3-D covariance")
        if self.kernel == "isotropic":
            widths = self.widths()
            return torch.diag_embed(widths.square().expand(-1, 3))
        raw = self.raw_cholesky
        if raw.dtype in (torch.float16, torch.bfloat16):
            raw = raw.float()
        d = F.softplus(raw[:, :3])
        zero = torch.zeros_like(d[:, 0])
        lower = torch.stack((d[:, 0], zero, zero,
                             raw[:, 3], d[:, 1], zero,
                             raw[:, 4], raw[:, 5], d[:, 2]), dim=-1).reshape(-1, 3, 3)
        eye = torch.eye(3, dtype=raw.dtype, device=raw.device)
        return lower @ lower.transpose(-1, -2) + self.sigma_floor**2 * eye

    def regularization_counts(self):
        per_atom = {"legacy": 2, "isotropic": 1, "anisotropic": 3}[self.kernel]
        return self.n_atom, per_atom * self.n_atom

    def regularization(self, limits=(0.1, 0.8, 1.0, 20.0),
                       global_atom_count=None, global_width_count=None):
        widths = self.widths().float()
        weights = self.atom_weights.float()
        lo, hi, wlo, whi = limits
        shape_violation = torch.relu(widths - hi) + torch.relu(lo - widths)
        amplitude_violation = torch.relu(weights - whi) + torch.relu(wlo - weights)
        if global_atom_count is None and global_width_count is None:
            # Exact expression/order of the original main-trainer penalty.
            return shape_violation.mean() + amplitude_violation.mean()
        if global_atom_count is None or global_width_count is None:
            raise ValueError("provide both global regularization counts")
        if global_atom_count <= 0 or global_width_count <= 0:
            raise ValueError("global regularization counts must be positive")
        # Match the old chain-parallel local contribution, NOT a mean per rank.
        return (shape_violation.sum() / float(global_width_count)
                + amplitude_violation.sum() / float(global_atom_count))

    def project_coordinates(self, atoms_coord, rotation, apix):
        if atoms_coord.ndim == 2:
            atoms_coord = atoms_coord[None]
        if atoms_coord.ndim != 3 or atoms_coord.shape[-2:] != (self.n_atom, 3):
            raise ValueError("coordinates must have shape [N,3], [1,N,3] or [B,N,3]")
        if rotation.ndim != 3 or rotation.shape[-2:] not in ((2, 3), (3, 3)):
            raise ValueError("rotation must have shape [B,2,3] or [B,3,3]")
        if atoms_coord.shape[0] not in (1, rotation.shape[0]):
            raise ValueError("coordinate batch must be one or equal to pose batch")
        if apix <= 0:
            raise ValueError("apix must be positive")
        return centers_rotation(atoms_coord / apix, rotation)

    def render_raw(self, projected_coordinates, rotation, global_origin, resolution,
                   box_size, cutoff_range=5, sigma_factor=1 / (math.pi * math.sqrt(2))):
        """Uncentered local contribution; all components must share global_origin.

        projected_coordinates is R*X/apix, before division by step. The return
        includes the old common scale but NOT per-image normalization/centering.
        """
        if resolution <= 0 or box_size <= 0:
            raise ValueError("resolution and box_size must be positive")
        if projected_coordinates.shape[1:] != (self.n_atom, 2):
            raise ValueError("projected_coordinates must have shape [B,N,2]")
        if global_origin.shape != (projected_coordinates.shape[0], 1, 2):
            raise ValueError("global_origin must have shape [B,1,2]")
        step = float(resolution) / 3
        centers = (projected_coordinates - global_origin) / step + 3 * float(resolution)
        weights = self.atom_weights.to(centers.device)
        if self.amplitude_convention == "peak_2d":
            widths = self.widths()
            if self.kernel == "isotropic":
                widths = widths.expand(-1, 2)
            image = sum_of_gaussians_2d_torch(
                centers=centers, coef=weights, sdev=widths.to(centers.device),
                maxrange=cutoff_range,
                matrices=torch.zeros(centers.shape[0], box_size, box_size,
                                     dtype=centers.dtype, device=centers.device),
            )
        else:
            cov2 = project_gaussian_covariances(self.covariance(), rotation)
            mass = (2 * math.pi * self.sigma_init**2) * weights
            image = sum_of_gaussians_2d_covariance(
                centers, mass, cov2, box_size, self.atom_chunk_size,
                cutoff_range, self.checkpoint_chunks,
            )
        # Preserve the source chain-parallel arithmetic order and common scale.
        scalar_sdev = float(resolution) * float(sigma_factor)
        image = image * ((2 * math.pi)**-1 * scalar_sdev**-2)
        return (image / step).unsqueeze(1)

    @staticmethod
    def finalize(image, trans, resolution, box_size, apix, density_center):
        return translation_2d(image, trans.clone() / (float(resolution) / 3),
                              box_size, apix, density_center.to(image.device))

    def forward(self, atoms_coord, rotation, trans, resolution, density_center,
                box_size=256, apix=1, cutoff_range=5,
                sigma_factor=1 / (math.pi * math.sqrt(2))):
        if atoms_coord.ndim == 2:
            atoms_coord = atoms_coord[None]
        if self.amplitude_convention == "peak_2d":
            widths = self.widths()
            if self.kernel == "isotropic":
                widths = widths.expand(-1, 2)
            # Direct call: preserve legacy operations, defaults, and normalization.
            return pdb2img(atoms_coord, resolution, self.atom_weights, rotation,
                           trans, density_center, box_size=box_size,
                           cutoff_range=cutoff_range, sigma_factor=sigma_factor,
                           apix=apix, sdevs=widths)
        projected = self.project_coordinates(atoms_coord, rotation, apix)
        origin = projected.amin(dim=1, keepdim=True)
        raw = self.render_raw(projected, rotation, origin, resolution, box_size,
                              cutoff_range, sigma_factor)
        return self.finalize(raw, trans, resolution, box_size, apix, density_center)

    def config(self):
        return {
            "kernel": self.kernel, "sigma_init": self.sigma_init,
            "sigma_floor": self.sigma_floor,
            "amplitude_convention": self.amplitude_convention,
            "atom_chunk_size": self.atom_chunk_size,
            "checkpoint_chunks": self.checkpoint_chunks,
            "units": "internal_render_grid",
            "frame": "aligned_reference",
        }

    def export_checkpoint(self):
        return {"format_version": self.VERSION, "config": self.config(),
                "state_dict": {k: v.detach().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        """Load a GMM payload, a refinement checkpoint, or old explicit GMM fields.

        Restores Gaussian state only, not a complete diffusion/optimizer resume.
        Old 2-D tensors are loaded only as legacy; there is no silent conversion.
        """
        if "gmm" in checkpoint:
            checkpoint = checkpoint["gmm"]
        if "format_version" not in checkpoint:
            if checkpoint.get("gmm_kernel", "legacy") != "legacy":
                raise ValueError("non-legacy checkpoint requires its GMM payload")
            weights, widths = checkpoint["atom_weights"], checkpoint["sdevs"]
            if widths is None or widths.shape != (weights.numel(), 2):
                raise ValueError("old GMM checkpoint requires [N,2] sdevs")
            weights = weights.to(device) if device is not None else weights
            widths = widths.to(device) if device is not None else widths
            model = cls(weights, shape_device=widths.device)
            with torch.no_grad():
                model.sdevs.copy_(widths)
            return model
        if checkpoint["format_version"] != cls.VERSION:
            raise ValueError("unsupported GMM checkpoint version")
        config = dict(checkpoint["config"])
        if config.pop("units") != "internal_render_grid" or config.pop("frame") != "aligned_reference":
            raise ValueError("unsupported GMM coordinate convention")
        state = checkpoint["state_dict"]
        weights = state["atom_weights"]
        if device is not None:
            weights = weights.to(device)
        shape_name = {"legacy": "sdevs", "isotropic": "raw_sigma",
                      "anisotropic": "raw_cholesky"}[config["kernel"]]
        shape_device = device if device is not None else state[shape_name].device
        model = cls(weights, shape_device=shape_device, **config)
        model.load_state_dict(state, strict=True)
        return model


def add_gmm_arguments(parser):
    parser.add_argument("--gmm-kernel", choices=GaussianProjector.MODES, default="legacy")
    parser.add_argument("--gmm-amplitude", choices=("auto", "peak_2d", "reference_mass"), default="auto")
    parser.add_argument("--gmm-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--gmm-atom-chunk-size", type=int, default=64,
                        help="Atom chunk size for new covariance kernels only")
    parser.add_argument("--gmm-checkpoint-chunks", action=argparse.BooleanOptionalAction,
                        default=True, help="Recompute new raster blocks during backward")


def gmm_from_arguments(weights, args, shape_device=None):
    return GaussianProjector(
        weights, kernel=args.gmm_kernel, sigma_floor=args.gmm_sigma_floor,
        amplitude_convention=args.gmm_amplitude,
        atom_chunk_size=args.gmm_atom_chunk_size,
        checkpoint_chunks=args.gmm_checkpoint_chunks, shape_device=shape_device,
    )
