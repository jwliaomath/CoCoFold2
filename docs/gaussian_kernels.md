# Unified interface for three Gaussian kernels

The interface is available in `src/train.py` and
`src/chain_parallel/train_chain_parallel_2d.py`. The default `legacy` retains
the original 2D kernel and ReLU regularization. Isotropic and anisotropic
kernels support `reference_mass` and optional `peak_3d` amplitudes; `auto`
selects `reference_mass` for these kernels. Kernel parameters are shared across
all particles, without a separate GMM parameter set for each particle.

## 1. Command-line options

Keep validated data, diffusion caches, poses, CTF and batch settings. Append
one of these argument combinations to the training command; they are not
standalone commands. Both trainers use the same option names. Give different
experiments different output prefixes.

```text
--gmm-kernel legacy
--gmm-kernel isotropic
--gmm-kernel anisotropic
--gmm-kernel isotropic --gmm-amplitude peak_2d
--gmm-kernel isotropic --gmm-amplitude peak_3d
--gmm-kernel anisotropic --gmm-amplitude peak_3d
```

| Option | Default | Meaning |
|---|---|---|
| `--gmm-kernel` | `legacy` | Select the kernel |
| `--gmm-amplitude` | `auto` | `peak_2d` for legacy, `reference_mass` otherwise |
| `--gmm-sigma-floor` | `0.0001` | Positive width floor for covariance kernels; does not change legacy |
| `--gmm-atom-chunk-size` | `1000` | Atoms per covariance rasterization chunk |
| `--gmm-checkpoint-chunks` | Enabled | Recompute covariance raster chunks during backward |
| `--no-gmm-checkpoint-chunks` | — | Disable that recomputation for time/memory comparisons |
| `--gmm-checkpoint-peak2d` | Disabled | Experimental atom-chunk recomputation for legacy/isotropic `peak_2d` |
| `--no-gmm-checkpoint-peak2d` | — | Retain the original `peak_2d` path without recomputation |

After resolving `auto`, legacy accepts only `peak_2d`; anisotropic accepts
`reference_mass` and `peak_3d`; isotropic accepts all three. Invalid combinations
fail. Covariance chunk options apply to both mass and 3D-peak paths. The separate
peak-2D switch defaults to off; both old-kernel paths retain 1000-atom chunks and
the original rectangular crop policy.

### Optional physical-width initialization

Width initialization is independent of `--gmm-kernel`. Existing commands default
to `--gmm-sdev-init-mode legacy`, retaining the original floating-point expression
`3 / (pi * sqrt(2))` for the initial internal width. Historical experiments retain
this initialization. To change only the fresh width, append:

```text
--gmm-sdev-init-mode molmap --gmm-molmap-resolution-A 3.0
```

With pixel size `a = apix` and legacy scale `R_C = resolution`, one internal-grid
unit corresponds to `h = a * R_C / 3` Angstrom. The new mode initializes
`s = 3 * R_M / (pi * sqrt(2) * a * R_C)`, giving physical standard deviation
`sigma = R_M / (pi * sqrt(2))` Angstrom. Here `R_M` is the requested molmap
resolution. All three inputs must be finite and positive, and the internal width
must exceed the existing `--gmm-sigma-floor` constructor bound.

`resolution` is a legacy CoCoFold renderer parameter that participates in
coordinate/grid scaling and the historical fresh-GMM parameterization. It is
not generally a ChimeraX molmap resolution in Angstrom. Fresh legacy widths
correspond to `R_M = apix * resolution`; for example, `apix=3, resolution=3`
gives a molmap-equivalent width resolution of 9 Angstrom. With the new mode and
`R_M=3`, the internal width is approximately 0.225079 rather than 0.675237.
Neither the coordinate transform nor `resolution` is changed automatically.

This matches **Gaussian physical width only**, not the complete ChimeraX
renderer. Atom-weight initialization, normalization conventions, cutoff,
regularization, and subsequent width learning remain unchanged. In particular,
`reference_mass` retains its historical amplitude baseline `sigma_init`, separate
from the new initial shape width. Under `peak_3d`, integrated mass still depends
on covariance through the existing formula. Narrower widths can interact with
the unchanged width penalty; no penalty thresholds are adjusted automatically.
Use `--no-learn-gmm` to freeze both widths and amplitudes, as before.

Fresh-width provenance is recorded in the resolved run configuration and the
GMM config's `width_initialization` metadata (mode, pixel size, legacy scale,
internal spacing, initial width, physical sigma and equivalent resolution).
These are initialization values, not current learned widths. The unused
`gmm_physical_sigma_A` field is null; a direct-sigma mode is not implemented.
Legacy mode ignores the optional molmap target.

Existing checkpoint tensors take precedence on resume/warm-start. Historical
GMM payloads without width metadata remain loadable; their original physical
initialization is not inferred. Tensor keys, shapes and format version are
unchanged. New metadata requires this version of the loader: loading new
checkpoints in older software is not guaranteed.

## 2. Learned quantities and regularization

| Kernel | Learned quantities per atom | Regularized quantities | Projection |
|---|---|---|---|
| legacy | One amplitude and two directly optimized 2D widths | Original widths and amplitude | Original `pdb2img`; widths remain on image axes |
| isotropic | One amplitude and one raw width | Actual `softplus(raw)+floor` width and amplitude | Spherical 3D kernel or optional old 2D peak path |
| anisotropic | One amplitude and six raw Cholesky parameters | Three principal-axis standard deviations and amplitude | Rotate 3D covariance, then project the full 2D ellipse |

Isotropic kernels still permit atom-specific sigma values; anisotropic kernels
add directional broadening.

### Preserved legacy behavior

The original renderer's executable expressions remain unchanged by comment and
docstring cleanup. Legacy calls the original `pdb2img`, bypassing covariance
rasterization. Initialization, both GMM optimizer groups and learning rates,
AdamW defaults and ReLU expressions are retained, including the single-GPU
placement of CPU amplitude parameters and GPU widths. The interval penalty is:

```python
mean(relu(sdevs.float() - 0.8) + relu(0.1 - sdevs.float())) \
  + mean(relu(atom_weights.float() - 20) + relu(1 - atom_weights.float()))
```

Legacy does not gain softplus, determinant factors or new normalization. The
original `translation_2d` normalization, centroid shift and interpolation remain,
including existing coordinate conventions, cropped tails and numerical behavior.

### Positive parameterization

Isotropic uses `sigma = softplus(raw_sigma) + sigma_floor`. Anisotropic constructs
a lower-triangular L with three softplus-transformed diagonal entries and three
directly learned off-diagonal entries:

```python
Sigma = L @ L.T + sigma_floor**2 * I
```

Initialization is `Sigma = sigma_init**2 * I`, with
`sigma_init = 3/(pi*sqrt(2))`, matching the old initial width. L's initial diagonal
accounts for the floor's variance contribution. Both kernels retain ReLU interval
penalties. Anisotropic regularization uses `sqrt(eigvalsh(Sigma))`, not the six raw
parameters. Positive parameterization changes the optimization scale: retaining
the shape learning rate `5e-3` does not imply the same actual-width step as legacy.

## 3. Anisotropic projection and units

Sigma is defined in the molecular reference frame **after coordinate alignment**.
For each particle, U contains the first two rows of its pose matrix:

```python
U = rotation[:, :2, :]
C = U[:, None] @ Sigma[None] @ U[:, None].transpose(-1, -2)
```

C has shape `[B, N_atom, 2, 2]` and retains the cross term Cxy. A kernel elongated
along reference x becomes elongated along image y after a 90-degree rotation
about the viewing axis: covariance rotates along with atom centers.
For `C = [[a,c],[c,b]]`, an atom contributes:

```python
det = a*b - c*c
d2 = (b*dx*dx - 2*c*dx*dy + a*dy*dy) / det
image = q / (2*pi*sqrt(det)) * exp(-0.5*d2)
```

This is the analytic line integral of a 3D Gaussian, without a voxel volume or
per-pixel matrix inverses. Synthetic tests compare it with numerical line
integration under general rotations.

Widths use the existing renderer's internal grid units. Coordinate conversion
includes `atoms/apix` and `step=resolution/3`. Sigma is already in squared grid
units, so projection does not divide it again by `(apix*step)**2`. Conversion
to physical covariance is `Sigma_A2 = Sigma*(apix*step)**2`.

Sigma is neither in the unaligned diffusion frame nor a moving domain-local
frame. Covariances following local structural motion are not implemented here.

## 4. Amplitude conventions and comparisons

`peak_2d` retains the old effective 2D peak w, whose continuous integral varies
with width. Isotropic `peak_2d` ties both widths to one positive width while
retaining the original rasterizer.

`reference_mass` still learns `atom_weights = w` with the original initialization
scale and `[1,20]` ReLU interval, then converts it to:

```python
q = 2*pi*sigma_init**2 * w
```

sigma_init is fixed, not the current sigma. Thus changing width at fixed w
preserves the continuous integral; the projected peak depends on det(C).
At spherical initialization, peaks match legacy up to floating-point and crop
differences. q is the integral **before** shared renderer scaling and image
normalization. Finite support, pixel sampling and cutoff approximate it.
Subsequent `translation_2d` normalization means the final image sum is not an
absolute scattering mass.

### Three-dimensional peak amplitudes

With `peak_3d`, w is the 3D Gaussian peak before shared renderer scaling:

```python
rho(x) = w * exp(-0.5 * (x-mu).T @ inv(Sigma) @ (x-mu))
q = (2*pi)**1.5 * w * sqrt(det(Sigma))
peak_2d_for_this_pose = q / (2*pi*sqrt(det(C)))
```

Only q changes; covariance projection, 2D rasterization and postprocessing are
shared with `reference_mass`. No 3D volume function is called. The determinant
factor uses Cholesky diagonals of the **actual** Sigma, including the floor,
not the raw L. Its gradients remain attached so width changes affect the integral.

At fixed w, multiplying all principal widths by k preserves the 3D peak and
multiplies the integral by k³; `reference_mass` preserves the integral instead.
Isotropic `peak_3d` also differs from `peak_2d`: its projected peak is
`sqrt(2*pi)*w*sigma` and varies with sigma.

The original atom-weight and spherical-width initialization is retained.
Initially, raw `peak_3d` images differ from `reference_mass` by the global factor
`sqrt(2*pi)*sigma_init`, which final image normalization removes. Tests compare
normalized initial images, but width gradients and later learning may differ;
the modes are not equivalent. ReLU intervals remain `[1,20]` for amplitudes and
`[0.1,0.8]` for actual widths. An atom's 3D peak is shared across poses while
its projected peak varies with pose; no per-particle kernel parameters are added.

Controlled comparisons include:

1. Legacy versus isotropic `peak_2d`: retain the peak convention while restricting
   directional widths, accounting for the new positive parameterization.
2. Default isotropic versus default anisotropic: use the same covariance renderer,
   mass convention and spherical initialization. Synthetic spherical-limit images
   and coordinate/amplitude gradients agree within floating-point error.
3. Mass versus 3D-peak amplitudes within one kernel: retain renderer and cutoff
   while changing the amplitude convention.

## 5. Python and component-parallel interfaces

`src/gmm.py` manages parameters, penalties, projection and state loading:

```python
from gmm import GaussianProjector

gmm = GaussianProjector(initial_atom_weights, kernel="anisotropic",
                        shape_device=device)
optimizer.add_param_group({"params": gmm.amplitude_parameters(), "lr": 1e-2})
optimizer.add_param_group({"params": gmm.shape_parameters(), "lr": 5e-3})
projection = gmm(
    atoms_coord=aligned_coordinates,  # [N,3], [1,N,3] or [B,N,3]
    rotation=particle_rotations,       # [B,2,3] or [B,3,3]
    trans=particle_translations,
    resolution=resolution,
    density_center=density_center,
    box_size=box_size,
    apix=apix,
)
loss = particle_loss(projection) + gmm.regularization()
```

The renderer supports `[B,N,3]`, including one structure per particle, with
synthetic projection/coordinate-gradient checks. Public refinement still uses
one structure for multiple images; this does not add an image encoder or
latent-to-pair network.

Distributed projection shares the coordinate origin, calls component `render_raw`,
differentiably sums raw densities, then calls `finalize` once for normalization
and centroid shifting. Normalizing components before summing changes the objective.
Local ReLU penalties use global atom and actual-width counts: 2, 1 and 3 widths
per atom for legacy, isotropic and anisotropic, rather than six raw parameters
for anisotropic. Existing `loss_frc/world_size + local_penalty` scaling is retained
and has synthetic two-process Gloo gradient checks.

## 6. Computation and memory

Each atom stores a 3×3 covariance; projection produces 2×2 per view. Most added
work is ellipse rasterization and backward, not voxel generation. Atoms are
chunked; pixels are evaluated within each chunk's joint support bounding box.
The CLI default is 1000 atoms per chunk; direct Python construction retains 64.
Covariance activation checkpointing defaults to enabled and recomputes pixel
intermediates during backward rather than retaining all `[B,N,H,W]` activations.
Means, covariances, inputs and output images still occupy memory.

This pure PyTorch implementation has Python loops and GPU synchronization for
crop bounds, while checkpointing adds recomputation. No fixed speedup is claimed.
Chunk sizes such as 16/32/64/128 can be measured on the target system; they do
not change the covariance kernels' per-atom cutoff definition.

## 7. Saving and loading

Checkpoints include a `gmm` payload with version, kernel, amplitude convention,
units, reference frame and parameters, plus `gmm_kernel`. Legacy also retains
`atom_weights` and `sdevs`. Covariance kernels set the old 2D `sdevs` to None.
`peak_3d` uses format version 1 and records its convention; w and raw shape are
saved and q is reconstructed. Older mass/2D-peak payloads remain readable.

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
gmm = GaussianProjector.from_checkpoint(checkpoint, device="cuda:0")
# A payload from gmm.export_checkpoint() can also be passed directly.
# Legacy reconstruction requires atom_weights and [N,2] sdevs.
```

Load trusted checkpoints only. This method restores **only GMM**. Complete-epoch
training resume and warm-start are separate interfaces described in
[outputs and restart](outputs_and_restart.md). Learned 2D ellipses are not
automatically converted to 3D covariance. External code reading `sdevs` must
use the GMM loader for covariance kernels. Old `pdb2mrc`/volume exporters do
not automatically understand full covariance payloads; coordinate export
remains available.

## 8. Verification scope

Current public CPU checks, from the repository root:

```bash
python tests/run_public_tests.py --groups gmm --output results/new_gmm_check
```

Detailed development checks are also available:

```bash
python -m unittest discover -s tests -p 'test_gmm*.py' -v
python tests/check_gmm_distributed.py
```

Historical 2026-09-09 validation recorded 13 original and nine added `peak_3d`
checks (22 total) on PyTorch 2.1.2+cu121 and an RTX 4060 Laptop GPU (8 GB).
This is historical kernel evidence, not the size of the current public suite.
It covered executable renderer equivalence; legacy images, penalties, gradients
and one AdamW update; numerical 3D integration; finite differences; spherical
limits; chunk recomputation; checkpoint round trips; CLI parsing; and CUDA
forward/backward. CUDA checks skip when unavailable; the public CPU gate
explicitly lists its CUDA exclusions.

The four pre-existing configurations were compared before/after adding 3D-peak
amplitudes. Extra checks covered general rotations, viewing-direction width
gradients, the floor-aware determinant and loading updated parameters. Unequal
component sizes tested Gloo density sums, global penalties and gradient scales.
Legacy matched the old distributed implementation. Covariance kernels matched
whole-structure calculations within floating-point error; legacy partitioning
can show small tail differences from its original chunk crops.

Historical isotropic/`peak_3d` and anisotropic/`peak_3d` Gloo checks reported
maximum image differences of approximately `2.57e-16` and `2.36e-16`, with
double-precision gradient agreement. These used synthetic coordinates and
Fourier/CTF targets, not complete Protenix, real-particle refinement or NCCL.
They do not establish better experimental structures or full-training performance.

The local test Fourier objective uses `real**2 + imag**2`, equivalent to squared
complex magnitude, to avoid a Windows CUDA complex-abs UnicodeDecodeError.
Production FRC/CTF code was not changed for that workaround. Gloo uses project-local
temporary paths to avoid a historical non-English path issue. See
[release validation](release_validation.md) for separate real-model evidence.
