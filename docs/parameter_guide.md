# Choosing parameters

Start from the tutorial for your workflow. The [full CLI reference](cli_reference.md)
contains the public entrypoints' help output; this page groups the options by
purpose rather than duplicating that reference. Not every option applies to every
entrypoint. Consult that entrypoint's help before constructing a command.

## Inputs and coordinate conventions

Check STAR/MRCS paths, pixel size, box size and the reference coordinate frame
before loading the model. Training uses a reference CIF placed in the experimental
frame. Template-free structure export does not remove this training requirement.
See [data requirements](data_requirements.md).

Options such as `--transR` and `--particle_sign` describe conventions of the input
data. Use the settings appropriate to your processing pipeline; do not assume
that the values in one example apply to every dataset.

## Projection frame (single-GPU and component-parallel refinement)

`src/train.py` and `src/chain_parallel/train_chain_parallel_2d.py` offer two ways to place a rendered structure in each particle
image. This choice is separate from `--gmm-kernel`, `--gmm-sdev-init-mode`,
particle pose conventions and the experimental pixel size.

| `--projection-frame` | Image placement |
|---|---|
| `fixed` (default for new runs) | Keep one 3D reference origin in the experimental coordinate frame. Project that origin through each particle pose so it lands at `--density_center`, then apply only the recorded particle translation. Do not recenter by the current image centroid or normalize each rendered image by its total intensity. |
| `legacy` | Recompute the projected atom minimum for each view, normalize the rendered image by its total intensity, then shift its image centroid to `--density_center` while applying the recorded particle translation. This preserves the historical projection behavior. |

For `fixed`, `--projection-origin X_A Y_A Z_A` specifies that 3D reference
point in Angstrom. It is **required** for a new fixed-frame run: there is no
universal numerical origin that is safe for all map coordinate conventions.
An omitted origin fails input validation before model construction. The placed
reference CIF and origin must use the same 3D experimental frame, with particle
poses and `--density_center` consistent with that geometry. Specify
`--projection-frame legacy` to reproduce the historical image placement; omit
`--projection-origin` in that mode.

### Choosing the projection origin

The MRC header supplies a **candidate** map center, not a complete answer by
itself. It does not establish that the fitted CIF uses the map's coordinates
or that the STAR poses and shifts are from the same reconstruction. Check all
three before committing to an origin.

To inspect the header with `mrcfile`, replace the path and run:

```bash
python - /path/to/reconstruction.mrc <<'PY'
import sys
import mrcfile

with mrcfile.open(sys.argv[1], permissive=True) as m:
    h = m.header
    print('size_xyz:', int(h.nx), int(h.ny), int(h.nz))
    print('voxel_A:', float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z))
    print('origin_A:', float(h.origin.x), float(h.origin.y), float(h.origin.z))
    print('starts:', int(h.nxstart), int(h.nystart), int(h.nzstart))
    print('axes:', int(h.mapc), int(h.mapr), int(h.maps))
PY
```

Then use this procedure:

1. Confirm the header's dimensions, voxel size, origin, start indices and axis
   order. Open the reconstructed map and the **placed** reference CIF together
   in a viewer; verify that the CIF is still fitted to that map. Confirm that
   the STAR poses and particle shifts belong to this reconstruction.
   `--check-inputs` checks file/metadata consistency but cannot prove their
   common coordinate frame.
2. Choose one fixed point in that common **3D map/CIF coordinate frame** whose
   projection should land at `--density_center` in each particle image. The
   image center defaults to `(boxsize/2, boxsize/2)` **pixels**; the origin uses
   **Angstrom**, not pixels. All components in a parallel run share the same
   origin. Do not substitute the current CIF centroid: that would change as the
   model changes and defeat the fixed-frame convention.
3. For a zero-origin, zero-start, standard-axis cubic map with `N=boxsize` and
   voxel size equal to `apix`, whose placed CIF uses the same map coordinates,
   the projection box center is `(N × apix / 2)` on each axis. For the checked
   288-pixel, 1.073 Å/pixel 6ZBH-style map this gives
   `--projection-origin 154.512 154.512 154.512`. Recheck these conditions for
   each dataset; the formula does not apply automatically to cropped,
   reoriented or nonzero-origin maps.
4. A map centered at physical `(0,0,0)` instead uses
   `--projection-origin 0 0 0` if the placed CIF is in that frame. The supplied
   7ZDT/7ZD5 molmap example uses a grid origin of `(-96,-96,-96)` Å and a
   physical center of `(0,0,0)` Å, so its wrapper supplies `0 0 0` explicitly.
5. Run the chosen command once with `--check-inputs`, inspect the resolved
   projection settings and compare a simulated projection against particles or
   the reconstructed map before interpreting a full refinement. Pay particular
   attention to pose convention (`--transR` when appropriate), particle shifts,
   pixel size, sign and image center. A valid preflight does not establish
   geometric alignment.

For a nonstandard map, transform the box-center grid point
`(nx/2, ny/2, nz/2)` into physical coordinates using the map's voxel-to-world
transform, including its origin, starts and axis mapping as interpreted by
the viewer that was used to fit the CIF. Read that physical point in the viewer
and verify the CIF placement there. Do not blindly add header origin and starts
without checking the map software's convention, or treat the MRC header origin,
the map center and `--density_center` as interchangeable. The header origin is
a 3D physical offset, whereas `--density_center` is a 2D pixel position in
each particle image.

In component-parallel refinement, every rank must use the same origin and
particle pose. Each rank renders its component in that frame; the component
images are summed before the recorded particle translation is applied. The
reference component CIFs and `--projection-origin` must share the experimental
map frame. Both training entrypoints use `fixed` for new runs.

The results in the initial CoCoFold2 manuscript were obtained with the legacy
implementation. Fixed-frame projection preserves a shared reference origin
and more closely follows the intended particle projection geometry. In
subsequent author-reported tests, it performed comparably or better in backbone
and Cα RMSD for most targets, with substantial improvements in several cases.

The distinction matters when coordinates change: a global translation can be
absorbed by `legacy` recentering, whereas `fixed` retains its projected in-plane
displacement. A local change can move the legacy image centroid and thereby
shift an otherwise unchanged part; the fixed reference point does not follow
that centroid. The two modes also differ in image-intensity normalization, so
their raw projections and training trajectories need not match. These
implementation differences alone do not establish which mode performs better
on experimental data.

Projection settings are saved with new checkpoints and used by the case audit.
An exact `--resume` inherits the saved scientific settings, including `legacy`
for older checkpoints; the new CLI default does not silently change them. Use
a new run with `--warm-start` if changing the projection frame from a checkpoint.

## Memory and optimization

Distinguish `--batch_size` from `--mini_batch_size` when adapting a tutorial to
available GPU memory. See the entrypoint's help for their meanings and defaults.
Use the documented learning-rate options for the relevant entrypoint rather than
editing source constants.

GMM learning is enabled by default. `--no-learn-gmm` disables learning of both
atom amplitudes and widths. Kernel and amplitude choices are described in
[Gaussian rendering kernels](gaussian_kernels.md); they are not interchangeable
performance switches.

Fresh width initialization defaults to `--gmm-sdev-init-mode legacy`. To match
a molmap Gaussian width, explicitly add `--gmm-sdev-init-mode molmap
--gmm-molmap-resolution-A 3.0`. The existing `--resolution` controls legacy
coordinate/grid scaling and is not generally a molmap resolution in Angstrom.
Saved GMM widths take precedence when loading a checkpoint. See the
[width mapping](gaussian_kernels.md#optional-physical-width-initialization).

## Randomness and alignment

The CLI reference documents `--seed`, diffusion/data seed overrides and RNG modes.
For single-GPU training, an independent data seed requires isolated RNG mode;
parallel training uses a dedicated data generator. Preserve the tutorial settings
when first reproducing an example.

For separate chain or block transforms, follow [the alignment guide](block_rigid_alignment.md).
Component grouping across GPUs and chain-wise alignment are separate choices.

## Saving and continuing a run

See [outputs and restart](outputs_and_restart.md) before changing output prefixes,
export formats or restart options. Resume inherits saved settings and uses a
cumulative epoch target; warm-start starts a new optimization run. A partial-epoch
short-test checkpoint must not be treated as a complete-epoch resume point.

Check `requested_config.json` and `resolved_config.json` to see the requested and
effective settings, especially when resuming. Reference documentation describes
available controls; it does not imply that every combination has been tested.
