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

## Projection frame (single-GPU refinement)

`src/train.py` offers two ways to place a rendered structure in each particle
image. This choice is separate from `--gmm-kernel`, `--gmm-sdev-init-mode`,
particle pose conventions and the experimental pixel size.

| `--projection-frame` | Image placement |
|---|---|
| `legacy` (default) | Recompute the projected atom minimum for each view, normalize the rendered image by its total intensity, then shift its image centroid to `--density_center` while applying the recorded particle translation. This preserves the historical projection behavior. |
| `fixed` | Keep one 3D reference origin in the experimental coordinate frame. Project that origin through each particle pose so it lands at `--density_center`, then apply only the recorded particle translation. Do not recenter by the current image centroid or normalize each rendered image by its total intensity. |

For `fixed`, `--projection-origin X_A Y_A Z_A` specifies that 3D reference
point in Angstrom; its default is `0 0 0`. The reference CIF and origin must be
in the same 3D experimental frame, with particle poses and `--density_center`
consistent with that geometry. A nonzero origin requires
`--projection-frame fixed`. For example, append
`--projection-frame fixed --projection-origin 0 0 0` to a single-GPU training
command to select the fixed frame with the default origin.

The distinction matters when coordinates change: a global translation can be
absorbed by `legacy` recentering, whereas `fixed` retains its projected in-plane
displacement. A local change can move the legacy image centroid and thereby
shift an otherwise unchanged part; the fixed reference point does not follow
that centroid. The two modes also differ in image-intensity normalization, so
their raw projections and training trajectories need not match. These
implementation differences alone do not establish which mode performs better
on experimental data. The public two-GPU
component-parallel trainer has not adopted the fixed-frame option.

Projection settings are saved with new checkpoints and used by the case audit.
An exact `--resume` keeps the saved scientific settings; use a new run with
`--warm-start` if changing the projection frame from a checkpoint.

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
