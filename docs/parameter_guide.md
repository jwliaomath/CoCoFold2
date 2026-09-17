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

## Memory and optimization

Distinguish `--batch_size` from `--mini_batch_size` when adapting a tutorial to
available GPU memory. See the entrypoint's help for their meanings and defaults.
Use the documented learning-rate options for the relevant entrypoint rather than
editing source constants.

GMM learning is enabled by default. `--no-learn-gmm` disables learning of both
atom amplitudes and widths. Kernel and amplitude choices are described in
[Gaussian rendering kernels](gaussian_kernels.md); they are not interchangeable
performance switches.

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
