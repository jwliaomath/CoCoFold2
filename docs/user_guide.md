# CoCoFold2 User Guide

CoCoFold2 refines protein structure predictions from limited-particle cryo-EM
observations using a frozen Protenix-v1 diffusion prior. This guide covers
installation, input preparation, refinement and inspection of saved results.

[Project website](https://jwliaomath.github.io/CoCoFold2/) ·
[Source code](https://github.com/jwliaomath/CoCoFold2) ·
[Releases](https://github.com/jwliaomath/CoCoFold2/releases)

## Start here

1. Follow [Installation](installation.md) and validate your environment.
2. Read [Input data requirements](data_requirements.md), including the coordinate
   frame and particle-path conventions.
   For fixed-frame refinement, choose the required 3D
   [projection origin](parameter_guide.md#choosing-the-projection-origin) before
   starting a new run.
3. Run the [7ZDT/7ZD5 minimal example](../examples/7zdt_7zd5/README.md) before
   adapting the workflow to your own data.
4. Inspect the exported structure and [structured run records](outputs_and_restart.md).

Run shell commands from the cloned CoCoFold2 repository root unless a tutorial
explicitly says otherwise. Replace example paths with your own paths. Weights,
common resources and sequence/MSA inputs are separate from the source checkout.

## Choose a workflow

| Task | Guide |
| --- | --- |
| Refine a complete structure on one GPU | [Single-GPU 6ZBH tutorial](particle_tutorial_6zbh.md) |
| Refine components across two GPUs | [Parallel 6ZBH example](../examples/6zbh_parallel/README.md) |
| Prepare Independent or Contextual component caches | [Component preparation](component_parallel_tutorial.md) |
| Use separate rigid transforms for chains or blocks | [Alignment guide](block_rigid_alignment.md) |
| Understand options and saved states | [Parameters](parameter_guide.md) and [outputs/restart](outputs_and_restart.md) |

## Interpreting results

A successful smoke test checks execution and output consistency; it does not
establish structural accuracy for a new target. Inspect agreement with your
experimental map and, when appropriate, compare matched atoms with an independent
reference. Read the [validation scope](release_validation.md) for the tested
workflows and their limits.

For errors, start with [Troubleshooting](troubleshooting.md). Report reproducible
issues through [GitHub Issues](https://github.com/jwliaomath/CoCoFold2/issues),
including the command, environment and relevant error messages.
