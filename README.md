# CoCoFold2

CoCoFold2 refines a protein structure against cryo-EM particle observations
using a frozen Protenix-v1 diffusion prior. It optimizes a target-specific latent
perturbation and, by default, Gaussian renderer amplitudes and widths. Network
weights remain frozen. Fixed stochasticity supports deterministic latent
refinement within the selected sampling setup; it does not optimize an
expectation over all diffusion samples.

The public workflow includes initial prediction/cache generation, single-GPU
refinement, optional per-chain rigid alignment, component-parallel refinement,
CIF/PDB export and complete-epoch restart. It requires a sufficiently accurate
initial prior, informative observations, upstream particle poses/CTF estimates
and a compatible initial structure placed in the experimental coordinate frame.
CoCoFold2 does not estimate particle poses or CTF parameters.

## Start with the CPU checks

Follow [installation](docs/installation.md) to create a separate CPU environment:

```bash
python tests/run_public_tests.py --output results/first_cpu_check
```

This checks an independent public copy without Protenix, weights, private code
or GPUs. Each test and a total pass/fail summary are written to JSON, Markdown
and JUnit XML. An analytic decoder is used where needed; this is not a
real-model test. See [test levels and commands](tests/README_public_tests.md).

## Try the small real-model example

Use the [7ZDT/7ZD5 walkthrough](examples/7zdt_7zd5/README.md): generate a 3 Å
map from the supplied 7ZD5 CIF, simulate 1000 SNR=1 particles, predict with
Protenix-v1, place the prediction in the map, then run smoke and refine stages.
The example explicitly freezes both GMM parameter groups. The general trainer
keeps GMM learning enabled by default. The author accepted the real smoke and
refine results in the existing server environment.

For experimental particle inputs, see [data requirements](docs/data_requirements.md)
and the [single-GPU 6ZBH tutorial](docs/particle_tutorial_6zbh.md).
For independent rigid transforms within one structure, see
[per-chain alignment](docs/block_rigid_alignment.md).
For two GPUs, use the [6ZBH Contextual 1+3 example](examples/6zbh_parallel/README.md)
and [component preparation](docs/component_parallel_tutorial.md). One GPU owns
one component cache; a component may contain several chains. Per-chain fitting
does not require one GPU per chain. Each epoch writes both component structures
and a merged CIF.

## Parameters, outputs and restart

- `python src/train.py --help` lists controls and parameter explanations;
  [CLI reference](docs/cli_reference.md) records the public entrypoints.
- Defaults remain 10 epochs, seed 42, legacy RNG, GMM learning enabled and
  learning rates 0.01/0.01/0.005 for latent bias/amplitudes/widths. Explicit
  `--train_deterministic` documents the default fixed-stochasticity single-GPU run;
  `--no-train_deterministic` explicitly disables it.
- `train.py` requires an aligned reference CIF and defaults to CIF output;
  choose `--output-format pdb` or `both` when needed. `get_pdb.py` can export
  without a template when cache topology is sufficient.
- `--check-inputs` diagnoses inputs before model construction. It does not
  prove GPU memory sufficiency or successful model execution.
- Runs record the captured argument vector, requested/resolved configuration,
  provenance, metrics JSONL and artifacts. See [outputs and restart](docs/outputs_and_restart.md).
- Complete new epoch checkpoints support `--resume`; older results use
  `--warm-start`. `--epochs` is a cumulative target. No mid-epoch resume or
  emergency checkpoint on OOM is provided.

## Validation and limits

[Validation scope](docs/release_validation.md) separates CPU tests, real model
checks and manual structural review. The accepted Contextual 6ZBH long case
covers the first **four complete epochs**, not completion of the original
ten-epoch job. Structure/map agreement and Cα RMSD remain manual assessments.
Changing microbatch size can change legacy loss scaling, so preserve it in
comparisons. Half-map weighting options currently do not enter the active
particle FRC objective.

Random, fine-tuning and the main heterogeneity experiment code are outside this
public release. Weights, MSA resources and experimental particle stacks are not
bundled. The author accepted a fresh Linux installation, CPU tests and a short
real-weight smoke run with GNU 12.2.0 configured for extension compilation.
This does not establish support on every platform; final GitHub review remains
pending. See [installation](docs/installation.md) and
[troubleshooting](docs/troubleshooting.md).

## Citation and license

The existing author and repository metadata are in [CITATION.cff](CITATION.cff).
The planned release is CoCoFold2 v1.0.0. A manuscript DOI is not yet available
and is omitted from the citation metadata.
The code retains its [Apache-2.0 license](LICENSE); upstream software and data
retain their respective terms and attribution requirements.
