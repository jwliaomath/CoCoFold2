# CoCoFold2

**Preprint:** [CoCoFold2: scalable latent refinement of diffusion-based protein structure predictions from limited-particle cryo-EM data](https://doi.org/10.65215/LTSpreprints.2026.09.15.000338) · LangTaoSha, 2026.

**Project website and tutorials:** https://jwliaomath.github.io/CoCoFold2/

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

This tests an isolated copy of the public code without requiring Protenix,
model weights or GPUs. Each test and a total pass/fail summary are written
to JSON, Markdown and JUnit XML. An analytic decoder is used where needed;
this is not a real-model test.
See [test levels and commands](tests/README_public_tests.md).

## Try the small real-model example

Use the [7ZDT/7ZD5 walkthrough](examples/7zdt_7zd5/README.md): generate a 3 Å
map from the supplied 7ZD5 CIF, simulate 1000 SNR=1 particles, predict with
Protenix-v1, place the prediction in the map, then run smoke and refine stages.
The example explicitly freezes both GMM parameter groups. The general trainer
keeps GMM learning enabled by default. The smoke and refinement stages were
tested with real Protenix-v1 weights in the development server environment.

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

Validation includes automated CPU tests, real-model smoke and refinement
runs, checkpoint/restart checks, and manual structural inspection.
The Contextual 6ZBH two-GPU workflow was checked through four complete
epochs. See [validation details](docs/release_validation.md) for the
tested configurations and scope.

Changing microbatch size can change loss scaling, so keep it consistent
across comparisons. Half-map weighting options currently do not affect
the particle FRC objective.

The resampled-stochasticity and diffusion fine-tuning ablation code is
not included in this release. Model weights, MSA resources and experimental
particle stacks must be obtained separately.

Installation and real-weight smoke testing were completed in a fresh
Linux environment using GNU 12.2.0 for extension compilation.
See [installation](docs/installation.md) and
[troubleshooting](docs/troubleshooting.md) for environment requirements
and known issues.

## Citation and license

The [CoCoFold2 preprint](https://doi.org/10.65215/LTSpreprints.2026.09.15.000338) is now available on
[LangTaoSha Preprint Server](https://langtaosha.org.cn/lts/en/preprint/view/338)
(15 September 2026). Please cite the manuscript when using CoCoFold2:

Liao, J., Hu, M., & Bao, C. (2026). CoCoFold2: scalable latent refinement of diffusion-based protein structure predictions from limited-particle cryo-EM data. LangTaoSha Preprint Server. https://doi.org/10.65215/LTSpreprints.2026.09.15.000338

```bibtex
@article{liao2026cocofold2,
  title = {CoCoFold2: scalable latent refinement of diffusion-based protein structure predictions from limited-particle cryo-EM data},
  author = {Liao, Junwen and Hu, Mingxu and Bao, Chenglong},
  journal = {LangTaoSha Preprint Server},
  year = {2026},
  doi = {10.65215/LTSpreprints.2026.09.15.000338},
  url = {https://doi.org/10.65215/LTSpreprints.2026.09.15.000338}
}
```

Machine-readable author and citation metadata are in [CITATION.cff](CITATION.cff).
The code retains its [Apache-2.0 license](LICENSE); upstream software and data
retain their respective terms and attribution requirements.
