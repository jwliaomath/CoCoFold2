# Shared decoding during refinement

New training runs default to `--decode-mode auto`. With fixed diffusion
stochasticity (the training default), multiple particle microbatches share one
decoded coordinate tensor. Each microbatch still renders its own projections
and backpropagates its image loss. Coordinate gradients accumulate before one
decoder recomputation and one complete decoder backward pass.

For example, `--batch_size 32 --mini_batch_size 2` processes 16 microbatches.
Shared mode calls the decoder twice per optimizer update: once without an
activation graph, then once with gradients. It does **not** retain all 16
renderer graphs. GMM parameter gradients and regularization retain their
existing microbatch accumulation. The optimizer updates once per batch.

| Option | Behavior |
| --- | --- |
| `--decode-mode auto` | Shared for fixed stochasticity; legacy for resampled stochasticity. |
| `--decode-mode shared` | Requires fixed stochasticity. |
| `--decode-mode legacy` | Decode and backpropagate separately for each microbatch. |

A batch containing only one microbatch always uses one ordinary forward and
backward pass, including under `shared`. No speedup is expected in that case.
With `--update_affine_mat`, shared mode checks placement after the first decode
and holds that transform for the optimizer batch. Legacy mode retains its
original placement checks. Projection frames, loss normalization, diffusion
steps, learning rates, and cache formats are unchanged.

Both `src/train.py` and `src/chain_parallel/train_chain_parallel_2d.py` support
these options. Component-parallel training retains the global summed projection
and its existing distributed gradient scaling. Independent and contextual
component caches use the same training implementation; upstream Big prediction
and cache export do not need to change.

## Reproducibility and checkpoints

The replay restores the first decode's RNG state and restores the caller's RNG
state afterwards. Floating-point accumulation order differs from repeated
decoder backward passes. This is not a promise of bitwise equivalence or
identical long-run trajectories, and process-wide RNG side effects can differ
from repeatedly calling the legacy decoder. The RNG mode itself is unchanged.

`train_step` records contain `decode_mode`, `microbatches`, `decoder_calls`, and
`shared_decode_active`. Counts exclude initialization and epoch-end exports.
Saving still decodes the updated parameters for synchronized CIF/checkpoint
output. Allocated and reserved CUDA memory are different quantities; shared
decoding does not guarantee lower reserved memory on every workload.

Completed-epoch resume inherits the saved resolved decode mode. Older public
checkpoints without that field resume with `legacy`. Explicitly changing the
mode during resume is rejected; use `--warm-start` to start a new experiment
from the saved structure/latent/GMM state with reset optimizer and progress.
Historical experimental shared-decoding checkpoints may also lack the field:
do not silently relabel them. Use their original experimental runner or a new
warm-start; exact continuation through this public interface is not inferred.

## Tests

The public CPU gate includes nonlinear VJP/optimizer comparisons, activation
checkpoint replay, uneven microbatches, single-microbatch fallback, real trainer
integration with an analytic denoiser, and checkpoint compatibility. The
two-process Gloo gate checks distributed projection gradients, shared-versus-
legacy refinement, complete-epoch resume, and coordinated failures. These do not
replace GPU validation with real Protenix weights on the target environment.
