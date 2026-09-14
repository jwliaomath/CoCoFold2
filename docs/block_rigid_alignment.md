# Per-chain and block rigid alignment

Single-GPU training uses one complete diffusion cache, one decoder and one
whole-structure latent. Per-chain alignment applies a separate R/t to each
atom group after decoding; four chains do not require four models. Supply one
reference CIF already placed in the experimental map frame. Its atom identities,
count and order must match the cache, and all chains must share that reference frame.

## Generate a per-chain manifest

```bash
python src/prepare_block_alignment.py --target /path/to/placed.cif \
  --by-chain --fit-atoms ca --output /path/to/new_chains.json
```

Each CIF `auth_asym_id` becomes one group. By default, fitting uses all C-alpha
atoms in each chain; `--fit-atoms all` uses all atoms. Either choice retains
every atom and moves it with its assigned chain. Each fitting core requires at
least three non-collinear atoms; an invalid core raises an error rather than
falling back to a global fit. A manifest can manually restrict the fitting core
while retaining atom identities and complete group assignments.

Add these options to the normal `train.py` command:

```text
--coordinate-mode blocks --block-alignment /path/to/new_chains.json
--train_deterministic
```

Continue to supply the same `--cif_path`. For new per-chain manifests,
`--alignment-sampler auto` uses the original train sampler and applies per-chain
transforms to its output coordinates. The original dtype, cache and RNG behavior
are retained. Old block checkpoints retain their original block decoder;
sampling paths are not switched implicitly. The public block decoder supports
single-structure decoding without private heterogeneity modules.

## Fitting and optional updates

Each initial coordinate core is fitted to its reference core by rigid least
squares. These transforms contain rotation and translation, without scaling or
shear. The coordinate module uses the row-vector convention:
`Y[a] = X[a] @ R[body[a]] + t[body[a]]`.

Transforms remain fixed during training by default. With explicit
`--update_affine_mat`, chains independently test the common
`--block-update-trace-threshold` (default 2.5). A chain updates both R and t only
when `trace(R_new @ R_old.T) < threshold`. Fitting is not differentiated;
updates are recorded in JSONL and transforms are saved in the checkpoint.
Global mode retains its existing update logic.

## Saving, export and restart

Epoch saving synchronously decodes the updated parameters, so the structure and
checkpoint represent the same step. New checkpoints retain grouping, fitting
cores, target coordinates, transforms and sampling information. `get_pdb.py`
defaults to the saved experimental reference frame; raw export is optional.
Use `--resume` for complete new epochs and `--warm-start` for older files. See
[outputs and restart](outputs_and_restart.md).

The legacy `--atom-map` path accepts atom tables with `six_body_file` and
`suggested_fit_core` fields for existing block manifests. It is a separate
manifest preparation method from automatic one-group-per-chain assignment.

## Relationship to GPU grouping

Rigid fitting groups and compute groups are independent. Parallel uses one
component cache per GPU; each cache may contain several chains and optionally
fit them separately. Four chains can be optimized on one GPU as a whole or
with two already prepared two-chain caches in a 2+2 layout. The program does
not automatically combine four independent single-chain caches into two caches.

The accepted Contextual 6ZBH example uses 1+3: A and BCD each occupy one GPU.
It fits each component as a whole, without per-chain transforms inside a
component. Each epoch exports component structures and a merged CIF. See the
[two-GPU example](../examples/6zbh_parallel/README.md).
