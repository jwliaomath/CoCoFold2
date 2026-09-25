# 6ZBH component-parallel refinement (1+3)

Use two GPUs: rank 0 owns chain A, rank 1 owns chains B/C/D. Each rank has
one cache and one frozen diffusion decoder. Both projections contribute to
the same particle loss. The default example fits each component as a whole;
the three-chain component does not require three GPUs.

The B7b short Contextual run passed the author's real Protenix-v1 validation.
B8 was subsequently accepted for the first four complete epochs (69/69 checks), with structural quality accepted by the author. This does not claim completion of the original ten-epoch job. Independent
inputs use the same trainer, but a full Independent run is not part of this
batch's acceptance. CPU tests of the scripts are not model validation.

## Reuse prepared inputs

For the author's existing case, use the supplied server manifest and already
placed component CIFs. Do not regenerate caches, repeat fitting or rerun the
B7b smoke/resume tests before this full run.

For another installation, copy `contextual.yaml` (or `independent.yaml`) and
edit its paths. Paths are relative to the manifest, unless absolute. The BCD
references used in this example have local chain IDs A/B/C: `chain_id_map`
maps them to global B/C/D in the merged CIF. If a new reference already uses
B/C/D, use the identity mapping instead. Never guess a chain correspondence.

The reference CIFs must retain the cache's atom identities and ordering and be
rigidly placed in the same map/particle coordinate frame. A chain-name mapping
does not place a structure in the map and does not change its coordinates.

## Prepare inputs when no compatible caches exist

See [the component tutorial](../../docs/component_parallel_tutorial.md) for
initial Protenix input JSON/MSA requirements and the alternative `z_trunk`
workflow. This example does not download weights, MSA data or particle stacks.

- **Contextual:** generate a full 6ZBH cache with `src/inference.py`, inspect
  numeric `asym_id` assignments, then split the A and BCD groups. A full-cache
  `asym_id` is not inferred from the local CIF chain letter.
- **Independent:** prepare two Protenix JSON inputs, one for A and one for BCD,
  and generate caches separately with the same model and MSA policy. Replacing
  a Contextual cache path with an Independent cache changes its conditioning;
  neither method is equivalent to full-complex diffusion refinement.

From the repository root, after configuring `PROTENIX_ROOT_DIR` to contain
`common/` and `checkpoint/`:

```bash
python src/chain_parallel/prepare_contextual_diffusion_caches.py inspect \
  --cache /path/to/6zbh_diffusion_data.pth --output-json /path/to/inspection.json
python src/chain_parallel/prepare_contextual_diffusion_caches.py split \
  --spec examples/6zbh_parallel/contextual_split.yaml --device cuda:0
```

Edit the split spec paths first. Its 1683 tokens / 13464 atoms and groups
717/5754 versus 966/7710 come from the author's supplied cache metadata;
inspect new caches before reusing those expectations. Existing output caches
are protected from overwrite.

Decode each cache with `src/get_pdb.py --diffusion_data_dir CACHE --pdbid NAME
--out_dir NEW_DIRECTORY --seed 42 --device cuda:0 --output-format cif`.
Template-free export is available for compatible cache topology. Rigidly place
the exported prediction in the experimental map, save a fitted CIF and update
the training manifest. Keep atom identities/order. Placing the initial prediction
is a user step; the scripts do not silently align it to a deposited truth model.

## Full refinement without Slurm

Activate the validated environment, expose two suitable GPUs and run:

```bash
python examples/6zbh_parallel/run_case.py run \
  --manifest /path/to/contextual.yaml \
  --star /path/to/366.star \
  --mrc-dir /path/to/particle/stacks \
  --resource-root /path/to/protenix_resources \
  --output /path/to/new_6zbh_run \
  --epochs 10
```

`--resource-root` contains `common/` and `checkpoint/`. Absolute MRCS paths in
the STAR are used directly; relative paths use the specified `--mrc-dir`.
Add `--dry-run` to print the exact command without creating files or loading
models. The output directory must be new. A Slurm job runs the same entrypoint;
cluster partition and node exclusions belong in the local submission script.

The example uses the **entire supplied STAR**, 10 epochs, batch=32,
mini-batch=16, box=288, apix=1.073, GMM resolution=3, FRC cutoff=2.146,
particle sign=-1, transR enabled, fixed stochasticity, seed=42 and legacy RNG.
It explicitly uses fixed-frame projection with origin
`154.512 154.512 154.512` Å, the box center only for a verified zero-origin,
zero-start, standard-axis 288-pixel map at 1.073 Å/pixel. Check the map header
and all placed component CIFs before using this command with another dataset;
see [choosing the origin](../../docs/parameter_guide.md#choosing-the-projection-origin).
GMM weight/width learning stays enabled, with the trainer's unchanged default
learning rates (0.01/0.01/0.005). Component affine updates and per-chain fitting
are off. The peak-2D memory-checkpoint switch remains at its original default.
Mini-batch=16 is this validated case's setting; the generic trainer default is 12.

If the STAR contains N particles, one epoch has `ceil(N / 32)` optimizer
steps. The filename `366.star` does not specify its row count. All ranks see
all N particles in the same order; particles are not sharded between GPUs.

## Outputs and automatic checks

Each epoch writes two component `.pth`/`.cif` pairs named
`model_COMPONENT_ID_rankRANK_EPOCH`, plus `model_merged_EPOCH.cif` and
`model_epoch_EPOCH.json` in the new run directory. The wrapper's `--output`
names that run **directory**; it supplies the `model_` filename prefix to the
trainer. The merged CIF is in the experimental reference
frame and has chains A/B/C/D. **No manual merging is needed.** The initially
exported unrefined files have a different name and are raw decoder coordinates;
use the numbered merged CIF for the refined assembly.

`records/rank0/` and `records/rank1/` contain commands, requested/resolved
configuration, source/input provenance, JSONL events and summaries. The wrapper
also writes `launcher.json`, `preflight.log`, `training.log`, `summary.json`
and `summary.txt`. Supplied submission scripts are copied into each rank record.

The automatic audit checks every epoch's progress, full particle coverage,
matching rank order, finite metrics, component/merged CIF identities and finite
coordinates. It verifies final checkpoint hashes, finite latent/GMM/optimizer
states and consistency between saved parameters and exported coordinates.
Earlier checkpoints are checked for existence; full SHA-256 reading is limited
to the final two checkpoints to avoid rereading all large model files.

Timing reports distinguish rank run time (including model setup and export)
from training-step time. A distributed step uses the maximum time across the
two ranks. Peak allocated/reserved GPU memory is reported per rank in MiB;
these are the trainer's per-batch peaks, not measurements of peak initialization
or serialization memory. No extra decoding or benchmark training is performed.

To rerun only the audit on existing outputs, without GPUs or Protenix execution:

```bash
python examples/6zbh_parallel/run_case.py check \
  --output /path/to/new_6zbh_run --epochs 10
```

Automatic acceptance is `passed=true`, `failed_count=0`, `error=null`.
**Map agreement and RMSD remain the author's manual assessment.** No structural
quality threshold, monotonic-loss requirement or convergence claim is inferred
from automatic completion.

## Interrupted training

An explicitly revised acceptance scope may inspect the first N complete epochs
without waiting for the original target. For example, to accept four epochs
of an originally ten-epoch fresh run:

```bash
python examples/6zbh_parallel/run_case.py check \
  --output /path/to/existing_run --epochs 4 --completed-prefix \
  --reason "Author revised B8 acceptance from 10 to 4 complete epochs"
```

Wait until `model_epoch_4.json` exists. This audits epochs 1–4 and the epoch-4
checkpoint pair using the same numerical and file-integrity checks. It reads
only the corresponding recorded steps, so later epochs and partially written
later JSONL lines are excluded. The report is `acceptance_epoch_4.json/.txt`,
leaving the original full-run `summary.json` untouched. Original requested
epochs, observed rank/launcher statuses, errors and the scope-change reason
are preserved. A recorded `running` status can be stale after a process kill;
the checker does not query the scheduler or claim the job is still alive.

A passing prefix audit claims only that these complete epochs passed. It does
not claim the original 10-epoch job completed or automatically stop training.
Whole-job duration is not attributed to the four-epoch prefix; step timing and
memory reports use only the selected epochs. Running/stopped/failed job status
is recorded separately. Incomplete or corrupt selected epochs still fail.
Without `--completed-prefix`, the original strict full-run requirements remain.

The wrapper starts a fresh run. The trainer separately supports `--resume`
with a complete `model_epoch_EPOCH.json`, unchanged manifest/STAR/grouping and
an increased cumulative `--epochs`, in a new output directory. See the component
tutorial. OOM does not save a rescue checkpoint; a partial epoch is not resumable.
This full-run checker expects all epochs in one fresh-run directory, so do not
use it to claim a resumed segment alone contains the entire original run.
