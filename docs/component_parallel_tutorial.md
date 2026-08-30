# Component-parallel CoCoFold2 refinement

This tutorial describes the component-parallel particle-refinement workflow
implemented in [`src/chain_parallel`](../src/chain_parallel). It covers:

1. **Independent-component refinement**, in which Protenix caches are generated
   separately for user-defined component groups;
2. **Contextual-component refinement**, in which one full-complex Pairformer
   cache is divided into contextual component caches; and
3. a memory-conscious contextual path that saves full-complex `z_trunk`, splits
   it first, and then materializes component-local `pair_z` caches.

The implementation is not restricted to two GPUs. With the current manifest
format, a run uses one component group per process/rank. A manifest containing
`K` component groups is therefore launched with `K` processes, normally on
`K` GPUs. A component group may contain one chain or several chains.

## Scope and approximation boundary

Each rank owns:

- one frozen Protenix-v1 diffusion decoder;
- one component conditioning cache;
- one component-specific latent perturbation (`z_bias`); and
- component-local Gaussian renderer parameters.

For each particle, all ranks render their components in the same experimental
coordinate frame. CoCoFold2 uses differentiable distributed operations to
obtain a shared projected origin and sum the component projections. Particle
sign, translation, CTF and FRC loss are then applied to the assembled
projection, allowing one particle-space objective to update every component.

The two cache strategies differ before this shared objective:

| Strategy | Pairformer calculation | Information retained in each component cache |
|---|---|---|
| Independent-component | Run separately for each component group | Component-only sequence context |
| Contextual-component | Run once for the complete assembly | Full-complex contextual single rows and the selected pair diagonal block |

Neither strategy is exact full-complex Protenix refinement. Component-local
diffusion omits cross-component diffusion attention and joint coordinate
generation. Contextual-component refinement also requires the full Pairformer
and its full pair representation to fit once.

## 1. Installation and repository layout

Follow the main [installation instructions](../README.md#installation) and
run the commands below from the repository root:

```bash
git clone https://github.com/jwliaomath/CoCoFold2.git
cd CoCoFold2
conda env create -f environment.yml
conda activate cocofold2
```

The compatible Protenix checkpoint and common resources must be available as
described in the main README. The public source layout used here is:

```text
CoCoFold2/
├── checkpoint/
├── common/
├── src/
│   ├── inference.py
│   ├── get_pdb.py
│   ├── train.py
│   └── chain_parallel/
│       ├── train_chain_parallel_2d.py
│       ├── manifest.py
│       ├── distributed_gmm.py
│       ├── contextual_cache.py
│       ├── prepare_contextual_diffusion_caches.py
│       └── materialize_local_diffusion_cache.py
└── docs/
```

Configure the Python paths once per shell:

```bash
export REPO_ROOT="$(pwd)"
export COCOFOLD2_ROOT="${REPO_ROOT}/src"
export CHAIN_PARALLEL_DIR="${REPO_ROOT}/src/chain_parallel"
export PYTHONPATH="${COCOFOLD2_ROOT}:${CHAIN_PARALLEL_DIR}:${PYTHONPATH:-}"
```

`COCOFOLD2_ROOT` must directly contain `ctf.py`, `particledataset.py`,
`pts2img.py`, `utils.py` and `model/protenix.py`.

## 2. Required inputs

In addition to the standard particle-workflow inputs described in
[Data requirements](data_requirements.md), prepare:

- a chain-to-component partition;
- one diffusion cache per component group;
- one component topology file with exactly the same atom order as its cache;
- one fitted initial structure per component group; and
- one component manifest assigning component groups to ranks.

All fitted component structures must be placed in the same coordinate frame
defined by the experimental reconstruction and upstream particle poses. Fit
the sequence-derived initial predictions, not the deposited evaluation
structure.

The STAR file, MRC/MRCS particle stacks, particle subset, pose/CTF metadata,
box size, pixel size and loss settings must be identical across component
strategies used in a controlled comparison.

## 3. Select component groups and GPU count

Partition the assembly into `K` complete-chain groups. The current
implementation requires:

```text
number of manifest entries = torchrun world size = number of ranks
```

and each rank may occur only once in the manifest.

For example, a three-GPU partition could be:

```text
rank 0: chain A
rank 1: chains B+C
rank 2: chains D+E+F
```

Choose groups according to available GPU memory and component sizes. If there
are more chains than GPUs, place multiple chains in one component group rather
than assigning multiple manifest entries to one rank.

The manuscript examples use two groups (6ZBH: A versus B--D; 6O77: A--B versus
C--D), but these are example partitions rather than a two-GPU software limit.

## 4. Independent-component refinement

### 4.1 Generate one Protenix cache per component group

Create one Protenix input JSON for each group. Each JSON should contain only
the chains assigned to that group, while using the same checkpoint and the
same MSA/template policy.

For a three-group example:

```bash
mkdir -p runs/target/independent/cache runs/target/independent/protenix

python -u src/inference.py \
  --input_json_path inputs/target_A.json \
  --sample_name target_A \
  --output_model_dir runs/target/independent/cache/ \
  --dump_dir runs/target/independent/protenix/A

python -u src/inference.py \
  --input_json_path inputs/target_BC.json \
  --sample_name target_BC \
  --output_model_dir runs/target/independent/cache/ \
  --dump_dir runs/target/independent/protenix/BC

python -u src/inference.py \
  --input_json_path inputs/target_DEF.json \
  --sample_name target_DEF \
  --output_model_dir runs/target/independent/cache/ \
  --dump_dir runs/target/independent/protenix/DEF
```

The trailing slash on `--output_model_dir` is required by the current filename
construction. Expected files include:

```text
runs/target/independent/cache/target_A_diffusion_data.pth
runs/target/independent/cache/target_BC_diffusion_data.pth
runs/target/independent/cache/target_DEF_diffusion_data.pth
```

### 4.2 Generate and rigidly place each initial component

Use `src/get_pdb.py` with the Protenix topology output corresponding to the
same component cache:

```bash
python -u src/get_pdb.py \
  --pdbid target_A \
  --diffusion_data_dir runs/target/independent/cache/target_A_diffusion_data.pth \
  --cif_path /path/to/target_A_protenix_topology.cif \
  --out_dir runs/target/independent/initial/A \
  --device cuda:0
```

Repeat for every component. Rigidly fit the generated component structures
into one common experimental frame and preserve their atom ordering. The
fitted files, not deposited reference structures, are supplied to the
component trainer.

### 4.3 Create the component manifest

Save a YAML file such as
`runs/target/independent/components.yaml`:

```yaml
schema_version: 1
components:
  - id: target_A_independent
    rank: 0
    diffusion_data_dir: cache/target_A_diffusion_data.pth
    cif_path: fitted/target_A_fitted.cif

  - id: target_BC_independent
    rank: 1
    diffusion_data_dir: cache/target_BC_diffusion_data.pth
    cif_path: fitted/target_BC_fitted.cif

  - id: target_DEF_independent
    rank: 2
    diffusion_data_dir: cache/target_DEF_diffusion_data.pth
    cif_path: fitted/target_DEF_fitted.cif
```

Relative paths are resolved from the manifest directory. Component IDs and
ranks must be unique, and ranks must cover `0` through `K-1`.

### 4.4 Launch on `K` GPUs

Set the number of processes to the number of component groups:

```bash
NUM_COMPONENTS=3
OUTPUT_PREFIX=runs/target/independent/results/model_
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

torchrun --standalone --nproc_per_node="${NUM_COMPONENTS}" \
  src/chain_parallel/train_chain_parallel_2d.py \
  --component_manifest runs/target/independent/components.yaml \
  --star_data_dir /path/to/particles.star \
  --mrc_data_dir /path/to/particle/stacks/ \
  --output_trained_model_dir "${OUTPUT_PREFIX}" \
  --backend nccl \
  --boxsize REPLACE_WITH_BOX_SIZE \
  --apix REPLACE_WITH_PIXEL_SIZE \
  --resolution REPLACE_WITH_GMM_RESOLUTION \
  --map_resolution REPLACE_WITH_FRC_CUTOFF_RESOLUTION \
  --batch_size REPLACE_WITH_OUTER_BATCH_SIZE \
  --mini_batch_size REPLACE_WITH_MICROBATCH_SIZE \
  --particle_sign -1 \
  --train_deterministic \
  --update_affine_mat
```

Add `--transR` only when required by the validated upstream orientation
convention. Add `--update_affine_mat` only when the flip safeguard used in the
registered experiment is intended.

The particle-stack root must end in `/` because the current loader concatenates
it with the path stored in `rlnImageName`.

## 5. Contextual-component refinement

Contextual-component refinement uses one complete-assembly Pairformer cache
and extracts one contextual cache for every component group.

### 5.1 Generate a full-complex cache

For assemblies that can construct the standard full cache:

```bash
mkdir -p runs/target/contextual/full_cache runs/target/contextual/protenix

python -u src/inference.py \
  --input_json_path inputs/target_full_complex.json \
  --sample_name target_full \
  --output_model_dir runs/target/contextual/full_cache/ \
  --dump_dir runs/target/contextual/protenix
```

The expected cache is:

```text
runs/target/contextual/full_cache/target_full_diffusion_data.pth
```

If generating full-complex shared diffusion variables is impractical, use the
`z_trunk` workflow in Section 6 instead.

### 5.2 Inspect numeric `asym_id` assignments

```bash
python src/chain_parallel/prepare_contextual_diffusion_caches.py \
  inspect \
  --cache runs/target/contextual/full_cache/target_full_diffusion_data.pth \
  --output-json runs/target/contextual/full_cache/inspection.json
```

Use the reported Protenix `asym_id`, token count and atom count to define the
split. Do not infer `asym_id` solely from the visible CIF chain label,
particularly for repeated subunits.

### 5.3 Define an arbitrary `K`-component contextual split

Create `runs/target/contextual/split.yaml`:

```yaml
schema_version: 1
source_cache: full_cache/target_full_diffusion_data.pth
require_complete_partition: true
expected_source_n_token: REPLACE_WITH_INSPECTED_INTEGER
expected_source_n_atom: REPLACE_WITH_INSPECTED_INTEGER

components:
  - id: target_A_contextual
    asym_ids: [REPLACE_WITH_ASYM_ID]
    expected_n_token: REPLACE_WITH_INSPECTED_INTEGER
    expected_n_atom: REPLACE_WITH_INSPECTED_INTEGER
    output_cache: cache/target_A_contextual.pth

  - id: target_BC_contextual
    asym_ids: [REPLACE_WITH_ASYM_IDS]
    expected_n_token: REPLACE_WITH_INSPECTED_INTEGER
    expected_n_atom: REPLACE_WITH_INSPECTED_INTEGER
    output_cache: cache/target_BC_contextual.pth

  - id: target_DEF_contextual
    asym_ids: [REPLACE_WITH_ASYM_IDS]
    expected_n_token: REPLACE_WITH_INSPECTED_INTEGER
    expected_n_atom: REPLACE_WITH_INSPECTED_INTEGER
    output_cache: cache/target_DEF_contextual.pth

report_json: cache/contextual_split_report.json
```

`require_complete_partition: true` rejects missing, unknown, overlapping or
duplicated `asym_id` assignments.

### 5.4 Split and rebuild component-local atom features

```bash
python src/chain_parallel/prepare_contextual_diffusion_caches.py \
  split \
  --spec runs/target/contextual/split.yaml \
  --device cuda:0
```

For each group, the utility:

- selects the corresponding `s_inputs` and `s_trunk` rows;
- extracts the exact diagonal block of full-complex `z_trunk` or `pair_z`;
- selects component atoms and remaps atom-to-token indices;
- rebuilds `d_lm`, `v_lm` and `pad_info`; and
- rebuilds the atom shared cache when the source already contains `pair_z`.

The source cache is not overwritten. Output caches are saved on CPU for
portability and record the source-cache SHA-256.

### 5.5 Generate, place and register contextual components

Run `src/get_pdb.py` for every contextual component cache, then rigidly place
the generated structures in the same experimental frame. Create a training
manifest with one contextual cache and fitted CIF per rank:

```yaml
schema_version: 1
components:
  - id: target_A_contextual
    rank: 0
    diffusion_data_dir: cache/target_A_contextual.pth
    cif_path: fitted/target_A_contextual_fitted.cif

  - id: target_BC_contextual
    rank: 1
    diffusion_data_dir: cache/target_BC_contextual.pth
    cif_path: fitted/target_BC_contextual_fitted.cif

  - id: target_DEF_contextual
    rank: 2
    diffusion_data_dir: cache/target_DEF_contextual.pth
    cif_path: fitted/target_DEF_contextual_fitted.cif
```

Do not mix independent and contextual caches in one run. The trainer also
requires all contextual caches to record the same full source-cache hash.

Launch the contextual run with the same `torchrun` command used in Section
4.4, changing only the manifest and output prefix. For a controlled comparison,
keep the component partition, particles, checkpoint, frame placement policy,
diffusion settings, batch sizes and checkpoint-selection rule matched.

## 6. Large-complex contextual cache: `z_trunk` first, local `pair_z` later

The standard shared-cache path constructs full-complex `pair_z`, `p_lm` and
`c_l` during inference. For a large assembly, these additional full-complex
objects may be undesirable even when the full Pairformer and `z_trunk` can be
computed once.

The alternative workflow is:

```text
full-complex input
  -> full Pairformer and full z_trunk
  -> split contextual z_trunk diagonal blocks
  -> materialize pair_z, p_lm and c_l separately for each smaller component
  -> component-parallel refinement
```

The order is important: split the full `z_trunk` before creating `pair_z`.

### 6.1 Save full-complex `z_trunk`

Pass the Protenix configuration override as an explicit key-value pair:

```bash
mkdir -p runs/large_target/contextual/full_cache

python -u src/inference.py \
  --input_json_path inputs/large_target_full.json \
  --sample_name large_target_full \
  --output_model_dir runs/large_target/contextual/full_cache/ \
  --dump_dir runs/large_target/contextual/protenix \
  --enable_diffusion_shared_vars_cache false
```

Confirm that the log reports:

```text
enable_diffusion_shared_vars_cache False
shared_vars_cache=False
```

The saved cache should contain `z_trunk`, with `pair_z`, `p_lm` and `c_l` set
to `None`. Inspect this before continuing:

```bash
export FULL_CACHE=runs/large_target/contextual/full_cache/large_target_full_diffusion_data.pth
python -c "import os, torch; d=torch.load(os.environ['FULL_CACHE'], map_location='cpu', weights_only=False); print({'z_trunk': None if d['z_trunk'] is None else tuple(d['z_trunk'].shape), 'pair_z': d['pair_z'], 'p_lm': d['p_lm'], 'c_l': d['c_l']})"
```

In the current source, the diffusion cache is written before the final
full-complex coordinate-sampling call. If that later call fails, retain the
error log and validate the cache explicitly; a saved cache does not imply that
full-complex coordinate generation succeeded.

This pathway still requires the full Pairformer and full `z_trunk` to fit. It
does not solve the Pairformer memory boundary.

### 6.2 Inspect and split the `z_trunk` cache

Use `prepare_contextual_diffusion_caches.py inspect` and `split` exactly as in
Section 5. After splitting, every component cache should contain local
`z_trunk` while `pair_z`, `p_lm` and `c_l` remain `None`.

### 6.3 Materialize a local shared diffusion cache for each component

Run the following command once per component output:

```bash
python src/chain_parallel/materialize_local_diffusion_cache.py \
  --input-cache runs/large_target/contextual/cache/group_A_z_trunk.pth \
  --output-cache runs/large_target/contextual/cache/group_A_pair_z.pth \
  --device cuda:0 \
  --report-json runs/large_target/contextual/cache/group_A_materialization.json
```

Repeat for all `K` components. The input files are not modified. Each output
contains component-local `pair_z`, `p_lm` and `c_l`, sets `z_trunk` to `None`,
and records source/output hashes, tensor shapes, finite-value checks, wall
time and peak GPU memory for the materialization step.

Use the materialized `*_pair_z.pth` files in the final component manifest.

### 6.4 Record the optimized representation level

The trainer supports both cache forms, but their latent updates occur at
different representation levels:

- for a `z_trunk` cache, `z_bias` is added to `z_trunk` before diffusion
  conditioning is constructed;
- for a materialized cache, `z_bias` is added directly to cached `pair_z`.

Consequently, local `z_trunk -> pair_z` materialization is not merely a file
format conversion. It selects the cached-`pair_z` optimization path. Record
the active conditioning tensor in each experiment and do not describe a
`z_trunk` and a `pair_z` run as optimizing identical variables.

If `z_trunk`-level optimization is required, skip materialization and provide
the split `z_trunk` component caches directly to the trainer. The component
diffusion-conditioning cache will then be constructed during sampling.

## 7. Generic Slurm launch

Request one GPU per component group and launch one `torchrun` process per GPU.
A cluster-neutral skeleton is:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:K
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Activate the environment using the commands required by your cluster.
conda activate cocofold2

cd /path/to/CoCoFold2
export COCOFOLD2_ROOT="${PWD}/src"
export CHAIN_PARALLEL_DIR="${PWD}/src/chain_parallel"
export PYTHONPATH="${COCOFOLD2_ROOT}:${CHAIN_PARALLEL_DIR}:${PYTHONPATH:-}"

NUM_COMPONENTS=K
mkdir -p logs runs/target/results

python -c "import torch; expected=${NUM_COMPONENTS}; assert torch.cuda.device_count() == expected, f'expected {expected} visible GPUs, got {torch.cuda.device_count()}'; assert torch.distributed.is_nccl_available()"

srun --ntasks=1 \
  torchrun --standalone --nproc_per_node="${NUM_COMPONENTS}" \
  src/chain_parallel/train_chain_parallel_2d.py \
  --component_manifest runs/target/components.yaml \
  --star_data_dir /path/to/particles.star \
  --mrc_data_dir /path/to/particle/stacks/ \
  --output_trained_model_dir runs/target/results/model_ \
  --backend nccl \
  --boxsize REPLACE_WITH_BOX_SIZE \
  --apix REPLACE_WITH_PIXEL_SIZE \
  --resolution REPLACE_WITH_GMM_RESOLUTION \
  --map_resolution REPLACE_WITH_FRC_CUTOFF_RESOLUTION \
  --batch_size REPLACE_WITH_OUTER_BATCH_SIZE \
  --mini_batch_size REPLACE_WITH_MICROBATCH_SIZE \
  --particle_sign -1 \
  --train_deterministic \
  --update_affine_mat
```

Replace `K`, all paths and all data-dependent parameters. Add the partition,
account, time and memory directives required by the local scheduler.

The current implementation assumes all processes run within one `torchrun`
job and all ranks load the same particle minibatches. Multi-node execution has
not been documented or validated by this tutorial.

## 8. Outputs and validation

For every rank, the trainer writes:

- the unrefined component structure;
- one component structure and `.pth` checkpoint per epoch;
- `chain_parallel_2d_metrics_rank<rank>.jsonl`; and
- run metadata containing component IDs, cache metadata, world size and
  numerical settings.

Rank 0 also writes a summary metrics file. For a distributed batch, use the
maximum `batch_time_seconds` across rank-specific files as the observed step
time; do not report rank-0 time alone as total distributed time.

Before interpreting a run, confirm:

1. the manifest entry count equals `WORLD_SIZE`;
2. all ranks used the same frozen diffusion weights;
3. all contextual caches have the same full source-cache hash;
4. every fitted CIF matches its cache in atom count and ordering;
5. every component is in the same experimental frame;
6. all ranks received the same particle indices;
7. loss, gradients and renderer parameters remain finite; and
8. the reported checkpoint was selected without access to the deposited evaluation structure.