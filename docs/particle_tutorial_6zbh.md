# Particle-guided CoCoFold2 refinement: the 6ZBH case

This tutorial walks through a research-scale CoCoFold2 particle-refinement case. It does not use a toy dataset.

## Case summary

| Field | Value |
|---|---|
| Target | 6ZBH |
| EMPIAR accession | `EMPIAR-10437` |
| EMDB accession | `EMD-11155` |
| PDB accession | 6ZBH |
| Box size | 288 pixels |
| Pixel size | 1.073 Å/pixel |
| Frequency cutoff setting | 2.146 Å |

CoCoFold2 requires upstream particle poses and CTF parameters and does not estimate them. The Protenix network weights remain frozen. The current implementation optimizes `z_bias`, Gaussian-rendering atom weights and Gaussian widths by default. Epoch count and seeds are configurable; the historical defaults remain 10 and 42. For a smaller first run, use the [7ZDT/7ZD5 example](../examples/7zdt_7zd5/README.md).

## Stage A — Repository and environment

```bash
git clone https://github.com/jwliaomath/CoCoFold2.git
cd CoCoFold2

conda env create -f environment.yml
conda activate cocofold2
```

The repository is executed directly from its root. Do not use `pip install -e .` because this release has no CoCoFold2 package definition.

Set `PROTENIX_ROOT_DIR` before starting Python to a directory containing the compatible Protenix 1.0.2 resources (see [installation](installation.md)):

```bash
export PROTENIX_ROOT_DIR=/absolute/path/to/protenix_resources
```

```text
checkpoint/
common/
```

## Stage B — Expected data layout

```text
data/6zbh/
├── input/
│   └── 6zbh.json
├── particles/
│   ├── 366.star
│   └── PARTICLE_STACK.mrcs
└── fitted/
    └── 6zbh_fitted.cif

params/
outputs/
logs/
```

Download and prepare the public data using:

```bash
wget -nH -m ftp://ftp.ebi.ac.uk/empiar/world_availability/10437/data/particles/MSP1_altconf5/
```

This download command does not create the example layout above. Prepare your
STAR file and particle stacks under your chosen data root, or change the command
paths below to match their actual locations. `366.star` is the example's prepared
STAR filename, not a file that this download command is guaranteed to produce.

Verify that `rlnImageName` paths in `366.star` resolve when prefixed with `data/6zbh/particles/`. Absolute STAR image paths are used directly. Relative paths resolve under `--mrc_data_dir`, or under the STAR directory when that option is omitted; no trailing slash is required.

Choose and record the particle subset for your experiment; the filename does not encode its row count.

## Stage C — Frozen Protenix inference and cache generation

Run this command from the CoCoFold2 repository root:

```bash
python -u src/inference.py \
  --resource-root "$PROTENIX_ROOT_DIR" \
  --input_json_path data/6zbh/input/6zbh.json \
  --sample_name 6zbh \
  --output_model_dir params/ \
  --dump_dir outputs/protenix_6zbh
```

The output path is a directory and does not require a trailing slash. Single-target/single-seed runs retain the cache name below; multiple targets or seeds use separate sample/seed directories. Existing caches are refused rather than overwritten.

Use a single-target, single-seed inference input for the filename in this
walkthrough. The configured Protenix prediction seed is 101 by default; the
refinement/export diffusion seed defaults to 42. These control different stages
and need not be identical. Record any seed overrides with the run configuration.

The expected cache is:

```text
params/6zbh_diffusion_data.pth
```

The cache contains the frozen diffusion-module state and the cached conditional representations used by CoCoFold2, including `s_inputs`, `s_trunk`, cached `pair_z` or `z_trunk`, atom-level caches, the noise schedule and configuration.

Protenix also writes its standard prediction outputs under
`outputs/protenix_6zbh/`. These are separate from the diffusion cache in
`params/`. A Protenix sample CIF is optional for the direct `get_pdb.py` command
below; the supplied `run_initial_prediction.sh` wrapper still uses one as a
topology template.

## Stage D — Deterministic initial prediction

Generate the deterministic initial structure from the cache:

```bash
python src/get_pdb.py \
  --pdbid 6ZBH \
  --diffusion_data_dir params/6zbh_diffusion_data.pth \
  --out_dir outputs/6zbh_initial \
  --output-format cif
```

For a compatible cache, `get_pdb.py` reconstructs atom topology and ordering
from its saved features, so no reference CIF is needed at this stage. The
expected output is:

```text
outputs/6zbh_initial/6ZBH_initial_prediction.cif
outputs/6zbh_initial/6ZBH_initial_prediction_topology.json
```

If an older cache lacks the atom identities needed for template-free export,
provide `--cif_path` with the **matching Protenix sample CIF/PDB** as a topology
template. Never use the deposited evaluation structure for this purpose. With
a template and no explicit output format, the historical initial export is
`6ZBH_initial_prediction.pdb`; add `--output-format cif` to request CIF.
Training still requires the fitted initial CIF from Stage E.

Open this file in a molecular viewer and verify that the topology is not scrambled before proceeding.

## Stage E — One-time rigid-body placement

The initial model must be placed into the experimental coordinate frame before particle refinement. Use the reconstructed density or another experiment-derived frame, not the deposited reference model.

Apply the validated ChimeraX procedure same as [step 4 in CoCoFold](https://github.com/jwliaomath/CoCoFold).

Save the fitted initial model as:

```text
data/6zbh/fitted/6zbh_fitted.cif
```

This fitted model is the optimization-frame topology and placement template supplied to `train.py`.

## Stage F — Particle-guided refinement

### Example configuration with fixed stochasticity

```bash
python -u src/train.py \
  --star_data_dir data/6zbh/particles/366.star \
  --mrc_data_dir data/6zbh/particles/ \
  --output_trained_model_dir outputs/6zbh/checkpoint_ \
  --cif_path data/6zbh/fitted/6zbh_fitted.cif \
  --diffusion_data_dir params/6zbh_diffusion_data.pth \
  --boxsize 288 \
  --apix 1.073 \
  --projection-frame fixed \
  --projection-origin 154.512 154.512 154.512 \
  --batch_size 32 \
  --mini_batch_size 6 \
  --map_resolution 2.146 \
  --transR \
  --update_affine_mat \
  --train_deterministic
```

The origin shown is the geometric center of a 288-pixel map at 1.073 Å/pixel
whose MRC origin and starts are zero and whose axes have the standard order:
`288 × 1.073 / 2 = 154.512 Å`. Use it **only** after checking that this is the
frame of your reconstructed map and fitted CIF. A different map origin, crop,
pixel size or coordinate convention requires a different `--projection-origin`.
See [choosing the projection origin](parameter_guide.md#choosing-the-projection-origin).

The retained defaults are:

- 10 epochs;
- random seed 42;
- latent-bias learning rate `1e-2`;
- atom-weight learning rate `1e-2`;
- Gaussian-width learning rate `5e-3`.

Override them with `--epochs`, `--seed`, `--lr-bias`, `--lr-atom-weights` and `--lr-sdevs`. `--no-learn-gmm` freezes amplitudes and widths together. `--diffusion-seed` overrides the sampling seed; an independent single-GPU `--data-seed` requires `--rng-mode isolated`. The default RNG mode remains legacy.

## Stage G — Outputs

`--output_trained_model_dir` is used as a filename prefix. With:

```text
outputs/6zbh/checkpoint_
```

the current code writes files such as:

```text
outputs/6zbh/checkpoint__.cif
outputs/6zbh/checkpoint_1.cif
outputs/6zbh/checkpoint_1.pth
...
outputs/6zbh/checkpoint_10.cif
outputs/6zbh/checkpoint_10.pth
```

This prefix is different from the inference cache **directory** `params/`
and the initial-export **directory** `outputs/6zbh_initial/`. Invocation
records are written separately; see [outputs and restart](outputs_and_restart.md).

Each epoch checkpoint contains:

- the frozen diffusion-module state;
- optimizer state;
- Gaussian renderer atom weights and widths;
- cached Protenix representations;
- `z_bias`;
- current predicted coordinates and configuration.

CIF is the training default; `--output-format pdb` or `both` is optional. The initial `checkpoint__.cif` is written before optimization. The `.pth` files can be large and should not normally be committed to Git.

## Stage H — Successful-run checks

Before a full GPU run, append `--check-inputs` to the Stage F command to validate
its inputs before model construction. Passing this check does not establish
structural accuracy or guarantee sufficient GPU memory. Use a new output prefix
for a new run; existing training outputs are protected against overwrite.

Inspect the structured records and numbered CIF/checkpoint outputs described in
[outputs and restart](outputs_and_restart.md), in addition to terminal messages.

A successful run should print messages similar to:

```text
The dataset contains N particles.
batch_size ...
mini_batch_size ...
frc_loss ...
penalty ...
peak memory ...
model_path ...
```

Fixed stochasticity is intended to make optimization reproducible within a given environment, but small numerical differences can occur across GPU models, PyTorch/CUDA builds and compiled kernels.

## Portable example scripts

Copy and edit the environment template:

```bash
cp examples/6zbh/env.sh.example examples/6zbh/env.sh
```

Then run:

```bash
bash examples/6zbh/run_inference.sh
source examples/6zbh/env.sh
python src/get_pdb.py \
  --pdbid 6ZBH \
  --diffusion_data_dir "$DIFFUSION_DATA" \
  --out_dir "$OUTPUT_ROOT/6zbh_initial" \
  --output-format cif
# Complete Stage E before the next command.
bash examples/6zbh/run_refinement.sh
```

`run_initial_prediction.sh` is an optional template-based wrapper. If using it,
set `PROTENIX_SAMPLE_CIF` to the matching Protenix output in `env.sh`; that
wrapper keeps the historical PDB export. The direct Stage D command does not
need this variable.

See [records, export and restart](outputs_and_restart.md) for JSONL, captured arguments, synchronized checkpoints and complete-epoch resume. Check inputs before model construction by appending `--check-inputs` to the training command. The active loss does not currently consume the optional half-map weights.
