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
├── protenix/
│   └── 6zbh_sample_0.cif
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

The expected cache is:

```text
params/6zbh_diffusion_data.pth
```

The cache contains the frozen diffusion-module state and the cached conditional representations used by CoCoFold2, including `s_inputs`, `s_trunk`, cached `pair_z` or `z_trunk`, atom-level caches, the noise schedule and configuration.

Protenix also writes its standard prediction outputs under `outputs/protenix_6zbh/`. Locate the generated sample CIF and set `PROTENIX_SAMPLE_CIF` in `examples/6zbh/env.sh`.

## Stage D — Deterministic initial prediction

Generate the deterministic initial structure from the cache:

```bash
python src/get_pdb.py \
  --pdbid 6ZBH \
  --diffusion_data_dir params/6zbh_diffusion_data.pth \
  --cif_path outputs/protenix_6zbh/SAMPLE_CIF_PATH \
  --out_dir outputs/6zbh_initial
```

`--cif_path` supplies atom topology and ordering from the Protenix output. It must not point to the deposited reference structure. The expected output is:

```text
outputs/6zbh_initial/6ZBH_initial_prediction.pdb
```

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
  --batch_size 32 \
  --mini_batch_size 6 \
  --map_resolution 2.146 \
  --transR \
  --update_affine_mat \
  --train_deterministic
```

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

Each epoch checkpoint contains:

- the frozen diffusion-module state;
- optimizer state;
- Gaussian renderer atom weights and widths;
- cached Protenix representations;
- `z_bias`;
- current predicted coordinates and configuration.

CIF is the training default; `--output-format pdb` or `both` is optional. The initial `checkpoint__.cif` is written before optimization. The `.pth` files can be large and should not normally be committed to Git.

## Stage H — Successful-run checks

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
bash examples/6zbh/run_initial_prediction.sh
# Complete Stage E before the next command.
bash examples/6zbh/run_refinement.sh
```


See [records, export and restart](outputs_and_restart.md) for JSONL, captured arguments, synchronized checkpoints and complete-epoch resume. Check inputs before model construction by appending `--check-inputs` to the training command. The active loss does not currently consume the optional half-map weights.
