# Small 7ZDT/7ZD5 example

This map-derived example uses the 7ZDT sequence/MSA to generate a Protenix-v1
cache and the complete original 7ZD5 CIF to generate supervision. It does not
trim the CIF to common residues or simulate heterogeneous particles. The
author accepted smoke and refine in the existing server environment.

Run commands from the repository root after [installing dependencies](../../docs/installation.md).
Choose new output directories; prediction caches and training results refuse
overwrite. The sequence/MSA JSON, weights and common resources are user inputs.
The two original CIFs are under `examples/7zdt_7zd5/inputs/`.

## 1. Generate the map locally

Create a JSON configuration, replacing the paths:

```json
{
  "target_cif": "/absolute/path/to/CoCoFold2/examples/7zdt_7zd5/inputs/7ZD5.cif",
  "output_dir": "/absolute/path/to/new_map",
  "boxsize": 192,
  "apix": 1,
  "resolution": 3
}
```

Set `COCOFOLD2_MOLMAP_CONFIG` to this JSON and run
`examples/7zdt_7zd5/make_molmap.py` inside ChimeraX:

```bash
export COCOFOLD2_MOLMAP_CONFIG=/absolute/path/to/molmap_config.json
ChimeraX --nogui --exit --script /absolute/path/to/CoCoFold2/examples/7zdt_7zd5/make_molmap.py
```

Use your installed ChimeraX executable path. The accepted map was generated
with ChimeraX 1.7.1. The script uses all atoms loaded from the CIF, translates
their electron-weighted center to zero, and creates a 192³ grid with origin
(-96, -96, -96) Å. `molmap.json` records the exact translation, source/map hashes
and atom count. It does not modify the input CIF. Transfer the generated
`7zd5_full_3A.mrc` and `molmap.json` together if training on a server.

## 2. Simulate particles

```bash
python src/simulate_particles.py --map /path/to/new_map/7zd5_full_3A.mrc \
  --out-dir results/small_particles --n-particles 1000 --snr 1 --apix 1 --seed 42
```

The output contains a STAR file, noisy and clean CTF particle stacks, simulation
parameters and readback checks. The two stacks together occupy about 281 MiB.
STAR paths are relative to the simulation directory. This step needs no model.

## 3. Predict, then place the initial model

```bash
export PROTENIX_ROOT_DIR=/absolute/path/to/protenix_resources
python examples/7zdt_7zd5/run_case.py predict \
  --input-json /path/to/7zdt.json --resource-root "$PROTENIX_ROOT_DIR" \
  --output results/small_prediction --seed 42
```

The resources directory contains `common/` and `checkpoint/`. Prediction retains
the upstream seed 101; `--seed 42` here selects the initial export/refinement
diffusion seed. The stage ends with an initial CIF and a completion report.
Use this same successful prediction directory in both training commands.

Fit `results/small_prediction/initial/7zdt_initial_prediction.cif` to the
**generated map**, then save only the fitted atomic model as `/path/to/placed.cif`.
Preserve all atom identities and order. Do not fit the prediction to a deposited
truth structure instead. The deposited 7ZDT CIF is not automatically a compatible
training reference for a newly generated cache.

An independently prepared reference with matching topology but a different
conformation is allowed via `--reference-structure`; its map placement and atom
identity/order remain the user's responsibility. Do not add this flag merely
to bypass a failed rigid-movement check without understanding the difference.

## 4. Smoke, then refine

```bash
python examples/7zdt_7zd5/run_case.py train --kind smoke \
  --prediction results/small_prediction --particles results/small_particles \
  --placed-cif /path/to/placed.cif --output results/small_smoke
python examples/7zdt_7zd5/run_case.py train --kind refine \
  --prediction results/small_prediction --particles results/small_particles \
  --placed-cif /path/to/placed.cif --smoke-result results/small_smoke \
  --output results/small_refine
```

Smoke uses the first 32 particles for two epochs/two optimizer steps. Refine
uses all 1000 for ten epochs/320 steps. Both use batch=32, mini-batch=16, seed=42,
fixed stochasticity and explicitly frozen GMM amplitudes/widths. Keep these
settings and the inputs identical between smoke and refine. The generic trainer
defaults are unchanged by this example. Slurm is optional; submit the same
commands on a suitable GPU compute node if required by your cluster.

Inspect `validation.json`/`validation.txt` in each output directory. Checks cover
finite values, progress, latent updates, exact GMM freezing, output consistency
and records without repeating diffusion decoding. The reported eight-particle
FRC comparison uses training views, not held-out validation. If only the checker
fails, inspect the error and recheck the existing results before retraining.

## 5. Inspect structural quality

Inspect the final CIF in the map. A Cα RMSD within 2 Å is a manual inspection
target, not an automated condition. 7ZD5 and the modeled 7ZDT input can differ
in sequence and atoms; when using an external alignment tool, report chain
mapping, matched residues and coverage along with RMSD. Automated success alone
does not establish structural accuracy. See [detailed test scope](RUN_TESTS.md).
