# 6ZBH example

This directory contains portable wrapper scripts for the research-scale 6ZBH particle-guided CoCoFold2 tutorial. For a smaller first run, start with the [7ZDT/7ZD5 example](../7zdt_7zd5/README.md).

## Before running

1. Read [`../../docs/particle_tutorial_6zbh.md`](../../docs/particle_tutorial_6zbh.md).
2. Replace the paths needed by the wrapper scripts you use in the environment template.
3. Copy the environment template:

   ```bash
   cp examples/6zbh/env.sh.example examples/6zbh/env.sh
   ```

4. Edit the relevant paths in `env.sh`.
   Check and edit the three `PROJECTION_ORIGIN_*` values to match the physical
   map/CIF frame; the template's 154.512 Å applies only to the documented
   zero-origin 288-pixel map at 1.073 Å/pixel.
5. Set `PROTENIX_ROOT_DIR` to the directory containing compatible Protenix 1.0.2 `checkpoint/` and `common/` resources. See [installation](../../docs/installation.md).
6. Complete the external rigid-body placement before running refinement.

## Run

```bash
bash examples/6zbh/run_inference.sh
source examples/6zbh/env.sh
python src/get_pdb.py \
  --pdbid 6ZBH \
  --diffusion_data_dir "$DIFFUSION_DATA" \
  --out_dir "$OUTPUT_ROOT/6zbh_initial" \
  --output-format cif
# Rigidly fit the exported initial CIF before refinement.
bash examples/6zbh/run_refinement.sh
```

The optional `run_initial_prediction.sh` wrapper still requires
`PROTENIX_SAMPLE_CIF` from the matching Protenix output and retains its
historical PDB export. Direct `get_pdb.py` can instead reconstruct topology
from a compatible cache without that template; see
[the single-GPU tutorial](../../docs/particle_tutorial_6zbh.md).
In both cases, refinement requires a separately fitted initial CIF.
