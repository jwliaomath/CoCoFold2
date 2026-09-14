# 6ZBH example

This directory contains portable wrapper scripts for the research-scale 6ZBH particle-guided CoCoFold2 tutorial. For a smaller first run, start with the [7ZDT/7ZD5 example](../7zdt_7zd5/README.md).

## Before running

1. Read [`../../docs/particle_tutorial_6zbh.md`](../../docs/particle_tutorial_6zbh.md).
2. Replace all placeholder paths in the environment template.
3. Copy the environment template:

   ```bash
   cp examples/6zbh/env.sh.example examples/6zbh/env.sh
   ```

4. Edit every path in `env.sh`.
5. Set `PROTENIX_ROOT_DIR` to the directory containing compatible Protenix 1.0.2 `checkpoint/` and `common/` resources. See [installation](../../docs/installation.md).
6. Complete the external rigid-body placement before running refinement.

## Run

```bash
bash examples/6zbh/run_inference.sh
bash examples/6zbh/run_initial_prediction.sh
bash examples/6zbh/run_refinement.sh
```
