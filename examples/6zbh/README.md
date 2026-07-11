# 6ZBH example

This directory contains portable wrapper scripts and a human-readable manifest for the research-scale 6ZBH particle-guided CoCoFold2 tutorial.

## Before running

1. Read [`../../docs/particle_tutorial_6zbh.md`](../../docs/particle_tutorial_6zbh.md).
2. Replace all `TODO_*` placeholders in the documentation and manifest.
3. Copy the environment template:

   ```bash
   cp examples/6zbh/env.sh.example examples/6zbh/env.sh
   ```

4. Edit every path in `env.sh`.
5. Ensure that compatible Protenix 1.0.2 `checkpoint/` and `common/` resources are available at the repository root.
6. Complete the external rigid-body placement before running refinement.

## Run

```bash
bash examples/6zbh/run_inference.sh
bash examples/6zbh/run_initial_prediction.sh
bash examples/6zbh/run_refinement.sh
```
