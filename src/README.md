# Public source entrypoints

Run these from the repository root: `src/inference.py` generates prediction
caches; `src/train.py` refines one complete latent; `src/get_pdb.py` exports
initial or refined structures; `src/prepare_block_alignment.py` prepares rigid
alignment mappings. `src/chain_parallel/train_chain_parallel_2d.py` runs one
component cache per GPU. Use `--help` before loading a model.

Start at the [main README](../README.md), [CLI reference](../docs/cli_reference.md)
and [small example](../examples/7zdt_7zd5/README.md). Public distribution is
defined by `tests/public_files.json`; private research scripts in a development
checkout are not supported public entrypoints.
