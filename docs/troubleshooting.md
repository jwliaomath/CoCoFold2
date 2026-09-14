# Troubleshooting the CoCoFold2 particle workflow

## `AttributeError: module 'torch.utils' has no attribute 'checkpoint'`

Some PyTorch environments do not expose the checkpoint submodule until it is explicitly imported. The source should include:

```python
import torch.utils.checkpoint
```

in the relevant execution path before `torch.utils.checkpoint.checkpoint` is accessed. The public implementation imports this dependency explicitly. If the error persists, check the traceback/import location to ensure the expected source and environment are in use.

## CUDA out of memory

Reduce `--batch_size` and/or `--mini_batch_size`. Peak memory also increases with:

- sequence and chain length;
- number of generated atoms;
- particle box size;
- diffusion attention/chunk settings;
- cached representation size.

Changing mini-batch size affects throughput and may change legacy loss scaling; it is not guaranteed to be a numerically neutral memory adjustment. Record any changed settings when comparing runtimes.

## Missing STAR columns

`ParticleDataset` validates required fields at startup. Review [Data requirements](data_requirements.md) and confirm that the STAR file has either:

- one pre-RELION-3.1-style image table; or
- both `optics` and `particles` blocks for RELION 3.1+.

Do not rename columns without updating the loader.

## Incorrect `rlnImageName` paths

Absolute image paths in STAR are used directly. Relative paths use the
supplied `--mrc_data_dir`, or the STAR directory if omitted. A trailing slash
is unnecessary. Run `train.py` with your inputs and `--check-inputs` before
launching a long job; it does not construct the diffusion model.

## Particle sign mismatch

The current default is:

```text
--particle_sign -1
```

If observed and rendered particles use the same sign convention rather than opposite conventions, an incorrect sign can prevent FRC optimization. Verify the upstream preprocessing convention and test both signs on a short diagnostic run when uncertain.

## Incorrect orientation convention and `--transR`

`--transR` applies the fixed transformation:

```text
diag(1, 1, -1)
```

to the particle rotation matrices. It is intended to reconcile coordinate/orientation conventions used by the upstream particle metadata and the renderer. Use it only when validated for the dataset. A wrong convention can make otherwise correct structures appear incompatible with the particles.

## Loss does not decrease

Check, in this order:

1. the initial model is in the experimental coordinate frame;
2. particle paths resolve correctly;
3. box size and pixel size match the stack;
4. orientation convention and `--transR` are correct;
5. particle sign is correct;
6. CTF fields and units are correct;
7. the cached diffusion tensors and topology CIF come from the same target and Protenix version;
8. FRC loss and renderer penalties are finite.

Do not use the deposited reference structure to force the initial placement.

## `z_trunk` is `None`

When shared diffusion variables are cached, the cache may store `pair_z` and set `z_trunk` to `None`. In that case `train.py` initializes `z_bias` with the shape of cached `pair_z` and injects the bias there. When `z_trunk` is present, the bias is initialized and injected at `z_trunk` instead.

Do not transfer a bias between these representation levels without explicitly confirming where it was learned.

## Output CIF/PDB looks scrambled

This usually indicates that the topology template and generated coordinate tensor do not use the same atom order. Confirm that:

- the CIF/PDB supplied to `get_pdb.py --cif_path` is the exact Protenix output for the cached target;
- the fitted model supplied to `train.py --cif_path` preserves that atom order;
- no atoms, chains, ligands or residues were added, removed or reordered during rigid fitting.

An equal atom count is necessary but not sufficient; ordering must also match.

## Protenix or CUDA kernel compilation errors

If the traceback ends in `Error building extension 'fast_layer_norm_cuda_v2'`,
read the compiler error above it. In the fresh-installation test, PyTorch's
headers reported `We need GCC 9 or later`: the batch job had selected an old
system compiler because its GNU module was missing. Loading the site's
`gnu/12.2.0` module and setting `CC`/`CXX` resolved this failure. See
[compiler setup](installation.md#prepare-the-compiler-in-the-job-environment).

A preceding `ModuleNotFoundError` for this extension can be the trigger for
its first build; it does not by itself mean that Protenix was not installed.
Successful `pip check`, basic imports or CPU tests do not validate all CUDA
extensions. If smoke fails before validation, absence of `validation.json`
does not mean success: inspect the training log and recorded exception.
After correcting the toolchain, use a new smoke output directory and retain
the failed report. There is no need to reinstall the whole Python environment
merely to select the correct system compiler.

The custom model modules can depend on Triton, cuequivariance and compiled CUDA/C++ extensions. Confirm:

- the NVIDIA driver supports the installed CUDA-enabled PyTorch build;
- a compatible compiler and Ninja are available;
- the PyTorch, Triton and cuequivariance versions match the documented environment;
- the cache directory is writable;
- no extension built under an older PyTorch/CUDA version is being reused.

Deleting stale build caches may help, but record the action and rebuild in a clean environment.

## PyTorch 2.6+ `torch.load` safe-global behavior

Newer PyTorch versions use stricter deserialization defaults. The current `inference.py` adds `argparse.Namespace` to PyTorch safe globals. Only load `.pth` files from trusted sources, and retain `weights_only=False` only where the cached object structure requires it.

## Initial prediction differs across runs

Confirm that the same cached tensor file, model state, diffusion schedule and software environment are used. `get_pdb.py` replays a new checkpoint's saved sampling state unless an explicit seed is supplied; older results use their recorded diffusion seed or fall back to 42. Keep `--seed` and `--rng-mode` consistent. Differences may still arise if compiled kernels or deterministic settings differ across GPU architectures.

## Missing components.cif or model checkpoint

Export `PROTENIX_ROOT_DIR` before Python starts, pointing to the directory containing `common/` and `checkpoint/`. An inference completion report with failed targets does not constitute a completed prediction stage; resolve its error and use a new output directory. See [installation](installation.md).
