# Outputs, records and restart

The three similarly named output arguments have different meanings:
`src/inference.py --output_model_dir` names a **directory** for diffusion
caches, `src/get_pdb.py --out_dir` names a **directory** for exported structures,
and the trainers' `--output_trained_model_dir` names a **filename prefix**.
Inference also writes Protenix predictions and `inference_summary.json` under
its separate `--dump_dir`. None of these paths requires resources in the
source checkout; `--resource-root` points inference to the compatible
`checkpoint/` and `common/` directories.

## Structure files and checkpoints

Single-GPU `--output_trained_model_dir` remains a **filename prefix**. For
`outputs/run/model_`, the initial file is `model__.cif` and epoch 1 writes
`model_1.cif` and `model_1.pth`. The initial file is a diagnostic before
optimization; use the numbered structure for the refined result. Optional
chain alignment also writes an aligned-initial diagnostic. Select
`--output-format pdb` or `both` explicitly if required.

Saving synchronously decodes the updated parameters, preserving and restoring
training RNG. The numbered structure and checkpoint thus represent the same
saved state. This adds a decode at save time, not at each training batch.
Checkpoints include effective latent parameters, renderer state, transforms,
sampling state, configuration, progress and optimizer state when resumable.

Parallel training additionally writes a component structure/checkpoint pair per
rank, a merged CIF and an epoch manifest. A complete manifest is the resume
entrypoint. Chain ID mappings change exported labels, not coordinates. See the
[parallel example](../examples/6zbh_parallel/README.md) for filenames.

## Structured records

Each invocation writes a separate record directory, or the new directory
specified with `--record-dir`. Automatically generated names use Beijing time
and a Slurm job ID when available, otherwise a short random ID; the full UUID
stays in JSON. Non-Slurm commands work normally. Records include:

- `command.json`: captured argument vector, executable and working directory;
  `command.txt` is a POSIX shell reconstruction, not original shell quoting.
- `requested_config.json` and `resolved_config.json`: parsed values/defaults
  and the settings actually resolved from the cache, CLI and restart source.
- `environment.json`, source/input provenance and artifact entries.
- `metrics.jsonl` (or rank-specific JSONL) and a run summary with failures.

Regular log text remains useful for diagnostics but is not the sole record.
Large input identities may use path/size/mtime instead of SHA-256; consult the
record's `identity_method` and `hash_verified` rather than assuming every input
was hashed. `--check-inputs` normally does not create run records.

## Resume versus warm-start

For a new complete-epoch single-GPU checkpoint:

```bash
python src/train.py --resume /path/to/model_4.pth \
  --epochs 10 --output_trained_model_dir /path/to/new_run/model_
```

The checkpoint supplies the inherited inputs and scientific configuration.
Epochs is a cumulative target: four completed epochs and `--epochs 10` means
six more. Resume restores optimizer/progress and epoch-level randomness.
Parallel resume requires the same component grouping and GPU count, and uses
the completed epoch manifest. Changing scientific parameters requires a new
warm-start, not a silent override of resume configuration.

Use `--warm-start CHECKPOINT` with the normal required training inputs for an
old result or a deliberately new experiment. Its effective latent and existing
GMM/R/t are inherited; absent GMM/transform state is rebuilt using the current
configuration and aligned CIF. Optimizers and progress start from zero. The
source and reconstruction decisions are recorded.

OOM and other exceptions do not trigger emergency saves. An explicitly requested
`--max_steps` stop in mid-epoch retains an exportable/warm-start result but
cannot resume from that partial epoch. Use the preceding complete checkpoint.

## Export an existing result

```bash
python src/get_pdb.py --pdbid TARGET --diffusion_data_dir /path/to/model_4.pth \
  --out_dir /path/to/new_export --output-format cif
```

Compatible caches supply atom topology without `--cif_path`. Otherwise provide
a topology template corresponding to the cached atom identities/order.
Refinement exports default to the saved experimental reference frame, applying
saved latent parameters and R/t. Initial prediction caches remain in raw model
coordinates; supplying a CIF does not silently fit them to it.

Very old files without rotation/translation support explicit
`--coordinate-frame raw`; a reference-frame export cannot reconstruct missing
historical transforms. A new checkpoint normally replays its saved export
sampling state. An explicit `--seed` requests that fixed diffusion sample;
otherwise the saved diffusion seed is used where available, falling back to
42 for old files. This is not a bitwise reproducibility guarantee across
different hardware, CUDA kernels or software versions.
