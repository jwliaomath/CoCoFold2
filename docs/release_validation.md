# Validation scope and release checklist

## Recorded evidence (2026-09-14)

The author reported successful real Protenix-v1 tests in the existing server
environment. These reports are accepted as author-provided evidence; they are
not a claim that CI reproduced them. A separate fresh-installation acceptance
is recorded below; it does not repeat every earlier model test.

| Scope | Evidence | Limit |
|---|---|---|
| Single structure, inputs, seeds, records, export and restart | B1–B5 accepted by the author; legacy output exceptions recorded during development | Legacy files missing R/t require explicit raw export; no bitwise cross-GPU claim |
| Small single-state case | B6 smoke and 1000-particle 7ZDT/7ZD5 refine accepted; final structure acceptable to author | Map-derived supervision, SNR=1; RMSD is manual, not an automatic pass threshold |
| Contextual 6ZBH 1+3 short two-GPU test | B7b real Protenix and weights: 39/39 checks passed | Short continuation/resume and fault checks, not a convergence benchmark |
| Contextual 6ZBH, full 3324-particle STAR | B8 first four complete epochs: 69/69 checks passed, 416 steps | Original target was ten; later epochs and whole-job completion are outside acceptance |
| Fresh Linux installation | B9 environment report and full public CPU gate passed, as reported by the author; real-weight 7ZDT/7ZD5 smoke report: 10/10 checks passed | Existing B6 inputs reused; no new prediction, long refinement or two-GPU test in this new environment |

The first fresh-environment smoke attempt failed while compiling
`fast_layer_norm_cuda_v2` because the selected system GCC was too old. The author
loaded `gnu/12.2.0`, selected GCC/G++ through `CC`/`CXX`, and reported a successful
retry. The earlier CPU pass is retained separately; it was not rerun for the
compiler-only retry. See [compiler setup](installation.md#prepare-the-compiler-in-the-job-environment).

The supplied smoke report confirms finite and updated latent parameters,
frozen GMM amplitudes/widths, optimizer progress, complete-epoch checkpoint,
matching CIF coordinates/identities and complete records. Its runtime fields
report elapsed time 269.49 s, train-step time 9.36 s and peak memory
17,732,314,624 bytes. These are case-specific report values, not hardware
requirements or a performance comparison. The fixed-particle mean FRC decreased
from 0.28019 to 0.14819; this is report-only and provides no evidence of quality
improvement in this short smoke run. Structural RMSD remains a manual check.

For B8 the author reported a structure within 2 Å after the third epoch. This
is a manual structural assessment, not an automatically measured release result.
The four-epoch audit reported mean distributed step time 4.10223 s and summed
step time 1706.53 s on two NVIDIA A100-SXM4-80GB GPUs. A distributed step takes
the maximum rank duration. These times exclude initialization, epoch export
and file I/O; they are not whole-job wall time. Recorded peak allocated memory
was 47965.59/59394.84 MiB for ranks 0/1; reserved memory was 78834/78780 MiB.
These measurements describe this case and environment, not minimum hardware
requirements or a performance guarantee.

Independent and Contextual paths share public controls but have different
conditioning. A real long Independent run has not been accepted here. Random,
fine-tuning and the main heterogeneity experiments are outside this release.

## Remaining release gates

- Retain the accepted server CPU report with its source hashes. For subsequent
  changes, run the affected checks on the independent candidate; documentation
  edits do not require repeating accepted real-weight training. CPU substitute
  decoders must remain labelled.
- Run the new CI workflow after pushing the reviewed public repository; local
  tests do not establish a successful GitHub Actions run.
- Preserve the accepted fresh-installation reports and the earlier working
  environment. Each new platform still needs its own installation validation;
  see [installation](installation.md).
- Author reviews the final public file list, documentation, citation metadata,
  license notices and GitHub diff. The planned release version is v1.0.0;
  confirm the release after CI passes. No manuscript DOI is available, so it is
  omitted from the citation metadata.
- Only after that review, publish a version and separately agree on a project
  page and documentation site. This candidate does not publish either.

## Build the public candidate

```bash
python tools/build_public_release.py --output /path/to/new_delivery
```

The new directory contains `CoCoFold2/`, `CoCoFold2-public.zip` and a SHA-256
`manifest.json`. The builder copies only `tests/public_files.json`, verifies
the archive bytes and refuses an existing destination. It does not use Git,
upload anything or include local results, weights or private experiments.

The root `.gitignore` follows the same explicit public inventory. Update both
when intentionally adding a public file. An ignore rule does not remove files
already tracked by Git or erase private data in existing history. Review and
publish the independent candidate; do not push the research working directory
on the assumption that `.gitignore` alone makes it public-safe.
