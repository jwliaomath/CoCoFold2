# Public tests

Install the [separate CPU dependencies](../docs/installation.md), then run from
the repository root. Each command requires a new result directory:

```bash
python tests/run_public_tests.py --output results/public_cpu
python tests/run_parallel_tests.py --output results/public_gloo
```

The CPU runner copies only `tests/public_files.json` into an isolated directory.
It blocks private/model imports, CUDA initialization and network access. It
records source hashes, environment, each test description/result, totals and
JUnit XML. A pass requires no failed or unexpectedly skipped selected tests.
Four CUDA-only tests are explicitly excluded and named in the summary.

| Level | What is checked | Model use |
|---|---|---|
| T0 | Public inventory, compilation, imports and isolation guards | None |
| T1 | CLI/help and argument errors | None |
| T2 | STAR/MRCS/cache and configuration preflight | Small fixtures, no model weights |
| T3 | Sampler, loss, GMM gradients, alignment, export, records, restart and case audits | Analytic decoder where needed; real public numerical code |
| T4 | Two Gloo processes, summed projections/gradients, rank failure and epoch restart | Analytic decoder, no real Protenix |

To rerun only affected groups, use `--groups`, for example:

```bash
python tests/run_public_tests.py --groups b9 --output results/release_checks
python tests/run_public_tests.py --groups b8 --output results/case_audit_checks
```

Other groups are `public`, `entrypoints`, `gmm`, `b6`, `b6_cli`, `b7b` and
`b7b_cli`. Omitting groups runs the complete public CPU gate. The default
900-second limit covers all pytest subprocesses together; `--timeout` can
increase it for a slower machine. Gloo has its own 300-second timeout.

These results do not establish real-model accuracy. Existing real-weight
acceptance is documented under [validation scope](../docs/release_validation.md).
Run real smoke/refine on GPU compute nodes; CPU tests can run on a CPU node
subject to local cluster policy. Do not repeat passed real-model training after
a report-only or documentation change. A fresh GPU installation is separately
validated. Random/fine-tuning remain outside the current public scope.
