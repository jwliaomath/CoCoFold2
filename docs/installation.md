# Installation and environment validation

CoCoFold2 runs directly from the repository root; it has no editable-install
package. The full model workflow targets Linux, Python 3.11 and Protenix-v1
(`protenix==1.0.2`). Model weights, common data and MSA resources are separate
inputs, not included in the public source archive.

## Software requirements

The full GPU environment uses Python 3.11, PyTorch 2.7.1 with CUDA 12.6,
Protenix 1.0.2 and NumPy 2.4.1. Install the versions specified in
`environment.yml` or `requirements.txt`. These core versions agree with
[Protenix v1.0.2's requirements](https://github.com/bytedance/Protenix/blob/v1.0.2/requirements.txt).

Use a dedicated environment to avoid changing dependencies in other projects.
The dependency files pin direct packages, but do not fully lock transitive
dependencies or compiled extensions. GPU execution also requires an NVIDIA
driver, a CUDA toolkit and a compatible C++ compiler. GNU 12.2.0 was used for
the validated Linux installation; see [compiler setup](#prepare-the-compiler-in-the-job-environment).

After installation, run the CPU checks and a short GPU smoke test below to
verify your environment. See [validation scope](release_validation.md) for
the workflows and configurations that have been tested.

## CPU tests without Protenix or weights

Create a separate environment, then run from the repository root:

```bash
conda create -n cocofold2_cpu python=3.11 pip -y
conda activate cocofold2_cpu
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python -m pip check
python tests/run_public_tests.py --output results/cpu_check
python tests/run_parallel_tests.py --output results/gloo_check
```

The PyTorch command follows the official
[2.7.1 CPU installation instructions](https://pytorch.org/get-started/previous-versions/#v271).
The remaining CPU dependencies use the public workflow's existing package
versions; pytest supports versions 7–9. The setuptools pin retains
`pkg_resources`, used by starfile 0.4.12. The test runner blocks model imports,
CUDA initialization and network calls for the T0–T3 gate. Gloo is a separate
two-process CPU check with an analytic decoder. Neither check validates weights.

## Create a full GPU environment

Create and activate a dedicated environment:

```bash
conda env create --name cocofold2_gpu --file environment.yml
conda activate cocofold2_gpu
python -m pip install 'pytest>=7,<10'
export PROTENIX_ROOT_DIR=/absolute/path/to/protenix_resources
# On a GPU compute node, prepare the compiler as described below before imports.
python tools/inspect_environment.py --model-imports --output results/new_environment.json
python tests/run_public_tests.py --output results/new_environment_cpu
```

With a shared system Mamba installation, choose a writable prefix instead of
placing the environment in the system's `envs/` directory. For example:

```bash
export COCOFOLD2_ENV="$HOME/conda-envs/cocofold2"
mamba env create --prefix "$COCOFOLD2_ENV" --file environment.yml
# Initialize the appropriate Conda shell first if required by your site.
conda activate "$COCOFOLD2_ENV"
python -m pip install 'pytest>=7,<10'
python -m pip check
python -c "import sys; print(sys.executable)"
```

Use the same absolute prefix in batch jobs. An environment created with
`--prefix` may not be discoverable by its final directory name when a different
Conda/Mamba installation is active. Do not recreate it solely because activation
by name fails. Mamba can run its pip installation quietly; wait for the command
to finish and check its exit status and `pip check` before starting validation.

### Prepare the compiler in the job environment

The first real model import can compile `fast_layer_norm_cuda_v2`. Package
installation and `pip check` do not prove that this extension can build.
This PyTorch version requires GCC 9 or newer for compilation. A compiler must also
be compatible with the installed CUDA toolkit; GCC >=9 alone is not a complete
compatibility check. On a cluster providing a `gnu/12.2.0` module, add the
following **after environment activation and before Python**:

```bash
# Site-specific example: replace the module name with your cluster's equivalent.
module load gnu/12.2.0
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
gcc --version
g++ --version
# Use the actual CUDA toolkit path on your system.
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
```

Keep the module setup in the batch script, not only in the login shell. On a
system without environment modules, select the installed compatible GCC/G++
and CUDA toolkit using that system's configuration. Retain the compiler and
nvcc version output with the run logs. Compilation may take time on the first
run; a successful build must be followed by an actual smoke run.

`PROTENIX_ROOT_DIR` must contain `common/` (including `components.cif`) and
`checkpoint/`. Set it **before starting Python**; changing only the shell
working directory does not reliably select those resources. MSA paths in the
prediction JSON must also resolve. The existing model/resource files can be
reused without copying them into the source tree.

When invoking `src/inference.py` directly with resources outside the repository,
also pass `--resource-root "$PROTENIX_ROOT_DIR"`. The environment variable
selects upstream module resources; the CLI flag selects CoCoFold2's explicit
checkpoint/data configuration. The example wrappers provide both.

Follow the cluster's rules for installation and compilation. Run model imports,
GPU checks and actual prediction/refinement on a compute node. The inspector
records installed packages, import errors, the Protenix import location and
visible GPUs without loading weights. It imports inference data dependencies,
not every diffusion module or compiled kernel. Check that the import location belongs
to the intended new installation rather than an old checkout on `PYTHONPATH`.

Verify model execution with a short real-weight smoke run using the
[small example](../examples/7zdt_7zd5/README.md) and a new output directory.
Keep its report alongside the environment and CPU test reports. A long
refinement is not needed to check installation. If dependency resolution,
extension compilation or GPU execution fails, inspect the error and
[troubleshooting guide](troubleshooting.md) before changing package versions.

No model weights, source-changing monkey patches or guessed resource download
commands are supplied by the environment inspector.
