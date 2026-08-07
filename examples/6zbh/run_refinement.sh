#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} is missing. Copy env.sh.example to env.sh and edit it." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${COCOFOLD2_ROOT:?COCOFOLD2_ROOT is required}"
: "${STAR_FILE:?STAR_FILE is required}"
: "${MRC_ROOT:?MRC_ROOT is required}"
: "${FITTED_INITIAL_CIF:?FITTED_INITIAL_CIF is required}"
: "${DIFFUSION_DATA:?DIFFUSION_DATA is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

[[ -f "${STAR_FILE}" ]] || { echo "ERROR: missing STAR file: ${STAR_FILE}" >&2; exit 1; }
[[ -d "${MRC_ROOT}" ]] || { echo "ERROR: missing particle root directory: ${MRC_ROOT}" >&2; exit 1; }
[[ -f "${FITTED_INITIAL_CIF}" ]] || { echo "ERROR: missing fitted initial CIF: ${FITTED_INITIAL_CIF}. Complete the external rigid-body fit first." >&2; exit 1; }
[[ -f "${DIFFUSION_DATA}" ]] || { echo "ERROR: missing diffusion cache: ${DIFFUSION_DATA}" >&2; exit 1; }
[[ -f "${COCOFOLD2_ROOT}/src/train.py" ]] || { echo "ERROR: src/train.py not found under COCOFOLD2_ROOT" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/6zbh" "${LOG_ROOT}"
MRC_WITH_SLASH="${MRC_ROOT%/}/"
OUTPUT_PREFIX="${OUTPUT_ROOT}/6zbh/checkpoint_"

TRANS_FLAGS=()
if [[ "${USE_TRANS_R:-0}" == "1" ]]; then
  TRANS_FLAGS+=(--transR)
fi

AFFINE_FLAGS=()
if [[ "${USE_UPDATE_AFFINE_MAT:-0}" == "1" ]]; then
  AFFINE_FLAGS+=(--update_affine_mat)
fi

cd "${COCOFOLD2_ROOT}"
CMD=(
  python -u src/train.py
  --star_data_dir "${STAR_FILE}"
  --mrc_data_dir "${MRC_WITH_SLASH}"
  --output_trained_model_dir "${OUTPUT_PREFIX}"
  --cif_path "${FITTED_INITIAL_CIF}"
  --diffusion_data_dir "${DIFFUSION_DATA}"
  --boxsize 288
  --apix 1.073
  --batch_size 32
  --mini_batch_size 6
  --map_resolution 2.146
  "${TRANS_FLAGS[@]}"
  "${AFFINE_FLAGS[@]}"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_ROOT}/6zbh_refinement.log"

echo "Refinement completed. Output prefix: ${OUTPUT_PREFIX}"
