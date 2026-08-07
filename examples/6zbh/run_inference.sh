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
: "${INPUT_JSON:?INPUT_JSON is required}"
: "${PARAMS_ROOT:?PARAMS_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

[[ -f "${INPUT_JSON}" ]] || { echo "ERROR: missing INPUT_JSON: ${INPUT_JSON}" >&2; exit 1; }
[[ -f "${COCOFOLD2_ROOT}/src/inference.py" ]] || { echo "ERROR: inference.py not found under COCOFOLD2_ROOT" >&2; exit 1; }
[[ -d "${COCOFOLD2_ROOT}/checkpoint" ]] || { echo "ERROR: Protenix checkpoint/ directory is missing" >&2; exit 1; }
[[ -d "${COCOFOLD2_ROOT}/common" ]] || { echo "ERROR: Protenix common/ directory is missing" >&2; exit 1; }

mkdir -p "${PARAMS_ROOT}" "${OUTPUT_ROOT}/protenix_6zbh" "${LOG_ROOT}"
PARAMS_WITH_SLASH="${PARAMS_ROOT%/}/"

cd "${COCOFOLD2_ROOT}"
CMD=(
  python -u src/inference.py
  --input_json_path "${INPUT_JSON}"
  --sample_name 6zbh
  --output_model_dir "${PARAMS_WITH_SLASH}"
  --dump_dir "${OUTPUT_ROOT}/protenix_6zbh"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_ROOT}/6zbh_inference.log"

EXPECTED_CACHE="${PARAMS_WITH_SLASH}6zbh_diffusion_data.pth"
[[ -f "${EXPECTED_CACHE}" ]] || { echo "ERROR: expected cache not created: ${EXPECTED_CACHE}" >&2; exit 1; }
echo "Inference and cache generation completed: ${EXPECTED_CACHE}"
