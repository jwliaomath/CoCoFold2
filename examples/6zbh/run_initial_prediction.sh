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
: "${DIFFUSION_DATA:?DIFFUSION_DATA is required}"
: "${PROTENIX_SAMPLE_CIF:?PROTENIX_SAMPLE_CIF is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

if [[ "${PROTENIX_SAMPLE_CIF}" == TODO_* ]]; then
  echo "ERROR: replace TODO_PROTENIX_SAMPLE_CIF in env.sh with the generated Protenix CIF path." >&2
  exit 1
fi
[[ -f "${DIFFUSION_DATA}" ]] || { echo "ERROR: missing diffusion cache: ${DIFFUSION_DATA}" >&2; exit 1; }
[[ -f "${PROTENIX_SAMPLE_CIF}" ]] || { echo "ERROR: missing Protenix sample CIF: ${PROTENIX_SAMPLE_CIF}" >&2; exit 1; }
[[ -f "${COCOFOLD2_ROOT}/src/get_pdb.py" ]] || { echo "ERROR: get_pdb.py not found under COCOFOLD2_ROOT" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/6zbh_initial" "${LOG_ROOT}"
cd "${COCOFOLD2_ROOT}"
CMD=(
  python src/get_pdb.py
  --pdbid 6ZBH
  --diffusion_data_dir "${DIFFUSION_DATA}"
  --cif_path "${PROTENIX_SAMPLE_CIF}"
  --out_dir "${OUTPUT_ROOT}/6zbh_initial"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_ROOT}/6zbh_initial_prediction.log"

echo "Initial prediction written under ${OUTPUT_ROOT}/6zbh_initial"
