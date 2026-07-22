#!/usr/bin/env bash
# Merge ORAS5 sea-ice thickness files for 2019-01 through 2025-12.
# Run from any directory:
#   bash /fs6/home/daihaijin4/data/myice/data_bak/merge_sithick.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OLD_FILE="${1:-${PROJECT_ROOT}/data/oras5/sithick.nc}"
NEW_DIR="${2:-${PROJECT_ROOT}/data_bak/oras5/sithick}"
DEFAULT_TARGET_GRID="${PROJECT_ROOT}/data/target.nc"
if [[ ! -f "${DEFAULT_TARGET_GRID}" ]]; then
    DEFAULT_TARGET_GRID="${PROJECT_ROOT}/target.nc"
fi
if [[ ! -f "${DEFAULT_TARGET_GRID}" ]]; then
    DEFAULT_TARGET_GRID="${PROJECT_ROOT}/target.txt"
fi
TARGET_GRID="${3:-${DEFAULT_TARGET_GRID}}"
OUTPUT_FILE="${4:-${PROJECT_ROOT}/data/oras5/sithick_1979-2025.nc}"

for command in cdo find sort awk mktemp; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "Error: required command not found: ${command}" >&2
        exit 1
    }
done

for path in "${OLD_FILE}" "${NEW_DIR}" "${TARGET_GRID}"; do
    [[ -e "${path}" ]] || {
        echo "Error: path does not exist: ${path}" >&2
        exit 1
    }
done

if [[ -e "${OUTPUT_FILE}" ]]; then
    echo "Error: output already exists, refusing to overwrite: ${OUTPUT_FILE}" >&2
    exit 1
fi

declare -a NEW_FILES=()
while IFS= read -r file; do
    filename="$(basename "${file}")"
    # Explicitly exclude 2018-12, which is already in the old file.
    if [[ "${filename}" =~ _20(19|20|21|22|23|24|25)(0[1-9]|1[0-2])_ ]]; then
        NEW_FILES+=("${file}")
    fi
done < <(find "${NEW_DIR}" -maxdepth 1 -type f -name 'iicethic*' -print | sort)

if [[ ${#NEW_FILES[@]} -ne 84 ]]; then
    echo "Error: expected 84 monthly files for 2019-01 through 2025-12; found ${#NEW_FILES[@]}." >&2
    printf '  %s\n' "${NEW_FILES[@]}" >&2
    exit 1
fi

OLD_VARIABLE="$(cdo -s showname "${OLD_FILE}" | awk '{print $1}')"
NEW_VARIABLE="$(cdo -s showname "${NEW_FILES[0]}" | awk '{print $1}')"
if [[ -z "${OLD_VARIABLE}" || -z "${NEW_VARIABLE}" ]]; then
    echo "Error: unable to identify the NetCDF variable name." >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
RAW_MERGED="${WORK_DIR}/sithick_201901-202512_raw.nc"
NAMED_MERGED="${WORK_DIR}/sithick_201901-202512_named.nc"
REGRIDDED="${WORK_DIR}/sithick_201901-202512.nc"

echo "Old file:       ${OLD_FILE}"
echo "New directory:  ${NEW_DIR}"
echo "Target grid:    ${TARGET_GRID}"
echo "Output file:    ${OUTPUT_FILE}"
echo "Variables:      old=${OLD_VARIABLE}, new=${NEW_VARIABLE}"

echo "[1/4] Merging 84 monthly source files..."
cdo -O mergetime "${NEW_FILES[@]}" "${RAW_MERGED}"

echo "[2/4] Matching the variable name used by the old file..."
if [[ "${NEW_VARIABLE}" == "${OLD_VARIABLE}" ]]; then
    cp "${RAW_MERGED}" "${NAMED_MERGED}"
else
    cdo -O "chname,${NEW_VARIABLE},${OLD_VARIABLE}" "${RAW_MERGED}" "${NAMED_MERGED}"
fi

echo "[3/4] Regridding new data to the target grid..."
cdo -O "remapbil,${TARGET_GRID}" "${NAMED_MERGED}" "${REGRIDDED}"

echo "[4/4] Merging old and new data..."
cdo -O mergetime "${OLD_FILE}" "${REGRIDDED}" "${OUTPUT_FILE}"

echo
echo "Created: ${OUTPUT_FILE}"
echo "Variable: $(cdo -s showname "${OUTPUT_FILE}")"
echo "Time steps: $(cdo -s ntime "${OUTPUT_FILE}")"
cdo -s showdate "${OUTPUT_FILE}" | awk 'NR == 1 {first = $1} {last = $NF} END {print "Date range: " first " to " last}'
echo "The original file was not changed. Review this output before replacing it."
