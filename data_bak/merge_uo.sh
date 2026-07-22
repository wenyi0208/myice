#!/usr/bin/env bash
# Merge ORAS5 uo data for 2019-01 through 2025-12.
#
# Run from any directory:
#   bash /fs6/home/daihaijin4/data/myice/data_bak/merge_uo.sh
#
# Optional positional arguments:
#   1: directory holding the 2019-2025 raw vomecrty monthly files
#   2: target grid file (target.nc or target.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEW_DIR="${1:-${PROJECT_ROOT}/data_bak/oras5/uome}"
DEFAULT_TARGET_GRID="${PROJECT_ROOT}/data/target.nc"
if [[ ! -f "${DEFAULT_TARGET_GRID}" ]]; then
    DEFAULT_TARGET_GRID="${PROJECT_ROOT}/target.nc"
fi
if [[ ! -f "${DEFAULT_TARGET_GRID}" ]]; then
    DEFAULT_TARGET_GRID="${PROJECT_ROOT}/target.txt"
fi
TARGET_GRID="${2:-${DEFAULT_TARGET_GRID}}"

OLD_UO_0="${PROJECT_ROOT}/data/oras5/uo_0.nc"
OLD_UO_10="${PROJECT_ROOT}/data/oras5/uo_10.nc"
OUT_UO_0="${PROJECT_ROOT}/data/oras5/uo_0_1979-2025.nc"
OUT_UO_10="${PROJECT_ROOT}/data/oras5/uo_10_1979-2025.nc"

# The raw ORAS5 depth levels nearest to 0 m and 10 m, respectively.
DEPTH_0="0.505760014"
DEPTH_10="9.82275009"

for command in cdo find sort awk basename mktemp; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "Error: required command not found: ${command}" >&2
        exit 1
    }
done

for path in "${NEW_DIR}" "${TARGET_GRID}" "${OLD_UO_0}" "${OLD_UO_10}"; do
    [[ -e "${path}" ]] || {
        echo "Error: path does not exist: ${path}" >&2
        exit 1
    }
done

for output in "${OUT_UO_0}" "${OUT_UO_10}"; do
    if [[ -e "${output}" ]]; then
        echo "Error: output already exists, refusing to overwrite: ${output}" >&2
        exit 1
    fi
done

declare -a NEW_FILES=()
while IFS= read -r file; do
    filename="$(basename "${file}")"
    # Explicitly exclude 2018-12, which is already in the old files.
    if [[ "${filename}" =~ _20(19|20|21|22|23|24|25)(0[1-9]|1[0-2])_ ]]; then
        NEW_FILES+=("${file}")
    fi
done < <(find "${NEW_DIR}" -maxdepth 1 -type f -name 'vomecrty*' -print | sort)

if [[ ${#NEW_FILES[@]} -ne 84 ]]; then
    echo "Error: expected 84 monthly files for 2019-01 through 2025-12; found ${#NEW_FILES[@]}." >&2
    printf '  %s\n' "${NEW_FILES[@]}" >&2
    exit 1
fi

single_variable_name() {
    local file="$1"
    local names
    names="$(cdo -s showname "${file}")"
    local -a name_array=()
    read -r -a name_array <<< "${names}"
    if [[ ${#name_array[@]} -ne 1 ]]; then
        echo "Error: expected exactly one data variable in ${file}; found: ${names}" >&2
        exit 1
    fi
    printf '%s\n' "${name_array[0]}"
}

OLD_VAR_0="$(single_variable_name "${OLD_UO_0}")"
OLD_VAR_10="$(single_variable_name "${OLD_UO_10}")"
NEW_VAR="$(single_variable_name "${NEW_FILES[0]}")"

if [[ "${OLD_VAR_0}" != "${OLD_VAR_10}" ]]; then
    echo "Error: old uo files use different variable names: ${OLD_VAR_0} and ${OLD_VAR_10}" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

process_depth() {
    local label="$1"
    local depth="$2"
    local old_file="$3"
    local output_file="$4"
    local target_variable="$5"
    local level_dir="${WORK_DIR}/uo_${label}_monthly"
    local merged_file="${WORK_DIR}/uo_${label}_201901-202512_merged.nc"
    local named_file="${WORK_DIR}/uo_${label}_201901-202512_named.nc"
    local regridded_file="${WORK_DIR}/uo_${label}_201901-202512_regridded.nc"
    local -a selected_files=()
    local index=0

    mkdir -p "${level_dir}"
    echo "Extracting ${label} m target layer (raw depth ${depth} m)..."
    for input_file in "${NEW_FILES[@]}"; do
        local selected_file
        printf -v selected_file "%s/%03d.nc" "${level_dir}" "${index}"
        # Select the raw level, then reset its coordinate to level=0 to match
        # the existing uo_0.nc and uo_10.nc files.
        cdo -O -setlevel,0 -sellevel,"${depth}" "${input_file}" "${selected_file}"
        selected_files+=("${selected_file}")
        ((index += 1))
    done

    echo "Merging extracted ${label} m monthly files..."
    cdo -O mergetime "${selected_files[@]}" "${merged_file}"

    if [[ "${NEW_VAR}" == "${target_variable}" ]]; then
        named_file="${merged_file}"
    else
        echo "Renaming ${NEW_VAR} to ${target_variable}..."
        cdo -O "chname,${NEW_VAR},${target_variable}" "${merged_file}" "${named_file}"
    fi

    echo "Regridding ${label} m data to ${TARGET_GRID}..."
    cdo -O "remapbil,${TARGET_GRID}" "${named_file}" "${regridded_file}"

    echo "Merging existing and new ${label} m data..."
    cdo -O mergetime "${old_file}" "${regridded_file}" "${output_file}"
}

echo "New directory:  ${NEW_DIR}"
echo "Target grid:    ${TARGET_GRID}"
echo "Output uo_0:    ${OUT_UO_0}"
echo "Output uo_10:   ${OUT_UO_10}"
echo "Variables:      old=${OLD_VAR_0}, new=${NEW_VAR}"
echo

process_depth "0" "${DEPTH_0}" "${OLD_UO_0}" "${OUT_UO_0}" "${OLD_VAR_0}"
process_depth "10" "${DEPTH_10}" "${OLD_UO_10}" "${OUT_UO_10}" "${OLD_VAR_10}"

for output in "${OUT_UO_0}" "${OUT_UO_10}"; do
    echo
    echo "Created: ${output}"
    echo "Variable: $(cdo -s showname "${output}")"
    echo "Time steps: $(cdo -s ntime "${output}")"
    cdo -s showdate "${output}" |
        awk 'NR == 1 {first = $1} {last = $NF} END {print "Date range: " first " to " last}'
done

echo
echo "The existing uo_0.nc and uo_10.nc files were not changed."
