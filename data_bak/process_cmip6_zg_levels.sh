#!/usr/bin/env bash
# Extract and regrid seven CMIP6 atmospheric geopotential-height levels.
#
# Default usage from any directory:
#   bash /fs6/home/daihaijin4/data/myice/data_bak/process_cmip6_zg_levels.sh
#
# Optional positional arguments:
#   1: source directory containing zg_Amon_*.nc
#   2: target grid NetCDF/text file
#   3: CMIP6 output root containing one directory per model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_DIR="${1:-${PROJECT_ROOT}/data_bak/cmip6/zg}"
TARGET_GRID="${2:-${PROJECT_ROOT}/data/target.nc}"
OUTPUT_ROOT="${3:-${PROJECT_ROOT}/data/cmip6}"

MODELS=(
    "CIESM"
    "CMCC-CM2-SR5"
    "CMCC-ESM2"
    "E3SM-1-0"
    "E3SM-1-1"
    "EC-Earth3"
    "EC-Earth3-Veg"
    "HadGEM3-GC31-MM"
    "MPI-ESM1-2-HR"
    "TaiESM1"
)

declare -A MODEL_MEMBER=(
    ["CIESM"]="r1i1p1f1"
    ["CMCC-CM2-SR5"]="r1i1p1f1"
    ["CMCC-ESM2"]="r1i1p1f1"
    ["E3SM-1-0"]="r1i1p1f1"
    ["E3SM-1-1"]="r1i1p1f1"
    ["EC-Earth3"]="r1i1p1f1"
    ["EC-Earth3-Veg"]="r1i1p1f1"
    ["HadGEM3-GC31-MM"]="r1i1p1f3"
    ["MPI-ESM1-2-HR"]="r1i1p1f1"
    ["TaiESM1"]="r1i1p1f1"
)

declare -A MODEL_GRID=(
    ["CIESM"]="gr"
    ["CMCC-CM2-SR5"]="gn"
    ["CMCC-ESM2"]="gn"
    ["E3SM-1-0"]="gr"
    ["E3SM-1-1"]="gr"
    ["EC-Earth3"]="gr"
    ["EC-Earth3-Veg"]="gr"
    ["HadGEM3-GC31-MM"]="gn"
    ["MPI-ESM1-2-HR"]="gn"
    ["TaiESM1"]="gn"
)

LEVELS=("92500" "85000" "50000" "30000" "10000" "5000" "1000")
OUTPUT_NAMES=(
    "zg_925.nc"
    "zg_850.nc"
    "zg_500.nc"
    "zg_300.nc"
    "zg_100.nc"
    "zg_50.nc"
    "zg_10.nc"
)
LEVEL_CSV="92500,85000,50000,30000,10000,5000,1000"

EXPECTED_FIRST_MONTH="185001"
EXPECTED_LAST_MONTH="201412"
EXPECTED_FIRST_DATE="1850-01-16"
EXPECTED_LAST_DATE="2014-12-16"
EXPECTED_NTIME="1980"
EXPECTED_XSIZE="360"
EXPECTED_YSIZE="151"

for required_command in cdo find sort awk basename mktemp mkdir mv rm tr wc; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "Error: required command not found: ${required_command}" >&2
        exit 1
    }
done

for required_path in "${SOURCE_DIR}" "${TARGET_GRID}" "${OUTPUT_ROOT}"; do
    [[ -e "${required_path}" ]] || {
        echo "Error: path does not exist: ${required_path}" >&2
        exit 1
    }
done

next_month() {
    local value="$1"
    local year=$((10#${value:0:4}))
    local month=$((10#${value:4:2}))
    month=$((month + 1))
    if ((month == 13)); then
        month=1
        year=$((year + 1))
    fi
    printf '%04d%02d\n' "${year}" "${month}"
}

contains_word() {
    local words="$1"
    local expected="$2"
    local word
    for word in ${words}; do
        [[ "${word}" == "${expected}" ]] && return 0
    done
    return 1
}

date_range() {
    local file="$1"
    cdo -s showdate "${file}" |
        awk 'NR == 1 {first = $1} {last = $NF} END {print first, last}'
}

declare -A MODEL_FILES_SERIALIZED=()
TOTAL_SELECTED_FILES=0

echo "Preflight checks"
echo "  Source: ${SOURCE_DIR}"
echo "  Target: ${TARGET_GRID}"
echo "  Output: ${OUTPUT_ROOT}"
echo

target_grid_description="$(cdo -s griddes "${TARGET_GRID}")"
target_xsize="$(awk '$1 == "xsize" {print $3; exit}' <<< "${target_grid_description}")"
target_ysize="$(awk '$1 == "ysize" {print $3; exit}' <<< "${target_grid_description}")"
if [[ "${target_xsize}" != "${EXPECTED_XSIZE}" || "${target_ysize}" != "${EXPECTED_YSIZE}" ]]; then
    echo "Error: target grid is ${target_xsize}x${target_ysize}; expected ${EXPECTED_XSIZE}x${EXPECTED_YSIZE}" >&2
    exit 1
fi

# Validate every model, source interval, pressure coordinate, and destination
# before starting the expensive regridding work.
for model in "${MODELS[@]}"; do
    member="${MODEL_MEMBER[${model}]}"
    grid="${MODEL_GRID[${model}]}"
    expected_prefix="zg_Amon_${model}_historical_${member}_${grid}_"
    output_dir="${OUTPUT_ROOT}/${model}"

    [[ -d "${output_dir}" ]] || {
        echo "Error: output model directory does not exist: ${output_dir}" >&2
        exit 1
    }

    declare -a all_model_files=()
    while IFS= read -r file; do
        [[ -n "${file}" ]] && all_model_files+=("${file}")
    done < <(find "${SOURCE_DIR}" -maxdepth 1 -type f \
        -name "zg_Amon_${model}_historical_*.nc" -print | sort)

    [[ ${#all_model_files[@]} -gt 0 ]] || {
        echo "Error: no historical zg files found for ${model}" >&2
        exit 1
    }

    declare -a model_files=()
    for file in "${all_model_files[@]}"; do
        filename="$(basename "${file}")"
        if [[ "${filename}" != "${expected_prefix}"*.nc ]]; then
            echo "Error: unexpected member/grid combination for ${model}: ${filename}" >&2
            echo "Expected prefix: ${expected_prefix}" >&2
            exit 1
        fi
        model_files+=("${file}")
    done

    previous_end=""
    first_start=""
    last_end=""
    for file in "${model_files[@]}"; do
        filename="$(basename "${file}")"
        interval="${filename#${expected_prefix}}"
        interval="${interval%.nc}"
        if [[ ! "${interval}" =~ ^([0-9]{6})-([0-9]{6})$ ]]; then
            echo "Error: cannot parse time interval from ${filename}" >&2
            exit 1
        fi

        start_month="${BASH_REMATCH[1]}"
        end_month="${BASH_REMATCH[2]}"
        [[ -n "${first_start}" ]] || first_start="${start_month}"
        if [[ -n "${previous_end}" ]]; then
            expected_start="$(next_month "${previous_end}")"
            if [[ "${start_month}" != "${expected_start}" ]]; then
                echo "Error: time gap or overlap for ${model}: expected ${expected_start}, got ${start_month}" >&2
                exit 1
            fi
        fi
        previous_end="${end_month}"
        last_end="${end_month}"
    done

    if [[ "${first_start}" != "${EXPECTED_FIRST_MONTH}" || "${last_end}" != "${EXPECTED_LAST_MONTH}" ]]; then
        echo "Error: ${model} covers ${first_start}-${last_end}; expected ${EXPECTED_FIRST_MONTH}-${EXPECTED_LAST_MONTH}" >&2
        exit 1
    fi

    source_variable="$(cdo -s showname "${model_files[0]}" | awk '{print $1}')"
    [[ "${source_variable}" == "zg" ]] || {
        echo "Error: ${model_files[0]} contains variable '${source_variable}', expected 'zg'" >&2
        exit 1
    }

    available_levels="$(cdo -s showlevel "${model_files[0]}")"
    for level in "${LEVELS[@]}"; do
        contains_word "${available_levels}" "${level}" || {
            echo "Error: ${model} does not contain pressure level ${level} Pa" >&2
            exit 1
        }
    done

    for output_name in "${OUTPUT_NAMES[@]}"; do
        final_file="${output_dir}/${output_name}"
        [[ ! -e "${final_file}" ]] || {
            echo "Error: output already exists, refusing to overwrite: ${final_file}" >&2
            exit 1
        }
    done

    # Paths cannot contain newlines in this dataset; newline serialization keeps
    # the preflight-selected list stable for the processing phase.
    MODEL_FILES_SERIALIZED["${model}"]="$(printf '%s\n' "${model_files[@]}")"
    TOTAL_SELECTED_FILES=$((TOTAL_SELECTED_FILES + ${#model_files[@]}))
    echo "  ${model}: ${#model_files[@]} files, ${member}/${grid}, ${first_start}-${last_end}"
done

TOTAL_DISCOVERED_FILES="$(find "${SOURCE_DIR}" -maxdepth 1 -type f -name 'zg_Amon_*.nc' -print | wc -l | awk '{print $1}')"
if [[ "${TOTAL_SELECTED_FILES}" != "${TOTAL_DISCOVERED_FILES}" ]]; then
    echo "Error: selected ${TOTAL_SELECTED_FILES} files, but found ${TOTAL_DISCOVERED_FILES} zg_Amon files in ${SOURCE_DIR}." >&2
    echo "There are files outside the ten configured model/member/grid combinations." >&2
    exit 1
fi
echo "  Total: ${TOTAL_SELECTED_FILES} source files"

WORK_DIR="$(mktemp -d "${SOURCE_DIR}/.zg_levels_work.XXXXXX")"
declare -a PART_FILES=()
declare -a FINAL_FILES=()

cleanup() {
    local part_file
    for part_file in "${PART_FILES[@]:-}"; do
        [[ -n "${part_file}" && -f "${part_file}" ]] && rm -f -- "${part_file}"
    done
    [[ -d "${WORK_DIR}" ]] && rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

validate_output() {
    local file="$1"
    local expected_level="$2"
    local variable
    local levels
    local ntime
    local first_date
    local last_date
    local unique_dates
    local grid_description
    local xsize
    local ysize

    variable="$(cdo -s showname "${file}" | awk '{print $1}')"
    [[ "${variable}" == "zg" ]] || {
        echo "Error: ${file} has variable '${variable}', expected 'zg'" >&2
        return 1
    }

    levels="$(cdo -s showlevel "${file}")"
    read -r -a level_array <<< "${levels}"
    if [[ ${#level_array[@]} -ne 1 || "${level_array[0]}" != "${expected_level}" ]]; then
        echo "Error: ${file} has level(s) '${levels}', expected only ${expected_level}" >&2
        return 1
    fi

    ntime="$(cdo -s ntime "${file}" | awk '{print $1}')"
    [[ "${ntime}" == "${EXPECTED_NTIME}" ]] || {
        echo "Error: ${file} has ${ntime} time steps; expected ${EXPECTED_NTIME}" >&2
        return 1
    }

    read -r first_date last_date < <(date_range "${file}")
    if [[ "${first_date}" != "${EXPECTED_FIRST_DATE}" || "${last_date}" != "${EXPECTED_LAST_DATE}" ]]; then
        echo "Error: ${file} covers ${first_date} to ${last_date}; expected ${EXPECTED_FIRST_DATE} to ${EXPECTED_LAST_DATE}" >&2
        return 1
    fi

    unique_dates="$(cdo -s showdate "${file}" | tr ' ' '\n' | awk 'NF' | sort -u | wc -l | awk '{print $1}')"
    [[ "${unique_dates}" == "${EXPECTED_NTIME}" ]] || {
        echo "Error: ${file} contains duplicate dates (${unique_dates} unique of ${EXPECTED_NTIME})" >&2
        return 1
    }

    grid_description="$(cdo -s griddes "${file}")"
    xsize="$(awk '$1 == "xsize" {print $3; exit}' <<< "${grid_description}")"
    ysize="$(awk '$1 == "ysize" {print $3; exit}' <<< "${grid_description}")"
    if [[ "${xsize}" != "${EXPECTED_XSIZE}" || "${ysize}" != "${EXPECTED_YSIZE}" ]]; then
        echo "Error: ${file} grid is ${xsize}x${ysize}; expected ${EXPECTED_XSIZE}x${EXPECTED_YSIZE}" >&2
        return 1
    fi
}

echo
echo "Processing models"

for model in "${MODELS[@]}"; do
    member="${MODEL_MEMBER[${model}]}"
    grid="${MODEL_GRID[${model}]}"
    output_dir="${OUTPUT_ROOT}/${model}"
    model_work_dir="${WORK_DIR}/${model}"
    segment_dir="${model_work_dir}/segments"
    merged_six_levels="${model_work_dir}/zg_six_levels_185001-201412.nc"
    mkdir -p "${segment_dir}"

    declare -a model_files=()
    while IFS= read -r file; do
        [[ -n "${file}" ]] && model_files+=("${file}")
    done <<< "${MODEL_FILES_SERIALIZED[${model}]}"

    declare -a regridded_segments=()
    segment_index=0
    echo
    echo "[${model}] Regridding ${#model_files[@]} source file(s), ${member}/${grid}"
    for source_file in "${model_files[@]}"; do
        printf -v segment_file '%s/%04d.nc' "${segment_dir}" "${segment_index}"
        echo "  [$((segment_index + 1))/${#model_files[@]}] $(basename "${source_file}")"
        cdo -O "remapbil,${TARGET_GRID}" -sellevel,"${LEVEL_CSV}" \
            "${source_file}" "${segment_file}"
        regridded_segments+=("${segment_file}")
        segment_index=$((segment_index + 1))
    done

    echo "[${model}] Merging regridded time segments"
    cdo -O mergetime "${regridded_segments[@]}" "${merged_six_levels}"

    for index in "${!LEVELS[@]}"; do
        level="${LEVELS[${index}]}"
        output_name="${OUTPUT_NAMES[${index}]}"
        final_file="${output_dir}/${output_name}"
        part_file="${final_file}.part.$$"
        PART_FILES+=("${part_file}")
        FINAL_FILES+=("${final_file}")

        echo "[${model}] Writing ${output_name} (${level} Pa)"
        cdo -O sellevel,"${level}" "${merged_six_levels}" "${part_file}"
        validate_output "${part_file}" "${level}"
    done

    rm -rf -- "${model_work_dir}"
done

echo
echo "All 60 outputs passed validation. Publishing final files."
for index in "${!PART_FILES[@]}"; do
    part_file="${PART_FILES[${index}]}"
    final_file="${FINAL_FILES[${index}]}"
    [[ ! -e "${final_file}" ]] || {
        echo "Error: output appeared during processing, refusing to overwrite: ${final_file}" >&2
        exit 1
    }
    mv -- "${part_file}" "${final_file}"
    echo "  Created: ${final_file}"
done

echo
echo "Completed: 60 files created for 10 models. Existing files were not changed."
