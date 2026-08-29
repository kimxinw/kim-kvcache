#!/usr/bin/env bash

set -euo pipefail

readonly script_directory="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"
readonly project_root="$(cd -- "${script_directory}/.." && pwd)"
cd "${project_root}"

readonly source_commit="$(git rev-parse --verify HEAD)"
readonly short_commit="${source_commit:0:12}"
readonly timestamp_utc="$(date -u +%Y%m%dT%H%M%SZ)"
readonly results_root="${KIM_KV_RESULTS_ROOT:-${project_root}/benchmarks/results}"
readonly result_directory="${results_root}/${short_commit}_${timestamp_utc}_k6"
readonly cpu_requests="${KIM_KV_CPU_REQUESTS:-10000}"
readonly cuda_warmup="${KIM_KV_CUDA_WARMUP:-3}"
readonly cuda_iterations="${KIM_KV_CUDA_ITERATIONS:-20}"
readonly build_jobs="${KIM_KV_BUILD_JOBS:-4}"
readonly cuda_device="${KIM_KV_CUDA_DEVICE:-0}"
readonly benchmark_seed="${KIM_KV_BENCHMARK_SEED:-0x4B564341434845}"
readonly capacity_budget_mib="${KIM_KV_CAPACITY_BUDGET_MIB:-12288}"
readonly concurrency="${KIM_KV_CONCURRENCY:-8}"
readonly maximum_sequence_length="${KIM_KV_MAX_SEQUENCE_LENGTH:-544}"
readonly -a workloads=(
    short
    mixed
    adversarial
    long
    shared_prompt
    fork_cow
)
readonly -a variants=(hetero fixed_8 fixed_16 fixed_32 fixed_64)

if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    echo "error: formal benchmarks require a clean, committed worktree" >&2
    git status --short >&2
    exit 2
fi

if [[ -e "${result_directory}" ]]; then
    echo "error: result directory already exists: ${result_directory}" >&2
    exit 2
fi

cuda_root="${KIM_KV_CUDA_ROOT:-}"
cuda_compiler="${CUDACXX:-}"
if [[ -z "${cuda_compiler}" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_compiler="$(command -v nvcc)"
fi
if [[ -z "${cuda_compiler}" ]] \
    && [[ -f build-k5-cuda-release/CMakeCache.txt ]]; then
    cuda_compiler="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            build-k5-cuda-release/CMakeCache.txt | head -n 1
    )"
fi
if [[ -z "${cuda_compiler}" ]] && [[ -n "${cuda_root}" ]]; then
    cuda_compiler="${cuda_root}/bin/nvcc"
fi
if [[ -z "${cuda_root}" ]] && [[ -n "${cuda_compiler}" ]]; then
    cuda_root="$(cd -- "$(dirname -- "${cuda_compiler}")/.." && pwd)"
fi
if [[ -z "${cuda_compiler}" ]] || [[ ! -x "${cuda_compiler}" ]]; then
    echo "error: CUDA compiler not found; set CUDACXX or KIM_KV_CUDA_ROOT" >&2
    exit 2
fi
readonly cuda_root
readonly cuda_compiler
readonly -a cuda_cmake_arguments=(
    "-DCMAKE_CUDA_COMPILER=${cuda_compiler}"
    "-DCUDAToolkit_ROOT=${cuda_root}"
)

mkdir -p "${result_directory}/cpu" "${result_directory}/cuda"

echo "Configuring, building, and testing CPU Release"
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel "${build_jobs}"
ctest --preset cpu-release

echo "Configuring, building, and testing CUDA Release"
cmake --preset cuda-release "${cuda_cmake_arguments[@]}"
cmake --build --preset cuda-release --parallel "${build_jobs}"
CUDA_VISIBLE_DEVICES="${cuda_device}" ctest --preset cuda-release

validate_json_report()
{
    local report_path="$1"
    grep -Fq '"successful": true' "${report_path}"
    grep -Fq "\"git_commit\":\"${source_commit}\"" "${report_path}"
    grep -Fq '"failed_operations":0' "${report_path}"
    grep -Fq '"invariants_ok":true' "${report_path}"
    grep -Fq '"resources_released":true' "${report_path}"
}

fixed_tokens_for_variant()
{
    local variant="$1"
    if [[ "${variant}" == hetero ]]; then
        echo 0
    else
        echo "${variant#fixed_}"
    fi
}

run_cpu_report()
{
    local variant="$1"
    local workload="$2"
    local fixed_tokens
    fixed_tokens="$(fixed_tokens_for_variant "${variant}")"
    local output="${result_directory}/cpu/${variant}/${workload}"
    local -a fixed_arguments=()
    local report_name="cpu_metadata.json"
    if [[ "${fixed_tokens}" != 0 ]]; then
        fixed_arguments=(--fixed-page-tokens "${fixed_tokens}")
        report_name="cpu_metadata_fixed_${fixed_tokens}.json"
    fi
    mkdir -p "${output}"
    echo "Running CPU Release: ${variant}/${workload}"
    "${project_root}/build-k5-cpu-release/kim_kv_cpu_benchmark" \
        --workload "${workload}" \
        --seed "${benchmark_seed}" \
        --requests "${cpu_requests}" \
        --concurrency "${concurrency}" \
        --max-sequence-length "${maximum_sequence_length}" \
        --warmup "${cuda_warmup}" \
        --iterations "${cuda_iterations}" \
        --capacity-budget-mib "${capacity_budget_mib}" \
        --git-commit "${source_commit}" \
        --output-dir "${output}" \
        "${fixed_arguments[@]}" \
        >"${output}/run.log" 2>&1
    validate_json_report "${output}/${report_name}"
}

run_cuda_report()
{
    local variant="$1"
    local workload="$2"
    local fixed_tokens
    fixed_tokens="$(fixed_tokens_for_variant "${variant}")"
    local output="${result_directory}/cuda/${variant}/${workload}"
    local -a fixed_arguments=()
    local report_name="cuda_data_path.json"
    if [[ "${fixed_tokens}" != 0 ]]; then
        fixed_arguments=(--fixed-page-tokens "${fixed_tokens}")
        report_name="cuda_data_path_fixed_${fixed_tokens}.json"
    fi
    mkdir -p "${output}"
    echo "Running CUDA Release: ${variant}/${workload}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
        "${project_root}/build-k5-cuda-release/kim_kv_cuda_benchmark" \
        --workload "${workload}" \
        --seed "${benchmark_seed}" \
        --requests "${cpu_requests}" \
        --concurrency "${concurrency}" \
        --max-sequence-length "${maximum_sequence_length}" \
        --warmup "${cuda_warmup}" \
        --iterations "${cuda_iterations}" \
        --capacity-budget-mib "${capacity_budget_mib}" \
        --git-commit "${source_commit}" \
        --output-dir "${output}" \
        "${fixed_arguments[@]}" \
        >"${output}/run.log" 2>&1
    validate_json_report "${output}/${report_name}"
}

for variant in "${variants[@]}"; do
    for workload in "${workloads[@]}"; do
        run_cpu_report "${variant}" "${workload}"
    done
done

for variant in "${variants[@]}"; do
    for workload in "${workloads[@]}"; do
        run_cuda_report "${variant}" "${workload}"
    done
done

first_csv="$(
    find "${result_directory}/cpu" -type f -name '*.csv' -print \
        | LC_ALL=C sort | head -n 1
)"
head -n 1 "${first_csv}" > "${result_directory}/summary.csv"
while IFS= read -r csv_path; do
    awk -F, 'NR > 1 && ($1 == "capacity" || $1 == "summary")' \
        "${csv_path}" >> "${result_directory}/summary.csv"
done < <(
    find "${result_directory}/cpu" "${result_directory}/cuda" \
        -type f -name '*.csv' -print | LC_ALL=C sort
)

python3 "${script_directory}/analyze_k6_results.py" \
    --result-dir "${result_directory}"

{
    echo "schema_version=1"
    echo "stage=K6"
    echo "source_commit=${source_commit}"
    echo "timestamp_utc=${timestamp_utc}"
    echo "working_tree_clean=true"
    echo "variants=${variants[*]}"
    echo "workloads=${workloads[*]}"
    echo "cpu_requests=${cpu_requests}"
    echo "cuda_warmup=${cuda_warmup}"
    echo "cuda_iterations=${cuda_iterations}"
    echo "benchmark_seed=${benchmark_seed}"
    echo "capacity_budget_mib=${capacity_budget_mib}"
    echo "concurrency=${concurrency}"
    echo "maximum_sequence_length=${maximum_sequence_length}"
    echo "cuda_visible_device=${cuda_device}"
    echo "cuda_root=${cuda_root}"
    echo "cuda_compiler=${cuda_compiler}"
    echo "cmake=$(cmake --version | head -n 1)"
    echo "cxx=$(c++ --version | head -n 1)"
    echo "nvcc=$("${cuda_compiler}" --version | tail -n 1)"
    if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        echo "gpu=$(
            /usr/lib/wsl/lib/nvidia-smi \
                --query-gpu=name,driver_version,memory.total \
                --format=csv,noheader | sed -n "$((cuda_device + 1))p"
        )"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu=$(
            nvidia-smi --query-gpu=name,driver_version,memory.total \
                --format=csv,noheader | sed -n "$((cuda_device + 1))p"
        )"
    fi
} > "${result_directory}/MANIFEST.txt"

find "${result_directory}/cpu" "${result_directory}/cuda" \
    -type f \( -name '*.json' -o -name '*.csv' \) -print0 \
    | xargs -0 gzip -9

(
    cd "${result_directory}"
    find . -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

echo "K6 Release benchmark matrix completed"
echo "Results: ${result_directory}"
