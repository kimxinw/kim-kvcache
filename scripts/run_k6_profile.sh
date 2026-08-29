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
readonly result_directory="${results_root}/${short_commit}_${timestamp_utc}_k6_profile"
readonly cuda_device="${KIM_KV_CUDA_DEVICE:-0}"
readonly benchmark_seed="${KIM_KV_BENCHMARK_SEED:-0x4B564341434845}"
readonly maximum_sequence_length="${KIM_KV_MAX_SEQUENCE_LENGTH:-544}"
readonly nsys_iterations="${KIM_KV_NSYS_ITERATIONS:-3}"
readonly ncu_launch_count="${KIM_KV_NCU_LAUNCH_COUNT:-24}"

if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    echo "error: formal profiling requires a clean, committed worktree" >&2
    git status --short >&2
    exit 2
fi

if [[ -e "${result_directory}" ]]; then
    echo "error: result directory already exists: ${result_directory}" >&2
    exit 2
fi

cuda_compiler="${CUDACXX:-}"
if [[ -z "${cuda_compiler}" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda_compiler="$(command -v nvcc)"
fi
if [[ -z "${cuda_compiler}" ]]; then
    for cache_path in \
        build-k5-cuda-release/CMakeCache.txt \
        build-impl-units-cuda/CMakeCache.txt; do
        if [[ -f "${cache_path}" ]]; then
            cuda_compiler="$(
                sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
                    "${cache_path}" | head -n 1
            )"
        fi
        if [[ -n "${cuda_compiler}" ]]; then
            break
        fi
    done
fi
if [[ -z "${cuda_compiler}" ]] || [[ ! -x "${cuda_compiler}" ]]; then
    echo "error: CUDA compiler not found" >&2
    exit 2
fi
readonly cuda_root="$(cd -- "$(dirname -- "${cuda_compiler}")/.." && pwd)"

nsys_bin="${KIM_KV_NSYS:-}"
ncu_bin="${KIM_KV_NCU:-}"
if [[ -z "${nsys_bin}" ]] || [[ -z "${ncu_bin}" ]]; then
    shopt -s nullglob
    nsight_roots=("${cuda_root}"/nsight-compute-*)
    shopt -u nullglob
    for nsight_root in "${nsight_roots[@]}"; do
        if [[ -z "${nsys_bin}" ]] \
            && [[ -x "${nsight_root}/host/target-linux-x64/nsys" ]]; then
            nsys_bin="${nsight_root}/host/target-linux-x64/nsys"
        fi
        if [[ -z "${ncu_bin}" ]] && [[ -x "${nsight_root}/ncu" ]]; then
            ncu_bin="${nsight_root}/ncu"
        fi
    done
fi
if [[ -z "${nsys_bin}" ]] || [[ ! -x "${nsys_bin}" ]]; then
    echo "error: Nsight Systems CLI not found; set KIM_KV_NSYS" >&2
    exit 2
fi
if [[ -z "${ncu_bin}" ]] || [[ ! -x "${ncu_bin}" ]]; then
    echo "error: Nsight Compute CLI not found; set KIM_KV_NCU" >&2
    exit 2
fi
readonly nsys_bin
readonly ncu_bin
readonly benchmark="${project_root}/build-k5-cuda-release/benchmarks/kim_kv_cuda_benchmark"
if [[ ! -x "${benchmark}" ]]; then
    echo "error: CUDA Release benchmark is missing; run K6 matrix first" >&2
    exit 2
fi

mkdir -p "${result_directory}"

profile_variant()
{
    local variant="$1"
    local output="${result_directory}/${variant}"
    local -a fixed_arguments=()
    if [[ "${variant}" == fixed_8 ]]; then
        fixed_arguments=(--fixed-page-tokens 8)
    fi
    mkdir -p "${output}/nsys_run" "${output}/ncu_run"

    echo "Running Nsight Systems: ${variant}/long"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
        "${nsys_bin}" profile \
        --trace=cuda,osrt \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${output}/timeline" \
        "${benchmark}" \
        --workload long \
        --seed "${benchmark_seed}" \
        --requests 10000 \
        --max-sequence-length "${maximum_sequence_length}" \
        --warmup 1 \
        --iterations "${nsys_iterations}" \
        --git-commit "${source_commit}" \
        --output-dir "${output}/nsys_run" \
        "${fixed_arguments[@]}" \
        >"${output}/nsys.log" 2>&1

    "${nsys_bin}" stats \
        --force-overwrite=true \
        --report cuda_gpu_kern_sum,cuda_api_sum \
        --format csv \
        --output "${output}/nsys_stats" \
        "${output}/timeline.nsys-rep" \
        >"${output}/nsys_stats.log" 2>&1

    echo "Running Nsight Compute: ${variant}/long"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
        "${ncu_bin}" \
        --target-processes all \
        --set basic \
        --launch-count "${ncu_launch_count}" \
        --export "${output}/kernels" \
        --force-overwrite \
        --csv \
        --log-file "${output}/ncu.csv" \
        "${benchmark}" \
        --workload long \
        --seed "${benchmark_seed}" \
        --requests 10000 \
        --max-sequence-length "${maximum_sequence_length}" \
        --warmup 0 \
        --iterations 1 \
        --git-commit "${source_commit}" \
        --output-dir "${output}/ncu_run" \
        "${fixed_arguments[@]}" \
        >"${output}/ncu.log" 2>&1
}

profile_variant hetero
profile_variant fixed_8

{
    echo "schema_version=1"
    echo "stage=K6"
    echo "source_commit=${source_commit}"
    echo "timestamp_utc=${timestamp_utc}"
    echo "working_tree_clean=true"
    echo "workload=long"
    echo "variants=hetero fixed_8"
    echo "cuda_visible_device=${cuda_device}"
    echo "maximum_sequence_length=${maximum_sequence_length}"
    echo "nsys_iterations=${nsys_iterations}"
    echo "ncu_launch_count=${ncu_launch_count}"
    echo "nsys=$(${nsys_bin} --version | head -n 1)"
    echo "ncu=$(${ncu_bin} --version | grep '^Version' | head -n 1)"
    if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        echo "gpu=$(
            /usr/lib/wsl/lib/nvidia-smi \
                --query-gpu=name,driver_version,memory.total \
                --format=csv,noheader | sed -n "$((cuda_device + 1))p"
        )"
    fi
} > "${result_directory}/MANIFEST.txt"

(
    cd "${result_directory}"
    find . -type f ! -name SHA256SUMS -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

echo "K6 profiling completed"
echo "Results: ${result_directory}"
