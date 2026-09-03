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
readonly result_directory="${results_root}/${short_commit}_${timestamp_utc}_e5"
readonly manifest="${KIM_KV_MODEL_MANIFEST:-/home/xinwang/workspaces/kim-kvcache-e2-model/tinyllama-1.1b-chat-fp16.manifest}"
readonly weights="${KIM_KV_MODEL_WEIGHTS:-/home/xinwang/workspaces/kim-kvcache-e2-model/tinyllama-1.1b-chat-fp16.weights}"
readonly warmup="${KIM_KV_E5_WARMUP:-1}"
readonly iterations="${KIM_KV_E5_ITERATIONS:-3}"
readonly kv_capacity_tokens="${KIM_KV_E5_KV_CAPACITY_TOKENS:-8192}"
readonly capacity_probe_tokens="${KIM_KV_E5_CAPACITY_PROBE_TOKENS:-512}"
readonly allow_dirty="${KIM_KV_ALLOW_DIRTY:-0}"
readonly build_jobs="${KIM_KV_BUILD_JOBS:-4}"
readonly cuda_device="${KIM_KV_CUDA_DEVICE:-0}"
readonly reference_python="${KIM_KV_REFERENCE_PYTHON:-/home/xinwang/miniconda3/envs/vllm/bin/python}"
readonly -a variants=(fixed_8 fixed_16 fixed_32 fixed_64 hetero)

if [[ ! -f "${manifest}" ]] || [[ ! -f "${weights}" ]]; then
    echo "error: set KIM_KV_MODEL_MANIFEST and KIM_KV_MODEL_WEIGHTS" >&2
    exit 2
fi
if [[ ! -x "${reference_python}" ]]; then
    echo "error: set KIM_KV_REFERENCE_PYTHON to Python with torch/transformers" >&2
    exit 2
fi

working_tree_clean=true
if [[ -n "$(git status --porcelain=v1 --untracked-files=normal)" ]]; then
    working_tree_clean=false
    if [[ "${allow_dirty}" != 1 ]]; then
        echo "error: formal E5 evidence requires a clean committed worktree" >&2
        echo "set KIM_KV_ALLOW_DIRTY=1 only for provisional development runs" >&2
        git status --short >&2
        exit 2
    fi
fi

if [[ -e "${result_directory}" ]]; then
    echo "error: result directory already exists: ${result_directory}" >&2
    exit 2
fi
mkdir -p "${result_directory}/variants"

cuda_compiler="${CUDACXX:-}"
if [[ -z "${cuda_compiler}" ]] && [[ -f build-k5-cuda-release/CMakeCache.txt ]]; then
    cuda_compiler="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            build-k5-cuda-release/CMakeCache.txt | head -n 1
    )"
fi
if [[ -z "${cuda_compiler}" ]] || [[ ! -x "${cuda_compiler}" ]]; then
    echo "error: CUDA compiler not found; set CUDACXX" >&2
    exit 2
fi
readonly cuda_compiler
readonly cuda_root="$(cd -- "$(dirname -- "${cuda_compiler}")/.." && pwd)"

cmake --preset cpu-release
cmake --build --preset cpu-release --parallel "${build_jobs}"
ctest --preset cpu-release
cmake --preset cuda-release \
    "-DCMAKE_CUDA_COMPILER=${cuda_compiler}" \
    "-DCUDAToolkit_ROOT=${cuda_root}"
cmake --build --preset cuda-release --parallel "${build_jobs}"
CUDA_VISIBLE_DEVICES="${cuda_device}" ctest --preset cuda-release

for variant in "${variants[@]}"; do
    report_variant="${variant}"
    if [[ "${variant}" == hetero ]]; then
        report_variant="hetero_8_64"
    fi
    echo "Running E5 ${report_variant}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
        build-k5-cuda-release/tools/kim_kv_tinyllama_e2e_benchmark \
        --manifest "${manifest}" \
        --weights "${weights}" \
        --variant "${variant}" \
        --output "${result_directory}/variants/${report_variant}.json" \
        --warmup "${warmup}" \
        --iterations "${iterations}" \
        --kv-capacity-tokens "${kv_capacity_tokens}" \
        --capacity-probe-tokens "${capacity_probe_tokens}" \
        --git-commit "${source_commit}" \
        >"${result_directory}/variants/${report_variant}.log" 2>&1
done

CUDA_VISIBLE_DEVICES="${cuda_device}" \
    "${reference_python}" scripts/validate_e5_reference.py \
    --manifest "${manifest}" \
    --weights "${weights}" \
    --runtime-json "${result_directory}/variants/fixed_8.json" \
    --output "${result_directory}/reference_validation.json" \
    >"${result_directory}/reference_validation.log"

python3 scripts/analyze_e5_results.py --result-dir "${result_directory}" \
    >"${result_directory}/analysis.log"

{
    echo "schema_version=1"
    echo "stage=E5"
    echo "source_commit=${source_commit}"
    echo "working_tree_clean=${working_tree_clean}"
    echo "timestamp_utc=${timestamp_utc}"
    echo "gpu_device=${cuda_device}"
    echo "warmup=${warmup}"
    echo "iterations=${iterations}"
    echo "kv_capacity_tokens=${kv_capacity_tokens}"
    echo "capacity_probe_tokens=${capacity_probe_tokens}"
    echo "variants=${variants[*]}"
    echo "model_manifest=${manifest}"
    echo "model_weights=${weights}"
    sha256sum "${manifest}" "${weights}"
    sha256sum build-k5-cuda-release/tools/kim_kv_tinyllama_e2e_benchmark
} >"${result_directory}/MANIFEST.txt"

find "${result_directory}" -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >"${result_directory}/SHA256SUMS"

echo "E5 evidence: ${result_directory}"
