# HeteroPageKV

HeteroPageKV is a lossless, dual-granularity paged KV-cache research
prototype for LLM serving. It keeps active tails and fork points in 8-token
Micro Pages, then transactionally coalesces eligible stable history into
64-token Extent Pages.

The project is intentionally independent from TensorRT-LLM. Its purpose is to
make page ownership, failure semantics, memory accounting, and performance
trade-offs reproducible before attempting an inference-engine integration.

## Current scope

- Fixed-capacity, generation-safe Micro and Extent page pools.
- Hierarchical block tables with append, seal, prefix fork, and partial-tail
  copy-on-write.
- Transactional promotion with explicit prepare, commit, and rollback.
- Page I/O leases that prevent CUDA work from racing slot reuse.
- CUDA append, gather, promotion copy, and reference attention paths.
- Deterministic CPU and CUDA benchmark harnesses with raw JSON/CSV output.

K0 through K4 are complete. K5 contains the benchmark implementation and is
being closed with a commit-bound Release matrix. No performance claim should
be made from Debug or smoke-test results.

## Build and test

CPU Release:

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel
ctest --preset cpu-release
```

CUDA Release for the registered RTX 3060 (SM 86) baseline:

```bash
CUDACXX=/path/to/cuda/bin/nvcc cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release
```

The default build remains CPU-only. CUDA support is enabled only by the CUDA
preset or by setting `HETEROPAGE_KV_ENABLE_CUDA=ON` explicitly. `CUDACXX` can
be omitted when `nvcc` is already available on `PATH`.

## Reproducible benchmark matrix

Formal results must be generated from a clean, committed worktree so every
report can be tied to an exact source revision:

```bash
./scripts/run_release_matrix.sh
```

The script builds and tests both Release presets, then runs all seven
registered workloads:

- `short`
- `mixed`
- `adversarial`
- `long`
- `shared_prompt`
- `fork_cow`
- `fault`

CPU metadata runs use 10,000 requests per workload. CUDA data-path runs use
three warmup and twenty measured iterations. The output directory contains
per-workload JSON/CSV files, logs, a consolidated summary, an environment
manifest, and SHA-256 checksums.

For a smoke run or a non-default destination, override the script inputs:

```bash
KIM_KV_CPU_REQUESTS=100 \
KIM_KV_CUDA_ITERATIONS=2 \
KIM_KV_CUDA_ROOT=/path/to/cuda \
KIM_KV_RESULTS_ROOT=/tmp/kim-kv-results \
./scripts/run_release_matrix.sh
```

`KIM_KV_CUDA_ROOT` is required only when the CUDA Toolkit is not discoverable
through `CUDACXX` or `PATH`.

See [`benchmark/results/README.md`](benchmark/results/README.md) for the
evidence policy and artifact layout.

## Registered baseline

The initial model baseline is TinyLlama 1.1B Chat with 22 layers, four KV
heads, head dimension 64, and FP16 KV data. The initial hardware baseline is
an NVIDIA GeForce RTX 3060 12 GB.

Capacity reports compare seven layouts:

1. contiguous maximum allocation;
2. fixed 8-token blocks;
3. fixed 16-token blocks;
4. fixed 32-token blocks;
5. fixed 64-token blocks;
6. heterogeneous 8/64 pages without promotion;
7. heterogeneous 8/64 pages with promotion.

## Correctness boundary

The prototype does not compress, prune, quantize, or approximate KV values.
It also does not replace TensorRT-LLM's internal KV-cache manager and does not
yet report end-to-end model-serving throughput. CPU metadata throughput and
CUDA kernel latency are reported as separate suites.

## Related work

- [vLLM](https://github.com/vllm-project/vllm): PagedAttention, block pools,
  and automatic prefix caching.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): production C++/CUDA
  KV-cache management, block reuse, and offloading.
- [SGLang](https://github.com/sgl-project/sglang): RadixAttention and
  hierarchical KV-cache storage.
- [vAttention](https://github.com/microsoft/vattention): CUDA virtual-memory
  allocation as an alternative to non-contiguous paged attention.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): optimized paged
  KV-cache attention kernels.

HeteroPageKV specifically studies whether small append pages can reduce tail
fragmentation while losslessly coalescing stable history into larger extents
to reduce mapping and data-path overhead.
