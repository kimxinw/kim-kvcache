# K6 Fixed-Page Runtime and Promotion Break-Even Report

## Scope

This report records the formal K6 comparison between the executable
Hetero-8/64 runtime and Fixed-8/16/32/64 runtimes. It covers the CPU metadata
path and the CUDA data path on the same workload definitions and timing
boundaries. It does not claim end-to-end model-serving acceleration.

## Reproducibility baseline

| Item | Value |
|---|---|
| Source commit | `f593161df222b06e4a23dd7fc6960dc430dd3c25` |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB |
| Driver | 560.94 |
| CUDA compiler | 12.6.85 |
| CPU requests per report | 10,000 |
| CUDA timing | 3 warmup + 20 measured iterations |
| Variants | Hetero, Fixed-8, Fixed-16, Fixed-32, Fixed-64 |
| Workloads | Short, Mixed, Adversarial, Long, Shared Prompt, Fork/COW |
| Reports | 30 CPU + 30 CUDA |

CPU Release contracts passed 8/8 and CUDA Release contracts passed 13/13.
All 60 formal reports returned `successful: true`, zero failed operations,
valid invariants, and complete resource release. `SHA256SUMS` covers every
artifact in this directory.

## CUDA access-path comparison

The table below uses mean CUDA Event latency. "Access" is Gather plus
Reference Attention where Attention exists.

| Workload | Variant | Gather (us) | Attention (us) | Access total (us) |
|---|---:|---:|---:|---:|
| Long | Hetero | 269.411 | 1,298.261 | 1,567.672 |
| Long | Fixed-8 | 664.320 | 1,685.256 | 2,349.576 |
| Long | Fixed-16 | 445.048 | 1,461.320 | 1,906.368 |
| Long | Fixed-32 | 330.629 | 1,352.189 | 1,682.818 |
| Long | Fixed-64 | 256.843 | 1,233.198 | 1,490.041 |
| Mixed | Hetero | 168.283 | 743.186 | 911.469 |
| Mixed | Fixed-8 | 323.922 | 901.989 | 1,225.911 |
| Mixed | Fixed-16 | 258.734 | 860.730 | 1,119.464 |
| Mixed | Fixed-32 | 204.147 | 778.110 | 982.257 |
| Mixed | Fixed-64 | 155.878 | 690.373 | 846.251 |

Hetero reduces the measured access cost relative to Fixed-8, but Fixed-64 is
still faster on this isolated data-path microbenchmark. Fixed-64 also has the
known capacity/fragmentation trade-off recorded by the K5 capacity model.

## Promotion break-even

The analytical repeat model is based only on measured primitives:

```text
net_saving(N) = N * (Fixed8_access - Hetero_access) - Promotion_cost
```

Promotion cost is the sum of successful Promotion CUDA Event samples divided
by the 20 measured iterations. This models repeated access to the same
promoted KV region; it is not an end-to-end serving prediction.

| Workload | Promotion/iteration (us) | Saving/access vs Fixed-8 (us) | Break-even |
|---|---:|---:|---:|
| Long | 819.403 | 781.904 | 2 accesses |
| Mixed | 436.883 | 314.442 | 2 accesses |
| Shared Prompt | N/A | -4.139 | No Promotion samples |
| Fork/COW | N/A | -12.941 | No Promotion samples |

The registered Shared Prompt and Fork/COW workload paths do not execute a
Promotion transaction. Their rows are therefore deliberately reported as
not applicable rather than extrapolated from unrelated samples. The complete
1/2/4/8/16/32/64/128-repeat curve is in
`promotion_break_even_curve.csv`.

The whole-workload single-access throughput remains lower for Hetero in the
two promotable cases: Long is 202.82 requests/s versus 230.55 for Fixed-8,
and Mixed is 334.35 versus 420.54. The measured break-even result therefore
does not establish a one-access or end-to-end speedup.

## Nsight profiling status

Nsight profiling is blocked by the current WSL2 dual-GPU environment:

- Nsight Systems 2024.3.2 and 2025.2.1 both collect a report container but
  fail during import with `Unrecognized GPU UUID` for the Quadro P4000 UUID,
  even when the RTX 3060 is selected explicitly. CUDA kernel/API summary
  tables are consequently empty.
- Nsight Compute 2024.3.2 and 2025.2.1 both attach to the benchmark and then
  return `Unknown Error on device 0`. Device metric discovery itself can
  identify the RTX 3060, so the failure occurs when collection starts.

The exact logs are retained in `profiling_attempts/`. K6 must remain open
until the case study is rerun in an environment where Nsight can collect
CUDA traces and performance counters, for example native Linux or a corrected
WSL GPU/driver configuration.

## Conclusion

The Fixed runtime matrix and Promotion break-even portions of K6 are complete
and reproducible. Promotion amortizes against Fixed-8 after approximately two
repeated Gather+Attention accesses for the measured Long and Mixed cases.
There is no evidence of an end-to-end speedup, and the Nsight case study is
an explicit remaining blocker.
