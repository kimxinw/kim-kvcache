# E5 TinyLlama 端到端证据

## 验证结论

| 项目 | 结果 |
|---|---|
| 五种 Page 策略配置一致 | PASS |
| CUDA KV Storage 字节预算一致 | PASS |
| 跨策略输出 Token 一致 | PASS |
| Transformers FP16 独立 Reference | PASS（8 个唯一 Prompt） |
| 每组测量轮数 | 3 或以上 |
| 结果类型 | 真实 TinyLlama 1.1B FP16 端到端 Generation |

## Fixed-8 与 Hetero-8/64

正值表示 Hetero 指标高于 Fixed-8；对于延迟，正值代表更慢。

| Workload | E2E p50 差值 | TPOT p50 差值 | Output tokens/s 差值 |
|---|---:|---:|---:|
| decode_short_c1 | +7.09% | +7.09% | -6.52% |
| decode_long_prompt_c1 | +6.16% | +6.02% | -5.75% |
| decode_short_c2 | +6.42% | +6.74% | -6.11% |
| decode_long_prompt_c2 | +10.12% | +12.97% | -9.78% |
| decode_short_c4 | +8.25% | +7.64% | -6.95% |
| decode_long_prompt_c4 | +7.03% | +3.13% | -6.85% |
| mixed_c4 | +6.34% | +4.02% | -6.47% |

## 绝对结果

| Variant | Workload | E2E p50 (ms) | TPOT p50 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| fixed_8 | decode_short_c1 | 540.996 | 9.139 | 59.079 |
| fixed_8 | decode_short_c4 | 2186.527 | 37.584 | 57.621 |
| fixed_8 | decode_long_prompt_c4 | 7137.512 | 61.447 | 17.918 |
| fixed_8 | mixed_c4 | 2251.932 | 30.629 | 27.146 |
| fixed_16 | decode_short_c1 | 570.982 | 9.573 | 56.077 |
| fixed_16 | decode_short_c4 | 2150.217 | 36.152 | 57.993 |
| fixed_16 | decode_long_prompt_c4 | 6791.097 | 56.389 | 18.781 |
| fixed_16 | mixed_c4 | 2353.285 | 28.478 | 28.211 |
| fixed_32 | decode_short_c1 | 519.818 | 8.725 | 61.564 |
| fixed_32 | decode_short_c4 | 2170.266 | 36.601 | 58.545 |
| fixed_32 | decode_long_prompt_c4 | 6641.935 | 49.745 | 19.277 |
| fixed_32 | mixed_c4 | 2273.423 | 26.017 | 28.454 |
| fixed_64 | decode_short_c1 | 613.850 | 10.271 | 53.697 |
| fixed_64 | decode_short_c4 | 2356.464 | 39.498 | 53.941 |
| fixed_64 | decode_long_prompt_c4 | 6921.099 | 52.184 | 18.424 |
| fixed_64 | mixed_c4 | 2367.682 | 26.291 | 27.385 |
| hetero_8_64 | decode_short_c1 | 579.369 | 9.786 | 55.229 |
| hetero_8_64 | decode_short_c4 | 2366.922 | 40.456 | 53.617 |
| hetero_8_64 | decode_long_prompt_c4 | 7639.587 | 63.370 | 16.691 |
| hetero_8_64 | mixed_c4 | 2394.594 | 31.860 | 25.390 |

## Capacity 与故障隔离

| Variant | Capacity 完成/失败/拒绝 | Fault 完成/失败 | Peak fragmentation tokens |
|---|---:|---:|---:|
| fixed_8 | 48/0/3 | 9/3 | 112 |
| fixed_16 | 48/0/3 | 9/3 | 240 |
| fixed_32 | 48/0/3 | 9/3 | 496 |
| fixed_64 | 24/24/3 | 9/3 | 504 |
| hetero_8_64 | 24/24/3 | 9/3 | 112 |

## 边界说明

- Hetero 的 Extent Page 分配次数为 `0`。当前 Engine Generation 路径没有自动 Promotion，因此 Hetero-8/64 在这些 E2E Workload 中实际主要走 Micro-8。
- 当前 Iteration Scheduler 在共享 CUDA Stream 上逐请求推进，并未使用融合 Batched GEMM/Attention；并发提高主要增加排队延迟，不代表 GPU Batch 加速。
- Microbenchmark 与本报告的 E2E 结果分开；K6 的 Gather/Promotion 收益不能直接替代模型端到端收益。
- Nsight Systems GPU Activity Timeline 受当前 WSL2/CUPTI 环境限制，本报告不以 CUDA API Duration 冒充 Kernel Timeline。
- 未与 vLLM 或 TensorRT-LLM 比较峰值性能。
