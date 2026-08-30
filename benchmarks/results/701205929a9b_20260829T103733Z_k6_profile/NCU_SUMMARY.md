# K6 Nsight Compute 案例分析

## 性能剖析标识

| 字段 | 值 |
|---|---|
| 源码提交 | `701205929a9bb99f56df91bcced233b3e64e5b5a` |
| 工作负载 | Long，10,000 个请求，最大序列长度 544 |
| 变体 | Hetero 和 Fixed-8 |
| GPU | NVIDIA GeForce RTX 3060，12 GiB |
| Windows 驱动程序 | 560.94 |
| 性能分析器 | Nsight Compute 2025.2.1 |
| 指标集 | `basic` |
| 请求的样本数 | 每个内核最多匹配 4 次启动 |

每个内核均保留原始 `.ncu-rep` 文件和导出的 `metrics.csv` 文件。
下表中的样本数是基准执行期间实际发生的匹配启动数量，不是预设值。

## 内核平均指标

| 变体 | 内核 | 样本数 | 耗时（µs） | DRAM（%） | L1/TEX（%） | Compute SM（%） | 实际占用率（%） |
|---|---|---:|---:|---:|---:|---:|---:|
| Hetero | Append | 4 | 5.848 | 16.553 | 10.905 | 40.890 | 79.535 |
| Fixed-8 | Append | 4 | 5.825 | 14.945 | 10.915 | 40.470 | 79.573 |
| Hetero | Gather | 2 | 278.590 | 23.500 | 49.385 | 74.150 | 89.490 |
| Fixed-8 | Gather | 2 | 796.740 | 8.810 | 85.985 | 85.740 | 94.885 |
| Hetero | Attention scores | 1 | 902.210 | 2.200 | 69.470 | 59.530 | 50.980 |
| Fixed-8 | Attention scores | 1 | 902.940 | 2.150 | 69.600 | 59.380 | 51.030 |
| Hetero | Attention output | 1 | 71.620 | 28.190 | 17.740 | 19.820 | 15.170 |
| Fixed-8 | Attention output | 1 | 71.870 | 28.260 | 17.800 | 19.650 | 15.200 |
| Hetero | Promotion | 4 | 53.920 | 78.003 | 81.505 | 33.740 | 85.380 |

在当前启动配置下，以上所有内核的理论占用率均为 100%。
`NCU_SUMMARY.csv` 还保留了内存吞吐量、L2、寄存器和每个 SM 的 Wave 数等字段。

## 分析结果

| 对比项 | 结果 | 解释边界 |
|---|---:|---|
| Gather 耗时 | Hetero 快 2.86 倍，耗时降低 65.03% | 这是性能剖析访问路径中的主要差异。 |
| Attention 内核合计 | 973.830 µs 对 974.810 µs；Hetero 低 0.10% | 此处 Attention 计算基本不受页面布局影响。 |
| Append 耗时 | Hetero 高 0.39% | 在当前采样规模下，差异可以忽略。 |
| 页面晋升内核 | 53.920 µs | 这只是一个被选中的 GPU 内核，不代表每次迭代的完整页面晋升事务。 |

Fixed-8 Gather 的实际占用率和报告的 L1/TEX 吞吐量更高，但耗时却是 Hetero
的 2.86 倍。因此，仅凭占用率无法解释该工作负载。`basic` 指标集表明 Fixed-8
Gather 的缓存侧工作成本明显更高，但若要作出因果判断，还需要更深入的指标集来分析
指令级行为和 Warp 停顿。

此前的发布版矩阵测得 Long 工作负载中每次迭代的完整页面晋升事务耗时为
819.403 µs。该数值包含全部页面晋升工作及其基准计时边界，不能用上表中选定的
`promotionKernel` 耗时替代。

## Nsight Systems 状态

禁用 Quadro P4000 后，先前的 `Unrecognized GPU UUID` 导入故障已经消失。
Nsight Systems 2024.3.2 和 2025.2.1 现在可以记录 CUDA API 活动，探测运行中
包括 76 次 `cudaLaunchKernel` 调用，但在使用 Windows 驱动程序 560.94 的 WSL2
环境中，两份报告均不包含 CUDA GPU 内核事件。因此，没有把空的 Systems 报告
登记为成功的时间线。现在，只要内核或 API 汇总任一为空，性能剖析脚本就会
明确失败。

该案例分析完成了 K6 的 Nsight Compute 部分。完整的 Systems 时间线仍需在兼容的
Windows NVIDIA 驱动程序/WSL 跟踪环境中单独运行，并设置
`KIM_KV_PROFILE_COMPONENT=nsys`。
