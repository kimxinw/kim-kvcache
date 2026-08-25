# Benchmark result policy

This directory stores formal, commit-bound Release benchmark evidence. Debug
and smoke outputs belong in `/tmp` and must not be presented as performance
results.

`scripts/run_release_matrix.sh` creates one directory per run:

```text
benchmark/results/<source-commit>_<UTC timestamp>/
├── MANIFEST.txt
├── SHA256SUMS
├── summary.csv
├── cpu/<workload>/
│   ├── cpu_metadata.json
│   ├── cpu_metadata.csv
│   └── run.log
└── cuda/<workload>/
    ├── cuda_data_path.json
    ├── cuda_data_path.csv
    └── run.log
```

A formal run must satisfy all of the following:

1. The source worktree is clean before execution.
2. Every report records the exact full source commit.
3. Release contract tests pass before benchmarks start.
4. Every workload reports `successful: true`, zero unexpected failures,
   valid invariants, and complete resource release.
5. Raw JSON/CSV files remain unedited after generation.
6. `SHA256SUMS` matches every artifact in the run directory.

The consolidated `summary.csv` contains only capacity and latency-summary
records. Per-operation raw samples remain in the per-workload JSON/CSV files.
