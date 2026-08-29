#!/usr/bin/env python3

"""Build K6 runtime and promotion break-even tables from formal JSON reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable


COMPARISON_WORKLOADS = ("long", "mixed", "shared_prompt", "fork_cow")
CURVE_REPEATS = (1, 2, 4, 8, 16, 32, 64, 128)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_reports(result_directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    cuda_directory = result_directory / "cuda"
    candidates = sorted(cuda_directory.glob("*/*/*.json"))
    candidates.extend(sorted(cuda_directory.glob("*/*/*.json.gz")))
    for path in candidates:
        relative = path.relative_to(cuda_directory)
        variant, workload = relative.parts[:2]
        with open_text(path) as source:
            report = json.load(source)
        if not report.get("successful", False):
            raise ValueError(f"unsuccessful report: {path}")
        if len(report.get("workloads", [])) != 1:
            raise ValueError(f"expected one workload per report: {path}")
        reports[(variant, workload)] = report
    if not reports:
        raise ValueError(f"no CUDA JSON reports found below {cuda_directory}")
    return reports


def workload_result(report: dict[str, Any]) -> dict[str, Any]:
    return report["workloads"][0]


def latency_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        value["operation"]: value
        for value in workload_result(report).get("latency", [])
    }


def access_mean_ns(latency: dict[str, dict[str, Any]]) -> int:
    return sum(
        int(latency[operation]["mean_ns"])
        for operation in ("gather", "reference_attention")
        if operation in latency
    )


def promotion_cost_per_iteration(report: dict[str, Any]) -> tuple[int, int]:
    promotions = [
        sample
        for sample in workload_result(report).get("samples", [])
        if sample.get("operation") == "promote" and sample.get("success")
    ]
    measured_iterations = int(report["config"]["measured_iterations"])
    if measured_iterations <= 0:
        raise ValueError("measured_iterations must be positive")
    total_ns = sum(int(sample["duration_ns"]) for sample in promotions)
    return len(promotions), int(round(total_ns / measured_iterations))


def write_runtime_comparison(
    destination: Path,
    reports: dict[tuple[str, str], dict[str, Any]],
) -> None:
    fields = (
        "variant",
        "workload",
        "operation",
        "sample_count",
        "mean_ns",
        "p50_ns",
        "p95_ns",
        "p99_ns",
        "effective_bandwidth_gbps",
        "requests_per_second",
    )
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for (variant, workload), report in sorted(reports.items()):
            result = workload_result(report)
            for latency in result.get("latency", []):
                writer.writerow({
                    "variant": variant,
                    "workload": workload,
                    "operation": latency["operation"],
                    "sample_count": latency["sample_count"],
                    "mean_ns": latency["mean_ns"],
                    "p50_ns": latency["p50_ns"],
                    "p95_ns": latency["p95_ns"],
                    "p99_ns": latency["p99_ns"],
                    "effective_bandwidth_gbps": latency[
                        "effective_bandwidth_gbps"
                    ],
                    "requests_per_second": result["requests_per_second"],
                })


def break_even_rows(
    reports: dict[tuple[str, str], dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for workload in COMPARISON_WORKLOADS:
        hetero = reports.get(("hetero", workload))
        fixed8 = reports.get(("fixed_8", workload))
        if hetero is None or fixed8 is None:
            yield {
                "workload": workload,
                "status": "missing_report",
            }
            continue

        hetero_latency = latency_map(hetero)
        fixed8_latency = latency_map(fixed8)
        promotion_samples, promotion_ns = promotion_cost_per_iteration(hetero)
        hetero_access_ns = access_mean_ns(hetero_latency)
        fixed8_access_ns = access_mean_ns(fixed8_latency)
        saving_ns = fixed8_access_ns - hetero_access_ns

        if promotion_samples == 0:
            status = "no_promotion_samples"
            break_even = ""
        elif saving_ns <= 0:
            status = "no_positive_access_saving"
            break_even = ""
        else:
            status = "break_even_found"
            break_even = math.ceil(promotion_ns / saving_ns)

        yield {
            "workload": workload,
            "status": status,
            "promotion_samples": promotion_samples,
            "promotion_cost_per_iteration_ns": promotion_ns,
            "hetero_access_mean_ns": hetero_access_ns,
            "fixed8_access_mean_ns": fixed8_access_ns,
            "access_saving_per_repeat_ns": saving_ns,
            "break_even_repeats": break_even,
        }


def write_break_even(destination: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "workload",
        "status",
        "promotion_samples",
        "promotion_cost_per_iteration_ns",
        "hetero_access_mean_ns",
        "fixed8_access_mean_ns",
        "access_saving_per_repeat_ns",
        "break_even_repeats",
    )
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_curve(destination: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "workload",
        "repeat_count",
        "promotion_cost_ns",
        "gross_access_saving_ns",
        "net_saving_ns",
        "profitable",
        "status",
    )
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            promotion_ns = int(row.get("promotion_cost_per_iteration_ns") or 0)
            saving_ns = int(row.get("access_saving_per_repeat_ns") or 0)
            for repeats in CURVE_REPEATS:
                gross = repeats * saving_ns
                net = gross - promotion_ns
                writer.writerow({
                    "workload": row["workload"],
                    "repeat_count": repeats,
                    "promotion_cost_ns": promotion_ns,
                    "gross_access_saving_ns": gross,
                    "net_saving_ns": net,
                    "profitable": (
                        "true"
                        if row.get("status") == "break_even_found" and net >= 0
                        else "false"
                    ),
                    "status": row.get("status", ""),
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="K6 result directory containing cuda/<variant>/<workload>",
    )
    arguments = parser.parse_args()
    result_directory = arguments.result_dir.resolve()
    reports = load_reports(result_directory)
    rows = list(break_even_rows(reports))
    write_runtime_comparison(
        result_directory / "runtime_comparison.csv",
        reports,
    )
    write_break_even(result_directory / "promotion_break_even.csv", rows)
    write_curve(result_directory / "promotion_break_even_curve.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
