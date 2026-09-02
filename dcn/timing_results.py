"""Analyze and render paired critical-path results from DCN timing logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any


LINE_KINDS = (
    "DCN_TIMING_CONFIG",
    "DCN_TIMING_FIRST_CALL",
    "DCN_TIMING_WARMUP",
    "DCN_TIMING_REP",
)
DEFAULT_PAYLOAD_GBITS_PER_HOST = 17.179869184
DEFAULT_LINK_GBPS = 400.0


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 5),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def parse(lines: list[str]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        for kind in LINE_KINDS:
            marker = kind + " "
            if marker not in line:
                continue
            raw = line.split(marker, 1)[1]
            try:
                records[kind].append(json.loads(raw))
            except json.JSONDecodeError:
                pass
            break
    return records


def nic_imbalance(row: dict[str, Any], direction: str) -> float:
    delta = row["network_delta"]
    first = float(delta["eth1"][direction])
    second = float(delta["eth2"][direction])
    return abs(first - second) / (first + second) * 100 if first + second else 0.0


def analyze(
    records: dict[str, list[dict[str, Any]]],
    *,
    payload_gbits_per_host: float = DEFAULT_PAYLOAD_GBITS_PER_HOST,
    link_gbps: float = DEFAULT_LINK_GBPS,
) -> dict[str, Any]:
    configs = {row["experiment"]: row for row in records["DCN_TIMING_CONFIG"]}
    warmups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records["DCN_TIMING_WARMUP"]:
        warmups[row["experiment"]][int(row["rank"])] = row

    first_calls: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records["DCN_TIMING_FIRST_CALL"]:
        first_calls[row["experiment"]][int(row["rank"])] = row

    grouped: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records["DCN_TIMING_REP"]:
        grouped[row["experiment"]][int(row["rank"])][int(row["rep"])] = row

    experiments: list[dict[str, Any]] = []
    for experiment, ranks in grouped.items():
        common_reps = sorted(set(ranks.get(0, {})) & set(ranks.get(1, {})))
        if not common_reps:
            continue
        critical_collective = [
            max(
                float(ranks[0][rep]["collective_ms_per_iteration"]),
                float(ranks[1][rep]["collective_ms_per_iteration"]),
            )
            for rep in common_reps
        ]
        critical_legacy = [
            max(
                float(ranks[0][rep]["legacy_ms_per_iteration"]),
                float(ranks[1][rep]["legacy_ms_per_iteration"]),
            )
            for rep in common_reps
        ]
        rank_medians = {
            str(rank): statistics.median(
                float(rows[rep]["collective_ms_per_iteration"]) for rep in common_reps
            )
            for rank, rows in sorted(ranks.items())
            if all(rep in rows for rep in common_reps)
        }
        all_rows = [ranks[rank][rep] for rank in (0, 1) for rep in common_reps]
        collective = stats(critical_collective)
        legacy = stats(critical_legacy)
        bandwidth = payload_gbits_per_host * 1000 / collective["median"]

        experiment_warmups = warmups.get(experiment, {})
        warmup_critical = None
        if 0 in experiment_warmups and 1 in experiment_warmups:
            warmup_critical = max(
                float(experiment_warmups[0]["collective_ms_per_iteration"]),
                float(experiment_warmups[1]["collective_ms_per_iteration"]),
            )
        experiment_first_calls = first_calls.get(experiment, {})
        first_call_critical = None
        if 0 in experiment_first_calls and 1 in experiment_first_calls:
            first_call_critical = max(
                float(experiment_first_calls[0]["compile_execute_ms"]),
                float(experiment_first_calls[1]["compile_execute_ms"]),
            )

        experiments.append(
            {
                "experiment": experiment,
                "paired_reps": len(common_reps),
                "libtpu_init_args": configs.get(experiment, {}).get(
                    "libtpu_init_args", ""
                ),
                "rank_collective_median_ms": rank_medians,
                "first_call_critical_ms": first_call_critical,
                "warmup_critical_ms": warmup_critical,
                "critical_collective_ms": collective,
                "critical_legacy_ms": legacy,
                "critical_gbps": bandwidth,
                "link_efficiency_pct": bandwidth / link_gbps * 100,
                "tx_nic_imbalance_pct": stats(
                    [nic_imbalance(row, "tx_bytes") for row in all_rows]
                ),
                "rx_nic_imbalance_pct": stats(
                    [nic_imbalance(row, "rx_bytes") for row in all_rows]
                ),
            }
        )

    baseline_start = next(
        (row for row in experiments if row["experiment"].endswith("baseline_start")),
        None,
    )
    baseline_end = next(
        (row for row in experiments if row["experiment"].endswith("baseline_end")),
        None,
    )
    if baseline_start and baseline_end:
        reference = (
            baseline_start["critical_gbps"] + baseline_end["critical_gbps"]
        ) / 2
    elif baseline_start:
        reference = baseline_start["critical_gbps"]
    else:
        reference = None
    for row in experiments:
        row["relative_to_baseline_pct"] = (
            (row["critical_gbps"] / reference - 1) * 100 if reference else None
        )
    return {
        "payload_gbits_per_host": payload_gbits_per_host,
        "link_gbps": link_gbps,
        "baseline_reference_gbps": reference,
        "experiments": experiments,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "| Experiment | Paired reps | Warmup critical | Critical median | p05-p95 | Bandwidth | Efficiency | vs baseline | TX/RX NIC imbalance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["experiments"]:
        collective = row["critical_collective_ms"]
        warmup = row["warmup_critical_ms"]
        relative = row["relative_to_baseline_pct"]
        lines.append(
            f"| {row['experiment']} | {row['paired_reps']} | "
            f"{warmup:.3f} ms | {collective['median']:.3f} ms | "
            f"{collective['p05']:.3f}-{collective['p95']:.3f} ms | "
            f"{row['critical_gbps']:.3f} Gbps | "
            f"{row['link_efficiency_pct']:.2f}% | {relative:+.2f}% | "
            f"{row['tx_nic_imbalance_pct']['median']:.2f}%/"
            f"{row['rx_nic_imbalance_pct']['median']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def load_log_lines(paths: list[Path]) -> list[str]:
    lines = []
    for path in paths:
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--payload-gbits-per-host",
        type=float,
        default=DEFAULT_PAYLOAD_GBITS_PER_HOST,
    )
    parser.add_argument("--link-gbps", type=float, default=DEFAULT_LINK_GBPS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze(
        parse(load_log_lines(args.logs)),
        payload_gbits_per_host=args.payload_gbits_per_host,
        link_gbps=args.link_gbps,
    )
    output = (
        json.dumps(report, indent=2, sort_keys=True) if args.json else render(report)
    )
    print(output, end="" if output.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
