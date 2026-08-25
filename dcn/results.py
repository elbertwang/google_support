"""Summarize DCN benchmark JSONL and compare it with the documented baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_GBPS = {
    "ppermute_uni": 364.726,
    "ppermute_bidi": 274.932,
    "all_gather": 254.010,
    "all_reduce": 172.084,
}


def load_rows(path: Path) -> list[dict[str, Any]]:
  rows = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    row = json.loads(line)
    if row.get("benchmark") != "dcn_google_baseline":
      raise ValueError(f"{path}:{line_number}: unexpected benchmark")
    rows.append(row)
  return rows


def select_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  selected = {}
  for variant in BASELINE_GBPS:
    candidates = [row for row in rows if row.get("variant") == variant]
    if not candidates:
      raise ValueError(f"missing variant: {variant}")
    selection_key = (
        "host_ring_equivalent_bus_GBps_best"
        if variant == "all_reduce"
        else "host_tx_GBps_best"
    )
    selected[variant] = max(candidates, key=lambda row: float(row[selection_key]))
  return selected


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  output = []
  for variant, row in select_results(rows).items():
    metric = (
        "host_ring_equivalent_bus_GBps_median"
        if variant == "all_reduce"
        else "host_tx_GBps_median"
    )
    measured_gbps = float(row[metric]) * 8.0
    baseline_gbps = BASELINE_GBPS[variant]
    output.append({
        "variant": variant,
        "matrix_dim": int(row["matrix_dim"]),
        "shard_mib_per_device": float(row["shard_bytes_per_device"]) / 2**20,
        "median_ms": float(row["time_ms_median"]),
        "measured_gbps": measured_gbps,
        "baseline_gbps": baseline_gbps,
        "ratio": measured_gbps / baseline_gbps,
    })
  return output


def render_markdown(comparisons: list[dict[str, Any]]) -> str:
  lines = [
      "| Variant | Dim | Shard/device | Median | Reproduced | Baseline | Ratio |",
      "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for row in comparisons:
    lines.append(
        f"| `{row['variant']}` | {row['matrix_dim']:,} | "
        f"{row['shard_mib_per_device']:,.0f} MiB | {row['median_ms']:.3f} ms | "
        f"{row['measured_gbps']:.3f} Gbps | {row['baseline_gbps']:.3f} Gbps | "
        f"{row['ratio']:.3f}x |"
    )
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("metrics", type=Path)
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--min-ratio", type=float, default=None)
  args = parser.parse_args()
  comparisons = comparison_rows(load_rows(args.metrics))
  print(json.dumps(comparisons, indent=2) if args.json else render_markdown(comparisons), end="")
  if args.min_ratio is not None:
    failed = [row for row in comparisons if row["ratio"] < args.min_ratio]
    if failed:
      raise SystemExit(1)


if __name__ == "__main__":
  main()
