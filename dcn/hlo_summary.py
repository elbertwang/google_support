"""Summarize collective-related tokens in XLA HLO text dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKENS = (
    "all-reduce",
    "all-reduce-start",
    "all-reduce-done",
    "collective",
    "megascale",
    "sparse",
    "aggregator",
)


def summarize(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.txt"))
    optimized = [path for path in files if "after_optimizations" in path.name]
    selected = optimized or files
    token_counts = {token: 0 for token in TOKENS}
    matching_files = []
    for path in selected:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        counts = {token: text.count(token) for token in TOKENS}
        for token, count in counts.items():
            token_counts[token] += count
        if counts["all-reduce"] or counts["sparse"] or counts["aggregator"]:
            matching_files.append({"file": path.name, "counts": counts})
    return {
        "root": str(root),
        "text_file_count": len(files),
        "selected_file_count": len(selected),
        "token_counts": token_counts,
        "matching_files": matching_files[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--experiment", default="unknown")
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    report.update({"experiment": args.experiment, "rank": args.rank})
    print("DCN_HLO_SUMMARY " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
