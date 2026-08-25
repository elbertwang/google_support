import json
from pathlib import Path
import tempfile
import unittest

from dcn.results import comparison_rows, load_rows, select_results


def _row(variant, dim, median, best):
  row = {
      "benchmark": "dcn_google_baseline",
      "variant": variant,
      "matrix_dim": dim,
      "shard_bytes_per_device": 268435456,
      "time_ms_median": 100.0,
  }
  if variant == "all_reduce":
    row["host_ring_equivalent_bus_GBps_median"] = median
    row["host_ring_equivalent_bus_GBps_best"] = best
  else:
    row["host_tx_GBps_median"] = median
    row["host_tx_GBps_best"] = best
  return row


class ResultsTest(unittest.TestCase):

  def test_selection_uses_best_bandwidth_and_reports_corresponding_median(self):
    rows = []
    for variant in ("ppermute_uni", "ppermute_bidi", "all_gather", "all_reduce"):
      rows.extend([_row(variant, 8192, 1.0, 2.0), _row(variant, 16384, 3.0, 4.0)])
    with tempfile.TemporaryDirectory() as tmp_dir:
      metrics = Path(tmp_dir) / "metrics.jsonl"
      metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

      loaded = load_rows(metrics)
      selected = select_results(loaded)
      self.assertEqual({row["matrix_dim"] for row in selected.values()}, {16384})
      compared = comparison_rows(loaded)
      self.assertEqual({row["measured_gbps"] for row in compared}, {24.0})
