import json
import unittest

from dcn.timing_results import analyze, parse, render


def _line(kind, **payload):
    return f"prefix DCN_TIMING_{kind} {json.dumps(payload)}"


def _experiment_lines(name, rank, collective_values, legacy_values):
    lines = [
        _line(
            "CONFIG",
            experiment=name,
            libtpu_init_args="--example=true",
            grpc_experiments="",
        ),
        _line(
            "FIRST_CALL",
            experiment=name,
            rank=rank,
            compile_execute_ms=250.0 + rank,
        ),
        _line(
            "WARMUP",
            experiment=name,
            rank=rank,
            collective_ms_per_iteration=100.0 + rank,
        ),
    ]
    for rep, (collective, legacy) in enumerate(
        zip(collective_values, legacy_values, strict=True)
    ):
        lines.append(
            _line(
                "REP",
                experiment=name,
                rank=rank,
                rep=rep,
                collective_ms_per_iteration=collective,
                legacy_ms_per_iteration=legacy,
                network_delta={
                    "eth1": {"tx_bytes": 49, "rx_bytes": 50},
                    "eth2": {"tx_bytes": 51, "rx_bytes": 50},
                },
            )
        )
    return lines


class TimingResultsTest(unittest.TestCase):

    def test_analysis_pairs_ranks_and_excludes_trailing_barrier(self):
        lines = []
        cases = {
            "test_baseline_start": ([10.0, 14.0], [11.0, 15.0]),
            "test_candidate": ([9.0, 13.0], [10.0, 14.0]),
            "test_baseline_end": ([12.0, 16.0], [13.0, 17.0]),
        }
        for name, (rank0, rank1) in cases.items():
            lines += _experiment_lines(name, 0, rank0, [value + 10 for value in rank0])
            lines += _experiment_lines(name, 1, rank1, [value + 10 for value in rank1])

        report = analyze(parse(lines), payload_gbits_per_host=1.0, link_gbps=100.0)
        rows = {row["experiment"]: row for row in report["experiments"]}
        baseline_start = rows["test_baseline_start"]
        candidate = rows["test_candidate"]

        # Paired critical samples are max(rank0, rank1): [11, 15]. The trailing
        # barrier appears only in legacy_ms_per_iteration and must not affect them.
        self.assertEqual(baseline_start["critical_collective_ms"]["median"], 13.0)
        self.assertEqual(baseline_start["critical_legacy_ms"]["median"], 23.0)
        self.assertEqual(baseline_start["paired_reps"], 2)
        self.assertEqual(baseline_start["first_call_critical_ms"], 251.0)
        self.assertEqual(baseline_start["warmup_critical_ms"], 101.0)
        self.assertAlmostEqual(baseline_start["critical_gbps"], 1000 / 13)

        expected_reference = ((1000 / 13) + (1000 / 15)) / 2
        self.assertAlmostEqual(report["baseline_reference_gbps"], expected_reference)
        self.assertAlmostEqual(
            candidate["relative_to_baseline_pct"],
            ((1000 / 12) / expected_reference - 1) * 100,
        )
        self.assertAlmostEqual(candidate["tx_nic_imbalance_pct"]["median"], 2.0)
        self.assertIn("test_candidate", render(report))

    def test_analysis_uses_only_rep_ids_present_on_both_ranks(self):
        lines = _experiment_lines("test_baseline_start", 0, [10.0, 20.0], [11.0, 21.0])
        lines += _experiment_lines("test_baseline_start", 1, [12.0], [13.0])

        row = analyze(parse(lines))["experiments"][0]
        self.assertEqual(row["paired_reps"], 1)
        self.assertEqual(row["critical_collective_ms"]["median"], 12.0)


if __name__ == "__main__":
    unittest.main()
