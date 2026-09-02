from pathlib import Path
import tempfile
import unittest

from dcn.hlo_summary import summarize


class HloSummaryTest(unittest.TestCase):

    def test_prefers_optimized_hlo_and_counts_collective_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "module.before_optimizations.txt").write_text(
                "all-reduce sparse", encoding="utf-8"
            )
            (root / "module.after_optimizations.txt").write_text(
                "all-reduce all-reduce collective megascale aggregator",
                encoding="utf-8",
            )

            report = summarize(root)

        self.assertEqual(report["text_file_count"], 2)
        self.assertEqual(report["selected_file_count"], 1)
        self.assertEqual(report["token_counts"]["all-reduce"], 2)
        self.assertEqual(report["token_counts"]["collective"], 1)
        self.assertEqual(report["token_counts"]["megascale"], 1)
        self.assertEqual(report["token_counts"]["aggregator"], 1)
        self.assertEqual(report["token_counts"]["sparse"], 0)


if __name__ == "__main__":
    unittest.main()
