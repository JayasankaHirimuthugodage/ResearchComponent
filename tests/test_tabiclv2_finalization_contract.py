from __future__ import annotations

from pathlib import Path
import unittest


class TabICLv2FinalizationContractTests(unittest.TestCase):
    @staticmethod
    def source() -> str:
        return Path("training/experiments/run_tabiclv2_finalization.py").read_text(
            encoding="utf-8"
        )

    def test_runner_is_development_only_and_has_approval_gate(self):
        source = self.source()
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("secondary-tabiclv2-test-evaluation", source)
        self.assertIn('"final_test_evaluated": False', source)

    def test_final_threshold_is_not_fold_threshold_average(self):
        source = self.source()
        self.assertIn("final_tab_raw = np.mean", source)
        self.assertIn("select_balanced_accuracy_threshold(y, final_p)", source)
        self.assertNotIn("mean_threshold", source)

    def test_checkpoint_logical_filename_is_recorded(self):
        source = self.source()
        self.assertIn('"checkpoint_filename": TABICLV2_CHECKPOINT_NAME', source)


if __name__ == "__main__":
    unittest.main()
