from __future__ import annotations

import json
import unittest

from training.contracts import repository_root


class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = repository_root() / "notebooks" / "development-model-comparison.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_notebook_is_valid_basic_v4_structure(self) -> None:
        self.assertEqual(4, self.notebook["nbformat"])
        self.assertGreater(len(self.notebook["cells"]), 0)
        self.assertTrue(
            all(cell["cell_type"] in {"markdown", "code"} for cell in self.notebook["cells"])
        )

    def test_committed_notebook_contains_successful_saved_outputs(self) -> None:
        code_cells = [
            cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]

        self.assertTrue(code_cells)
        self.assertTrue(all(cell["execution_count"] is not None for cell in code_cells))
        self.assertTrue(any(cell["outputs"] for cell in code_cells))
        self.assertFalse(
            any(
                output.get("output_type") == "error"
                for cell in code_cells
                for output in cell["outputs"]
            )
        )

    def test_notebook_records_frozen_evidence_and_privacy_boundary(self) -> None:
        source = "\n".join(
            line
            for cell in self.notebook["cells"]
            for line in cell.get("source", [])
        )

        self.assertIn("comparison['final_test_evaluated'] is False", source)
        self.assertIn("final['final_test_evaluated'] is True", source)
        self.assertIn("models_refit_or_adjusted_on_final_test", source)
        self.assertIn("thresholds_adjusted_on_final_test", source)
        self.assertIn("development-model-comparison-report.json", source)
        self.assertIn("development-calibration-report.json", source)
        self.assertIn("development-decision-report.json", source)
        self.assertIn("final-locked-test-report.json", source)
        self.assertIn("Youden's J", source)
        self.assertIn("Five predefined model types", source)
        self.assertIn("Comparison with the Logistic Regression benchmark", source)
        self.assertIn("benchmark_candidate = 'logistic_l2_balanced'", source)
        self.assertIn("Locked-test results were not used for this comparison", source)
        self.assertNotIn("data/processed", source)


if __name__ == "__main__":
    unittest.main()
