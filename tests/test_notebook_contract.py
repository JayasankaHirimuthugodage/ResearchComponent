from __future__ import annotations

import json
import unittest

from training.contracts import (
    DECISION_THRESHOLD_APPROVAL_STATEMENT,
    repository_root,
)


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

    def test_committed_notebook_has_no_execution_outputs(self) -> None:
        code_cells = [
            cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]

        self.assertTrue(code_cells)
        self.assertTrue(all(cell["execution_count"] is None for cell in code_cells))
        self.assertTrue(all(cell["outputs"] == [] for cell in code_cells))

    def test_notebook_records_approval_and_final_test_guard(self) -> None:
        source = "\n".join(
            line
            for cell in self.notebook["cells"]
            for line in cell.get("source", [])
        )

        self.assertIn(DECISION_THRESHOLD_APPROVAL_STATEMENT, source)
        self.assertIn("report['final_test_evaluated'] is False", source)
        self.assertIn("development-model-comparison-report.json", source)
        self.assertIn("extra_trees_balanced_leaf5", source)
        self.assertIn("experimental_not_approved_for_model", source)
        self.assertNotIn("data/processed", source)


if __name__ == "__main__":
    unittest.main()
