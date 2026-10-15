from __future__ import annotations

import json
import unittest

from training.contracts import repository_root


class HybridNotebookContractTests(unittest.TestCase):
    def test_notebook_is_a_report_reader_and_preserves_test_boundary(self) -> None:
        path = repository_root() / "notebooks" / "development-hybrid-model-comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("development-hybrid-model-comparison-report.json", source)
        self.assertIn("final_test_evaluated", source)
        self.assertIn("new untouched cohort", source)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("run_hybrid_development_evaluation", source)


if __name__ == "__main__":
    unittest.main()
