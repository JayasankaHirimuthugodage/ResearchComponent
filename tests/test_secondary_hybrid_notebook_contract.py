from __future__ import annotations

import json
import unittest

from training.contracts import repository_root


class SecondaryHybridNotebookContractTests(unittest.TestCase):
    def test_notebook_reads_only_aggregate_secondary_report(self) -> None:
        path = repository_root() / "notebooks" / "secondary-hybrid-test-evaluation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("secondary-exploratory-hybrid-evaluation-report.json", source)
        self.assertIn("cohort_previously_evaluated", source)
        self.assertIn("original_individual", source)
        self.assertIn("equal_weight_hybrid", source)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("run_secondary_hybrid_evaluation", source)
        self.assertNotIn("fit_candidate_pipeline", source)


if __name__ == "__main__":
    unittest.main()
