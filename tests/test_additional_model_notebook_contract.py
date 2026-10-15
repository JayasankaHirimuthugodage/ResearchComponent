from __future__ import annotations

import json
import unittest

from training.contracts import repository_root


class AdditionalModelNotebookContractTests(unittest.TestCase):
    def test_notebook_reads_only_aggregate_development_report(self) -> None:
        path = repository_root() / "notebooks" / "development-additional-model-comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("development-additional-model-comparison-report.json", source)
        self.assertIn("final_test_evaluated", source)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("run_additional_model_evaluation", source)


if __name__ == "__main__":
    unittest.main()
