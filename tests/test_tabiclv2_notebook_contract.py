from __future__ import annotations

import json
import unittest

from training.contracts import repository_root


class TabICLv2NotebookContractTests(unittest.TestCase):
    def test_notebook_reads_only_aggregate_reports(self) -> None:
        path = repository_root() / "notebooks" / "development-tabiclv2-model-comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("development-tabiclv2-model-comparison-report.json", source)
        self.assertIn("development-additional-model-comparison-report.json", source)
        self.assertIn("final_test_evaluated", source)
        self.assertNotIn("load_development_dataset", source)
        self.assertNotIn("run_tabiclv2_evaluation", source)
        self.assertNotIn("load_final_test_dataset", source)


if __name__ == "__main__":
    unittest.main()

