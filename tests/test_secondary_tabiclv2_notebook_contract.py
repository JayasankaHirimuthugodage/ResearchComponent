from __future__ import annotations

import json
from pathlib import Path
import unittest


class SecondaryTabICLv2NotebookContractTests(unittest.TestCase):
    def test_notebook_is_aggregate_only_and_executed(self) -> None:
        path = Path("notebooks/secondary-tabiclv2-test-evaluation.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("secondary-tabiclv2-test-evaluation-report.json", source)
        self.assertIn("previously_evaluated_test_cohort", source)
        self.assertNotIn("load_final_test_dataset", source)
        self.assertNotIn("participant_number", source)
        self.assertNotIn("predict_proba", source)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is not None for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
