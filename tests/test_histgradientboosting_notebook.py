from __future__ import annotations
import json
from pathlib import Path
import unittest
class HGBNotebookTests(unittest.TestCase):
 def test_notebook_aggregate_only_and_executed(self):
  nb=json.loads(Path('notebooks/development-histgradientboosting-comparison.ipynb').read_text());src='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
  self.assertIn('development-histgradientboosting-comparison-report.json',src);self.assertNotIn('load_final_test_dataset',src);self.assertNotIn('participant_number',src);self.assertTrue(all(c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code'))
if __name__=='__main__':unittest.main()
