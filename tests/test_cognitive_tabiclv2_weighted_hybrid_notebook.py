from __future__ import annotations
import json
from pathlib import Path
import unittest
class WeightedHybridNotebookTests(unittest.TestCase):
 def test_notebook_is_aggregate_only_and_executed(self):
  nb=json.loads(Path('notebooks/development-cognitive-tabiclv2-weighted-hybrid.ipynb').read_text(encoding='utf-8'))
  source='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
  self.assertIn('development-cognitive-tabiclv2-weighted-hybrid-report.json',source)
  self.assertNotIn('load_development_dataset',source); self.assertNotIn('predict_proba',source); self.assertNotIn('participant_number',source)
  self.assertTrue(all(c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code'))
if __name__=='__main__': unittest.main()
