from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from training.experiments.prepare_cognitive_tabiclv2_kv_cache import ENGINEERING_TOLERANCE, TOLERANCE, sigmoid, comparison, persist_equivalence_diagnostic, synthetic_features
class KVCacheRunnerTests(unittest.TestCase):
 def test_frozen_calibration_and_inclusive_threshold_are_stable(self):
  import numpy as np
  values=sigmoid(np.asarray([.01,.15,.99])); self.assertTrue(np.all((values>=0)&(values<=1))); self.assertGreater(TOLERANCE,0)
 def test_default_execution_is_blocked(self):
  from training.experiments.prepare_cognitive_tabiclv2_kv_cache import main
  self.assertEqual(2,main([]))
 def test_aggregate_diagnostic_captures_material_difference_without_rows(self):
  report=comparison([.1,.2,.3],[.1,.20001,.3],.1503614235173848)
  self.assertEqual(3,report['count']); self.assertGreater(report['raw_max_absolute_difference'],TOLERANCE)
  self.assertNotIn('raw',str(report).split('raw_max_absolute_difference')[0])
  self.assertNotIn('probabilities',str(report).lower())
 def test_threshold_change_is_reported_independently_of_tolerance(self):
  report=comparison([.1387],[.1389],.1503614235173848)
  self.assertIn('changed_threshold_decisions',report)
 def test_aggregate_diagnostic_is_persisted_before_a_caller_raises(self):
  with tempfile.TemporaryDirectory() as value:
   diagnostic={'comparisons':{'direct_context_vs_cache_before_serialization':comparison([.1],[.2],.1503614235173848)},'synthetic_assessments':1}
   path=persist_equivalence_diagnostic(Path(value),diagnostic)
   loaded=json.loads(path.read_text()); self.assertEqual(1,loaded['synthetic_assessments'])
   self.assertNotIn('participant',json.dumps(loaded).lower())
 def test_frozen_engineering_suite_is_128_requests(self):
  self.assertEqual((128,16),synthetic_features(True).shape)
  self.assertGreater(ENGINEERING_TOLERANCE,TOLERANCE)
