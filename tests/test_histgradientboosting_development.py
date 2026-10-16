from __future__ import annotations
import inspect,unittest
from training.experiments.run_histgradientboosting_development import CONFIG,NAME,predict
class HGBTests(unittest.TestCase):
 def test_fixed_safe_configuration(self):
  self.assertEqual(CONFIG['early_stopping'],False);self.assertEqual(CONFIG['class_weight'],'balanced');self.assertEqual(CONFIG['max_iter'],150)
 def test_runner_is_development_only(self):
  s=inspect.getsource(__import__('training.experiments.run_histgradientboosting_development',fromlist=['x']));self.assertNotIn('load_final_test_dataset',s);self.assertIn(NAME,s)
if __name__=='__main__':unittest.main()
