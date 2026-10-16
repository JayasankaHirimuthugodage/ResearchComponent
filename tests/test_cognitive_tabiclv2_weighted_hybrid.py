from __future__ import annotations
import unittest
import numpy as np
from training.experiments.run_cognitive_tabiclv2_weighted_hybrid import WEIGHTS, select_weight
class WeightedHybridContractTests(unittest.TestCase):
 def test_weight_grid_is_locked(self): self.assertEqual(WEIGHTS,(0.0,0.25,0.5,0.75,1.0))
 def test_selector_returns_predeclared_weight(self):
  y=np.array([0,0,1,1]); w,s=select_weight(y,np.array([.1,.2,.7,.9]),np.array([.2,.3,.6,.8])); self.assertIn(w,WEIGHTS); self.assertEqual(set(s),set(WEIGHTS))
if __name__=='__main__': unittest.main()
