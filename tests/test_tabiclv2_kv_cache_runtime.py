from __future__ import annotations
import unittest
from pathlib import Path
from unittest.mock import patch
from inference.tabiclv2_kv_cache_runtime import CACHE_RUNTIME_AUTHORIZATION, COGNITIVE_EXPLANATION_STATUS, KVCacheReleaseContract, validate_loaded_cache, verify_cache_artifact, build_kv_cache_experimental_runtime_service
from inference.questionnaire import load_questionnaire_catalog

class _Generator:
 X_=None; y_=None; preprocessors_={}
class _CacheModel:
 ensemble_generator_=_Generator(); model_kv_cache_={"x":object()}
 def __init__(self,path): self.model_path_=str(path)

class KVCacheRuntimeTests(unittest.TestCase):
 def test_untrusted_or_tampered_cache_is_rejected_before_deserialization(self):
  with self.assertRaises(ValueError): verify_cache_artifact(Path('untrusted.pkl'),'x',1)
 def test_loaded_cache_requires_no_training_arrays_and_checkpoint_match(self):
  checkpoint=Path('checkpoint.ckpt').resolve(); validate_loaded_cache(_CacheModel(checkpoint),checkpoint)
  bad=_CacheModel(checkpoint); bad.ensemble_generator_=type('BadGenerator',(),{'X_':object(),'y_':None,'preprocessors_':{}})()
  with self.assertRaises(ValueError): validate_loaded_cache(bad,checkpoint)
  with self.assertRaises(ValueError): validate_loaded_cache(_CacheModel(Path('other.ckpt').resolve()),checkpoint)
  self.assertEqual('unavailable_pending_model_specific_validation',COGNITIVE_EXPLANATION_STATUS)
 def test_factory_blocks_before_any_io_without_explicit_authorization(self):
  with self.assertRaises(PermissionError):
   build_kv_cache_experimental_runtime_service(load_questionnaire_catalog(),repository_root=Path('.'),composite_manifest=Path('missing.json'),checkpoint=Path('missing.ckpt'))
