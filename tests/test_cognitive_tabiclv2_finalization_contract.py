from pathlib import Path
import json, unittest
from unittest.mock import patch
import tempfile
SOURCE=Path('training/experiments/run_cognitive_tabiclv2_finalization.py')
class ContractTests(unittest.TestCase):
 def test_safeguards_are_declared(self):
  s=SOURCE.read_text(encoding='utf-8');self.assertIn('target_cognitive_concern',s);self.assertIn('FEATURE_COLUMNS',s);self.assertIn('FINGERPRINT',s);self.assertIn('tabicl version mismatch',s);self.assertIn('Finalization is disabled by default',s)
 def test_aggregate_privacy_guard(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import validate_aggregate
  validate_aggregate({'scope':'development_only','metrics':{'accuracy':.8}})
  with self.assertRaises(ValueError):validate_aggregate({'participant_key':'x'})
 def test_template_is_json(self):
  value={'status':'experimental_template','domain':'cognitive','privacy':{'participant_identifiers_written':False}};json.dumps(value)
 def test_detached_index_rejects_changed_artifact(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import write_integrity_index,verify_integrity_index
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=root/'report.json';a.write_text('{"ok":true}');i=root/'integrity.json';write_integrity_index(i,{'report':a});verify_integrity_index(i,root);a.write_text('{"changed":true}')
   with self.assertRaises(ValueError):verify_integrity_index(i,root)
 def test_manifest_integrity_and_atomic_output(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import _atomic_json,verify_release_manifest
  from training.contracts import FEATURE_COLUMNS
  value={'artifact_schema_version':'x','status':'development_only_experimental_not_deployed','domain':'cognitive','checkpoint_sha256':'x','tabicl_version':'2.2.0','configuration':{},'feature_contract':list(FEATURE_COLUMNS),'calibration':{},'threshold':{'value':.2},'privacy':{'participant_identifiers_written':False,'row_level_predictions_written':False},'limitations':[]}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'release.json';_atomic_json(p,value);self.assertEqual('cognitive',verify_release_manifest(p)['domain'])
   with self.assertRaises(FileExistsError):_atomic_json(p,value)
   value['threshold']['value']=2;Path(d,'bad.json').write_text(json.dumps(value));
   with self.assertRaises(ValueError):verify_release_manifest(Path(d)/'bad.json')
 def test_repeat_alignment_and_averaging(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import validate_repeat_probabilities,record_outer_probabilities
  repeats=[__import__('numpy').full(500,float(i)) for i in range(5)]
  average=validate_repeat_probabilities(repeats);self.assertEqual((500,),average.shape);self.assertTrue((average==2).all())
  with self.assertRaises(ValueError):validate_repeat_probabilities(repeats[:4])
  duplicate=__import__('numpy').full(500,__import__('numpy').nan)
  with self.assertRaises(ValueError):validate_repeat_probabilities(repeats[:-1]+[duplicate])
  with self.assertRaises(ValueError):validate_repeat_probabilities([__import__('numpy').zeros(499)]*5)
  destination=__import__('numpy').full(500,__import__('numpy').nan);record_outer_probabilities(destination,__import__('numpy').array([0,499]),__import__('numpy').array([.1,.9]));self.assertEqual(.9,destination[499])
  with self.assertRaises(ValueError):record_outer_probabilities(destination,__import__('numpy').array([499]),__import__('numpy').array([.8]))
  with self.assertRaises(ValueError):record_outer_probabilities(destination,__import__('numpy').array([500]),__import__('numpy').array([.8]))
 def test_runtime_limit_stops_before_release_completion(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import enforce_runtime_limit,LIMIT
  with self.assertRaises(RuntimeError):enforce_runtime_limit(0,1,clock=lambda:LIMIT)
  enforce_runtime_limit(0,0,clock=lambda:LIMIT+1)
 def test_checkpoint_and_package_contract(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import validate_checkpoint_and_package
  with tempfile.TemporaryDirectory() as d:
   checkpoint=Path(d)/'missing.ckpt'
   with self.assertRaises(ValueError):validate_checkpoint_and_package(checkpoint)
   checkpoint=Path(d)/'tabicl-classifier-v2-20260212.ckpt';checkpoint.write_bytes(b'synthetic')
   with patch('training.experiments.run_cognitive_tabiclv2_finalization.sha',return_value='BAD'):
    with self.assertRaises(ValueError):validate_checkpoint_and_package(checkpoint)
   with patch('training.experiments.run_cognitive_tabiclv2_finalization.sha',return_value='BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0'),patch('training.experiments.run_cognitive_tabiclv2_finalization.importlib.metadata.version',return_value='wrong'):
    with self.assertRaises(ValueError):validate_checkpoint_and_package(checkpoint)
   with patch('training.experiments.run_cognitive_tabiclv2_finalization.sha',return_value='BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0'),patch('training.experiments.run_cognitive_tabiclv2_finalization.importlib.metadata.version',return_value='2.2.0'):
    validate_checkpoint_and_package(checkpoint)
 def test_mocked_end_to_end_output_contract(self):
  from training.experiments.run_cognitive_tabiclv2_finalization import _atomic_json,write_integrity_index,verify_integrity_index,verify_release_manifest,verify_release_bundle
  from training.contracts import FEATURE_COLUMNS
  release={'artifact_schema_version':'experimental-cognitive-tabiclv2-v1','status':'development_only_experimental_not_deployed','domain':'cognitive','checkpoint_sha256':'synthetic','tabicl_version':'2.2.0','configuration':{'n_estimators':8},'feature_contract':list(FEATURE_COLUMNS),'calibration':{'method':'synthetic'},'threshold':{'value':.5},'privacy':{'participant_identifiers_written':False,'row_level_predictions_written':False},'limitations':['synthetic']}
  report={'status':'completed_development_only_experimental_finalization','scope':'development_only','test_cohort_loaded':False,'final_release':release,'privacy':release['privacy']}
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);rp=root/'report.json';mp=root/'release.json';ip=root/'integrity.json';_atomic_json(rp,report);_atomic_json(mp,release);self.assertEqual('cognitive',verify_release_manifest(mp)['domain']);write_integrity_index(ip,{'report':rp,'manifest':mp},root);verify_integrity_index(ip,root);verify_release_bundle(mp,ip,root)
   with self.assertRaises(FileExistsError):_atomic_json(rp,report)
   mp.unlink()
   with self.assertRaises(ValueError):verify_integrity_index(ip,root)
 def test_default_execution_is_refused_before_loading_data(self):
  from training.experiments import run_cognitive_tabiclv2_finalization as runner
  with patch('sys.argv',['runner']),patch.object(runner,'load_development_dataset',side_effect=AssertionError('must not load')):
   with self.assertRaises(RuntimeError):runner.main()
if __name__=='__main__':unittest.main()
