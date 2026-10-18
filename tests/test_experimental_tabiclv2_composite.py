from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.contracts import AssessmentStatus, QuestionnaireSubmission
from inference.experimental_composite import build_experimental_service, load_experimental_composite_release
from inference.questionnaire import load_questionnaire_catalog
from inference.tabiclv2_adapter import CognitiveReleaseContract, TabICLv2CognitiveAdapter
from inference.tabiclv2_context import InMemoryContextCache
from training.contracts import FEATURE_COLUMNS
from training.tabiclv2.candidate import TABICLV2_CHECKPOINT_SHA256


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()

class FakeModel:
    def __init__(self, probability: float) -> None: self.probability, self.calls = probability, 0
    def positive_probability(self, values): self.calls += 1; return self.probability

class FakeTabICL:
    def __init__(self, probability: float) -> None: self.probability, self.calls = probability, 0
    def predict_proba(self, values): self.calls += 1; return np.array([[1-self.probability, self.probability]])

class CompositeTests(unittest.TestCase):
    def release(self, root: Path) -> Path:
        for name in ('communication.joblib','social.joblib','rollback.joblib','rollback.json'):(root/name).write_text(name)
        (root/'communication.json').write_text(json.dumps({'domain':'communication','threshold':{'value':.3}}));(root/'social.json').write_text(json.dumps({'domain':'social_emotional','threshold':{'value':.8}}))
        cognitive=root/'cognitive.json'; cognitive.write_text(json.dumps({'status':'development_only_experimental_not_deployed','domain':'cognitive','checkpoint_sha256':TABICLV2_CHECKPOINT_SHA256,'feature_contract':list(FEATURE_COLUMNS),'calibration':{'coefficient':1.069105693175016,'intercept':.255897163477147},'threshold':{'value':.1503614235173848}}))
        index=root/'index.json'; index.write_text(json.dumps({'artifacts':{'release_manifest':{'sha256':sha(cognitive)}}}))
        pair=lambda artifact,manifest:{'artifact':artifact,'artifact_sha256':sha(root/artifact),'manifest':manifest,'manifest_sha256':sha(root/manifest)}
        value={'schema':'experimental-three-domain-composite-v1','release_id':'slc-development-models-tabiclv2-cognitive-v1','status':'experimental_not_deployed','domains':{'communication':pair('communication.joblib','communication.json'),'cognitive':{'manifest':'cognitive.json','manifest_sha256':sha(cognitive),'integrity_index':'index.json','integrity_index_sha256':sha(index)},'social_emotional':pair('social.joblib','social.json')},'rollback_cognitive':pair('rollback.joblib','rollback.json')}
        path=root/'composite.json';path.write_text(json.dumps(value));return path
    def adapter(self, raw=.2): return TabICLv2CognitiveAdapter(InMemoryContextCache(lambda:FakeTabICL(raw)),CognitiveReleaseContract(TABICLV2_CHECKPOINT_SHA256,1.069105693175016,.255897163477147,.1503614235173848))
    def submission(self, incomplete=False):
        catalog=load_questionnaire_catalog(); checkpoint='CP24M'; return QuestionnaireSubmission(catalog.questionnaire_schema_version,checkpoint,{q.question_id:('NOT_SURE' if incomplete and i==0 else 'YES_USUALLY') for i,q in enumerate(catalog.questions_for_checkpoint(checkpoint))})
    def test_validates_explicit_composite_and_preserves_rollback_reference(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); release=load_experimental_composite_release(self.release(root),root);self.assertEqual(root/'rollback.joblib',release.rollback_cognitive_artifact)
    def test_tampered_composite_reference_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); path=self.release(root);(root/'communication.joblib').write_text('changed')
            with self.assertRaises(ValueError):load_experimental_composite_release(path,root)
    def test_complete_and_incomplete_contracts_and_or_rule(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); release=load_experimental_composite_release(self.release(root),root); loaded={'communication.joblib':FakeModel(.1),'social.joblib':FakeModel(.9)};service=build_experimental_service(load_questionnaire_catalog(),release,self.adapter(.2),artifact_loader=lambda path:loaded[path.name])
            complete=service.predict(self.submission());self.assertIs(complete.status,AssessmentStatus.COMPLETE);self.assertEqual(('communication','cognitive','social_emotional'),tuple(item.domain for item in complete.domain_predictions));self.assertTrue(complete.overall_concern);self.assertEqual((.3,.1503614235173848,.8),tuple(item.threshold for item in complete.domain_predictions))
            before=(loaded['communication.joblib'].calls,loaded['social.joblib'].calls)
            incomplete=service.predict(self.submission(True));self.assertIs(incomplete.status,AssessmentStatus.INCOMPLETE);self.assertEqual(before,(loaded['communication.joblib'].calls,loaded['social.joblib'].calls))
    def test_historical_default_identifier_is_unchanged(self):
        source=Path('inference/frozen_models.py').read_text(encoding='utf-8')
        self.assertIn('MODEL_RELEASE_VERSION = "slc-development-models-v1"',source)
