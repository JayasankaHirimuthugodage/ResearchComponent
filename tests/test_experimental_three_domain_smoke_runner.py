from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.contracts import AssessmentStatus, DomainPrediction, PredictionResult
from inference.questionnaire import load_questionnaire_catalog
from inference.tabiclv2_context import ContextSourceAuthorization
from training.experiments.run_experimental_three_domain_localhost_smoke_test import (
    GOVERNANCE_ATTESTATION_SCHEMA, RUNTIME_CONFIGURATION_SCHEMA, main, load_context_authorization,
    load_governance_attestation, require_non_elevated_process, run_real_smoke_test,
)


class FakeService:
    def __init__(self): self.calls = 0
    def predict(self, submission):
        self.calls += 1
        if any(value == 'NOT_SURE' for value in submission.responses.values()):
            return PredictionResult(AssessmentStatus.INCOMPLETE, (), None, tuple(submission.responses)[:1], 'x', 'x', 'x', None, None)
        domains=tuple(DomainPrediction(domain, .2, True, .1, 'x') for domain in ('communication','cognitive','social_emotional'))
        return PredictionResult(AssessmentStatus.COMPLETE, domains, True, (), 'x', 'x', 'x', None, None)


class SmokeRunnerTests(unittest.TestCase):
    def files(self, root):
        governance=root/'governance.json'; governance.write_text(json.dumps({'schema':GOVERNANCE_ATTESTATION_SCHEMA,'custodian_runtime_permission':True,'interactive_account_authorized':True,'retention_and_backup_addressed':True,'localhost_only':True}))
        config=root/'context.json'; config.write_text(json.dumps({'schema':RUNTIME_CONFIGURATION_SCHEMA,'source_path':str(root/'source.csv'),'provenance_path':str(root/'provenance.json'),'source_sha256':'x','membership_sha256':'x','dataset_sha256':'x','split_fingerprint_sha256':'x','forbidden_combined_source':str(root/'combined.csv')}))
        return governance, config
    def test_default_cli_blocks_without_io(self): self.assertEqual(2,main([]))
    def test_governance_and_combined_source_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); governance,config=self.files(root); load_governance_attestation(governance); self.assertIsInstance(load_context_authorization(config),ContextSourceAuthorization)
            config.write_text(json.dumps({'schema':RUNTIME_CONFIGURATION_SCHEMA,'source_path':str(root/'combined.csv'),'provenance_path':'x','source_sha256':'x','membership_sha256':'x','dataset_sha256':'x','split_fingerprint_sha256':'x','forbidden_combined_source':str(root/'combined.csv')}))
            with self.assertRaises(ValueError): load_context_authorization(config)
            with self.assertRaises(PermissionError): require_non_elevated_process(lambda: True)
    def test_synthetic_service_contract_and_safe_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); governance,config=self.files(root); authorization=load_context_authorization(config); calls=[]
            def factory(*args, **kwargs): calls.append(kwargs); return FakeService()
            summary=run_real_smoke_test(repository_root=root,checkpoint=root/'checkpoint.ckpt',context_authorization=authorization,governance_attestation=governance,catalog=load_questionnaire_catalog(),service_factory=factory,is_elevated=lambda:False)
            self.assertTrue(summary.complete_contract_ok and summary.incomplete_contract_ok); self.assertEqual(1,len(calls)); self.assertNotIn('probability',json.dumps(summary.safe_dict()).lower())
