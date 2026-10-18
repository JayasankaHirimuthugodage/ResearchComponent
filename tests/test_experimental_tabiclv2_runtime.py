from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.experimental_runtime import ExperimentalRuntimeDependencies, RUNTIME_AUTHORIZATION, build_experimental_runtime_service, verify_cognitive_integrity_index
from inference.questionnaire import load_questionnaire_catalog
from inference.tabiclv2_context import ContextSourceAuthorization, VerifiedCognitiveContext
from tests.test_experimental_tabiclv2_composite import CompositeTests, FakeModel, FakeTabICL


class RuntimeFactoryTests(unittest.TestCase):
    def authorization(self, root):
        return ContextSourceAuthorization(root/'context.csv', root/'provenance.json', 'x', 'x', 'x', 'x', root/'combined.csv')

    def dependencies(self, calls):
        def artifact(path): calls.append('artifact:' + path.name); return FakeModel(.2 if 'communication' in path.name else .1)
        def checkpoint(path): calls.append('checkpoint')
        def context(auth):
            calls.append('context'); import numpy as np
            return VerifiedCognitiveContext(np.asarray([['CP24M', *('1',) * 15]], dtype=object), np.asarray([0], dtype='int8'), 'synthetic')
        def classifier(context, checkpoint): calls.append('fit'); return FakeTabICL(.2)
        return ExperimentalRuntimeDependencies(artifact, checkpoint, context, classifier)

    def test_default_factory_is_disabled_before_any_io(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); calls=[]
            with self.assertRaises(PermissionError): build_experimental_runtime_service(load_questionnaire_catalog(), repository_root=root, composite_manifest=root/'missing.json', checkpoint=root/'missing.ckpt', context_authorization=self.authorization(root), dependencies=self.dependencies(calls))
            self.assertEqual([], calls)

    def test_synthetic_factory_validates_and_caches_context_only_on_prediction(self):
        root=Path(__file__).resolve().parents[1]; calls=[]
        with patch('inference.experimental_runtime.importlib.metadata.version', return_value='2.2.0'):
            service=build_experimental_runtime_service(load_questionnaire_catalog(), repository_root=root, composite_manifest=root/'models'/'experimental-three-domain-tabiclv2-v1'/'release-manifest.json', checkpoint=root/'synthetic-checkpoint.ckpt', context_authorization=self.authorization(root), execution_authorization=RUNTIME_AUTHORIZATION, dependencies=self.dependencies(calls))
        self.assertEqual('checkpoint',calls[0]); self.assertEqual(2,calls.count('artifact:model.joblib'))
        service.cognitive.positive_probability(('CP24M', *('1',) * 15)); service.cognitive.positive_probability(('CP24M', *('1',) * 15))
        self.assertEqual(1,calls.count('context')); self.assertEqual(1,calls.count('fit'))

    def test_detached_integrity_rejects_altered_or_incomplete_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=CompositeTests().release(root)
            from inference.experimental_composite import load_experimental_composite_release
            release=load_experimental_composite_release(path,root)
            with self.assertRaises(ValueError): verify_cognitive_integrity_index(release,root)
            index=json.loads((root/'index.json').read_text()); index['artifacts']['aggregate_report']={'path':'cognitive.json','sha256':'bad'}; (root/'index.json').write_text(json.dumps(index))
            with self.assertRaises(ValueError): verify_cognitive_integrity_index(release,root)
