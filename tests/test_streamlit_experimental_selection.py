from __future__ import annotations

import inspect
import unittest

from demo.development_prediction_demo import (
    DEFAULT_RELEASE_OPTION,
    EXPERIMENTAL_RELEASE_OPTION,
    HISTORICAL_RELEASE_OPTION,
    experimental_explanation_status,
    _experimental_explanation_key,
    selected_prediction_service,
    caregiver_domain_message,
)
from inference.contracts import AssessmentStatus, QuestionnaireSubmission
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from inference.tabiclv2_explanations import (
    TabICLv2ExplanationStatus,
)


class StreamlitExperimentalSelectionTests(unittest.TestCase):
    def encoded(self, incomplete=False):
        catalog=load_questionnaire_catalog(); checkpoint='CP24M'
        responses={item.question_id:'YES_USUALLY' for item in catalog.questions_for_checkpoint(checkpoint)}
        if incomplete: responses[next(iter(responses))]='NOT_SURE'
        return encode_submission(catalog,QuestionnaireSubmission(catalog.questionnaire_schema_version,checkpoint,responses))

    def test_experimental_release_is_the_default_and_historical_is_explicit(self):
        self.assertEqual(EXPERIMENTAL_RELEASE_OPTION,DEFAULT_RELEASE_OPTION)
        self.assertNotEqual(HISTORICAL_RELEASE_OPTION,EXPERIMENTAL_RELEASE_OPTION)

    def test_experimental_status_preserves_complete_and_incomplete_contracts(self):
        self.assertIs(self.encoded().status,AssessmentStatus.COMPLETE)
        self.assertIs(experimental_explanation_status(self.encoded()),TabICLv2ExplanationStatus.AVAILABLE_LOCAL_MODEL_BEHAVIOUR)
        self.assertIs(self.encoded(True).status,AssessmentStatus.INCOMPLETE)
        self.assertIs(experimental_explanation_status(self.encoded(True)),TabICLv2ExplanationStatus.NOT_APPLICABLE_INCOMPLETE)

    def test_demo_uses_only_explicit_lazy_cache_runtime(self):
        import demo.development_prediction_demo as module
        source=inspect.getsource(module)
        self.assertIn('load_frozen_prediction_service',source)
        self.assertIn('build_kv_cache_experimental_runtime_service',source)
        self.assertNotIn('load_authorized_cognitive_context',source)
        self.assertNotIn('TabICLClassifier',source)
    def test_service_factory_is_explicit_and_does_not_fallback(self):
        calls=[]
        historical=lambda: calls.append('historical') or object()
        experimental=lambda: calls.append('experimental') or object()
        _,is_experimental=selected_prediction_service(HISTORICAL_RELEASE_OPTION,historical,experimental)
        self.assertFalse(is_experimental); self.assertEqual(['historical'],calls)
        _,is_experimental=selected_prediction_service(EXPERIMENTAL_RELEASE_OPTION,historical,experimental)
        self.assertTrue(is_experimental); self.assertEqual(['historical','experimental'],calls)

    def test_experimental_explanation_key_changes_with_release_or_answer_vector(self):
        encoded=self.encoded()
        baseline=_experimental_explanation_key(EXPERIMENTAL_RELEASE_OPTION,encoded,'release')
        self.assertNotEqual(baseline,_experimental_explanation_key(HISTORICAL_RELEASE_OPTION,encoded,'release'))
        altered=list(encoded.feature_values); altered[1]='0' if altered[1] != '0' else '1'
        self.assertNotEqual(baseline,(EXPERIMENTAL_RELEASE_OPTION,'release',tuple(altered)))

    def test_caregiver_cognitive_summary_is_non_technical(self):
        self.assertIn('not a diagnosis', caregiver_domain_message(True).lower())
        self.assertIn('does not rule out', caregiver_domain_message(False).lower())

    def test_demo_reserves_alternative_analysis_for_researcher_view(self):
        import demo.development_prediction_demo as module
        source = inspect.getsource(module)
        self.assertIn('Cognitive development', source)
        self.assertIn('Researcher: Cognitive local model-behaviour analysis', source)
        self.assertIn('Run 30-alternative Cognitive analysis for researcher review', source)
        self.assertIn('if request_cognitive_explanation:', source)
