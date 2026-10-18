from __future__ import annotations

import unittest

from inference.contracts import AssessmentStatus, DomainPrediction, QuestionnaireSubmission
from inference.experimental_release import ExperimentalReleaseSelection, EXPERIMENTAL_COGNITIVE_RELEASE
from inference.questionnaire import encode_submission, load_questionnaire_catalog
from inference.tabiclv2_explanations import (
    TabICLv2ExplanationStatus,
    analyze_tabiclv2_cognitive_model_behaviour,
    explain_tabiclv2_cognitive_prediction,
)


class TabICLv2ExplanationContractTests(unittest.TestCase):
    def encoded(self, incomplete: bool = False):
        catalog=load_questionnaire_catalog(); checkpoint='CP24M'
        responses={item.question_id: 'YES_USUALLY' for item in catalog.questions_for_checkpoint(checkpoint)}
        if incomplete: responses[next(iter(responses))]='NOT_SURE'
        return encode_submission(catalog, QuestionnaireSubmission(catalog.questionnaire_schema_version,checkpoint,responses))
    def prediction(self): return DomainPrediction('cognitive', .2, True, .1503614235173848, 'experimental')

    def test_complete_cognitive_response_is_explicitly_unavailable_without_values(self):
        encoded=self.encoded(); original=encoded.feature_values; result=explain_tabiclv2_cognitive_prediction(encoded,self.prediction())
        self.assertIs(result.status,TabICLv2ExplanationStatus.UNAVAILABLE_PENDING_VALIDATION)
        self.assertFalse(result.is_available); self.assertEqual((),result.feature_contributions)
        self.assertIn('not evidence of causal',result.limitation); self.assertTrue(result.development_only)
        self.assertEqual(original,encoded.feature_values)
        self.assertNotIn('YES_USUALLY',repr(result))

    def test_incomplete_assessment_never_produces_or_requests_an_explanation(self):
        result=explain_tabiclv2_cognitive_prediction(self.encoded(True),self.prediction())
        self.assertIs(result.status,TabICLv2ExplanationStatus.NOT_APPLICABLE_INCOMPLETE)
        self.assertFalse(result.is_available); self.assertEqual((),result.feature_contributions)

    def test_invalid_domain_fails_closed_and_experimental_selection_remains_opt_in(self):
        with self.assertRaises(ValueError):
            explain_tabiclv2_cognitive_prediction(self.encoded(),DomainPrediction('communication',.2,False,.1,'x'))
        ExperimentalReleaseSelection(EXPERIMENTAL_COGNITIVE_RELEASE).require_experimental_cognitive()
        with self.assertRaises(ValueError): ExperimentalReleaseSelection('historical').require_experimental_cognitive()

    def test_contract_has_no_prediction_fitting_or_logging_hooks(self):
        import inspect
        import inference.tabiclv2_explanations as module
        source=inspect.getsource(module)
        self.assertNotIn('predict_proba',source); self.assertNotIn('fit(',source); self.assertNotIn('logging.',source)


class _RecordingCognitiveModel:
    def __init__(self):
        self.calls=[]
    def positive_probability(self, values):
        self.calls.append(values)
        return min(0.99, 0.10 + sum(int(value) for value in values[1:]) / 100.0)


class TabICLv2LocalModelBehaviourTests(unittest.TestCase):
    def encoded(self, incomplete=False):
        catalog=load_questionnaire_catalog(); checkpoint='CP24M'
        responses={item.question_id: 'YES_USUALLY' for item in catalog.questions_for_checkpoint(checkpoint)}
        if incomplete: responses[next(iter(responses))]='NOT_SURE'
        return catalog, encode_submission(catalog, QuestionnaireSubmission(catalog.questionnaire_schema_version,checkpoint,responses))

    def test_exactly_thirty_single_response_alternatives_use_baseline_prediction(self):
        catalog, encoded=self.encoded(); model=_RecordingCognitiveModel()
        baseline=0.40
        prediction=DomainPrediction('cognitive', baseline, True, .1503614235173848, 'experimental')
        analysis=analyze_tabiclv2_cognitive_model_behaviour(catalog,encoded,prediction,model)
        self.assertIs(analysis.status,TabICLv2ExplanationStatus.AVAILABLE_LOCAL_MODEL_BEHAVIOUR)
        self.assertTrue(analysis.is_available); self.assertEqual(30,len(analysis.variations)); self.assertEqual(30,len(model.calls))
        self.assertEqual(baseline,analysis.baseline_probability); self.assertEqual(.1503614235173848,analysis.threshold)
        for variation, altered in zip(analysis.variations,model.calls,strict=True):
            changed=[index for index,(old,new) in enumerate(zip(encoded.feature_values,altered,strict=True)) if old != new]
            self.assertEqual([list(__import__('training.contracts',fromlist=['FEATURE_COLUMNS']).FEATURE_COLUMNS).index(variation.feature)],changed)
            self.assertEqual(encoded.feature_values[0],altered[0])
            self.assertIn(int(altered[changed[0]]),(0,1,2))
            self.assertAlmostEqual(variation.alternative_probability-baseline,variation.probability_change)

    def test_incomplete_assessment_cannot_request_alternatives(self):
        catalog, encoded=self.encoded(True)
        with self.assertRaises(ValueError):
            analyze_tabiclv2_cognitive_model_behaviour(catalog,encoded,self._prediction(),_RecordingCognitiveModel())

    @staticmethod
    def _prediction():
        return DomainPrediction('cognitive', .2, True, .1503614235173848, 'experimental')
