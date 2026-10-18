# Phase 3Q Stage B: synthetic checkpoint-conditional permutation SHAP

## Scope and status

This is a synthetic-only methodology implementation. It does not load the real TabICLv2 checkpoint, the participant-derived KV-cache, the 500-participant development context, or any participant or test record. It does not activate SHAP in Streamlit. The existing 30-alternative Cognitive local model-behaviour analysis remains unchanged and independent.

Successful mock tests do **not** validate SHAP explanations for the fitted TabICLv2 model. The future method is model-specific only because a separately approved Stage C would connect this same prediction-function contract to the selected Cognitive TabICLv2 model.

## Prediction function

`CheckpointConditionalCalibratedProbabilityWrapper` in `inference/tabiclv2_shap.py` accepts exactly the 15 ordered questionnaire fields:

`communication_q1`–`communication_q5`, `cognitive_q1`–`cognitive_q5`, and `social_emotional_q1`–`social_emotional_q5`.

For every coalition it reconstructs the exact 16-column order:

1. `baseline_checkpoint_id`, represented as a categorical value with the established checkpoint categories;
2. the 15 responses in the established feature-contract order.

The checkpoint is fixed for one explanation and never enters the SHAP feature matrix. The wrapper validates the classifier's two distinct fitted `classes_` labels and locates positive Cognitive concern label `1`; it never assumes a probability-column position. It selects that raw probability, clips it to `[1e-12, 1 - 1e-12]`, applies the frozen calibration once, and returns calibrated probability only:

`sigmoid(1.069105693175016 × logit(raw_positive) + 0.255897163477147)`.

The frozen threshold `0.1503614235173848` is retained as metadata but is not applied by the function being explained.

## Baseline and interpretation

The SHAP background is one row of 15 NaN responses. This is an **imputation/missingness-based SHAP baseline conditional on the selected developmental checkpoint**. It is not physical feature removal, a normal child, an average child, a neutral clinical state, or a causal intervention.

The synthetic mock explicitly forwards NaNs to its preprocessing boundary and then deterministically imputes them to `1.0`. That verifies wrapper mechanics only; it does not establish the actual fitted TabICLv2 preprocessing response to missing inputs. Stage C must verify that separately.

Attributions are in calibrated probability units relative to this defined background. They are not XGBoost log-odds contributions, Logistic Regression coefficients, causal effects, clinical importance, or population-level importance.

## SHAP configuration and bounded workload

- Package and algorithm: `shap==0.47.2`, explicit `shap.PermutationExplainer`.
- Masker: `shap.maskers.Independent` with the one-row NaN background.
- Features: 15 response fields; fixed checkpoint outside the masker.
- Seed: `20260822`.
- Requested `max_evals`: `186 = 6 × (2 × 15 + 1)`.

Installed SHAP source calculates `npermutations = max_evals // (2 * len(inds) + 1)`. Thus 186 produces six permutations when all 15 response features vary. The bounded synthetic pilot directly measured 186 SHAP-evaluation rows and six wrapper calls per assessment. The separate unmasked fidelity call plus two-row batch-invariance control raised total rows per assessment to 189; those controls are not SHAP evaluation rows.

## Synthetic evidence

Three artificial complete assessments, one each at `CP6M`, `CP24M`, and `CP60M`, were used. No response vectors are persisted by the runner.

| Check | Result |
|---|---:|
| SHAP probability reconstruction error | 0.0 |
| Same-seed repeat attribution max difference | 0.0 |
| Batch-invariance difference | 0.0 |
| Max SHAP evaluation rows | 186 |
| NaN-masked SHAP rows | 180 |
| Total runtime (three assessments) | 7.328 s |
| Approximate process RSS increase | 111,919,104 bytes |

For this deterministic float64 mock, `1e-12` is a justified strict tolerance for unmasked fidelity, additivity, batch invariance, and repeat stability. These tolerances must not be transferred automatically to CPU float32 TabICLv2 inference. Before Stage C, predeclare distinct real-model tolerances based on expected CPU numerical precision and retain the hard requirement of no changed threshold decisions when assessing equivalence.

## Tests and limitations

`tests/test_tabiclv2_shap.py` contains 16 synthetic tests for feature count/order, checkpoint preservation, positive-class selection, calibration, NaN coalitions, finite output, malformed input rejection, normal incomplete-assessment blocking, additivity, seeded stability, batch invariance, and the observed workload bound. It does not fit a model or reconstruct a reference context.

Stage C requires separate approval to verify the unmasked wrapper against the deployed calibrated prediction path, real fitted preprocessing/missingness semantics, class ordering, batch composition, real-model numerical stability, latency, and the scientific suitability of this missingness baseline. A successful Stage B does not establish clinical validity or independent evaluation.

## Stage C3 reconstruction-accounting investigation

The original Stage C acceptance calculation compared the independently evaluated calibrated wrapper probability with `base_value + sum(SHAP values)`. In SHAP 0.47.2's `PermutationExplainer`, `expected_value` is assigned from `outputs[0]` of the final permutation batch, whereas feature contributions are averaged across all forward-and-reverse permutation paths. The full masking path assembles masked samples into model-call batches before invoking the wrapped prediction function.

Therefore, when a model has small batch-composition-dependent numerical variation, the independent standalone call and the permutation-path reconstruction need not be numerically identical even if SHAP's own path accounting is internally consistent. This is a mechanism identified from source inspection and synthetic tests; it is **not** an established explanation of the real TabICLv2 discrepancy.

The Stage C runner now retains only in-memory probability endpoints for its SHAP batches. A future separately authorized run can record three distinct aggregate quantities without persisting inputs or prediction vectors:

1. `base + sum(contributions)` versus an independently evaluated calibrated probability (the existing prospective acceptance check);
2. `base + sum(contributions)` versus the reconstruction calculated from SHAP's own permutation-batch endpoints;
3. the SHAP-sequence reconstruction versus the independent standalone probability.

Synthetic tests include a deliberately batch-dependent floating-point-like model. Its SHAP-sequence reconstruction is exact, while the standalone comparison exceeds `1e-6`. This demonstrates that the two comparisons test different numerical execution paths. It does not justify changing the existing tolerance, passing the prior C2 run, or inferring TabICLv2's cause without a separately approved real-model diagnostic run.
