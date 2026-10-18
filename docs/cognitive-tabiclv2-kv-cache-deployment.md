# Cognitive TabICLv2 KV-cache deployment artifact

The isolated cache-generation runner is disabled without `--execute`. It verifies the protected development-only provenance, checkpoint hash, package version, 16-feature contract, 500 participants, and 53 Cognitive positives before fitting a CPU eight-estimator cache. It persists a participant-derived KV cache without ordinary training arrays, a separate manifest, and no row-level predictions. The cache remains restricted derived research data and is not an application release or clinical validation.

Before accepting a cache, the runner writes an aggregate-only `equivalence-diagnostic.json`. It reports pairwise direct-context/cache/reloaded-cache maximum and mean raw/calibrated differences, changed frozen-threshold decisions, synthetic-request count, configuration, and runtime versions. It contains no requests, participant rows, identifiers, or individual probabilities. The fixed `1e-8` tolerance remains a diagnostic criterion, not an automatically adjustable acceptance rule; any changed decision fails acceptance regardless of tolerance.

## Phase 3I synthetic numerical investigation

`training/experiments/investigate_cognitive_tabiclv2_kv_cache_synthetic.py` is an isolated, synthetic-only verifier. It has no protected-context path and cannot run without `--execute-synthetic`. Its frozen verification suite contains 128 synthetic 16-feature requests: all 12 valid baseline checkpoints crossed with eight predefined response profiles, plus 32 fixed mixed-response profiles. The suite’s canonical SHA-256 is embedded in source and checked before model execution.

The verifier keeps the finalized CPU/eight-estimator/seed configuration and compares repeated direct calls, repeated cache calls, and direct versus cache calls. It reports aggregate differences and the minimum calibrated-probability distance to the frozen threshold, together with a predeclared 0.01 threshold-adjacency count. Its output is evidence about finite synthetic inputs only; it neither changes the historical `1e-8` result nor establishes universal equivalence.

## Phase 3J prospective engineering verification

The cache runner supports an explicit `--execute --engineering-verification` mode for one separately authorised controlled run. It uses the frozen 128-request suite and writes an aggregate-only engineering-equivalence report before any failure. The original `1e-8` status remains reported. A new, prospectively labelled `1e-6` policy requires maximum raw *and* calibrated differences at or below `1e-6` for every direct/cache/reloaded pair and zero changed decisions under the frozen inclusive threshold. A passing result is limited to engineering equivalence on this finite suite; it is not a release, integration, universal-equivalence claim, or independent validation.
