# Synthetic experimental TabICLv2 composite integration

`inference.experimental_composite` remains separate from the historical `load_frozen_prediction_service()` loader. The cache-backed composite release is explicitly experimental and development-only; it does not modify historical artifacts or provide a Logistic Regression fallback.

The cache-backed composite manifest references existing Communication and Social-emotional frozen artifacts, the finalized Cognitive manifest/index, and a separate cache-release manifest by SHA-256. It preserves the historical Cognitive Logistic Regression artifact/manifest as rollback metadata. Missing or altered references fail; no Logistic Regression fallback is performed inside the experimental path.

The synthetic service preserves existing complete/incomplete behavior and derives overall concern by OR. It applies the Cognitive adapter's manifest-derived calibrated probability and inclusive frozen threshold.

`inference.tabiclv2_kv_cache_runtime.build_kv_cache_experimental_runtime_service` is the opt-in cache-backed factory used by the local experimental demo. It verifies the cache release manifest, external cache path/size/SHA-256 before deserialization, both XGBoost references, the finalized Cognitive manifest/index, exact `tabicl==2.2.0`, and the approved checkpoint. It accepts no development-context authorization and has no Logistic Regression fallback.

After explicit selection and a complete submission, the factory lazily loads the verified cache and reuses it in-process. It rejects retained ordinary training arrays and never opens the 500-participant development CSV during inference. The cache remains a protected participant-derived artifact; this is not an independent validation, deployment, or validated TabICLv2 explainability implementation.

`training/experiments/run_experimental_three_domain_localhost_smoke_test.py` is a non-server command-line runner. Without `--execute-real-inference` it exits blocked before runtime I/O. A future real run additionally requires a local governance attestation and a non-sensitive context-authorization configuration, checks that the Windows process is not elevated, and emits only timing/memory booleans. It never binds a host or starts Streamlit.
