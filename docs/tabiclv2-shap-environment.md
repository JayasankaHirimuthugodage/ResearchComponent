# Phase 3Q Stage A: isolated SHAP environment

This record describes the isolated environment prepared for a future, synthetic-only Cognitive TabICLv2 SHAP methodology pilot. It does not authorize or document model loading, KV-cache loading, participant-data access, prediction, or SHAP explanation execution.

## Environment

- Path: `F:\4thYear\2026July\Research_Component\.venv-tabiclv2-xai`
- Interpreter: `C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe` (CPython 3.10.0)
- Installed inference-critical pins: `tabicl==2.2.0`, `einops==0.8.2`, `numpy==2.2.2`, `scikit-learn==1.6.1`, `torch==2.14.1`, `pandas==2.2.3`, `joblib==1.4.2`, and `huggingface-hub==1.33.0`.
- XAI pins: `shap==0.47.2`, `numba==0.68.0`, `llvmlite==0.50.0`, and `slicer==0.0.8`.
- Full, hash-locked binary-wheel dependency set: [`requirements-tabiclv2-xai.lock.txt`](../requirements-tabiclv2-xai.lock.txt).

The required APIs imported successfully: `tabicl`, `shap`, `numpy`, `scikit-learn`, `torch`, `pandas`, and `joblib`. `shap.PermutationExplainer`, `shap.maskers`, and `shap.maskers.Independent` are available. `pip check` reported no broken requirements.

## Provenance and installation

The binary wheels were obtained with pip using its default PyPI index configuration: the final configuration check found no `PIP_*` index override or pip configuration file. They are held outside the repository at `E:\Temp\tabicl-xai-wheelhouse-20261017`. Wheel filenames and SHA-256 values for every installed package are in the lock file. The resolver was run offline against that wheelhouse before installation; it selected no source distributions. SHA-256 locking verifies the preserved wheel bytes, but is not a substitute for an independent publisher-signature or supply-chain audit.

The primary environment's known working `huggingface-hub==1.33.0` was explicitly pinned rather than accepting the resolver's newer `2.2.0` candidate. The downloaded but uninstalled `2.2.0` wheel and its `httpx2`/`httpcore2` companions are not part of this environment or lock file.

Recreation, using a verified wheelhouse and a fresh target directory:

```powershell
C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe -m venv F:\4thYear\2026July\Research_Component\.venv-tabiclv2-xai
F:\4thYear\2026July\Research_Component\.venv-tabiclv2-xai\Scripts\python.exe -m pip install --no-index --find-links E:\Temp\tabicl-xai-wheelhouse-20261017 --require-hashes -r F:\4thYear\2026July\Research_Component\requirements-tabiclv2-xai.lock.txt
```

The environment is excluded from Git via `.venv-tabiclv2-xai/`. The external wheelhouse is outside the repository. To remove only this experimental environment, after work has concluded and with separate approval, use:

```powershell
Remove-Item -LiteralPath 'F:\4thYear\2026July\Research_Component\.venv-tabiclv2-xai' -Recurse -Force
```

## Scope and remaining risk

This is dependency compatibility only. Successful imports do not establish numerical equivalence, SHAP validity, real-model compatibility, or the appropriateness of an imputation-based baseline. A separately approved Stage B must use synthetic inputs first and preserve the existing response-perturbation analysis independently.
