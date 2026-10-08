# Local development prediction demo

This directory contains a thin Streamlit adapter for reviewing the current three-domain prediction workflow. It displays the committed questionnaire wording, accepts manual answers, provides clearly labelled synthetic quick-fill examples, and renders Logistic Regression and Extra Trees probabilities and decisions side by side from the same submission.

The app fits both predeclared candidate configurations from the frozen 500-participant development partition in memory: three regularized-logistic classifiers and three Extra Trees classifiers, one per outcome domain. It cannot load or evaluate the 125-participant final-test partition, and it does not serialize a fitted model. Manual answers remain in the local Streamlit session and are not written to files, a database or an API.

From the repository root, with `.venv` active:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/development_prediction_demo.py
```

Use **Load synthetic complete example** for a quick end-to-end demonstration. The synthetic values are test/demo data only: they are not participants, do not augment training and must not be presented as a representative clinical case.

Use **Load synthetic incomplete example** to demonstrate the safety boundary: `NOT_SURE` prevents ML prediction rather than being converted to zero.

The Logistic Regression threshold has the recorded pediatrician-approved status for that model version. The numerically identical Extra Trees cutoff remains explicitly experimental because approval does not automatically transfer between model configurations.

This adapter is intentionally separate from the prediction application code in `inference/`. A future microservice or monorepo integration can replace Streamlit with an HTTP adapter without rewriting questionnaire validation, encoding or threshold decisions.
