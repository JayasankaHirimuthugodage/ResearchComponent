# Local development prediction demo

This directory contains a thin Streamlit adapter for reviewing the final three-domain prediction workflow. It displays the committed questionnaire wording, accepts manual answers, keeps the clearly labelled synthetic quick-fill examples, and produces communication, cognitive and social-emotional decisions from the frozen research artifacts.

The app checksum-verifies and loads the frozen XGBoost communication model, Logistic Regression cognitive model and XGBoost social-emotional model. It does not fit or update a model, load the research dataset, or evaluate the locked-test partition. Manual answers remain in the local Streamlit session and are not written to files, a database or an API.

From the repository root, with `.venv` active:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/development_prediction_demo.py
```

Select any checkpoint and use **Load synthetic complete example** for a quick end-to-end demonstration. The control fills the 15 question IDs belonging to the selected checkpoint. The synthetic values are test/demo data only: they are not participants and do not augment training.

Use **Load synthetic incomplete example** to fill the selected checkpoint with one `NOT_SURE` response and demonstrate the no-prediction safety boundary. **Clear all answers** removes the current responses without changing the selected checkpoint.

Each domain uses the threshold selected from development-only out-of-fold predictions and stored in its frozen artifact. The questionnaire score cutoff and the three ML probability thresholds remain separate decisions.

The main result uses plain, non-diagnostic wording. Raw calibrated probabilities, thresholds and model versions appear only inside the explicitly labelled research/reviewer technical section. A future HTTP adapter can reuse the framework-neutral validation and frozen prediction service.

The overall summary names the specific domains identified for further attention instead of displaying only a generic overall flag. The underlying overall decision remains the logical OR of the three domain decisions.

When at least one domain is flagged, the demonstration asks four separate activity-suitability questions covering hearing, vision, fine-motor and gross-motor observations. These answers never enter the frozen 16-feature model contract. The app uses the flagged domains as retrieval gates, exact `NOT_YET` and `SOMETIMES` question wording as traceable skill context, and the safety observations to filter v0.4 activities before semantic ranking.

The app returns up to three eligible activities independently for each flagged domain. It returns fewer when fewer records pass all restrictions and does not relax safety or approval rules. A zero-concern result does not trigger targeted retrieval. A flagged domain without supported item-level skill context does not receive an invented query.

The activity stage is local safety-filtered semantic retrieval. It does not use a generative LLM and is not complete RAG. Returned activities remain internal research-demo candidates and are not represented as caregiver-deployment-approved interventions.

For each domain, the main result shows the strongest submitted answer supporting the final decision. A single shared notice explains that all answers contribute and that the highlighted observation did not determine the result alone. The reviewer section retains the threshold, supporting, opposing and numerical model-specific contributions: coefficient contributions for Logistic Regression and TreeSHAP contributions for XGBoost. These explanations do not change the prediction and do not claim causation.
