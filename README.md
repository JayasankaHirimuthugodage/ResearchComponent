# Socio-Emotional, Language/Communication and Cognitive/Problem-Solving Development Assessment Service

Research workspace for a privacy-conscious early-childhood Socio-Emotional,
Language/Communication and Cognitive/Problem-Solving developmental-monitoring component.

This project is non-diagnostic. It must not diagnose autism, developmental delay or any
medical condition, and it does not replace assessment by a qualified healthcare professional.

## Project information

- **Group ID:** J26-SE-349
- **Student name:** Hirimuthugodage J.
- **Student ID:** IT23297968

This repository contains an individual research component developed as part of the J26-SE-349
group project.

## Current capabilities

- Three separate domain-specific developmental-concern predictions.
- Required-field and incomplete-assessment handling for all 15 questionnaire responses.
- Checksum-verified offline research training and frozen inference artifacts.
- Caregiver-safe explanations without exposing raw probabilities.
- Deterministic safety checks and approved-activity recommendations.
- A local FastAPI research API and React caregiver interface.

Model predictions, deterministic rules, activity selection and any future LLM personalisation
remain separate boundaries. An LLM cannot alter predictions or safety rules.

## Privacy

Private research data, participant identifiers and participant-level records must not be
committed, transmitted, published, logged or placed in LLM prompts.

## Domain-specific experiment

The domain-specific notebook evaluates communication, cognitive and social-emotional models
separately. Each model uses only information available at baseline:

- `baseline_checkpoint_id`
- the five baseline questionnaire items belonging to the predicted domain

Overall concern is derived by OR-ing the three domain decisions; it is not trained as a fourth
model. Development selection uses shared repeated folds, fold-local preprocessing, PR-AUC
ranking, nested calibration and threshold evaluation.

## Local validation

Install the Python environment and run the synthetic test suite:

```powershell
python -m pip install -r requirements.lock
python -B -m unittest discover -v
```

Validate the React frontend:

```powershell
cd frontend
npm install
npm test
npm run build
```
