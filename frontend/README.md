# Child Care frontend

Local React + shadcn-compatible interface for the Child Care research prototype.

## Run locally

```powershell
npm run dev
```

Validate changes with:

```powershell
npm test
npm run build
```

The frontend connects only to the local FastAPI server at `http://127.0.0.1:8000` by default, using its HttpOnly cookie session. Copy `.env.example` to `.env.local` only if you need to override that localhost URL.

Start the backend before starting this frontend. The UI receives the age-routed questionnaire, caregiver-safe screening messages, and approved activity cards from the API; it does not load model artifacts in the browser.

## Source structure

- `src/app/` contains application-wide route types, constants, and pure formatting helpers.
- `src/components/layout/` contains reusable authenticated and unauthenticated page shells.
- `src/features/` groups screens by caregiver workflow, such as assessments and activities.
- `src/api.ts` remains the only browser boundary for backend requests.

Keep prediction results, deterministic safety rules, approved activity selection, and any
personalisation outside presentation components. The frontend displays the caregiver-safe API
contract and must not calculate or reinterpret model decisions.
