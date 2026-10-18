# Child Care frontend

Local React + shadcn-compatible interface for the Child Care research prototype.

## Run locally

```powershell
npm run dev
```

The frontend connects only to the local FastAPI server at `http://127.0.0.1:8000` by default, using its HttpOnly cookie session. Copy `.env.example` to `.env.local` only if you need to override that localhost URL.

Start the backend before starting this frontend. The UI receives the age-routed questionnaire, caregiver-safe screening messages, and approved activity cards from the API; it does not load model artifacts in the browser.
