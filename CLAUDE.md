# Palm Passport (GeoMata)

Per-plot oil palm intelligence for Indonesia on Google Earth Engine. Prototype area: Siak, Riau.

- Code the notebooks rely on lives in `src/palm_passport/`; notebooks stay thin and install the
  package from GitHub (`REPO_REF` in the first cell), because the owner runs them in Colab.
- Every notebook starts with a settings cell holding `PROJECT_ID` and calls `auth.init(PROJECT_ID)`.
  Earth Engine runs under the owner's own Google account; never hard-code a project.
- Keep notebooks free of saved outputs (`tests/test_notebooks.py` checks this).
- Catalog assets that get new versions go in `assets.CANDIDATES`, newest first.
- Earth Engine calls can't run in CI (they need sign-in). `pytest` covers pure-Python code and
  notebook syntax only; anything EE-side must be checked by running the notebook.
- Free tiers only while prototyping.
- `legacy/streamlit-demo/` is the old app, kept for reference (branch `v0-demo`). Don't build on it.
