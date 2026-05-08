# Session Handoff

Use this file as the small starting context for future Codex sessions. Read only the relevant unchecked section of `upgrade.md` after this file.

## Repo

Path: `C:\Users\aruba\OneDrive\Documents\1 Professional Documents\Projects\Real-Time Options Volatility Surface`

Current main head before this session commit: `b8fba24`

Branch state before this session commit: `main...origin/main [ahead 3]`

## Upgrade State

Phase 0-3 are complete.

Phase 3: `27/27` complete.

Phase 4: `20/27` complete.

Completed this session:

1. Saved workspaces
2. Snapshot comparison
3. Backtesting framework
4. Transaction cost model
5. Paper-trading simulator
6. Broker integration abstraction
7. Notebook export

Phase 4 next unchecked items:

1. Research report generator
2. ML anomaly detector
3. Vol regime classifier
4. Forecasting module
5. News/event overlay
6. WebSocket/async refresh engine
7. Multi-page app architecture

## Latest Verification Pattern

Run from the repo root:

```powershell
python -m ruff check src tests diagnostic.py dashboard_connector.py app.py main.py scripts
python -m compileall src tests scripts dashboard_connector.py diagnostic.py app.py main.py
python -m pytest -q
$env:PYTHONPATH='.'; python scripts\healthcheck.py
python diagnostic.py
```

For Streamlit smoke, start the app with `PYTHONPATH=.` on a free local port and check that `http://127.0.0.1:<port>` returns HTTP 200. Latest smoke passed on port `8532`.

## New Session Prompt

```text
Continue Real-Time Options Volatility Surface upgrade.

Repo:
C:\Users\aruba\OneDrive\Documents\1 Professional Documents\Projects\Real-Time Options Volatility Surface

Start by reading SESSION_HANDOFF.md and only the relevant unchecked section of upgrade.md.
Implement the next 7 unchecked Phase 4 items in order if they fit cleanly.
Preserve provenance, use deterministic offline tests, run full verification, update SESSION_HANDOFF.md, commit changes, and report the updated Phase 4 count.
```

## Notes

- Keep future handoffs short. Do not paste long summaries of completed phases into the chat.
- Prefer targeted searches/reads over loading whole files.
- Preserve generated plots/log/cache files unless the task explicitly touches them; diagnostic runs may regenerate plot artifacts, so restore or ignore those before committing unrelated work.
- The healthcheck needs `PYTHONPATH=.` in this environment.
