# GreekKV Capacity Planner (T29)

Static TypeScript app that translates `(model, VRAM × GPU count, context, scheme)` into the max concurrent users each KV cache scheme supports. Hardware math is a pure TypeScript port of `src/greekkv/memory/capacity.py` (28 parity + chart tests enforce byte-identical output). Perplexity and decode-latency columns come from `public/data.json`, which is regenerated from the committed `results/*.jsonl` by `scripts/build_report.py`.

The app is intentionally minimal: Vite + React + 1 CSS file, zero chart libraries, ~50 KB gzip.

## Development

```bash
cd planner
npm install
npm run dev        # http://localhost:5173
npm test           # vitest (parity + chart tests)
npm run build      # static bundle in dist/
npm run preview    # serve dist/ locally
```

## Regenerate the measurements file

```bash
cd ..
python scripts/build_report.py
# → writes docs/RESULTS.md  AND  planner/public/data.json (same source of truth)
```

The two outputs share the same JSONL inputs under `results/` so the docs and the web UI can never drift apart.

## Deploy (GitHub Pages)

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds the app with `GREEKKV_BASE=/greekkv/` and publishes `planner/dist/` to Pages on every push to `main` that touches `planner/**`, `scripts/build_report.py`, or `results/**`. Trigger manually via `gh workflow run pages.yml`.

Deployed URL: <https://georgebon5.github.io/greekkv/>

If Actions is still locked on this account (see root `docs/DECISIONS.md` ADR-0001), the workflow will not run but the local `npm run build && npm run preview` reproduces the deployed app bit-for-bit.
