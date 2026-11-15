#!/usr/bin/env bash
# bootstrap_github.sh — Idempotent bootstrap of labels, milestones, and issues
# for the GreekKV project. Safe to re-run. Uses gh CLI + jq.
#
# Usage:
#   scripts/bootstrap_github.sh [--dry-run] [--repo owner/name]
#
# Without --repo, the script uses the repo detected from the current directory.

set -euo pipefail

DRY_RUN=0
REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --repo)    REPO="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

require() { command -v "$1" >/dev/null || { echo "missing: $1" >&2; exit 1; } }
require gh
require jq

gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated. Run: gh auth login" >&2; exit 1; }

if [[ -z "$REPO" ]]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi
echo "Target repo: $REPO"
[[ $DRY_RUN -eq 1 ]] && echo "(dry-run: no changes will be made)"

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] $*"
  else
    eval "$@"
  fi
}

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
# Format: name|color(hex,no #)|description
LABELS=(
  "infra|0e8a16|Project scaffolding, CI, tooling"
  "data|1d76db|Datasets, loaders, fixtures"
  "research|5319e7|Analyses, figures, hypothesis checks"
  "core|b60205|Library code (quant, cache, eval primitives)"
  "eval|fbca04|Evaluation harness and metrics"
  "systems|d93f0b|Memory/latency profiling, kernels"
  "cpp|6f42c1|C++ / minillm.cpp cross-contribution"
  "planner|c2e0c6|Capacity planner web app"
  "docs|0366d6|Documentation, reports, outreach"
  "bug|d73a4a|Something behaves unexpectedly"
)

existing_labels() {
  gh label list --repo "$REPO" --limit 200 --json name -q '.[].name'
}

echo "--- Labels ---"
LABEL_CACHE="$(existing_labels || true)"
for row in "${LABELS[@]}"; do
  IFS='|' read -r name color desc <<<"$row"
  if grep -qx -- "$name" <<<"$LABEL_CACHE"; then
    echo "  update: $name"
    run "gh label edit \"$name\" --repo \"$REPO\" --color \"$color\" --description \"$desc\" >/dev/null"
  else
    echo "  create: $name"
    run "gh label create \"$name\" --repo \"$REPO\" --color \"$color\" --description \"$desc\" >/dev/null"
  fi
done

# -----------------------------------------------------------------------------
# Milestones
# -----------------------------------------------------------------------------
# Format: title|description
MILESTONES=(
  "M0 — Foundation|Repo, CI, configs, contribution scaffolding"
  "M1 — Greek tokenization & KV cost|Fertility study, KV calculator, Greek-vs-English figure"
  "M2 — Quantization library|Quantizer primitives, fake-quant cache, KIVI-style cache, baselines"
  "M3 — Evaluation|Perplexity, benchmarks, needle, fidelity metrics, runner"
  "M4 — Systems|Profiler, Triton kernel, kernel benchmarks, C++ contribution"
  "M5 — Product|Capacity model, planner web app, results dashboard"
  "M6 — Packaging & outreach|README, report, release, outreach drafts"
)

fetch_milestones() {
  gh api "repos/$REPO/milestones?state=all&per_page=100"
}

echo "--- Milestones ---"
MILESTONES_JSON="$(fetch_milestones || echo '[]')"
for row in "${MILESTONES[@]}"; do
  IFS='|' read -r title desc <<<"$row"
  num="$(jq -r --arg t "$title" '.[] | select(.title == $t) | .number' <<<"$MILESTONES_JSON" | head -1)"
  if [[ -n "$num" ]]; then
    echo "  update: $title (#$num)"
    run "gh api -X PATCH \"repos/$REPO/milestones/$num\" -f title=\"$title\" -f description=\"$desc\" >/dev/null"
  else
    echo "  create: $title"
    run "gh api -X POST \"repos/$REPO/milestones\" -f title=\"$title\" -f description=\"$desc\" >/dev/null"
  fi
done

# Refresh after creation so lookup by title works for issues below.
MILESTONES_JSON="$(fetch_milestones || echo '[]')"
milestone_number_by_title() {
  jq -r --arg t "$1" '.[] | select(.title == $t) | .number' <<<"$MILESTONES_JSON" | head -1
}

# -----------------------------------------------------------------------------
# Issues
# -----------------------------------------------------------------------------
# One issue per task. Bodies are heredocs so they stay readable.

ISSUES_JSON="$(gh issue list --repo "$REPO" --state all --limit 500 --json number,title || echo '[]')"

ensure_issue() {
  local title="$1" label="$2" milestone="$3" body="$4"
  local num
  num="$(jq -r --arg t "$title" '.[] | select(.title == $t) | .number' <<<"$ISSUES_JSON" | head -1)"
  local ms_num
  ms_num="$(milestone_number_by_title "$milestone")"
  if [[ -z "$ms_num" ]]; then
    echo "  milestone not found: $milestone" >&2; return 1
  fi
  if [[ -n "$num" ]]; then
    echo "  update: #$num  $title"
    if [[ $DRY_RUN -eq 0 ]]; then
      printf '%s' "$body" | gh issue edit "$num" --repo "$REPO" \
        --add-label "$label" --milestone "$milestone" --body-file - >/dev/null
    fi
  else
    echo "  create:      $title"
    if [[ $DRY_RUN -eq 0 ]]; then
      printf '%s' "$body" | gh issue create --repo "$REPO" \
        --title "$title" --label "$label" --milestone "$milestone" --body-file - >/dev/null
    fi
  fi
}

echo "--- Issues ---"

# --- M0 Foundation -----------------------------------------------------------

ensure_issue "T01 — Repo scaffold" "infra" "M0 — Foundation" "$(cat <<'EOF'
## Deliverable
`pyproject.toml` with `src/greekkv/` layout, ruff, pytest, pre-commit, MIT license, README skeleton.

## Acceptance criteria
- `pip install -e .` succeeds in a fresh venv.
- `pytest` runs (at least one trivial import test).
- `ruff check .` is green.
- README includes: project one-liner, planned deliverables, status.
EOF
)"

ensure_issue "T02 — CI" "infra" "M0 — Foundation" "$(cat <<'EOF'
## Deliverable
GitHub Actions workflow: lint + tests on CPU, Python 3.10/3.11 matrix, pip cache.

## Acceptance criteria
- `.github/workflows/ci.yml` runs on push + PR to `main`.
- Matrix: `python-version: [3.10, 3.11]`.
- Steps: checkout, setup-python (with pip cache), `pip install -e .[dev]`, `ruff check .`, `pytest`.
- README has a CI badge.
EOF
)"

ensure_issue "T03 — Config + results schema" "infra" "M0 — Foundation" "$(cat <<'EOF'
## Deliverable
YAML experiment configs, deterministic seeding utility, JSONL results schema with git SHA, model, config hash, timestamp.

## Acceptance criteria
- `src/greekkv/utils/config.py` loads YAML via pydantic (or dataclasses + jsonschema).
- `src/greekkv/utils/seeding.py` seeds `random`, `numpy`, `torch`, `transformers`.
- `src/greekkv/utils/results.py` writes JSONL with a validated schema.
- Unit test validates the schema against a canned record.
EOF
)"

ensure_issue "T04 — Contribution scaffolding" "docs" "M0 — Foundation" "$(cat <<'EOF'
## Deliverable
Issue/PR templates, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `docs/DECISIONS.md` (ADR index).

## Acceptance criteria
- `.github/ISSUE_TEMPLATE/*.yml` with at least bug + task templates.
- `.github/pull_request_template.md` with what/why/how-tested + `Closes #` field.
- CONTRIBUTING describes conventional commits, branch naming, PR flow.
- `docs/DECISIONS.md` contains ADR-0001 explaining the src/ layout + tooling choices.
EOF
)"

# --- M1 Greek tokenization & KV cost ----------------------------------------

ensure_issue "T05 — Parallel corpus loader" "data" "M1 — Greek tokenization & KV cost" "$(cat <<'EOF'
## Deliverable
Loader for Greek/English parallel text (FLORES-200 verified) + Greek Wikipedia sample.

## Acceptance criteria
- `src/greekkv/tokenization/corpora.py` exposes `load_flores_pair()` and `load_el_wikipedia_sample(n)`.
- Datasets cached under `data/` (gitignored); loader is deterministic given a seed.
- Unit tests use tiny on-disk fixtures (no network) to verify shapes/encodings.
EOF
)"

ensure_issue "T06 — Tokenizer fertility study" "research" "M1 — Greek tokenization & KV cost" "$(cat <<'EOF'
## Deliverable
Tokens-per-word and tokens-per-character for every candidate tokenizer, Greek vs English.

## Acceptance criteria
- Script `experiments/fertility.py` produces `results/fertility.csv` with bootstrap CIs.
- Runs on CPU in under 5 minutes using the corpus sample from T05.
- A table in `docs/` summarizes the result, with a note that this is **measured, not assumed**.
EOF
)"

ensure_issue "T07 — KV memory calculator" "core" "M1 — Greek tokenization & KV cost" "$(cat <<'EOF'
## Deliverable
`M_KV = 2 · L · H_kv · d_head · n_ctx · b` with GQA support, weights memory, overhead for quant scales/zero-points per group.

## Acceptance criteria
- `src/greekkv/memory/kv.py` exports `kv_bytes(model_cfg, n_ctx, bits, group_size)` and `weights_bytes(model_cfg, bits)`.
- Unit tests compare to the real tensor sizes of a tiny HF model.
- Handles GQA (`num_key_value_heads != num_attention_heads`).
EOF
)"

ensure_issue "T08 — Greek-vs-English KV cost figure" "research" "M1 — Greek tokenization & KV cost" "$(cat <<'EOF'
## Deliverable
Notebook + figure showing extra KV bytes for the same content in Greek vs English per model.

## Acceptance criteria
- `notebooks/greek_vs_english_kv.ipynb` consumes T06 fertility + T07 calculator.
- Figure saved to `results/figures/kv_greek_vs_english.png` and committed.
- Script path documented so the figure regenerates from a single command.
EOF
)"

# --- M2 Quantization library -------------------------------------------------

ensure_issue "T09 — Quantizer primitives" "core" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Symmetric/asymmetric, group-wise (32/64/128), per-token and per-channel, INT8/INT4, real bit-packing for INT4.

## Acceptance criteria
- `src/greekkv/quant/primitives.py`: `quantize`, `dequantize`, `pack_int4`, `unpack_int4`.
- Property tests: round-trip error bound, shape preservation, pack/unpack identity.
- CPU-only; no CUDA kernels yet.
EOF
)"

ensure_issue "T10 — Fake-quant HF cache" "core" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Custom HF-transformers cache that quantizes→dequantizes K/V on every `update()`.

## Acceptance criteria
- `src/greekkv/quant/fakequant_cache.py` subclasses the HF cache interface.
- FP16 mode is **bit-identical** to the vanilla cache (sanity test).
- Works with one Greek model and one baseline from the plan.
EOF
)"

ensure_issue "T11 — KIVI-style cache" "core" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Keys per-channel, values per-token, residual FP16 window of the most recent R tokens, configurable group size.

## Acceptance criteria
- `src/greekkv/quant/kivi_cache.py` with a config dataclass (`bits`, `group_size`, `residual`).
- Tests for residual window boundary (first R tokens, mid-stream, eviction into the quantized store).
- Numerical parity vs T10 when R = seq_len.
EOF
)"

ensure_issue "T12 — Baseline adapters" "core" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Wrappers for HF `QuantizedCache` (quanto/HQQ if installable) exposing the same interface; vLLM fp8 adapter OPTIONAL.

## Acceptance criteria
- `src/greekkv/quant/adapters/` with pluggable backends behind a common `KVCacheBackend` protocol.
- The same eval harness runs all backends via a config flag.
- vLLM adapter clearly marked OPTIONAL and skipped on free hardware.
EOF
)"

ensure_issue "T13 — Key/value outlier analysis" "research" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Per-layer, per-channel magnitude statistics of K and V on Greek text; figure + writeup.

## Acceptance criteria
- Script produces `results/kv_outliers.json` and `results/figures/kv_outliers.png`.
- `docs/kv_outliers.md` explains **why** per-channel key quantization helps (or doesn't) in terms of observed outliers.
EOF
)"

ensure_issue "T14 — Token-eviction baseline (stretch)" "research" "M2 — Quantization library" "$(cat <<'EOF'
## Deliverable
Simplified H2O / SnapKV-style token eviction for comparison with precision reduction.

## Acceptance criteria
- Same `KVCacheBackend` interface as T10/T11.
- Clearly labelled **simplified** (not a faithful reimplementation).
- Appears as a separate scheme in the eval sweep.

**Note:** Stretch — may be skipped if time-constrained; must be called out in README.
EOF
)"

# --- M3 Evaluation -----------------------------------------------------------

ensure_issue "T15 — Model loader" "core" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
4-bit / 8-bit / FP16 loader helper with memory guard for Kaggle T4; CPU fallback for tiny models.

## Acceptance criteria
- `src/greekkv/eval/loader.py`: `load_model(name, dtype, device, memory_budget_gb)`.
- Pre-flight check: estimates weights+KV memory, refuses to OOM.
- `docs/vram_table.md` lists tested configurations.
EOF
)"

ensure_issue "T16 — Context-then-continuation scorer" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Prefill → quantize cache → score continuation (per hard constraint of the project).

## Acceptance criteria
- `src/greekkv/eval/scorer.py` exposes `score_continuation(ctx, cont, cache_cfg)`.
- Sanity test 1: FP16 "quantization" bit-equal to vanilla.
- Sanity test 2: 2-bit (or random-noise) scheme visibly destroys quality.
EOF
)"

ensure_issue "T17 — Perplexity on Greek text" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Sliding-window perplexity with quantized cache across all schemes.

## Acceptance criteria
- `experiments/perplexity.py` writes `results/perplexity/*.jsonl`.
- Reads T05 corpus; runs per model × scheme × context length.
- Logged fields: model, scheme, bits, group, residual, n_ctx, nll, tokens, wall_time.
EOF
)"

ensure_issue "T18 — Greek benchmark runner" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Multiple-choice benchmarks (MMLU/ARC/HellaSwag/TruthfulQA/Winogrande el) via the T16 scorer.

## Acceptance criteria
- Dataset IDs verified on the HF Hub (documented in the PR).
- Sample size + seed recorded in each result row.
- Bootstrap CIs per scheme in the aggregated table.
EOF
)"

ensure_issue "T19 — Greek needle-in-a-haystack" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Synthetic retrieval test on Greek text, contexts 4k–16k (32k if memory allows), depth × length grid.

## Acceptance criteria
- `experiments/needle.py` produces a heatmap per scheme in `results/figures/needle/`.
- Needles/haystacks are Greek (names, numbers, places); not translated English.
- Grid parameters recorded in the config file.
EOF
)"

ensure_issue "T20 — Fidelity metrics vs FP16" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
KL divergence of next-token distributions and top-1 agreement rate vs the FP16 baseline.

## Acceptance criteria
- Added to every T17/T18/T19 run with negligible overhead.
- Output fields: `kl_mean`, `kl_p95`, `top1_agree`, `top5_agree`.
EOF
)"

ensure_issue "T21 — Experiment runner" "infra" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Resumable sweep runner (skips completed configs), progress logging, Kaggle/Colab notebook wrappers.

## Acceptance criteria
- `scripts/run_sweep.py` reads a sweep YAML, resumes from `results/` checkpoints.
- `notebooks/kaggle_runner.ipynb` and `notebooks/colab_runner.ipynb` are thin wrappers.
- Interrupted run can be restarted idempotently.
EOF
)"

ensure_issue "T22 — Statistical report generator" "eval" "M3 — Evaluation" "$(cat <<'EOF'
## Deliverable
Script that turns `results/` JSONL into tables and figures with CIs; `docs/RESULTS.md` regenerated by one command.

## Acceptance criteria
- `scripts/build_report.py` emits `docs/RESULTS.md` + figures.
- `make reproduce` (or equivalent) regenerates all tables/figures from raw results.
- Fails loudly if a referenced result file is missing.
EOF
)"

# --- M4 Systems --------------------------------------------------------------

ensure_issue "T23 — Memory & latency profiler" "systems" "M4 — Systems" "$(cat <<'EOF'
## Deliverable
Peak VRAM and per-token decode latency per scheme and context length.

## Acceptance criteria
- `src/greekkv/systems/profiler.py` + `experiments/profile.py`.
- Output JSONL + plots in `results/figures/profile/`.
- Reports both GPU and CPU paths where applicable.
EOF
)"

ensure_issue "T24 — Triton fused dequant-attention kernel (decode)" "systems" "M4 — Systems" "$(cat <<'EOF'
## Deliverable
Triton kernel: INT8 (then INT4) KV, single-query attention path for decode.

## Acceptance criteria
- Max abs error vs PyTorch reference under a stated tolerance.
- Runs on Kaggle T4 (compute 7.5).
- Unit + property tests.
EOF
)"

ensure_issue "T25 — Kernel benchmark" "systems" "M4 — Systems" "$(cat <<'EOF'
## Deliverable
Benchmark: T24 kernel vs unfused PyTorch vs FP16 baseline.

## Acceptance criteria
- Reports honest numbers, including when the kernel is **slower** than PyTorch on free hw.
- Plots + JSONL in `results/kernel_bench/`.
EOF
)"

ensure_issue "T26 — minillm.cpp cross-contribution" "cpp" "M4 — Systems" "$(cat <<'EOF'
## Deliverable
In the separate `minillm-cpp` repo (own PR): INT8 KV cache option with a NEON dequant path + benchmark.

## Acceptance criteria
- Existing C++ tests still pass.
- Benchmark documented in that repo's README.
- Link back to the GreekKV PR context.

**Blocker:** confirm minillm-cpp repo URL + access before starting.
EOF
)"

ensure_issue "T27 — Throughput sweep (OPTIONAL)" "systems" "M4 — Systems" "$(cat <<'EOF'
## Deliverable
vLLM / serving concurrency sweep **if** free credits allow; otherwise skip and say so.

## Acceptance criteria
- If skipped: a short note in `docs/RESULTS.md` explaining why; capacity numbers in M5 come from the memory model + measured latency, not fabricated throughput.
EOF
)"

# --- M5 Product --------------------------------------------------------------

ensure_issue "T28 — Capacity model + CLI" "core" "M5 — Product" "$(cat <<'EOF'
## Deliverable
Given model, VRAM, context, scheme → max concurrent users, using T07 + measured T23/T17 numbers.

## Acceptance criteria
- `src/greekkv/memory/capacity.py` with a `capacity(...)` function.
- CLI `greekkv-capacity ...` with `--measured|--estimate` flags.
- Each output row tags whether it used measured or estimated inputs.
EOF
)"

ensure_issue "T29 — Capacity planner web app" "planner" "M5 — Product" "$(cat <<'EOF'
## Deliverable
Static TypeScript app (model, GPU memory × count, context, quantization → users supported, memory breakdown, measured quality delta). Deployed on GitHub Pages (or Vercel).

## Acceptance criteria
- `planner/` builds with `pnpm build` or `npm run build`.
- Deployed URL in README.
- Reads a JSON data file generated from T22 so numbers stay in sync with `results/`.
EOF
)"

ensure_issue "T30 — Results dashboard" "planner" "M5 — Product" "$(cat <<'EOF'
## Deliverable
Interactive charts reading the JSONL results (quality vs bits vs context).

## Acceptance criteria
- Shares the T29 deployment.
- Charts are reproducible from the committed results file; no hidden data.
EOF
)"

# --- M6 Packaging & outreach -------------------------------------------------

ensure_issue "T31 — README" "docs" "M6 — Packaging & outreach" "$(cat <<'EOF'
## Deliverable
Headline result in the first 5 lines, one-command reproduction, results table, limitations section, badges.

## Acceptance criteria
- First paragraph answers the headline question with numbers + CIs.
- "What was measured vs estimated" is explicit.
- Reproduction command verified in a fresh venv.
EOF
)"

ensure_issue "T32 — Technical report" "docs" "M6 — Packaging & outreach" "$(cat <<'EOF'
## Deliverable
3–5 page report (markdown → PDF): question, method, results with CIs, limitations, what I'd do with production hardware.

## Acceptance criteria
- `docs/report.md` + exported `docs/report.pdf`.
- Every figure references its source script.
EOF
)"

ensure_issue "T33 — Release v0.1.0" "infra" "M6 — Packaging & outreach" "$(cat <<'EOF'
## Deliverable
Git tag, changelog, GitHub release with the report attached.

## Acceptance criteria
- `CHANGELOG.md` follows Keep a Changelog.
- `v0.1.0` tag pushed, release created via `gh release create`.
- Report PDF attached as an asset.
EOF
)"

ensure_issue "T34 — Outreach drafts" "docs" "M6 — Packaging & outreach" "$(cat <<'EOF'
## Deliverable
`docs/outreach/`: 6-line email (result → links → role), LinkedIn post with one figure, per-recipient tailoring checklist. **Drafts only.**

## Acceptance criteria
- Files committed under `docs/outreach/`.
- No messages are sent by the tooling; sending is a manual action by the author.
EOF
)"

echo
echo "Done. Review issues at: https://github.com/$REPO/issues"
