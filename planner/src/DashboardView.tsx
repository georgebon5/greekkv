// Dashboard view (T30). Charts come in subsequent commits (T30.3+).
// This commit lands the scaffold + the "reading the data" preamble so
// the tab is wired and the user can navigate even before any chart
// renders.

import type { Dataset } from "./measurements";

export function DashboardView({ dataset }: { dataset: Dataset | null }) {
  if (!dataset) {
    return (
      <section className="placeholder">
        <p>
          <strong>Offline.</strong> The dashboard reads{" "}
          <code>planner/public/data.json</code>, which was not available on
          this host. Regenerate it with{" "}
          <code>python scripts/build_report.py</code> and reload.
        </p>
      </section>
    );
  }

  const schemaOk = dataset.schema_version >= 2;
  const nPpl = dataset.perplexity_rows?.length ?? 0;
  const nProf = dataset.profile_rows?.length ?? 0;
  const nKernel = dataset.kernel_bench_rows?.length ?? 0;

  return (
    <>
      <p className="lede">
        All numbers below come straight from the committed{" "}
        <code>results/*.jsonl</code>. No smoothing, no interpolation — if a
        cell is missing it stays blank. Model coverage today is{" "}
        <code>Qwen/Qwen2.5-0.5B</code>; the M4 re-run will add Krikri and
        Meltemi on the same grid.
      </p>

      {!schemaOk && (
        <div className="placeholder">
          <p>
            <strong>Stale data.</strong> This snapshot was generated against
            an older schema (v{dataset.schema_version}). Re-run{" "}
            <code>python scripts/build_report.py</code> to regenerate.
          </p>
        </div>
      )}

      <section className="chart">
        <h2>Dataset overview</h2>
        <p className="muted">
          Generated {dataset.generated_at} · {nPpl} perplexity rows · {nProf}{" "}
          profile rows · {nKernel} kernel-bench rows.
        </p>
        <p className="muted">
          Charts (perplexity × n_ctx, decode latency × n_ctx, fidelity top-1
          vs KL) land in T30.3 – T30.5.
        </p>
      </section>
    </>
  );
}
