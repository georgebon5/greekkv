// Loader for the baked-in data.json snapshot written by
// scripts/build_report.py. The planner computes capacity numbers
// locally but needs measurements for PPL / decode latency columns.

export interface Measurement {
  model: string;
  scheme: string;
  n_ctx: number;
  perplexity?: number;
  kl_mean?: number;
  top1_agree?: number;
  decode_ms_per_token?: number;
  peak_mem_mb?: number;
}

export interface RawRow {
  model: string;
  scheme: string;
  metrics: Record<string, number | string | boolean | null>;
}

export interface Dataset {
  schema_version: number;
  generated_at: string;
  models: string[];
  schemes: string[];
  n_ctx_grid: number[];
  measurements: Measurement[];
  // Schema v2+: raw per-experiment rows for the dashboard. Older
  // snapshots omit them, hence the optional ?.
  perplexity_rows?: RawRow[];
  profile_rows?: RawRow[];
  kernel_bench_rows?: RawRow[];
}

export function pickMeasurement(
  ds: Dataset | null,
  model: string,
  scheme: string,
  nCtx: number,
): Measurement | null {
  if (!ds) return null;
  return (
    ds.measurements.find(
      (m) => m.model === model && m.scheme === scheme && m.n_ctx === nCtx,
    ) ?? null
  );
}

export async function loadDataset(basePath: string): Promise<Dataset | null> {
  const url = `${basePath.replace(/\/$/, "")}/data.json`;
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) return null;
    return (await res.json()) as Dataset;
  } catch {
    return null;
  }
}
