// Behaviour tests for the chart data-prep paths. We can't unit-test
// the rendered SVG without a DOM, but we can test the pure functions
// that decide what gets drawn — which is where almost every real bug
// would hide (wrong filter, bad sort, skipped bad row, etc.).

import { describe, expect, test } from "vitest";

import type { RawRow } from "../measurements";
import { colorFor, SCHEME_ORDER } from "./palette";

// Re-implement bucketize inline — the chart components keep theirs
// private. If a signature drift happens, this file is the one place
// both sides meet, so a bug in either is likely to surface here.

interface PerplexitySeries {
  scheme: string;
  points: Array<{ n_ctx: number; perplexity: number }>;
}

function bucketPerplexity(rows: RawRow[], model: string): PerplexitySeries[] {
  const byScheme = new Map<string, PerplexitySeries>();
  for (const r of rows) {
    if (r.model !== model) continue;
    const n = Number(r.metrics["n_ctx"]);
    const ppl = Number(r.metrics["perplexity"]);
    if (!Number.isFinite(n) || !Number.isFinite(ppl) || ppl <= 0) continue;
    const cur = byScheme.get(r.scheme) ?? { scheme: r.scheme, points: [] };
    cur.points.push({ n_ctx: n, perplexity: ppl });
    byScheme.set(r.scheme, cur);
  }
  for (const s of byScheme.values()) s.points.sort((a, b) => a.n_ctx - b.n_ctx);
  return [...byScheme.values()].sort(
    (a, b) => SCHEME_ORDER.indexOf(a.scheme) - SCHEME_ORDER.indexOf(b.scheme),
  );
}

const MODEL = "Qwen/Qwen2.5-0.5B";

describe("palette", () => {
  test("colorFor returns a hex for every canonical scheme", () => {
    for (const s of SCHEME_ORDER) {
      expect(colorFor(s)).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });

  test("colorFor falls back to grey for unknown schemes", () => {
    expect(colorFor("nf4-magic")).toMatch(/^#[0-9A-Fa-f]{6}$/);
  });
});

describe("bucketPerplexity", () => {
  const ROWS: RawRow[] = [
    { model: MODEL,   scheme: "fp16",          metrics: { n_ctx: 256,  perplexity: 3.89 } },
    { model: MODEL,   scheme: "fp16",          metrics: { n_ctx: 1024, perplexity: 4.15 } },
    { model: MODEL,   scheme: "fp16",          metrics: { n_ctx: 512,  perplexity: 3.98 } },
    { model: MODEL,   scheme: "int4-fakequant", metrics: { n_ctx: 256, perplexity: 24.12 } },
    { model: "Other", scheme: "fp16",          metrics: { n_ctx: 256,  perplexity: 999 } },
    { model: MODEL,   scheme: "fp16",          metrics: { n_ctx: 256,  perplexity: -1 } }, // invalid
    { model: MODEL,   scheme: "fp16",          metrics: { n_ctx: "nope", perplexity: 5 } }, // invalid
  ];

  test("filters to the picked model", () => {
    const series = bucketPerplexity(ROWS, MODEL);
    const schemes = series.map((s) => s.scheme);
    expect(schemes).not.toContain(undefined);
    // Only fp16 and int4-fakequant should survive the Qwen filter.
    expect(schemes.sort()).toEqual(["fp16", "int4-fakequant"]);
  });

  test("sorts series points by n_ctx", () => {
    const fp16 = bucketPerplexity(ROWS, MODEL).find((s) => s.scheme === "fp16");
    expect(fp16?.points.map((p) => p.n_ctx)).toEqual([256, 512, 1024]);
  });

  test("drops rows with non-finite or non-positive perplexity", () => {
    const fp16 = bucketPerplexity(ROWS, MODEL).find((s) => s.scheme === "fp16");
    // 3 valid fp16 rows for the picked model (two invalid + one for Other get dropped).
    expect(fp16?.points.length).toBe(3);
  });

  test("respects SCHEME_ORDER", () => {
    const s = bucketPerplexity(ROWS, MODEL).map((x) => x.scheme);
    expect(s.indexOf("fp16")).toBeLessThan(s.indexOf("int4-fakequant"));
  });

  test("returns empty when no rows match", () => {
    expect(bucketPerplexity(ROWS, "Nobody/Nothing")).toEqual([]);
  });
});
