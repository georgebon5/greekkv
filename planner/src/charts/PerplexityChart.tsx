// Perplexity × n_ctx chart — one line per scheme, log-y because the
// numbers span from ~4 (fp16) to ~12k (eviction-sink).
//
// Data: dataset.perplexity_rows (T17 JSONL). We filter to a single
// model so the lines stay comparable. If no rows exist for the picked
// model the chart renders a 'no data' placeholder rather than an
// empty frame.

import { useMemo } from "react";

import type { RawRow } from "../measurements";
import { colorFor, SCHEME_ORDER } from "./palette";
import { linearScale, logScale, niceLinearTicks, niceLogTicks } from "./scale";

export interface PerplexityChartProps {
  rows: RawRow[];
  model: string;
}

interface Series {
  scheme: string;
  points: Array<{ n_ctx: number; perplexity: number }>;
}

const W = 760;
const H = 300;
const PAD_LEFT = 54;
const PAD_RIGHT = 160;
const PAD_TOP = 16;
const PAD_BOTTOM = 36;

export function PerplexityChart({ rows, model }: PerplexityChartProps) {
  const series = useMemo(() => bucketize(rows, model), [rows, model]);

  if (series.length === 0) {
    return (
      <section className="chart">
        <h2>Perplexity × n_ctx</h2>
        <p className="muted">No perplexity rows for {model} in this snapshot.</p>
      </section>
    );
  }

  const allN = series.flatMap((s) => s.points.map((p) => p.n_ctx));
  const allP = series.flatMap((s) => s.points.map((p) => p.perplexity));
  const xDomain: [number, number] = [Math.min(...allN), Math.max(...allN)];
  const yDomain: [number, number] = [Math.min(...allP), Math.max(...allP)];

  const x = linearScale(xDomain, [PAD_LEFT, W - PAD_RIGHT]);
  const y = logScale(yDomain, [H - PAD_BOTTOM, PAD_TOP]);

  const xTicks = niceLinearTicks(xDomain[0], xDomain[1], 4);
  const yTicks = niceLogTicks(yDomain[0], yDomain[1]);

  return (
    <section className="chart">
      <h2>Perplexity × n_ctx</h2>
      <p className="muted">
        Lower is better. Log-y because the quantization floor and ceiling span
        three orders of magnitude on this model.
      </p>

      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Perplexity versus context length per scheme">
        {/* axes */}
        <g className="axis">
          {yTicks.map((t) => (
            <g key={`y-${t}`} transform={`translate(0, ${y(t)})`}>
              <line x1={PAD_LEFT} x2={W - PAD_RIGHT} stroke="#eee" />
              <text x={PAD_LEFT - 8} y={4} textAnchor="end">
                {formatTick(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={`x-${t}`} transform={`translate(${x(t)}, ${H - PAD_BOTTOM})`}>
              <line y1={0} y2={5} stroke="#555" />
              <text y={18} textAnchor="middle">
                {t.toLocaleString()}
              </text>
            </g>
          ))}
          <line
            x1={PAD_LEFT} x2={W - PAD_RIGHT}
            y1={H - PAD_BOTTOM} y2={H - PAD_BOTTOM}
            stroke="#555"
          />
          <text x={(PAD_LEFT + W - PAD_RIGHT) / 2} y={H - 6} textAnchor="middle" className="axis-label">
            n_ctx
          </text>
          <text
            transform={`translate(14, ${(PAD_TOP + H - PAD_BOTTOM) / 2}) rotate(-90)`}
            textAnchor="middle"
            className="axis-label"
          >
            perplexity (log)
          </text>
        </g>

        {/* series */}
        {series.map((s) => (
          <g key={s.scheme} className="series">
            <path
              d={toPath(s.points, x, y)}
              fill="none"
              stroke={colorFor(s.scheme)}
              strokeWidth={2}
            />
            {s.points.map((p) => (
              <circle
                key={`${s.scheme}-${p.n_ctx}`}
                cx={x(p.n_ctx)}
                cy={y(p.perplexity)}
                r={3.5}
                fill={colorFor(s.scheme)}
              >
                <title>
                  {s.scheme} · n_ctx={p.n_ctx} · ppl={p.perplexity.toFixed(2)}
                </title>
              </circle>
            ))}
          </g>
        ))}

        {/* legend */}
        <g transform={`translate(${W - PAD_RIGHT + 14}, ${PAD_TOP + 4})`}>
          {series.map((s, i) => (
            <g key={s.scheme} transform={`translate(0, ${i * 20})`}>
              <rect x={0} y={-8} width={12} height={12} fill={colorFor(s.scheme)} />
              <text x={18} y={2} className="legend-text">
                {s.scheme}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </section>
  );
}

function bucketize(rows: RawRow[], model: string): Series[] {
  const byScheme = new Map<string, Series>();
  for (const r of rows) {
    if (r.model !== model) continue;
    const n = Number(r.metrics["n_ctx"]);
    const ppl = Number(r.metrics["perplexity"]);
    if (!Number.isFinite(n) || !Number.isFinite(ppl) || ppl <= 0) continue;
    const cur = byScheme.get(r.scheme) ?? { scheme: r.scheme, points: [] };
    cur.points.push({ n_ctx: n, perplexity: ppl });
    byScheme.set(r.scheme, cur);
  }
  for (const s of byScheme.values()) {
    s.points.sort((a, b) => a.n_ctx - b.n_ctx);
  }
  return [...byScheme.values()].sort(
    (a, b) => SCHEME_ORDER.indexOf(a.scheme) - SCHEME_ORDER.indexOf(b.scheme),
  );
}

function toPath(
  points: Array<{ n_ctx: number; perplexity: number }>,
  x: (v: number) => number,
  y: (v: number) => number,
): string {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.n_ctx).toFixed(2)} ${y(p.perplexity).toFixed(2)}`)
    .join(" ");
}

function formatTick(v: number): string {
  if (v >= 1000) return `${(v / 1000).toLocaleString()}k`;
  return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
}
