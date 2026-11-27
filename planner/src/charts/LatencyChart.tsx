// Decode latency × n_ctx chart. Linear-y (ms / token). Highlights the
// T23 one-big-finding: int4-fakequant is ~20× slower per token than
// everything else because of the dispatch-overhead pile-up — a 2 ms
// vs 140 ms gap that reads better on a linear axis than on a log one
// (the outlier should feel like an outlier).

import { useMemo } from "react";

import type { RawRow } from "../measurements";
import { colorFor, SCHEME_ORDER } from "./palette";
import { linearScale, niceLinearTicks } from "./scale";

export interface LatencyChartProps {
  rows: RawRow[];
  model: string;
}

interface Series {
  scheme: string;
  points: Array<{ n_ctx: number; ms: number }>;
}

const W = 760;
const H = 300;
const PAD_LEFT = 54;
const PAD_RIGHT = 160;
const PAD_TOP = 16;
const PAD_BOTTOM = 36;

export function LatencyChart({ rows, model }: LatencyChartProps) {
  const series = useMemo(() => bucketize(rows, model), [rows, model]);

  if (series.length === 0) {
    return (
      <section className="chart">
        <h2>Decode latency × n_ctx</h2>
        <p className="muted">No profile rows for {model} in this snapshot.</p>
      </section>
    );
  }

  const allN = series.flatMap((s) => s.points.map((p) => p.n_ctx));
  const allMs = series.flatMap((s) => s.points.map((p) => p.ms));
  const xDomain: [number, number] = [Math.min(...allN), Math.max(...allN)];
  // Start the Y at 0 so the comparison is visually fair.
  const yDomain: [number, number] = [0, Math.max(...allMs) * 1.1];

  const x = linearScale(xDomain, [PAD_LEFT, W - PAD_RIGHT]);
  const y = linearScale(yDomain, [H - PAD_BOTTOM, PAD_TOP]);

  const xTicks = niceLinearTicks(xDomain[0], xDomain[1], 4);
  const yTicks = niceLinearTicks(yDomain[0], yDomain[1], 5);

  return (
    <section className="chart">
      <h2>Decode latency × n_ctx</h2>
      <p className="muted">
        Lower is better. The gap for <code>int4-fakequant</code> (~140 ms/token
        vs ~5 ms for everything else) is the T23 architectural finding — a
        full quantize→dequantize round trip firing on every layer at every
        decode step.
      </p>

      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Decode latency versus context length per scheme">
        {/* axes */}
        <g className="axis">
          {yTicks.map((t) => (
            <g key={`y-${t}`} transform={`translate(0, ${y(t)})`}>
              <line x1={PAD_LEFT} x2={W - PAD_RIGHT} stroke="#eee" />
              <text x={PAD_LEFT - 8} y={4} textAnchor="end">
                {t.toFixed(0)}
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
            decode ms / token
          </text>
        </g>

        {/* series */}
        {series.map((s) => (
          <g key={s.scheme} className="series">
            <path d={toPath(s.points, x, y)} fill="none" stroke={colorFor(s.scheme)} strokeWidth={2} />
            {s.points.map((p) => (
              <circle
                key={`${s.scheme}-${p.n_ctx}`}
                cx={x(p.n_ctx)}
                cy={y(p.ms)}
                r={3.5}
                fill={colorFor(s.scheme)}
              >
                <title>
                  {s.scheme} · n_ctx={p.n_ctx} · {p.ms.toFixed(2)} ms/tok
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
    const ms = Number(r.metrics["decode_ms_per_token"]);
    if (!Number.isFinite(n) || !Number.isFinite(ms) || ms < 0) continue;
    const cur = byScheme.get(r.scheme) ?? { scheme: r.scheme, points: [] };
    cur.points.push({ n_ctx: n, ms });
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
  points: Array<{ n_ctx: number; ms: number }>,
  x: (v: number) => number,
  y: (v: number) => number,
): string {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.n_ctx).toFixed(2)} ${y(p.ms).toFixed(2)}`)
    .join(" ");
}
