// Fidelity scatter — top-1 agreement vs KL(P_ref || P_q) per
// (scheme, n_ctx). The useful quadrant is top-left: high top-1
// agreement, low KL. Schemes that fall off the top-left (int4-kivi,
// eviction-sink) are the ones to argue against before deploying.
//
// Data: dataset.perplexity_rows (T17 — the fidelity fields live
// alongside the perplexity numbers).

import { useMemo } from "react";

import type { RawRow } from "../measurements";
import { colorFor, SCHEME_ORDER } from "./palette";
import { linearScale, niceLinearTicks } from "./scale";

export interface FidelityScatterProps {
  rows: RawRow[];
  model: string;
}

interface Point {
  scheme: string;
  n_ctx: number;
  top1: number;
  kl_mean: number;
}

const W = 760;
const H = 360;
const PAD_LEFT = 54;
const PAD_RIGHT = 160;
const PAD_TOP = 16;
const PAD_BOTTOM = 42;

export function FidelityScatter({ rows, model }: FidelityScatterProps) {
  const points = useMemo(() => bucketize(rows, model), [rows, model]);

  if (points.length === 0) {
    return (
      <section className="chart">
        <h2>Fidelity scatter — top-1 agreement vs KL</h2>
        <p className="muted">No fidelity rows for {model} in this snapshot.</p>
      </section>
    );
  }

  const maxKl = Math.max(...points.map((p) => p.kl_mean));
  const xDomain: [number, number] = [0, Math.max(1e-3, maxKl * 1.1)];
  const yDomain: [number, number] = [0, 1];

  const x = linearScale(xDomain, [PAD_LEFT, W - PAD_RIGHT]);
  const y = linearScale(yDomain, [H - PAD_BOTTOM, PAD_TOP]);

  const xTicks = niceLinearTicks(xDomain[0], xDomain[1], 5);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  const schemes = [...new Set(points.map((p) => p.scheme))].sort(
    (a, b) => SCHEME_ORDER.indexOf(a) - SCHEME_ORDER.indexOf(b),
  );

  return (
    <section className="chart">
      <h2>Fidelity — top-1 agreement vs KL(P_ref ∥ P_q)</h2>
      <p className="muted">
        Top-left is the goal: high argmax agreement with FP16, low KL
        divergence on the mass FP16 cares about. Each point is one{" "}
        <code>(scheme, n_ctx)</code> cell. The same scheme appears multiple
        times, once per measured context length.
      </p>

      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fidelity scatter: top-1 agreement versus KL divergence">
        {/* axes */}
        <g className="axis">
          {yTicks.map((t) => (
            <g key={`y-${t}`} transform={`translate(0, ${y(t)})`}>
              <line x1={PAD_LEFT} x2={W - PAD_RIGHT} stroke="#eee" />
              <text x={PAD_LEFT - 8} y={4} textAnchor="end">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={`x-${t}`} transform={`translate(${x(t)}, ${H - PAD_BOTTOM})`}>
              <line y1={0} y2={5} stroke="#555" />
              <text y={18} textAnchor="middle">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          <line
            x1={PAD_LEFT} x2={W - PAD_RIGHT}
            y1={H - PAD_BOTTOM} y2={H - PAD_BOTTOM}
            stroke="#555"
          />
          <text x={(PAD_LEFT + W - PAD_RIGHT) / 2} y={H - 12} textAnchor="middle" className="axis-label">
            KL(P_ref ∥ P_q), mean
          </text>
          <text
            transform={`translate(14, ${(PAD_TOP + H - PAD_BOTTOM) / 2}) rotate(-90)`}
            textAnchor="middle"
            className="axis-label"
          >
            top-1 agreement vs FP16
          </text>
        </g>

        {/* points */}
        {points.map((p) => (
          <circle
            key={`${p.scheme}-${p.n_ctx}`}
            cx={x(p.kl_mean)}
            cy={y(p.top1)}
            r={5}
            fill={colorFor(p.scheme)}
            fillOpacity={0.75}
            stroke="#fff"
            strokeWidth={1}
          >
            <title>
              {p.scheme} · n_ctx={p.n_ctx} · top1={p.top1.toFixed(3)} · KL={p.kl_mean.toFixed(3)}
            </title>
          </circle>
        ))}

        {/* "ideal" annotation on the top-left corner */}
        <g transform={`translate(${PAD_LEFT + 10}, ${PAD_TOP + 14})`} className="axis">
          <text className="axis-label" fill="#1F7A1F">
            ← ideal
          </text>
        </g>

        {/* legend */}
        <g transform={`translate(${W - PAD_RIGHT + 14}, ${PAD_TOP + 4})`}>
          {schemes.map((s, i) => (
            <g key={s} transform={`translate(0, ${i * 20})`}>
              <circle cx={6} cy={-2} r={5} fill={colorFor(s)} fillOpacity={0.75} />
              <text x={18} y={2} className="legend-text">
                {s}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </section>
  );
}

function bucketize(rows: RawRow[], model: string): Point[] {
  const out: Point[] = [];
  for (const r of rows) {
    if (r.model !== model) continue;
    const n_ctx = Number(r.metrics["n_ctx"]);
    const top1 = Number(r.metrics["top1_agree"]);
    const kl_mean = Number(r.metrics["kl_mean"]);
    if (!Number.isFinite(n_ctx) || !Number.isFinite(top1) || !Number.isFinite(kl_mean)) continue;
    out.push({ scheme: r.scheme, n_ctx, top1, kl_mean });
  }
  return out;
}
