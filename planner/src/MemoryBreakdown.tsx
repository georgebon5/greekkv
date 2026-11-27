// Stacked-bar chart of the VRAM breakdown per scheme, rendered as
// inline SVG so the planner stays zero-deps (no chart library).
//
// For each scheme we draw a horizontal bar sized to the VRAM budget
// and partition it into three bands:
//   1. weights (always present)
//   2. overhead (runtime: CUDA context, activations, framework)
//   3. KV cache for the computed concurrent-user count
// The remainder (if any) is slack — stays visually blank.
//
// The legend sits once above the bars so each row stays compact.

import type { CapacityRow } from "./capacity";
import { formatBytes, formatInt } from "./format";

const COLORS = {
  weights: "#4C6EF5",
  overhead: "#868E96",
  kv: "#20C997",
  slack: "#f1f3f5",
} as const;

interface MemoryBreakdownProps {
  rows: CapacityRow[];
  vramBytes: number;
}

export function MemoryBreakdown({ rows, vramBytes }: MemoryBreakdownProps) {
  if (vramBytes <= 0) return null;

  return (
    <section className="chart">
      <h2>Memory breakdown</h2>
      <p className="muted">
        One bar per scheme, each sized to the total VRAM budget. The KV band
        is the computed capacity × kv/user, i.e. the actual reserved cache
        for all concurrent users. Grey tail is slack.
      </p>

      <Legend />

      <div className="bars">
        {rows.map((r) => (
          <Bar key={r.scheme} row={r} vramBytes={vramBytes} />
        ))}
      </div>
    </section>
  );
}

function Legend() {
  return (
    <ul className="legend" aria-label="Legend">
      <LegendItem color={COLORS.weights} label="weights" />
      <LegendItem color={COLORS.overhead} label="overhead" />
      <LegendItem color={COLORS.kv} label="KV × users" />
      <LegendItem color={COLORS.slack} label="slack" />
    </ul>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <li>
      <span className="swatch" style={{ background: color }} aria-hidden="true" />
      {label}
    </li>
  );
}

function Bar({ row, vramBytes }: { row: CapacityRow; vramBytes: number }) {
  const kvTotal = row.kv_bytes_per_user * row.max_concurrent_users;
  // Clamp the reserved cache to the available free bytes so the stacked
  // bar never overshoots the VRAM budget — users × kv is always ≤ free
  // by construction, but keep the guard explicit.
  const kvReserved = Math.min(kvTotal, Math.max(0, row.free_bytes_after_weights));
  const slack = Math.max(0, vramBytes - row.weight_bytes - row.overhead_bytes - kvReserved);

  const toPct = (b: number) => (b / vramBytes) * 100;

  return (
    <div className="bar-row">
      <div className="bar-label">
        <code>{row.scheme}</code>
        <span className="muted">
          {formatInt(row.max_concurrent_users)} users · {formatBytes(row.kv_bytes_per_user)} / user
        </span>
      </div>
      <svg className="bar-svg" viewBox="0 0 100 10" preserveAspectRatio="none" role="img" aria-label={`${row.scheme} memory breakdown`}>
        <rect x={0} y={0} width={100} height={10} fill={COLORS.slack} />
        <Segment x={0} width={toPct(row.weight_bytes)} color={COLORS.weights} />
        <Segment x={toPct(row.weight_bytes)} width={toPct(row.overhead_bytes)} color={COLORS.overhead} />
        <Segment
          x={toPct(row.weight_bytes) + toPct(row.overhead_bytes)}
          width={toPct(kvReserved)}
          color={COLORS.kv}
        />
      </svg>
      <div className="bar-bytes muted">
        W {formatBytes(row.weight_bytes)} · O {formatBytes(row.overhead_bytes)} · KV {formatBytes(kvReserved)} · slack {formatBytes(slack)}
      </div>
    </div>
  );
}

function Segment({ x, width, color }: { x: number; width: number; color: string }) {
  if (width <= 0) return null;
  return <rect x={x} y={0} width={width} height={10} fill={color} />;
}
