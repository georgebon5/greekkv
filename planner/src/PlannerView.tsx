// The original T29 planner UI, extracted so T30 can swap it out for the
// dashboard view via the tab switcher in App.tsx. Nothing about this
// component's behaviour changed — only its position in the tree.

import { useMemo, useState } from "react";

import { capacityGrid, type CapacityRow } from "./capacity";
import { formatBytes, formatInt, formatNumber } from "./format";
import { MemoryBreakdown } from "./MemoryBreakdown";
import { type Dataset, pickMeasurement } from "./measurements";
import { KNOWN_MODELS, findModel } from "./models";

const N_CTX_CHOICES = [256, 512, 1024, 2048, 4096, 8192];
const DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B";
const DEFAULT_VRAM_GB = 16;
const DEFAULT_GPU_COUNT = 1;
const DEFAULT_N_CTX = 1024;
const DEFAULT_OVERHEAD_MB = 650;

export function PlannerView({ dataset }: { dataset: Dataset | null }) {
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [vramGb, setVramGb] = useState(DEFAULT_VRAM_GB);
  const [gpuCount, setGpuCount] = useState(DEFAULT_GPU_COUNT);
  const [nCtx, setNCtx] = useState(DEFAULT_N_CTX);
  const [overheadMb, setOverheadMb] = useState(DEFAULT_OVERHEAD_MB);

  const cfg = useMemo(() => findModel(modelId), [modelId]);

  const totalVramBytes = useMemo(
    () => Math.round(vramGb * gpuCount * 1024 ** 3),
    [vramGb, gpuCount],
  );

  const rows = useMemo(() => {
    const overheadBytes = Math.round(overheadMb * 1024 * 1024);
    return capacityGrid(cfg, totalVramBytes, nCtx, overheadBytes);
  }, [cfg, totalVramBytes, nCtx, overheadMb]);

  return (
    <>
      <p className="lede">
        Translate a GPU memory budget into max concurrent users per KV cache
        scheme. Hardware math is live; quality (PPL) and latency (ms/token)
        columns come from the committed <code>results/</code> JSONL files.
      </p>

      <Controls
        modelId={modelId}
        setModelId={setModelId}
        vramGb={vramGb}
        setVramGb={setVramGb}
        gpuCount={gpuCount}
        setGpuCount={setGpuCount}
        nCtx={nCtx}
        setNCtx={setNCtx}
        overheadMb={overheadMb}
        setOverheadMb={setOverheadMb}
      />

      <ResultsTable rows={rows} modelId={modelId} nCtx={nCtx} dataset={dataset} />

      <MemoryBreakdown rows={rows} vramBytes={totalVramBytes} />
    </>
  );
}

interface ControlsProps {
  modelId: string;
  setModelId: (v: string) => void;
  vramGb: number;
  setVramGb: (v: number) => void;
  gpuCount: number;
  setGpuCount: (v: number) => void;
  nCtx: number;
  setNCtx: (v: number) => void;
  overheadMb: number;
  setOverheadMb: (v: number) => void;
}

function Controls(p: ControlsProps) {
  return (
    <section className="controls">
      <label>
        Model
        <select value={p.modelId} onChange={(e) => p.setModelId(e.target.value)}>
          {KNOWN_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}
            </option>
          ))}
        </select>
      </label>

      <label>
        VRAM per GPU (GiB)
        <input
          type="number"
          min={1}
          max={1024}
          step={1}
          value={p.vramGb}
          onChange={(e) => p.setVramGb(Math.max(1, Number(e.target.value) || 0))}
        />
      </label>

      <label>
        GPU count
        <input
          type="number"
          min={1}
          max={64}
          step={1}
          value={p.gpuCount}
          onChange={(e) => p.setGpuCount(Math.max(1, Number(e.target.value) || 0))}
        />
      </label>

      <label>
        Context length (tokens)
        <select value={p.nCtx} onChange={(e) => p.setNCtx(Number(e.target.value))}>
          {N_CTX_CHOICES.map((n) => (
            <option key={n} value={n}>
              {n.toLocaleString()}
            </option>
          ))}
        </select>
      </label>

      <label>
        Runtime overhead (MiB)
        <input
          type="number"
          min={0}
          max={100_000}
          step={50}
          value={p.overheadMb}
          onChange={(e) =>
            p.setOverheadMb(Math.max(0, Number(e.target.value) || 0))
          }
        />
      </label>
    </section>
  );
}

interface ResultsTableProps {
  rows: CapacityRow[];
  modelId: string;
  nCtx: number;
  dataset: Dataset | null;
}

function ResultsTable({ rows, modelId, nCtx, dataset }: ResultsTableProps) {
  return (
    <section className="results">
      <table>
        <thead>
          <tr>
            <th>Scheme</th>
            <th className="num">Users</th>
            <th className="num">KV / user</th>
            <th className="num">Weights</th>
            <th className="num">PPL</th>
            <th className="num">ms / token</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const m = pickMeasurement(dataset, modelId, r.scheme, nCtx);
            const hasAny =
              m && (m.perplexity !== undefined || m.decode_ms_per_token !== undefined);
            return (
              <tr key={r.scheme}>
                <td>
                  <code>{r.scheme}</code>
                </td>
                <td className="num">{formatInt(r.max_concurrent_users)}</td>
                <td className="num">{formatBytes(r.kv_bytes_per_user)}</td>
                <td className="num">{formatBytes(r.weight_bytes)}</td>
                <td className="num">{formatNumber(m?.perplexity)}</td>
                <td className="num">{formatNumber(m?.decode_ms_per_token)}</td>
                <td>{hasAny ? "measured" : "estimated"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
