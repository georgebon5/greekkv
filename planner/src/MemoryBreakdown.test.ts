// Numerical invariants for the memory-breakdown chart. We don't test
// the SVG DOM (that's visual) — we test the arithmetic that drives it,
// mirrored out of MemoryBreakdown.tsx. If that math changes, this file
// changes with it, and we catch an off-by-byte before it ships.

import { describe, expect, test } from "vitest";
import { capacityGrid, DEFAULT_OVERHEAD_BYTES } from "./capacity";
import { findModel } from "./models";

const QWEN = findModel("Qwen/Qwen2.5-0.5B");
const VRAM = 16 * 1024 ** 3;

function partition(row: {
  weight_bytes: number;
  overhead_bytes: number;
  free_bytes_after_weights: number;
  kv_bytes_per_user: number;
  max_concurrent_users: number;
}, vramBytes: number) {
  const kvTotal = row.kv_bytes_per_user * row.max_concurrent_users;
  const kvReserved = Math.min(kvTotal, Math.max(0, row.free_bytes_after_weights));
  const slack = Math.max(0, vramBytes - row.weight_bytes - row.overhead_bytes - kvReserved);
  return { kvReserved, slack };
}

describe("memory breakdown arithmetic", () => {
  test("segments always sum to vramBytes", () => {
    const grid = capacityGrid(QWEN, VRAM, 1024, DEFAULT_OVERHEAD_BYTES);
    for (const row of grid) {
      const { kvReserved, slack } = partition(row, VRAM);
      const sum = row.weight_bytes + row.overhead_bytes + kvReserved + slack;
      expect(sum).toBe(VRAM);
    }
  });

  test("KV reservation never exceeds the VRAM free budget", () => {
    const grid = capacityGrid(QWEN, VRAM, 1024, DEFAULT_OVERHEAD_BYTES);
    for (const row of grid) {
      const { kvReserved } = partition(row, VRAM);
      expect(kvReserved).toBeLessThanOrEqual(row.free_bytes_after_weights);
    }
  });

  test("slack is zero or near-zero for schemes that fill the budget", () => {
    // fp16 is the greediest per-user; at n_ctx = 1024 the integer division
    // in capacity() leaves at most one kv_per_user worth of slack.
    const grid = capacityGrid(QWEN, VRAM, 1024, DEFAULT_OVERHEAD_BYTES);
    const fp16 = grid.find((r) => r.scheme === "fp16")!;
    const { slack } = partition(fp16, VRAM);
    expect(slack).toBeLessThan(fp16.kv_bytes_per_user);
  });

  test("tiny VRAM: everything is weights + slack, no KV, no users", () => {
    // 10 MiB budget for a 494M-param FP16 model → weights don't even fit.
    // kvReserved must collapse to 0 and slack stays non-negative.
    const tinyVram = 10 * 1024 * 1024;
    const grid = capacityGrid(QWEN, tinyVram, 1024, 0);
    for (const row of grid) {
      const { kvReserved } = partition(row, tinyVram);
      expect(kvReserved).toBe(0);
      expect(row.max_concurrent_users).toBe(0);
    }
  });
});
