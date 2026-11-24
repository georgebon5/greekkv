import { describe, expect, test } from "vitest";
import {
  ALL_SCHEMES,
  capacity,
  capacityGrid,
  DEFAULT_OVERHEAD_BYTES,
  DEFAULT_SCHEME_SPECS,
  kvBytes,
  type ArchConfig,
  type Scheme,
} from "./capacity";

// Qwen2.5-0.5B — mirrors the Python _qwen25_0_5b() fixture in tests/test_capacity.py.
const QWEN: ArchConfig = {
  num_hidden_layers: 24,
  num_attention_heads: 14,
  num_key_value_heads: 2,
  hidden_size: 896,
  head_dim: 64,
  num_parameters: 494_000_000,
};

const VRAM = 16 * 1024 ** 3;

// Hard-coded fixtures generated from the Python reference
// (see commit message of T29.3). Any divergence here means the TS
// port drifted — do not relax the assertion, fix the TS code.
const PARITY_FIXTURES: Array<{
  scheme: Scheme;
  n_ctx: number;
  weight_bytes: number;
  kv_bytes_per_user: number;
  max_concurrent_users: number;
}> = [
  { scheme: "fp16",           n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:  3145728, max_concurrent_users:  4930 },
  { scheme: "fp16",           n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user: 12582912, max_concurrent_users:  1232 },
  { scheme: "fp16",           n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user: 50331648, max_concurrent_users:   308 },
  { scheme: "int8-fakequant", n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:  1622016, max_concurrent_users:  9562 },
  { scheme: "int8-fakequant", n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user:  6488064, max_concurrent_users:  2390 },
  { scheme: "int8-fakequant", n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user: 25952256, max_concurrent_users:   597 },
  { scheme: "int8-kivi",      n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:  1836288, max_concurrent_users:  8446 },
  { scheme: "int8-kivi",      n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user:  6702336, max_concurrent_users:  2314 },
  { scheme: "int8-kivi",      n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user: 26166528, max_concurrent_users:   592 },
  { scheme: "int4-fakequant", n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:   835584, max_concurrent_users: 18562 },
  { scheme: "int4-fakequant", n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user:  3342336, max_concurrent_users:  4640 },
  { scheme: "int4-fakequant", n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user: 13369344, max_concurrent_users:  1160 },
  { scheme: "int4-kivi",      n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:  1160448, max_concurrent_users: 13365 },
  { scheme: "int4-kivi",      n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user:  3667200, max_concurrent_users:  4229 },
  { scheme: "int4-kivi",      n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user: 13694208, max_concurrent_users:  1132 },
  { scheme: "eviction-sink",  n_ctx:  256, weight_bytes: 988000000, kv_bytes_per_user:   442368, max_concurrent_users: 35061 },
  { scheme: "eviction-sink",  n_ctx: 1024, weight_bytes: 988000000, kv_bytes_per_user:   442368, max_concurrent_users: 35061 },
  { scheme: "eviction-sink",  n_ctx: 4096, weight_bytes: 988000000, kv_bytes_per_user:   442368, max_concurrent_users: 35061 },
];

describe("TS ↔ Python capacity parity (Qwen2.5-0.5B, 16 GiB)", () => {
  for (const f of PARITY_FIXTURES) {
    test(`${f.scheme} @ n_ctx=${f.n_ctx}`, () => {
      const row = capacity({
        cfg: QWEN,
        vram_bytes: VRAM,
        n_ctx: f.n_ctx,
        scheme: f.scheme,
      });
      expect(row.weight_bytes).toBe(f.weight_bytes);
      expect(row.kv_bytes_per_user).toBe(f.kv_bytes_per_user);
      expect(row.max_concurrent_users).toBe(f.max_concurrent_users);
    });
  }
});

describe("capacity edge cases", () => {
  test("unknown scheme throws", () => {
    expect(() => capacity({
      cfg: QWEN,
      vram_bytes: VRAM,
      n_ctx: 1024,
      // @ts-expect-error — testing the runtime guard against a bogus scheme
      scheme: "nf4-magic",
    })).toThrow(/unknown scheme/);
  });

  test("negative n_ctx throws", () => {
    expect(() => capacity({
      cfg: QWEN,
      vram_bytes: VRAM,
      n_ctx: -1,
      scheme: "fp16",
    })).toThrow(/n_ctx/);
  });

  test("zero n_ctx yields zero users", () => {
    const row = capacity({
      cfg: QWEN,
      vram_bytes: VRAM,
      n_ctx: 0,
      scheme: "int8-fakequant",
    });
    expect(row.kv_bytes_per_user).toBe(0);
    expect(row.max_concurrent_users).toBe(0);
  });

  test("tiny VRAM gives zero users", () => {
    const row = capacity({
      cfg: QWEN,
      vram_bytes: 10 * 1024 * 1024,
      n_ctx: 4096,
      scheme: "fp16",
    });
    expect(row.max_concurrent_users).toBe(0);
  });

  test("eviction-sink cache bytes do not grow with n_ctx", () => {
    const low  = capacity({ cfg: QWEN, vram_bytes: VRAM, n_ctx: 1024,  scheme: "eviction-sink" });
    const high = capacity({ cfg: QWEN, vram_bytes: VRAM, n_ctx: 32768, scheme: "eviction-sink" });
    expect(low.kv_bytes_per_user).toBe(high.kv_bytes_per_user);
  });

  test("KIVI degenerates to fp16 when n_ctx ≤ residual_tokens", () => {
    const spec = DEFAULT_SCHEME_SPECS["int8-kivi"];
    const row = capacity({ cfg: QWEN, vram_bytes: VRAM, n_ctx: spec.residual_tokens, scheme: "int8-kivi" });
    const expected = kvBytes(QWEN, spec.residual_tokens, 16, 0);
    expect(row.kv_bytes_per_user).toBe(expected);
  });
});

describe("capacityGrid", () => {
  test("returns one row per canonical scheme", () => {
    const grid = capacityGrid(QWEN, VRAM, 1024, DEFAULT_OVERHEAD_BYTES);
    expect(grid.map((r) => r.scheme)).toEqual([...ALL_SCHEMES]);
  });
});
