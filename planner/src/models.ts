// Mirror of the _KNOWN_MODELS registry in src/greekkv/cli/capacity_cli.py.
// Keep the two in sync by hand — the Python test suite + TS vitest both
// pin the numbers through their respective fixtures.

import type { ArchConfig } from "./capacity";

export interface KnownModel extends ArchConfig {
  id: string;
}

export const KNOWN_MODELS: KnownModel[] = [
  {
    id: "Qwen/Qwen2.5-0.5B",
    num_hidden_layers: 24,
    num_attention_heads: 14,
    num_key_value_heads: 2,
    hidden_size: 896,
    head_dim: 64,
    num_parameters: 494_000_000,
  },
  {
    id: "Qwen/Qwen2.5-7B",
    num_hidden_layers: 28,
    num_attention_heads: 28,
    num_key_value_heads: 4,
    hidden_size: 3584,
    head_dim: 128,
    num_parameters: 7_615_616_512,
  },
  {
    id: "ilsp/Llama-Krikri-8B-Instruct",
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    hidden_size: 4096,
    head_dim: 128,
    num_parameters: 8_030_000_000,
  },
  {
    id: "ilsp/Meltemi-7B-Instruct-v1.5",
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    hidden_size: 4096,
    head_dim: 128,
    num_parameters: 7_240_000_000,
  },
];

export function findModel(id: string): KnownModel {
  const m = KNOWN_MODELS.find((k) => k.id === id);
  if (!m) throw new Error(`unknown model ${id}`);
  return m;
}
