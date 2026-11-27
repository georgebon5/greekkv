// Scheme-stable colour palette, shared across every dashboard chart so
// a reader can learn a scheme → colour mapping once and have it hold
// everywhere. Picked from the Tailwind defaults to stay distinct on
// both light and dark backgrounds.

export const SCHEME_COLORS: Record<string, string> = {
  "fp16":            "#111827", // slate-900
  "int8-fakequant":  "#1D4ED8", // blue-700
  "int8-kivi":       "#2563EB", // blue-600 (near int8 — same family)
  "int4-fakequant":  "#D97706", // amber-600
  "int4-kivi":       "#F59E0B", // amber-500
  "eviction-sink":   "#DC2626", // red-600
};

export function colorFor(scheme: string): string {
  return SCHEME_COLORS[scheme] ?? "#6B7280"; // gray-500 fallback
}

// Canonical order — used both for the legend and the draw order so
// the "loudest" (worst) schemes land on top of the quieter ones.
export const SCHEME_ORDER: readonly string[] = [
  "fp16",
  "int8-fakequant",
  "int8-kivi",
  "int4-fakequant",
  "int4-kivi",
  "eviction-sink",
];
