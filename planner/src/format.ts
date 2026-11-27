// Small formatting helpers shared by the controls and the result table.

export function formatBytes(n: number): string {
  if (n === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"] as const;
  let x = n;
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i += 1;
  }
  return `${x.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[i]}`;
}

export function formatInt(n: number): string {
  return n.toLocaleString("en-US");
}

export function formatNumber(n: number | undefined, digits = 2): string {
  if (n === undefined || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}
