// Minimal linear + log scale helpers for the SVG charts. Zero deps,
// behaviour explicit — we never want a chart to silently clip a
// datapoint because a library decided to.

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  invert?: (p: number) => number;
}

export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  fn.domain = [d0, d1];
  fn.range = [r0, r1];
  fn.invert = (p: number) => d0 + ((p - r0) / (r1 - r0)) * span;
  return fn;
}

export function logScale(domain: [number, number], range: [number, number]): Scale {
  // log10 with a hard floor at 1e-9 so one bad input never explodes
  // the scale.
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const safe = (v: number) => Math.log10(Math.max(1e-9, v));
  const l0 = safe(d0);
  const l1 = safe(d1);
  const span = l1 - l0 || 1;
  const fn = ((v: number) => r0 + ((safe(v) - l0) / span) * (r1 - r0)) as Scale;
  fn.domain = [d0, d1];
  fn.range = [r0, r1];
  return fn;
}

export function niceLogTicks(d0: number, d1: number): number[] {
  if (d0 <= 0 || d1 <= 0 || d1 <= d0) return [d1];
  const start = Math.floor(Math.log10(d0));
  const end = Math.ceil(Math.log10(d1));
  const out: number[] = [];
  for (let e = start; e <= end; e++) {
    out.push(10 ** e);
  }
  return out;
}

export function niceLinearTicks(d0: number, d1: number, target = 5): number[] {
  if (d0 === d1) return [d0];
  const span = d1 - d0;
  const step0 = span / target;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const choices = [1, 2, 2.5, 5, 10].map((c) => c * mag);
  const step = choices.find((c) => span / c <= target * 1.2) ?? choices[choices.length - 1];
  const first = Math.ceil(d0 / step) * step;
  const out: number[] = [];
  for (let v = first; v <= d1 + 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}
