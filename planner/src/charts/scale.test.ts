import { describe, expect, test } from "vitest";
import { linearScale, logScale, niceLinearTicks, niceLogTicks } from "./scale";

describe("linearScale", () => {
  test("maps domain ends to range ends", () => {
    const s = linearScale([0, 100], [0, 1]);
    expect(s(0)).toBe(0);
    expect(s(100)).toBe(1);
    expect(s(50)).toBe(0.5);
  });

  test("invert reverses the mapping", () => {
    const s = linearScale([10, 20], [0, 100]);
    expect(s.invert?.(50)).toBeCloseTo(15);
  });

  test("zero-span domain does not divide by zero", () => {
    const s = linearScale([5, 5], [0, 100]);
    expect(Number.isFinite(s(5))).toBe(true);
  });
});

describe("logScale", () => {
  test("maps log decades linearly in range", () => {
    const s = logScale([1, 10_000], [0, 1]);
    expect(s(1)).toBeCloseTo(0);
    expect(s(10)).toBeCloseTo(0.25);
    expect(s(100)).toBeCloseTo(0.5);
    expect(s(10_000)).toBeCloseTo(1);
  });

  test("clamps non-positive inputs to a safe floor", () => {
    const s = logScale([1, 100], [0, 1]);
    expect(Number.isFinite(s(0))).toBe(true);
    expect(Number.isFinite(s(-5))).toBe(true);
  });
});

describe("niceLogTicks", () => {
  test("returns one tick per decade in range", () => {
    expect(niceLogTicks(1, 10_000)).toEqual([1, 10, 100, 1000, 10_000]);
  });

  test("handles non-power-of-ten bounds", () => {
    const t = niceLogTicks(3, 500);
    expect(t[0]).toBe(1);
    expect(t[t.length - 1]).toBe(1000);
  });
});

describe("niceLinearTicks", () => {
  test("hits round numbers near the target count", () => {
    const t = niceLinearTicks(0, 10, 5);
    expect(t).toEqual([0, 2, 4, 6, 8, 10]);
  });

  test("scales up for large spans", () => {
    const t = niceLinearTicks(0, 1000, 5);
    expect(t.length).toBeGreaterThanOrEqual(4);
    expect(t.length).toBeLessThanOrEqual(7);
  });
});
