import { describe, expect, it } from "vitest";

import {
  brushStrokeIndices,
  diskBrushIndices,
  diskOffsets,
  interpolateGridLine,
  sphereBrushIndices,
  sphereOffsets,
  validateBrushDiameter,
} from "./brush";
import { coordinateFromIndex } from "./volume";

describe("brush geometry", () => {
  it("accepts only odd diameters from 1 through 31", () => {
    expect(validateBrushDiameter(1)).toBe(1);
    expect(validateBrushDiameter(31)).toBe(31);
    for (const invalid of [0, 2, 15.5, 32, Number.NaN]) {
      expect(() => validateBrushDiameter(invalid)).toThrow(RangeError);
    }
  });

  it("builds true disk and sphere footprints", () => {
    expect(diskOffsets(2, 1)).toEqual([[0, 0, 0]]);
    expect(diskOffsets(2, 3)).toHaveLength(5);
    expect(sphereOffsets(3)).toHaveLength(7);
    expect(sphereOffsets(5)).toHaveLength(33);
  });

  it("clips footprints at volume boundaries", () => {
    expect([...diskBrushIndices([3, 3, 1], [0, 0, 0], 2, 3)]).toEqual([
      0, 3, 1,
    ]);
    expect(sphereBrushIndices([3, 3, 3], [0, 0, 0], 3)).toHaveLength(4);
  });

  it("interpolates gap-free and never emits out-of-bounds coordinates", () => {
    const points = interpolateGridLine([-5, -2, 1], [8, 6, 1], [5, 5, 3]);
    expect(points.length).toBeGreaterThan(1);
    for (const point of points) {
      expect(point[0]).toBeGreaterThanOrEqual(0);
      expect(point[0]).toBeLessThan(5);
      expect(point[1]).toBeGreaterThanOrEqual(0);
      expect(point[1]).toBeLessThan(5);
    }
    for (let index = 1; index < points.length; index += 1) {
      const current = points[index]!;
      const previous = points[index - 1]!;
      expect(
        Math.max(
          Math.abs(current[0] - previous[0]),
          Math.abs(current[1] - previous[1]),
          Math.abs(current[2] - previous[2]),
        ),
      ).toBeLessThanOrEqual(1);
    }
    expect(interpolateGridLine([-2, 0, 0], [-1, 4, 0], [5, 5, 1])).toEqual([]);
  });

  it("unions interpolated dabs into one sparse stroke", () => {
    const dimensions = [7, 5, 1] as const;
    const indices = brushStrokeIndices(
      dimensions,
      [0, 2, 0],
      [6, 2, 0],
      1,
      2,
    );
    expect([...indices].map((index) => coordinateFromIndex(dimensions, index))).toEqual([
      [0, 2, 0],
      [1, 2, 0],
      [2, 2, 0],
      [3, 2, 0],
      [4, 2, 0],
      [5, 2, 0],
      [6, 2, 0],
    ]);
  });
});
