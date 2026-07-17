import { describe, expect, it } from "vitest";

import {
  affineReciprocalCondition,
  createMatrix4,
  indexRayFromWorld,
  invertAffine,
  multiplyAffine,
  spacingFromAffine,
  transformDirection,
  transformPoint,
} from "./affine";

describe("affine helpers", () => {
  it("transforms points and directions without translating directions", () => {
    const affine = createMatrix4([
      0, -3, 0, 10,
      2, 0, 0, 20,
      0, 0, -4, 30,
      0, 0, 0, 1,
    ]);

    expect(transformPoint(affine, [1, 2, 3])).toEqual([4, 22, 18]);
    expect(transformDirection(affine, [1, 2, 3])).toEqual([-6, 2, -12]);
    expect(spacingFromAffine(affine)).toEqual([2, 3, 4]);
  });

  it("inverts an oblique affine in double precision", () => {
    const affine = createMatrix4([
      1.25, 0.3, -0.1, 12.5,
      -0.2, 2.5, 0.4, -7.25,
      0.15, -0.35, 3.75, 3.5,
      0, 0, 0, 1,
    ]);
    const inverse = invertAffine(affine);
    const original = [5.5, -2.25, 8.125] as const;
    const roundTrip = transformPoint(inverse, transformPoint(affine, original));

    expect(roundTrip[0]).toBeCloseTo(original[0], 12);
    expect(roundTrip[1]).toBeCloseTo(original[1], 12);
    expect(roundTrip[2]).toBeCloseTo(original[2], 12);
    for (const [index, value] of multiplyAffine(affine, inverse).entries()) {
      expect(value).toBeCloseTo(index % 5 === 0 ? 1 : 0, 12);
    }
  });

  it("accepts a well-conditioned affine expressed in tiny world units", () => {
    const affine = createMatrix4([
      1e-12, 0, 0, 3e-12,
      0, 2e-12, 0, -8e-12,
      0, 0, 4e-12, 20e-12,
      0, 0, 0, 1,
    ]);
    const inverse = invertAffine(affine);
    const original = [5.5, -2.25, 8.125] as const;
    const roundTrip = transformPoint(inverse, transformPoint(affine, original));

    expect(affineReciprocalCondition(affine)).toBeGreaterThan(0.1);
    roundTrip.forEach((value, axis) =>
      expect(value).toBeCloseTo(original[axis]!, 12),
    );
  });

  it("rejects a numerically invertible but ill-conditioned affine", () => {
    const affine = createMatrix4([
      1, 1, 0, 0,
      0, 1e-14, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    expect(affineReciprocalCondition(affine)).toBeLessThan(1e-12);
    expect(() => invertAffine(affine)).toThrow(/ill-conditioned/);
  });

  it("converts a world ray to index space", () => {
    const indexToWorld = createMatrix4([
      2, 0, 0, 10,
      0, 4, 0, -8,
      0, 0, 5, 2,
      0, 0, 0, 1,
    ]);

    const ray = indexRayFromWorld(
      invertAffine(indexToWorld),
      [12, 0, 12],
      [2, 4, 5],
    );
    ray.origin.forEach((value, axis) =>
      expect(value).toBeCloseTo([1, 2, 2][axis]!, 12),
    );
    ray.direction.forEach((value) => expect(value).toBeCloseTo(1, 12));
  });

  it("rejects non-affine and singular matrices", () => {
    expect(() =>
      createMatrix4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 1, 1,
      ]),
    ).toThrow(/final row/);
    expect(() =>
      invertAffine(
        createMatrix4([
          1, 0, 0, 0,
          1, 0, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ]),
      ),
    ).toThrow(/singular/);
  });
});
