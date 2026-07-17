import { describe, expect, it } from "vitest";

import { createMatrix4 } from "../core/affine";
import { geometryFromAffine, isValidAffine } from "./affine";

describe("codec affine validation", () => {
  it("accepts tiny but well-conditioned units", () => {
    const affine = createMatrix4([
      1e-12, 0, 0, 0,
      0, 2e-12, 0, 0,
      0, 0, 4e-12, 0,
      0, 0, 0, 1,
    ]);

    expect(isValidAffine(affine)).toBe(true);
    expect(geometryFromAffine(affine, ["m", "m", "m"]).spacing).toEqual([
      1e-12,
      2e-12,
      4e-12,
    ]);
  });

  it("rejects scale-relative ill-conditioning", () => {
    const affine = createMatrix4([
      1, 1, 0, 0,
      0, 1e-14, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    expect(isValidAffine(affine)).toBe(false);
    expect(() => geometryFromAffine(affine, [null, null, null])).toThrowError(
      expect.objectContaining({ code: "invalid-affine" }),
    );
  });
});
