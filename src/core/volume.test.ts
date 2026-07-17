import { describe, expect, it } from "vitest";

import { createMatrix4 } from "./affine";
import {
  coordinateFromIndex,
  linearIndex,
  VolumeDocument,
  voxelCount,
} from "./volume";

describe("volume document", () => {
  it("uses x + nx * (y + ny * z) indexing and reverses it", () => {
    const dimensions = [4, 3, 2] as const;
    expect(voxelCount(dimensions)).toBe(24);
    expect(linearIndex(dimensions, [0, 0, 0])).toBe(0);
    expect(linearIndex(dimensions, [3, 2, 1])).toBe(23);
    expect(linearIndex(dimensions, [2, 1, 1])).toBe(18);
    expect(coordinateFromIndex(dimensions, 18)).toEqual([2, 1, 1]);
    expect(() => linearIndex(dimensions, [4, 0, 0])).toThrow(RangeError);
    expect(() => coordinateFromIndex(dimensions, 24)).toThrow(RangeError);
  });

  it("owns binary occupancy and maintains its occupied count", () => {
    const occupancy = Uint8Array.from([0, 1, 0, 1, 0, 0, 0, 0]);
    const document = new VolumeDocument({
      dimensions: [2, 2, 2],
      occupancy,
    });
    expect(document.occupiedCount).toBe(2);
    const result = document.writeIndices([0, 0, 1, 7], 1);
    expect([...result.indices]).toEqual([0, 7]);
    expect(document.occupiedCount).toBe(4);
    expect(document.revision).toBe(1);
    expect(document.get([1, 1, 1])).toBe(1);
    expect(() =>
      new VolumeDocument({
        dimensions: [2, 1, 1],
        occupancy: Uint8Array.from([0, 2]),
      }),
    ).toThrow(/binary/);
  });

  it("derives inverse, spacing, units, and a bounded cursor", () => {
    const affine = createMatrix4([
      0, -3, 0, 10,
      2, 0, 0, 20,
      0, 0, -4, 30,
      0, 0, 0, 1,
    ]);
    const document = new VolumeDocument({
      dimensions: [5, 7, 9],
      indexToWorld: affine,
      units: "mm",
      cursor: [1, 2, 3],
    });

    expect(document.spacing).toEqual([2, 3, 4]);
    expect(document.units).toBe("mm");
    expect(document.cursor).toEqual([1, 2, 3]);
    expect(() => document.setCursor([5, 0, 0])).toThrow(RangeError);
  });

  it("marks normalized imports dirty and tracks an older saved snapshot", () => {
    const document = new VolumeDocument({
      dimensions: [2, 2, 1],
      normalizationState: "normalized-non-binary",
    });
    expect(document.revision).toBe(1);
    expect(document.savedRevision).toBe(0);
    expect(document.dirty).toBe(true);

    document.markSaved();
    expect(document.dirty).toBe(false);
    const savedRevision = document.revision;
    document.writeIndices([0], 1);
    document.markSaved(savedRevision);
    expect(document.dirty).toBe(true);
  });

  it("extracts consistently oriented slices along every axis", () => {
    const document = new VolumeDocument({ dimensions: [3, 2, 2] });
    document.writeIndices([0, 2, 4, 7, 9, 11], 1);

    expect(document.extractSlice(2, 0)).toMatchObject({
      width: 3,
      height: 2,
      data: Uint8Array.from([1, 0, 1, 0, 1, 0]),
    });
    expect(document.extractSlice(1, 1)).toMatchObject({
      width: 3,
      height: 2,
      data: Uint8Array.from([0, 1, 0, 1, 0, 1]),
    });
    expect(document.extractSlice(0, 1)).toMatchObject({
      width: 2,
      height: 2,
      data: Uint8Array.from([0, 1, 1, 0]),
    });
  });

  it("validates writes before mutating anything", () => {
    const document = new VolumeDocument({ dimensions: [2, 2, 1] });
    expect(() => document.writeIndices([0, 4], 1)).toThrow(RangeError);
    expect(document.occupiedCount).toBe(0);
    expect(document.revision).toBe(0);
  });
});
