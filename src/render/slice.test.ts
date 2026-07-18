import { describe, expect, it } from "vitest";
import { VolumeDocument } from "../core/volume";
import { createSliceRaster, updateSliceRaster } from "./slice";

describe("slice raster cache", () => {
  it("updates only edited pixels on the cached plane", () => {
    const document = new VolumeDocument({ dimensions: [3, 2, 2] });
    const raster = createSliceRaster(document, 2, 1);
    const dirty = updateSliceRaster(raster, document, [9, 11, 0], 1);

    expect(dirty).toEqual({ x: 0, y: 0, width: 3, height: 1 });
    expect(Array.from(raster.pixels.slice(0, 12))).toEqual([
      78, 211, 177, 255,
      22, 33, 39, 255,
      78, 211, 177, 255,
    ]);
    expect(Array.from(raster.pixels.slice(12, 16))).toEqual([22, 33, 39, 255]);
  });

  it("ignores edits outside the cached slice", () => {
    const document = new VolumeDocument({ dimensions: [2, 2, 2] });
    const raster = createSliceRaster(document, 0, 1);
    expect(updateSliceRaster(raster, document, [0, 2, 4, 6], 1)).toBeNull();
  });
});
