import { describe, expect, it } from "vitest";

import {
  checkedVoxelCount,
  MAX_AXIS_LENGTH,
  MAX_RENDER_CHUNKS,
  MAX_SLICE_VOXELS,
  MAX_VOXELS,
} from "./binary";
import { CodecError, type VolumeDimensions } from "./types";

const expectSizeLimit = (
  dimensions: VolumeDimensions,
  message: RegExp,
): void => {
  try {
    checkedVoxelCount(dimensions);
    throw new Error("Expected dimensions to exceed a size limit");
  } catch (error) {
    expect(error).toBeInstanceOf(CodecError);
    expect((error as CodecError).code).toBe("size-limit");
    expect((error as Error).message).toMatch(message);
  }
};

describe("checkedVoxelCount", () => {
  it("accepts the reference volume and each renderability boundary", () => {
    expect(checkedVoxelCount([512, 512, 256])).toBe(512 * 512 * 256);
    expect(checkedVoxelCount([MAX_AXIS_LENGTH, 1, 1])).toBe(
      MAX_AXIS_LENGTH,
    );
    expect(checkedVoxelCount([4_096, 4_096, 1])).toBe(MAX_SLICE_VOXELS);
    expect(checkedVoxelCount([8_192, 2_048, 1])).toBe(8_192 * 2_048);
  });

  it("rejects invalid dimensions separately from renderability limits", () => {
    expect(() => checkedVoxelCount([1, 0, 1])).toThrowError(
      expect.objectContaining({ code: "invalid-dimensions" }),
    );
  });

  it("rejects an overlong axis before any UI-sized allocation", () => {
    expectSizeLimit([MAX_AXIS_LENGTH + 1, 1, 1], /axis x.*8,192/i);
  });

  it("rejects an oversized slice even when total voxels are within bounds", () => {
    expectSizeLimit([4_097, 4_096, 1], /XY slices.*16,777,216/i);
  });

  it("rejects excessive chunk fan-out caused by partial edge chunks", () => {
    expectSizeLimit(
      [8_161, 2_049, 1],
      new RegExp(`render chunks.*${MAX_RENDER_CHUNKS.toLocaleString()}`, "i"),
    );
  });

  it("retains the aggregate voxel cap", () => {
    expectSizeLimit(
      [512, 512, 513],
      new RegExp(MAX_VOXELS.toLocaleString()),
    );
  });
});
