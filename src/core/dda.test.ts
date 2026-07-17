import { describe, expect, it } from "vitest";

import {
  adjacentCoordinate,
  firstOccupiedVoxel,
  intersectActiveSlice,
} from "./dda";
import { linearIndex } from "./volume";

describe("index-space DDA picking", () => {
  it("hits the first occupied voxel and reports its entered face", () => {
    const dimensions = [4, 3, 2] as const;
    const occupancy = new Uint8Array(24);
    occupancy[linearIndex(dimensions, [1, 1, 0])] = 1;
    occupancy[linearIndex(dimensions, [2, 1, 0])] = 1;

    const hit = firstOccupiedVoxel(occupancy, dimensions, {
      origin: [-2, 1, 0],
      direction: [1, 0, 0],
    });
    expect(hit).toMatchObject({
      coordinate: [1, 1, 0],
      normal: [-1, 0, 0],
      distance: 2.5,
    });
    expect(adjacentCoordinate(hit!, dimensions)).toEqual([0, 1, 0]);
  });

  it("traverses correctly in the negative direction", () => {
    const dimensions = [4, 1, 1] as const;
    const occupancy = Uint8Array.from([0, 1, 1, 0]);
    const hit = firstOccupiedVoxel(occupancy, dimensions, {
      origin: [5, 0, 0],
      direction: [-1, 0, 0],
    });

    expect(hit).toMatchObject({ coordinate: [2, 0, 0], normal: [1, 0, 0] });
    expect(adjacentCoordinate(hit!, dimensions)).toEqual([3, 0, 0]);
  });

  it("steps tied axes together on an exact diagonal", () => {
    const dimensions = [3, 3, 1] as const;
    const occupancy = new Uint8Array(9);
    occupancy[linearIndex(dimensions, [1, 1, 0])] = 1;
    const hit = firstOccupiedVoxel(occupancy, dimensions, {
      origin: [-1, -1, 0],
      direction: [1, 1, 0],
    });

    expect(hit).toMatchObject({ coordinate: [1, 1, 0], normal: [-1, 0, 0] });
  });

  it("returns zero entry normal when the ray starts inside a filled voxel", () => {
    const hit = firstOccupiedVoxel(Uint8Array.from([1]), [1, 1, 1], {
      origin: [0, 0, 0],
      direction: [1, 0, 0],
    });
    expect(hit).toMatchObject({
      coordinate: [0, 0, 0],
      normal: [0, 0, 0],
      distance: 0,
    });
    expect(adjacentCoordinate(hit!, [1, 1, 1])).toBeNull();
  });

  it("returns null when the ray misses the volume or all occupied cells", () => {
    expect(
      firstOccupiedVoxel(new Uint8Array(8), [2, 2, 2], {
        origin: [-1, 5, 0],
        direction: [1, 0, 0],
      }),
    ).toBeNull();
    expect(
      firstOccupiedVoxel(new Uint8Array(8), [2, 2, 2], {
        origin: [-1, 0, 0],
        direction: [1, 0, 0],
      }),
    ).toBeNull();
  });

  it("intersects the active slice plane to seed an empty volume", () => {
    expect(
      intersectActiveSlice(
        { origin: [1.2, 1.1, 10], direction: [0, 0, -2] },
        [4, 4, 4],
        2,
        2,
      ),
    ).toEqual([1, 1, 2]);
    expect(
      intersectActiveSlice(
        { origin: [10, 1, 10], direction: [0, 0, -1] },
        [4, 4, 4],
        2,
        2,
      ),
    ).toBeNull();
  });
});
