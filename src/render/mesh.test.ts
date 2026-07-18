import { describe, expect, it } from "vitest";
import {
  chunksAffectedByIndices,
  isCurrentMeshResult,
  meshChunk,
  shouldApplyMeshChanges,
} from "./mesh";

describe("chunk surface meshing", () => {
  it("emits six exposed faces for one voxel", () => {
    const result = meshChunk(new Uint8Array([1]), [1, 1, 1], { x: 0, y: 0, z: 0 });
    expect(result.quadCount).toBe(6);
    expect(result.unitFaceCount).toBe(6);
    expect(result.indices).toHaveLength(36);
    expect(result.positions).toHaveLength(72);
    expect(result.normals).toHaveLength(72);
    for (let face = 0; face < result.quadCount; face += 1) {
      const vertexOffset = face * 12;
      const ax = result.positions[vertexOffset] ?? 0;
      const ay = result.positions[vertexOffset + 1] ?? 0;
      const az = result.positions[vertexOffset + 2] ?? 0;
      const bx = result.positions[vertexOffset + 3] ?? 0;
      const by = result.positions[vertexOffset + 4] ?? 0;
      const bz = result.positions[vertexOffset + 5] ?? 0;
      const cx = result.positions[vertexOffset + 6] ?? 0;
      const cy = result.positions[vertexOffset + 7] ?? 0;
      const cz = result.positions[vertexOffset + 8] ?? 0;
      const edgeAb = [bx - ax, by - ay, bz - az] as const;
      const edgeAc = [cx - ax, cy - ay, cz - az] as const;
      const cross = [
        edgeAb[1] * edgeAc[2] - edgeAb[2] * edgeAc[1],
        edgeAb[2] * edgeAc[0] - edgeAb[0] * edgeAc[2],
        edgeAb[0] * edgeAc[1] - edgeAb[1] * edgeAc[0],
      ] as const;
      const normal = [
        result.normals[vertexOffset] ?? 0,
        result.normals[vertexOffset + 1] ?? 0,
        result.normals[vertexOffset + 2] ?? 0,
      ] as const;
      expect(cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2]).toBeGreaterThan(0);
      for (let corner = 1; corner < 4; corner += 1) {
        expect(Array.from(result.normals.slice(vertexOffset + corner * 3, vertexOffset + corner * 3 + 3))).toEqual(normal);
      }
    }
  });

  it("removes the shared face across a chunk boundary", () => {
    const occupancy = new Uint8Array(33);
    occupancy[31] = 1;
    occupancy[32] = 1;
    expect(meshChunk(occupancy, [33, 1, 1], { x: 0, y: 0, z: 0 }).unitFaceCount).toBe(5);
    expect(meshChunk(occupancy, [33, 1, 1], { x: 1, y: 0, z: 0 }).unitFaceCount).toBe(5);
  });

  it.each([
    { dimensions: [1, 33, 1] as const, first: 31, second: 32, chunk: { x: 0, y: 1, z: 0 } },
    { dimensions: [1, 1, 33] as const, first: 31, second: 32, chunk: { x: 0, y: 0, z: 1 } },
  ])("uses halo occupancy across Y and Z chunk boundaries", ({ dimensions, first, second, chunk }) => {
    const occupancy = new Uint8Array(33);
    occupancy[first] = 1;
    occupancy[second] = 1;
    expect(meshChunk(occupancy, dimensions, { x: 0, y: 0, z: 0 }).unitFaceCount).toBe(5);
    expect(meshChunk(occupancy, dimensions, chunk).unitFaceCount).toBe(5);
  });

  it("emits exact half-index face bounds and handles checkerboard occupancy", () => {
    const single = meshChunk(Uint8Array.of(1), [1, 1, 1], { x: 0, y: 0, z: 0 });
    expect(Math.min(...single.positions)).toBe(-0.5);
    expect(Math.max(...single.positions)).toBe(0.5);
    const checker = Uint8Array.from([1, 0, 0, 1, 0, 1, 1, 0]);
    const checkerMesh = meshChunk(checker, [2, 2, 2], { x: 0, y: 0, z: 0 });
    expect(checkerMesh.quadCount).toBe(24);
    expect(checkerMesh.unitFaceCount).toBe(24);
  });

  it("greedily merges a solid cuboid while preserving its exact surface area", () => {
    const result = meshChunk(new Uint8Array(4 * 3 * 2).fill(1), [4, 3, 2], { x: 0, y: 0, z: 0 });
    expect(result.quadCount).toBe(6);
    expect(result.unitFaceCount).toBe(52);
    expect(result.indices).toHaveLength(36);
  });

  it("invalidates both chunks for a boundary edit", () => {
    expect(chunksAffectedByIndices([32], [64, 1, 1]).map((chunk) => chunk.x).sort()).toEqual([0, 1]);
    expect(chunksAffectedByIndices([32], [1, 64, 1]).map((chunk) => chunk.y).sort()).toEqual([0, 1]);
    expect(chunksAffectedByIndices([32], [1, 1, 64]).map((chunk) => chunk.z).sort()).toEqual([0, 1]);
  });

  it("rejects stale revisions and results from cancelled generations", () => {
    const current = { revision: 9, generation: 4 };
    expect(isCurrentMeshResult({ revision: 9, generation: 4 }, current)).toBe(true);
    expect(isCurrentMeshResult({ revision: 8, generation: 4 }, current)).toBe(false);
    expect(isCurrentMeshResult({ revision: 9, generation: 3 }, current)).toBe(false);
  });

  it("accepts ordered undo updates while rejecting cancelled generations", () => {
    const current = { revision: 9, generation: 4 };
    expect(shouldApplyMeshChanges({ revision: 4, generation: 4 }, current)).toBe(true);
    expect(shouldApplyMeshChanges({ revision: 10, generation: 3 }, current)).toBe(false);
  });

});
