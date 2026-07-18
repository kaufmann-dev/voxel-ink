import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { MessageChannel, receiveMessageOnPort } from "node:worker_threads";
import { describe, expect, it } from "vitest";
import { sphereBrushIndices } from "../core/brush";
import { VoxelHistory } from "../core/history";
import {
  VolumeDocument,
  linearIndex,
  type Dimensions,
  type GridCoordinate,
} from "../core/volume";
import { CHUNK_SIZE, meshChunk } from "../render/mesh";

const REFERENCE_DIMENSIONS = [512, 512, 256] as const;
const CI_DIMENSIONS = [96, 96, 64] as const;
const MAX_EDIT_FEEDBACK_MS = 100;

interface GeneratedMask {
  readonly occupancy: Uint8Array;
  readonly occupiedCount: number;
}

interface EditorBenchmarkResult {
  readonly dimensions: Dimensions;
  readonly voxelCount: number;
  readonly occupiedCountBeforeEdit: number;
  readonly changedVoxelCount: number;
  readonly trustedInstallMs: number;
  readonly editAndThreeSlicesMs: number;
  readonly undoRedoMs: number;
  readonly snapshotCopyMs: number;
  readonly snapshotTransferMs: number;
  readonly boundedChunkMeshMs: number;
  readonly boundedChunkQuads: number;
  readonly boundedChunkUnitFaces: number;
  readonly gpuNavigation: string;
}

function generateRepresentativeMask(dimensions: Dimensions): GeneratedMask {
  const [nx, ny, nz] = dimensions;
  const occupancy = new Uint8Array(nx * ny * nz);
  const center = [nx * 0.33, ny * 0.5, nz * 0.5] as const;
  const radii = [nx * 0.2, ny * 0.31, nz * 0.34] as const;
  let occupiedCount = 0;

  for (let z = 0; z < nz; z += 1) {
    const normalizedZ = (z - center[2]) / radii[2];
    const zSquared = normalizedZ * normalizedZ;
    if (zSquared > 1) continue;

    for (let y = 0; y < ny; y += 1) {
      const normalizedY = (y - center[1]) / radii[1];
      const remaining = 1 - zSquared - normalizedY * normalizedY;
      if (remaining < 0) continue;

      const halfWidth = radii[0] * Math.sqrt(remaining);
      const startX = Math.max(0, Math.ceil(center[0] - halfWidth));
      const endX = Math.min(nx - 1, Math.floor(center[0] + halfWidth));
      if (startX > endX) continue;

      const rowStart = nx * (y + ny * z);
      occupancy.fill(1, rowStart + startX, rowStart + endX + 1);
      occupiedCount += endX - startX + 1;
    }
  }

  return { occupancy, occupiedCount };
}

function brushCenter(dimensions: Dimensions): GridCoordinate {
  return [
    Math.floor(dimensions[0] * 0.78),
    Math.floor(dimensions[1] / 2),
    Math.floor(dimensions[2] / 2),
  ];
}

function runEditorBenchmark(dimensions: Dimensions): EditorBenchmarkResult {
  const generated = generateRepresentativeMask(dimensions);

  const installStarted = performance.now();
  const document = new VolumeDocument({
    dimensions,
    occupancy: generated.occupancy,
    occupiedCount: generated.occupiedCount,
    sourceFormat: "generated-reference-mask",
  });
  const history = new VoxelHistory(document);
  const trustedInstallMs = performance.now() - installStarted;

  assert.equal(document.occupancy, generated.occupancy);
  expect(document.occupiedCount).toBe(generated.occupiedCount);

  const center = brushCenter(dimensions);
  const editStarted = performance.now();
  const brushIndices = sphereBrushIndices(dimensions, center, 31);
  const commit = history.commitStroke(brushIndices, 1);
  const slices = [
    document.extractSlice(0, center[0]),
    document.extractSlice(1, center[1]),
    document.extractSlice(2, center[2]),
  ];
  const editAndThreeSlicesMs = performance.now() - editStarted;

  expect(commit.stored).toBe(true);
  expect(commit.changedCount).toBe(brushIndices.length);
  expect(slices.map((slice) => slice.data.length)).toEqual([
    dimensions[1] * dimensions[2],
    dimensions[0] * dimensions[2],
    dimensions[0] * dimensions[1],
  ]);

  const centerIndex = linearIndex(dimensions, center);
  const countAfterEdit = document.occupiedCount;
  const historyStarted = performance.now();
  const undone = history.undo();
  expect(undone?.indices.length).toBe(brushIndices.length);
  expect(document.getIndex(centerIndex)).toBe(0);
  const redone = history.redo();
  const undoRedoMs = performance.now() - historyStarted;
  assert.equal(redone, undone);
  expect(document.getIndex(centerIndex)).toBe(1);
  expect(document.occupiedCount).toBe(countAfterEdit);

  const copyStarted = performance.now();
  const snapshot = document.copyOccupancy();
  const snapshotCopyMs = performance.now() - copyStarted;
  assert.notEqual(snapshot, document.occupancy);
  expect(snapshot.byteLength).toBe(document.occupancy.byteLength);
  expect(snapshot[centerIndex]).toBe(document.occupancy[centerIndex]);
  expect(snapshot[0]).toBe(document.occupancy[0]);
  expect(snapshot.at(-1)).toBe(document.occupancy.at(-1));

  const snapshotBuffer = snapshot.buffer as ArrayBuffer;
  const transferChannel = new MessageChannel();
  const transferStarted = performance.now();
  transferChannel.port1.postMessage(snapshot, [snapshotBuffer]);
  const transferMessage = receiveMessageOnPort(transferChannel.port2);
  const snapshotTransferMs = performance.now() - transferStarted;
  transferChannel.port1.close();
  transferChannel.port2.close();
  const transferredSnapshot = transferMessage?.message as Uint8Array | undefined;
  expect(snapshot.byteLength).toBe(0);
  expect(transferredSnapshot?.byteLength).toBe(document.occupancy.byteLength);
  expect(transferredSnapshot?.[centerIndex]).toBe(1);

  const chunk = {
    x: Math.floor(center[0] / CHUNK_SIZE),
    y: Math.floor(center[1] / CHUNK_SIZE),
    z: Math.floor(center[2] / CHUNK_SIZE),
  };
  const meshStarted = performance.now();
  const mesh = meshChunk(document.occupancy, dimensions, chunk);
  const boundedChunkMeshMs = performance.now() - meshStarted;
  expect(mesh.quadCount).toBeGreaterThan(0);
  expect(mesh.positions.length).toBe(mesh.quadCount * 12);
  expect(mesh.normals.length).toBe(mesh.positions.length);
  expect(mesh.indices.length).toBe(mesh.quadCount * 6);

  return {
    dimensions,
    voxelCount: document.occupancy.length,
    occupiedCountBeforeEdit: generated.occupiedCount,
    changedVoxelCount: commit.changedCount,
    trustedInstallMs,
    editAndThreeSlicesMs,
    undoRedoMs,
    snapshotCopyMs,
    snapshotTransferMs,
    boundedChunkMeshMs,
    boundedChunkQuads: mesh.quadCount,
    boundedChunkUnitFaces: mesh.unitFaceCount,
    gpuNavigation:
      "Not measured by this CPU benchmark; 30 fps navigation is verified separately by the native/browser smoke check.",
  };
}

describe("real editor path benchmark", () => {
  it("keeps editing synchronous while snapshot and bounded mesh work remain separable", () => {
    const referenceRun = process.env.VOXELINK_REFERENCE_BENCH === "1";
    const dimensions = referenceRun ? REFERENCE_DIMENSIONS : CI_DIMENSIONS;
    const result = runEditorBenchmark(dimensions);

    console.info("VoxelInk real editor CPU benchmark", result);

    expect(result.changedVoxelCount).toBeGreaterThan(0);
    expect(result.voxelCount).toBe(
      dimensions[0] * dimensions[1] * dimensions[2],
    );
    if (referenceRun) {
      expect(result.dimensions).toEqual(REFERENCE_DIMENSIONS);
      expect(result.trustedInstallMs).toBeLessThan(MAX_EDIT_FEEDBACK_MS);
      expect(result.editAndThreeSlicesMs).toBeLessThan(MAX_EDIT_FEEDBACK_MS);
    }
  }, 120_000);
});
