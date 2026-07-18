import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { describe, expect, it } from "vitest";
import { decodeVolume } from "../codecs";
import { brushStrokeIndices } from "../core/brush";
import { VoxelHistory } from "../core/history";
import { VolumeDocument, type GridCoordinate } from "../core/volume";
import { chunksAffectedByIndices, chunksForDimensions, meshChunk } from "../render/mesh";
import { createSliceRaster, updateSliceRaster } from "../render/slice";

const fixturePath = process.env.VOXELINK_VOLUME_BENCH;
const benchmark = fixturePath ? it : it.skip;

describe("large local volume paint benchmark", () => {
  benchmark("keeps painting independent from full-volume copies and global remeshing", async () => {
    const decodeStarted = performance.now();
    const decoded = await decodeVolume(new Uint8Array(readFileSync(fixturePath!)), fixturePath!);
    const decodeMs = performance.now() - decodeStarted;
    const document = new VolumeDocument({
      dimensions: decoded.dimensions,
      occupancy: decoded.occupancy,
      occupiedCount: decoded.occupiedCount,
      indexToWorld: decoded.geometry.indexToWorld,
      sourceFormat: decoded.sourceFormat,
      saveState: decoded.saveState,
    });
    const history = new VoxelHistory(document);
    const center = document.cursor;
    const rasters = [0, 1, 2].map((axis) =>
      createSliceRaster(document, axis as 0 | 1 | 2, center[axis]!),
    );
    const start: GridCoordinate = [Math.max(0, center[0] - 48), center[1], center[2]];
    const end: GridCoordinate = [Math.min(document.dimensions[0] - 1, center[0] + 48), center[1], center[2]];
    const value = document.get(center) === 1 ? 0 : 1;

    const feedbackStarted = performance.now();
    const candidates = brushStrokeIndices(document.dimensions, start, end, 31, 2);
    const stroke = history.beginStroke(value);
    const changed = stroke.apply(candidates);
    for (const raster of rasters) updateSliceRaster(raster, document, changed, value);
    const commit = stroke.commit();
    const paintFeedbackMs = performance.now() - feedbackStarted;

    const affectedChunks = chunksAffectedByIndices(changed, document.dimensions);
    const updateMeshStarted = performance.now();
    let updatedQuads = 0;
    for (const chunk of affectedChunks) {
      updatedQuads += meshChunk(document.occupancy, document.dimensions, chunk).quadCount;
    }
    const affectedMeshMs = performance.now() - updateMeshStarted;

    let fullMeshMs: number | null = null;
    let geometryMiB: number | null = null;
    let fullQuads: number | null = null;
    if (process.env.VOXELINK_FULL_MESH_BENCH === "1") {
      const fullMeshStarted = performance.now();
      let geometryBytes = 0;
      fullQuads = 0;
      for (const chunk of chunksForDimensions(document.dimensions)) {
        const payload = meshChunk(document.occupancy, document.dimensions, chunk);
        fullQuads += payload.quadCount;
        geometryBytes += payload.positions.byteLength
          + payload.normals.byteLength
          + payload.indices.byteLength;
      }
      fullMeshMs = performance.now() - fullMeshStarted;
      geometryMiB = geometryBytes / (1024 * 1024);
    }

    console.info("VoxelInk volume paint benchmark", {
      dimensions: document.dimensions,
      voxelCount: document.occupancy.length,
      occupiedCount: document.occupiedCount,
      decodeMs,
      candidateCount: candidates.length,
      changedCount: changed.length,
      paintFeedbackMs,
      meshBatchBytes: commit.transaction?.indices.byteLength ?? 0,
      fullSnapshotBytesAvoided: document.occupancy.byteLength,
      affectedChunkCount: affectedChunks.length,
      affectedMeshMs,
      updatedQuads,
      fullMeshMs,
      fullQuads,
      geometryMiB,
    });

    expect(commit.transaction?.indices).toEqual(changed);
    expect(paintFeedbackMs).toBeLessThan(100);
    expect(affectedChunks.length).toBeLessThan(chunksForDimensions(document.dimensions).length);
  }, 120_000);
});
