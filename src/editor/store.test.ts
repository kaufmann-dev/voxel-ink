import { beforeEach, describe, expect, it } from "vitest";
import { IDENTITY_AFFINE } from "../core/affine";
import {
  MAX_PENDING_MESH_BATCHES,
  combineMeshChangeIndices,
  useEditorStore,
} from "./store";

const dimensions = [MAX_PENDING_MESH_BATCHES + 2, 1, 1] as const;

beforeEach(() => {
  useEditorStore.getState().clearDocument();
  useEditorStore.getState().installDocument({
    dimensions,
    occupancy: new Uint8Array(dimensions[0]),
    occupiedCount: 0,
    geometry: {
      indexToWorld: IDENTITY_AFFINE,
      worldToIndex: IDENTITY_AFFINE,
      spacing: [1, 1, 1],
      units: [null, null, null],
    },
    sourceFormat: "binvox",
    sourceVariant: ".binvox",
    saveState: {
      kind: "binvox",
      variant: ".binvox",
      version: 1,
      translation: [0, 0, 0],
      scale: 1,
      dimensions,
    },
    normalization: { convertedNonBinary: false },
  }, {
    sourceId: "00000000-0000-4000-8000-000000000003",
    name: "mesh-journal.binvox",
    extension: ".binvox",
    size: dimensions[0],
  });
  const state = useEditorStore.getState();
  state.acknowledgeMeshChanges(state.meshChangeSequence);
});

describe("mesh change journal", () => {
  it("retains every synchronous stroke extension until the renderer consumes it", () => {
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke([2]);
    store.extendStroke([7]);
    store.extendStroke([11]);

    const state = useEditorStore.getState();
    expect(state.pendingMeshChanges).toHaveLength(3);
    const indices = combineMeshChangeIndices(state.pendingMeshChanges);
    expect(Array.from(indices)).toEqual([2, 7, 11]);

    const workerCache = new Uint8Array(dimensions[0]);
    for (const index of indices) workerCache[index] = state.document!.occupancy[index]!;
    expect(workerCache).toEqual(state.document!.occupancy);

    store.cancelStroke();
  });

  it("collapses an unconsumed burst to a bounded full-snapshot marker", () => {
    const store = useEditorStore.getState();
    store.beginStroke(1);
    for (let index = 0; index <= MAX_PENDING_MESH_BATCHES; index += 1) {
      store.extendStroke([index]);
    }

    const state = useEditorStore.getState();
    expect(state.pendingMeshChanges).toHaveLength(1);
    expect(state.pendingMeshChanges[0]).toMatchObject({ full: true });
    expect(state.pendingMeshChanges[0]!.indices).toHaveLength(0);

    store.cancelStroke();
  });

  it("acknowledges only batches at or before the consumed sequence", () => {
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke([3]);
    const firstSequence = useEditorStore.getState().pendingMeshChanges[0]!.sequence;
    store.extendStroke([4]);

    store.acknowledgeMeshChanges(firstSequence);

    const pending = useEditorStore.getState().pendingMeshChanges;
    expect(pending).toHaveLength(1);
    expect(Array.from(pending[0]!.indices)).toEqual([4]);
    store.cancelStroke();
  });
});
