import { beforeEach, describe, expect, it } from "vitest";
import { IDENTITY_AFFINE } from "../core/affine";
import { subscribeVoxelChanges, useEditorStore } from "./store";

const dimensions = [70_000, 1, 1] as const;

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
});

describe("voxel change journal", () => {
  it("keeps provisional stroke edits out of the mesh journal", () => {
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke([2]);
    store.extendStroke([7]);
    store.extendStroke([11]);
    expect(useEditorStore.getState().pendingMeshChanges).toHaveLength(0);

    store.finishStroke();
    const pending = useEditorStore.getState().pendingMeshChanges;
    expect(pending).toHaveLength(1);
    expect(pending[0]).toMatchObject({ value: 1 });
    expect(Array.from(pending[0]!.indices)).toEqual([2, 7, 11]);
  });

  it("retains a large stroke as one delta without a snapshot fallback", () => {
    const indices = Uint32Array.from({ length: 65_537 }, (_, index) => index);
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke(indices);
    store.finishStroke();

    const pending = useEditorStore.getState().pendingMeshChanges;
    expect(pending).toHaveLength(1);
    expect(pending[0]!.indices).toHaveLength(indices.length);
    expect(pending[0]!.value).toBe(1);
  });

  it("cancels provisional edits without publishing mesh work", () => {
    const events: number[][] = [];
    const unsubscribe = subscribeVoxelChanges((event) => events.push(Array.from(event.indices)));
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke([3, 4]);
    store.cancelStroke();
    unsubscribe();

    expect(useEditorStore.getState().pendingMeshChanges).toHaveLength(0);
    expect(useEditorStore.getState().document!.occupancy.slice(3, 5)).toEqual(Uint8Array.of(0, 0));
    expect(events).toEqual([[3, 4], [3, 4]]);
  });

  it("acknowledges only completed batches at or before the consumed sequence", () => {
    const store = useEditorStore.getState();
    store.beginStroke(1);
    store.extendStroke([3]);
    store.finishStroke();
    const firstSequence = useEditorStore.getState().pendingMeshChanges[0]!.sequence;
    store.beginStroke(1);
    store.extendStroke([4]);
    store.finishStroke();

    store.acknowledgeMeshChanges(firstSequence);

    const pending = useEditorStore.getState().pendingMeshChanges;
    expect(pending).toHaveLength(1);
    expect(Array.from(pending[0]!.indices)).toEqual([4]);
  });
});
