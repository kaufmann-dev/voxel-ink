import { describe, expect, it } from "vitest";

import { VoxelHistory } from "./history";
import { VolumeDocument } from "./volume";

describe("voxel history", () => {
  it("makes a multi-dab pointer stroke one undo transaction", () => {
    const document = new VolumeDocument({ dimensions: [4, 1, 1] });
    const history = new VoxelHistory(document);
    const stroke = history.beginStroke(1);

    expect([...stroke.apply([0, 1])]).toEqual([0, 1]);
    expect(document.occupiedCount).toBe(2);
    expect([...stroke.apply([1, 2])]).toEqual([2]);
    const result = stroke.commit();

    expect(result.changedCount).toBe(3);
    expect(history.undoCount).toBe(1);
    expect([...document.occupancy]).toEqual([1, 1, 1, 0]);
    history.undo();
    expect([...document.occupancy]).toEqual([0, 0, 0, 0]);
    expect(document.occupiedCount).toBe(0);
    history.redo();
    expect([...document.occupancy]).toEqual([1, 1, 1, 0]);
  });

  it("restores saved-revision cleanliness on undo without reusing branched IDs", () => {
    const document = new VolumeDocument({ dimensions: [3, 1, 1] });
    const history = new VoxelHistory(document);
    history.commitStroke([0], 1);
    document.markSaved();
    const savedRevision = document.savedRevision;

    history.commitStroke([1], 1);
    const abandonedRevision = document.revision;
    expect(document.dirty).toBe(true);
    history.undo();
    expect(document.revision).toBe(savedRevision);
    expect(document.dirty).toBe(false);

    history.commitStroke([2], 1);
    expect(document.revision).toBeGreaterThan(abandonedRevision);
    expect(history.redoCount).toBe(0);
  });

  it("cancels a live stroke without changing history or saved state", () => {
    const document = new VolumeDocument({ dimensions: [3, 1, 1] });
    const history = new VoxelHistory(document);
    const stroke = history.beginStroke(1);
    stroke.apply([0, 2]);
    stroke.cancel();

    expect([...document.occupancy]).toEqual([0, 0, 0]);
    expect(document.revision).toBe(0);
    expect(document.dirty).toBe(false);
    expect(history.undoCount).toBe(0);
  });

  it("enforces transaction-count and byte budgets by dropping oldest entries", () => {
    const countDocument = new VolumeDocument({ dimensions: [4, 1, 1] });
    const countHistory = new VoxelHistory(countDocument, {
      maxTransactions: 2,
    });
    countHistory.commitStroke([0], 1);
    countHistory.commitStroke([1], 1);
    countHistory.commitStroke([2], 1);
    expect(countHistory.undoCount).toBe(2);
    countHistory.undo();
    countHistory.undo();
    expect(countDocument.getIndex(0)).toBe(1);

    const byteDocument = new VolumeDocument({ dimensions: [4, 1, 1] });
    const byteHistory = new VoxelHistory(byteDocument, { maxBytes: 8 });
    byteHistory.commitStroke([0], 1);
    byteHistory.commitStroke([1, 2], 1);
    expect(byteHistory.retainedBytes).toBe(8);
    expect(byteHistory.undoCount).toBe(1);
  });

  it("applies an oversized stroke but does not retain an unbounded delta", () => {
    const document = new VolumeDocument({ dimensions: [4, 1, 1] });
    const history = new VoxelHistory(document, { maxBytes: 4 });
    const result = history.commitStroke([0, 1], 1);

    expect(result).toMatchObject({ changedCount: 2, stored: false });
    expect([...document.occupancy]).toEqual([1, 1, 0, 0]);
    expect(history.undoCount).toBe(0);
  });
});
