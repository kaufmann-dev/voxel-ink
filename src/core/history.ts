import {
  type BinaryVoxel,
  type VolumeDocument,
} from "./volume";

export const DEFAULT_MAX_HISTORY_TRANSACTIONS = 256;
export const DEFAULT_MAX_HISTORY_BYTES = 128 * 1024 * 1024;

export interface HistoryOptions {
  maxTransactions?: number;
  maxBytes?: number;
}

export interface StrokeTransaction {
  readonly indices: Uint32Array;
  readonly value: BinaryVoxel;
  readonly beforeRevision: number;
  readonly afterRevision: number;
  readonly byteSize: number;
}

export interface StrokeCommitResult {
  readonly changedCount: number;
  readonly stored: boolean;
  readonly transaction: StrokeTransaction | null;
}

function validateLimit(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer`);
  }
  return value;
}

export class VoxelStroke<TSaveState = unknown> {
  private readonly changedIndices: number[] = [];
  private expectedRevision: number;
  private closed = false;

  constructor(
    private readonly owner: VoxelHistory<TSaveState>,
    readonly value: BinaryVoxel,
    readonly beforeRevision: number,
  ) {
    this.expectedRevision = beforeRevision;
  }

  get changedCount(): number {
    return this.changedIndices.length;
  }

  apply(indices: Iterable<number>): Uint32Array {
    this.assertOpen();
    this.assertUninterrupted();
    const result = this.owner.document.writeIndices(indices, this.value);
    this.expectedRevision = result.afterRevision;
    for (const index of result.indices) {
      this.changedIndices.push(index);
    }
    return result.indices;
  }

  commit(): StrokeCommitResult {
    this.assertOpen();
    this.assertUninterrupted();
    this.closed = true;
    return this.owner.finishStroke(this, this.changedIndices, this.expectedRevision);
  }

  cancel(): void {
    this.assertOpen();
    this.assertUninterrupted();
    this.closed = true;
    if (this.changedIndices.length > 0) {
      this.owner.document.applyHistoryState(
        Uint32Array.from(this.changedIndices),
        this.value === 1 ? 0 : 1,
        this.beforeRevision,
      );
    }
    this.owner.cancelStroke(this);
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new Error("Stroke is already closed");
    }
  }

  private assertUninterrupted(): void {
    if (this.owner.document.revision !== this.expectedRevision) {
      throw new Error("Document changed outside the active stroke");
    }
  }
}

export class VoxelHistory<TSaveState = unknown> {
  readonly document: VolumeDocument<TSaveState>;
  readonly maxTransactions: number;
  readonly maxBytes: number;

  private readonly undoStack: StrokeTransaction[] = [];
  private readonly redoStack: StrokeTransaction[] = [];
  private undoBytes = 0;
  private activeStroke: VoxelStroke<TSaveState> | null = null;

  constructor(
    document: VolumeDocument<TSaveState>,
    options: HistoryOptions = {},
  ) {
    this.document = document;
    this.maxTransactions = validateLimit(
      options.maxTransactions ?? DEFAULT_MAX_HISTORY_TRANSACTIONS,
      "Maximum transaction count",
    );
    this.maxBytes = validateLimit(
      options.maxBytes ?? DEFAULT_MAX_HISTORY_BYTES,
      "Maximum history bytes",
    );
  }

  get undoCount(): number {
    return this.undoStack.length;
  }

  get redoCount(): number {
    return this.redoStack.length;
  }

  get retainedBytes(): number {
    return this.undoBytes;
  }

  beginStroke(value: BinaryVoxel): VoxelStroke<TSaveState> {
    if (this.activeStroke) {
      throw new Error("A stroke is already active");
    }
    if (value !== 0 && value !== 1) {
      throw new Error("Stroke value must be 0 or 1");
    }
    const stroke = new VoxelStroke(this, value, this.document.revision);
    this.activeStroke = stroke;
    return stroke;
  }

  commitStroke(
    indices: Iterable<number>,
    value: BinaryVoxel,
  ): StrokeCommitResult {
    const stroke = this.beginStroke(value);
    try {
      stroke.apply(indices);
      return stroke.commit();
    } catch (error) {
      if (this.activeStroke === stroke) {
        stroke.cancel();
      }
      throw error;
    }
  }

  undo(): StrokeTransaction | null {
    this.assertNoActiveStroke();
    const transaction = this.undoStack.at(-1);
    if (!transaction) {
      return null;
    }
    if (this.document.revision !== transaction.afterRevision) {
      throw new Error("Document revision does not match the undo history");
    }

    this.undoStack.pop();
    this.undoBytes -= transaction.byteSize;
    this.document.applyHistoryState(
      transaction.indices,
      transaction.value === 1 ? 0 : 1,
      transaction.beforeRevision,
    );
    this.redoStack.push(transaction);
    return transaction;
  }

  redo(): StrokeTransaction | null {
    this.assertNoActiveStroke();
    const transaction = this.redoStack.at(-1);
    if (!transaction) {
      return null;
    }
    if (this.document.revision !== transaction.beforeRevision) {
      throw new Error("Document revision does not match the redo history");
    }

    this.redoStack.pop();
    this.document.applyHistoryState(
      transaction.indices,
      transaction.value,
      transaction.afterRevision,
    );
    this.undoStack.push(transaction);
    this.undoBytes += transaction.byteSize;
    return transaction;
  }

  clear(): void {
    this.assertNoActiveStroke();
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.undoBytes = 0;
  }

  finishStroke(
    stroke: VoxelStroke<TSaveState>,
    changedIndices: readonly number[],
    afterRevision: number,
  ): StrokeCommitResult {
    this.assertActiveStroke(stroke);
    this.activeStroke = null;

    if (changedIndices.length === 0) {
      return { changedCount: 0, stored: false, transaction: null };
    }

    const indices = Uint32Array.from(changedIndices);
    const transaction: StrokeTransaction = Object.freeze({
      indices,
      value: stroke.value,
      beforeRevision: stroke.beforeRevision,
      afterRevision,
      byteSize: indices.byteLength,
    });

    this.redoStack.length = 0;
    if (transaction.byteSize > this.maxBytes) {
      this.undoStack.length = 0;
      this.undoBytes = 0;
      return {
        changedCount: indices.length,
        stored: false,
        transaction,
      };
    }

    this.undoStack.push(transaction);
    this.undoBytes += transaction.byteSize;
    this.trimUndoStack();
    return {
      changedCount: indices.length,
      stored: true,
      transaction,
    };
  }

  cancelStroke(stroke: VoxelStroke<TSaveState>): void {
    this.assertActiveStroke(stroke);
    this.activeStroke = null;
  }

  private trimUndoStack(): void {
    while (
      this.undoStack.length > this.maxTransactions ||
      this.undoBytes > this.maxBytes
    ) {
      const removed = this.undoStack.shift();
      if (removed) {
        this.undoBytes -= removed.byteSize;
      }
    }
  }

  private assertNoActiveStroke(): void {
    if (this.activeStroke) {
      throw new Error("Finish or cancel the active stroke first");
    }
  }

  private assertActiveStroke(stroke: VoxelStroke<TSaveState>): void {
    if (this.activeStroke !== stroke) {
      throw new Error("Stroke is not active in this history");
    }
  }
}
