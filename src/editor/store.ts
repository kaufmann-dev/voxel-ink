import { create } from "zustand";
import type { CodecSaveState, DecodedVolume, SourceVariant } from "../codecs/types";
import { VoxelHistory, type VoxelStroke } from "../core/history";
import {
  VolumeDocument,
  type BinaryVoxel,
  type GridCoordinate,
  type SliceAxis,
} from "../core/volume";

export type EditorMode = "inspect" | "brush" | "eraser";

export interface SourceHandle {
  readonly sourceId: string;
  readonly name: string;
  readonly extension: SourceVariant;
  readonly size: number;
  readonly hash?: string;
}

export interface VoxelChangeBatch {
  readonly sequence: number;
  readonly revision: number;
  readonly indices: Uint32Array;
  readonly value: BinaryVoxel;
}

export interface VoxelChangeEvent {
  readonly document: VolumeDocument<CodecSaveState>;
  readonly revision: number;
  readonly indices: Uint32Array;
  readonly value: BinaryVoxel;
}

type VoxelChangeListener = (event: VoxelChangeEvent) => void;
const voxelChangeListeners = new Set<VoxelChangeListener>();

export function subscribeVoxelChanges(listener: VoxelChangeListener): () => void {
  voxelChangeListeners.add(listener);
  return () => voxelChangeListeners.delete(listener);
}

function emitVoxelChanges(
  document: VolumeDocument<CodecSaveState>,
  indices: Uint32Array,
  value: BinaryVoxel,
): void {
  if (indices.length === 0) return;
  const event = { document, revision: document.revision, indices, value };
  for (const listener of voxelChangeListeners) listener(event);
}

export interface PreparedEditorDocument {
  readonly document: VolumeDocument<CodecSaveState>;
  readonly history: VoxelHistory<CodecSaveState>;
  readonly normalizationPending: boolean;
  readonly normalizationNotice: string | null;
}

interface EditorState {
  document: VolumeDocument<CodecSaveState> | null;
  history: VoxelHistory<CodecSaveState> | null;
  source: SourceHandle | null;
  revision: number;
  savedRevision: number;
  meshChangeSequence: number;
  pendingMeshChanges: readonly VoxelChangeBatch[];
  mode: EditorMode;
  brushDiameter: number;
  activeAxis: SliceAxis;
  busyMessage: string | null;
  errorMessage: string | null;
  noticeMessage: string | null;
  normalizationPending: boolean;
  strokeActive: boolean;
  installDocument: (decoded: DecodedVolume, source: SourceHandle) => void;
  installPreparedDocument: (prepared: PreparedEditorDocument, source: SourceHandle) => void;
  clearDocument: () => void;
  setMode: (mode: EditorMode) => void;
  setBrushDiameter: (diameter: number) => void;
  setActiveAxis: (axis: SliceAxis) => void;
  setCursor: (cursor: GridCoordinate) => void;
  beginStroke: (value: BinaryVoxel) => void;
  extendStroke: (indices: Iterable<number>) => Uint32Array;
  finishStroke: () => void;
  cancelStroke: () => void;
  acknowledgeMeshChanges: (throughSequence: number) => void;
  undo: () => void;
  redo: () => void;
  markSaved: (revision: number, source?: SourceHandle) => void;
  resolveNormalization: () => void;
  setBusy: (message: string | null) => void;
  setError: (message: string | null) => void;
  setNotice: (message: string | null) => void;
}

let activeStroke: VoxelStroke<CodecSaveState> | null = null;

function asUnits(decoded: DecodedVolume): string {
  const units = decoded.geometry.units.filter((unit): unit is string => unit !== null);
  return units.length === 0 ? "unknown" : [...new Set(units)].join("/");
}

function publishChanges(
  state: Pick<EditorState, "meshChangeSequence" | "pendingMeshChanges">,
  revision: number,
  indices: Uint32Array,
  value: BinaryVoxel,
): Pick<EditorState, "meshChangeSequence" | "pendingMeshChanges"> {
  const sequence = state.meshChangeSequence + 1;
  return {
    meshChangeSequence: sequence,
    pendingMeshChanges: [...state.pendingMeshChanges, { sequence, revision, indices, value }],
  };
}

export function prepareEditorDocument(decoded: DecodedVolume): PreparedEditorDocument {
  const document = new VolumeDocument<CodecSaveState>({
    dimensions: decoded.dimensions,
    occupancy: decoded.occupancy,
    occupiedCount: decoded.occupiedCount,
    indexToWorld: Array.from(decoded.geometry.indexToWorld),
    units: asUnits(decoded),
    sourceFormat: decoded.sourceFormat,
    saveState: decoded.saveState,
    normalizationState: decoded.normalization.convertedNonBinary
      ? "normalized-non-binary"
      : "binary",
  });
  return {
    document,
    history: new VoxelHistory(document),
    normalizationPending: decoded.normalization.convertedNonBinary,
    normalizationNotice: decoded.normalization.convertedNonBinary
      ? "Non-binary source values were normalized to empty and filled voxels."
      : null,
  };
}

export const useEditorStore = create<EditorState>()((set, get) => ({
  document: null,
  history: null,
  source: null,
  revision: 0,
  savedRevision: 0,
  meshChangeSequence: 0,
  pendingMeshChanges: [],
  mode: "inspect",
  brushDiameter: 1,
  activeAxis: 2,
  busyMessage: null,
  errorMessage: null,
  noticeMessage: null,
  normalizationPending: false,
  strokeActive: false,

  installDocument: (decoded, source) => {
    get().installPreparedDocument(prepareEditorDocument(decoded), source);
  },

  installPreparedDocument: (prepared, source) => {
    if (activeStroke) activeStroke.cancel();
    activeStroke = null;
    const { document, history } = prepared;
    set({
      document,
      history,
      source,
      revision: document.revision,
      savedRevision: document.savedRevision,
      pendingMeshChanges: [],
      activeAxis: 2,
      strokeActive: false,
      normalizationPending: prepared.normalizationPending,
      errorMessage: null,
      noticeMessage: prepared.normalizationNotice,
    });
  },

  clearDocument: () => {
    if (activeStroke) activeStroke.cancel();
    activeStroke = null;
    set({
      document: null,
      history: null,
      source: null,
      revision: 0,
      savedRevision: 0,
      pendingMeshChanges: [],
      strokeActive: false,
      normalizationPending: false,
      noticeMessage: null,
      errorMessage: null,
    });
  },

  setMode: (mode) => {
    if (get().busyMessage !== null || activeStroke) return;
    set({ mode });
  },
  setBrushDiameter: (diameter) => {
    if (!Number.isInteger(diameter) || diameter < 1 || diameter > 31 || diameter % 2 === 0) {
      throw new RangeError("Brush diameter must be an odd integer from 1 through 31");
    }
    if (get().busyMessage !== null || activeStroke) return;
    set({ brushDiameter: diameter });
  },
  setActiveAxis: (activeAxis) => {
    if (get().busyMessage !== null) return;
    set({ activeAxis });
  },
  setCursor: (cursor) => {
    if (get().busyMessage !== null) return;
    get().document?.setCursor(cursor);
    set((state) => ({ revision: state.revision }));
  },

  beginStroke: (value) => {
    const state = get();
    if (state.busyMessage !== null) return;
    const history = state.history;
    if (!history) return;
    if (activeStroke) throw new Error("A stroke is already active");
    activeStroke = history.beginStroke(value);
    set({ strokeActive: true });
  },
  extendStroke: (indices) => {
    if (!activeStroke) return new Uint32Array();
    const value = activeStroke.value;
    const changed = activeStroke.apply(indices);
    if (changed.length > 0) {
      const document = get().document;
      if (document) {
        set({ revision: document.revision });
        emitVoxelChanges(document, changed, value);
      }
    }
    return changed;
  },
  finishStroke: () => {
    if (!activeStroke) return;
    const result = activeStroke.commit();
    activeStroke = null;
    const document = get().document;
    if (document) {
      set((state) => ({
        revision: document.revision,
        savedRevision: document.savedRevision,
        strokeActive: false,
        ...(result.transaction
          ? publishChanges(
              state,
              result.transaction.afterRevision,
              result.transaction.indices,
              result.transaction.value,
            )
          : {}),
      }));
    }
  },
  cancelStroke: () => {
    if (!activeStroke) return;
    const value = activeStroke.value === 1 ? 0 : 1;
    const changed = activeStroke.cancel();
    activeStroke = null;
    const document = get().document;
    if (document) {
      set({
        revision: document.revision,
        savedRevision: document.savedRevision,
        strokeActive: false,
      });
      emitVoxelChanges(document, changed, value);
    }
  },

  acknowledgeMeshChanges: (throughSequence) => {
    set((state) => {
      const pendingMeshChanges = state.pendingMeshChanges.filter(
        (change) => change.sequence > throughSequence,
      );
      return pendingMeshChanges.length === state.pendingMeshChanges.length
        ? state
        : { pendingMeshChanges };
    });
  },

  undo: () => {
    const { busyMessage, history, document, strokeActive } = get();
    if (busyMessage !== null || strokeActive) return;
    const transaction = history?.undo();
    if (!document || !transaction) return;
    const value = transaction.value === 1 ? 0 : 1;
    set((state) => ({
      revision: document.revision,
      savedRevision: document.savedRevision,
      ...publishChanges(state, document.revision, transaction.indices, value),
    }));
    emitVoxelChanges(document, transaction.indices, value);
  },
  redo: () => {
    const { busyMessage, history, document, strokeActive } = get();
    if (busyMessage !== null || strokeActive) return;
    const transaction = history?.redo();
    if (!document || !transaction) return;
    set((state) => ({
      revision: document.revision,
      savedRevision: document.savedRevision,
      ...publishChanges(state, document.revision, transaction.indices, transaction.value),
    }));
    emitVoxelChanges(document, transaction.indices, transaction.value);
  },

  markSaved: (revision, source) => {
    const document = get().document;
    if (!document) return;
    document.markSaved(revision);
    set({ savedRevision: document.savedRevision, ...(source ? { source } : {}) });
  },
  resolveNormalization: () => set({ normalizationPending: false }),
  setBusy: (busyMessage) => set({ busyMessage }),
  setError: (errorMessage) => set({ errorMessage }),
  setNotice: (noticeMessage) => set({ noticeMessage }),
}));

export const currentTargetVariant = (): SourceVariant | null =>
  useEditorStore.getState().source?.extension ?? null;
