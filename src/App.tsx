import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import appIcon from "../src-tauri/icons/icon.png";
import "./App.css";
import { CodecWorkerClient } from "./codecs/client";
import type { SourceVariant } from "./codecs/types";
import { Modal, type ModalAction } from "./components/Modal";
import { SliceView } from "./components/SliceView";
import { VolumeView3D } from "./components/VolumeView3D";
import {
  prepareEditorDocument,
  useEditorStore,
  type EditorMode,
  type SourceHandle,
} from "./editor/store";
import {
  cancelNativeExit,
  chooseAndReadVolume,
  chooseSaveDestination,
  confirmNativeExit,
  errorMessage,
  onNativeCloseRequested,
  releaseSource,
  saveTicketVariant,
  writeSavedVolume,
  type SaveMode,
} from "./native/files";

interface PromptOption extends ModalAction {
  readonly id: string;
}

interface PromptState {
  readonly title: string;
  readonly message: string;
  readonly options: readonly PromptOption[];
  readonly resolve: (choice: string) => void;
}

type SaveOutcome = "saved" | "copied" | "stale" | "cancelled" | "failed";

function makeSource(
  previous: SourceHandle,
  result: { sourceId: string; displayName: string; size: number },
  extension: SourceVariant,
): SourceHandle {
  return {
    ...previous,
    sourceId: result.sourceId,
    name: result.displayName,
    size: result.size,
    extension,
  };
}

function App() {
  const document = useEditorStore((state) => state.document);
  const source = useEditorStore((state) => state.source);
  const revision = useEditorStore((state) => state.revision);
  const savedRevision = useEditorStore((state) => state.savedRevision);
  const cursor = useEditorStore((state) => state.document?.cursor ?? null);
  const undoCount = useEditorStore((state) => state.history?.undoCount ?? 0);
  const redoCount = useEditorStore((state) => state.history?.redoCount ?? 0);
  const mode = useEditorStore((state) => state.mode);
  const brushDiameter = useEditorStore((state) => state.brushDiameter);
  const busyMessage = useEditorStore((state) => state.busyMessage);
  const errorMessageState = useEditorStore((state) => state.errorMessage);
  const noticeMessage = useEditorStore((state) => state.noticeMessage);
  const normalizationPending = useEditorStore((state) => state.normalizationPending);
  const strokeActive = useEditorStore((state) => state.strokeActive);
  const setMode = useEditorStore((state) => state.setMode);
  const setBrushDiameter = useEditorStore((state) => state.setBrushDiameter);
  const installPreparedDocument = useEditorStore((state) => state.installPreparedDocument);
  const clearDocument = useEditorStore((state) => state.clearDocument);
  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);
  const markSaved = useEditorStore((state) => state.markSaved);
  const resolveNormalization = useEditorStore((state) => state.resolveNormalization);
  const setBusy = useEditorStore((state) => state.setBusy);
  const setError = useEditorStore((state) => state.setError);
  const setNotice = useEditorStore((state) => state.setNotice);
  const [promptState, setPromptState] = useState<PromptState | null>(null);
  const promptRef = useRef<PromptState | null>(null);
  const codecClient = useMemo(() => new CodecWorkerClient(), []);
  const dirty = document !== null && revision !== savedRevision;
  const interactionLocked = busyMessage !== null || strokeActive;

  useEffect(() => () => {
    codecClient.dispose();
    promptRef.current?.resolve("cancel");
    promptRef.current = null;
  }, [codecClient]);

  const prompt = useCallback((title: string, message: string, options: readonly PromptOption[]): Promise<string> =>
    new Promise((resolve) => {
      promptRef.current?.resolve("cancel");
      const next = { title, message, options, resolve };
      promptRef.current = next;
      setPromptState(next);
    }), []);

  const choosePrompt = useCallback((choice: string) => {
    const current = promptRef.current;
    promptRef.current = null;
    setPromptState(null);
    current?.resolve(choice);
  }, []);

  const performSave = useCallback(async (saveMode: SaveMode): Promise<SaveOutcome> => {
    const initial = useEditorStore.getState();
    if (!initial.document || !initial.source || initial.busyMessage !== null || initial.strokeActive) {
      return "failed";
    }
    try {
      setError(null);
      setBusy(saveMode === "overwriteSource" ? "Checking destination…" : "Choosing destination…");
      const ticket = await chooseSaveDestination(initial.source, saveMode);
      if (!ticket) return "cancelled";
      const snapshot = useEditorStore.getState();
      if (
        snapshot.document !== initial.document ||
        snapshot.source?.sourceId !== initial.source.sourceId ||
        snapshot.strokeActive
      ) {
        setNotice("The document changed before encoding began; nothing was written.");
        return "stale";
      }
      const snapshotRevision = snapshot.document.revision;
      const targetVariant = saveTicketVariant(ticket);
      setBusy("Encoding binary volume…");
      const bytes = await codecClient.encode(snapshot.document, targetVariant);
      setBusy("Writing file atomically…");
      const result = await writeSavedVolume(ticket, bytes);
      const current = useEditorStore.getState();
      const sameDocument =
        current.document === snapshot.document &&
        current.source?.sourceId === snapshot.source.sourceId;
      const snapshotIsCurrent =
        sameDocument && current.document?.revision === snapshotRevision;
      if (result.adopted) {
        if (!sameDocument) {
          await releaseSource(result.sourceId);
          setNotice(`Saved ${result.displayName}, but it no longer belongs to the open document.`);
          return "stale";
        }
        const adoptedSource = makeSource(snapshot.source, result, targetVariant);
        markSaved(snapshotRevision, adoptedSource);
        resolveNormalization();
        if (snapshotIsCurrent) {
          setNotice(`Saved ${result.displayName}.`);
          return "saved";
        }
        setNotice(`Saved ${result.displayName}, but newer edits remain unsaved.`);
        return "stale";
      }
      if (!snapshotIsCurrent) {
        setNotice(`Saved a copy as ${result.displayName}, but newer edits remain unsaved.`);
        return "stale";
      }
      setNotice(`Saved a copy as ${result.displayName}; the current document remains unchanged.`);
      return "copied";
    } catch (error) {
      setError(errorMessage(error));
      return "failed";
    } finally {
      setBusy(null);
    }
  }, [codecClient, markSaved, resolveNormalization, setBusy, setError, setNotice]);

  const saveDocument = useCallback(async (saveMode: SaveMode = "overwriteSource"): Promise<SaveOutcome> => {
    const state = useEditorStore.getState();
    if (!state.document || !state.source || state.busyMessage !== null || state.strokeActive) {
      return "failed";
    }
    if (saveMode === "overwriteSource" && state.normalizationPending) {
      const choice = await prompt(
        "Overwrite normalized import?",
        "This source contained values other than 0 and 1. Overwriting will permanently save the normalized binary occupancy.",
        [
          { id: "copy", label: "Save a Copy", onSelect: () => undefined },
          { id: "overwrite", label: "Overwrite", tone: "danger", onSelect: () => undefined },
          { id: "cancel", label: "Cancel", autoFocus: true, onSelect: () => undefined },
        ],
      );
      if (choice === "cancel") return "cancelled";
      return performSave(choice === "copy" ? "saveCopy" : "overwriteSource");
    }
    return performSave(saveMode);
  }, [performSave, prompt]);

  const guardDirty = useCallback(async (
    actionName: string,
    action: () => void | Promise<void>,
    cancel: () => void | Promise<void> = () => undefined,
  ) => {
    const state = useEditorStore.getState();
    if (!state.document?.dirty) {
      await action();
      return;
    }
    const choice = await prompt(
      `Save before ${actionName}?`,
      "The current volume has unsaved voxel changes.",
      [
        { id: "save", label: "Save", tone: "primary", onSelect: () => undefined },
        { id: "discard", label: "Discard", tone: "danger", onSelect: () => undefined },
        { id: "cancel", label: "Cancel", autoFocus: true, onSelect: () => undefined },
      ],
    );
    if (choice === "discard") {
      await action();
      return;
    }
    if (choice === "save") {
      const outcome = await saveDocument();
      if (outcome === "saved" || outcome === "copied") {
        await action();
        return;
      }
    }
    await cancel();
  }, [prompt, saveDocument]);

  const openVolume = useCallback(() => {
    void guardDirty("opening another volume", async () => {
      let opened: Awaited<ReturnType<typeof chooseAndReadVolume>> = null;
      let adopted = false;
      try {
        setError(null);
        setBusy("Choosing volume…");
        opened = await chooseAndReadVolume();
        if (!opened) return;
        setBusy(`Decoding ${opened.source.name}…`);
        const decoded = await codecClient.decode(opened.bytes, opened.source.extension);
        const prepared = prepareEditorDocument(decoded);
        const previousSource = useEditorStore.getState().source;
        if (previousSource && previousSource.sourceId !== opened.source.sourceId) {
          await releaseSource(previousSource.sourceId);
        }
        installPreparedDocument(prepared, opened.source);
        adopted = true;
      } catch (error) {
        if (opened && !adopted) {
          try {
            await releaseSource(opened.source.sourceId);
          } catch {
            // The original error is the actionable one; the native broker also drops all authority on exit.
          }
        }
        setError(errorMessage(error));
      } finally {
        setBusy(null);
      }
    });
  }, [codecClient, guardDirty, installPreparedDocument, setBusy, setError]);

  const closeVolume = useCallback(() => {
    void guardDirty("closing the volume", async () => {
      const currentSource = useEditorStore.getState().source;
      try {
        setBusy("Closing volume…");
        if (currentSource) await releaseSource(currentSource.sourceId);
        clearDocument();
      } catch (error) {
        setError(errorMessage(error));
      } finally {
        setBusy(null);
      }
    });
  }, [clearDocument, guardDirty, setBusy, setError]);

  const save = useCallback(() => { void saveDocument(); }, [saveDocument]);
  const saveAs = useCallback(() => { void saveDocument("saveAsAdopt"); }, [saveDocument]);

  const handleNativeExit = useCallback(() => {
    const state = useEditorStore.getState();
    if (state.busyMessage !== null) {
      setNotice("Finish the current file operation before exiting VoxelInk.");
      void cancelNativeExit();
      return;
    }
    if (state.strokeActive) {
      setNotice("Finish the active brush stroke before exiting VoxelInk.");
      void cancelNativeExit();
      return;
    }
    void guardDirty(
      "exiting VoxelInk",
      () => confirmNativeExit(),
      () => cancelNativeExit(),
    );
  }, [guardDirty, setNotice]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void onNativeCloseRequested(handleNativeExit).then((stop) => { unlisten = stop; });
    return () => unlisten?.();
  }, [handleNativeExit]);

  useEffect(() => {
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editingControl = target?.matches("input, textarea, select, [contenteditable='true']") ?? false;
      const modifier = event.ctrlKey || event.metaKey;
      const commandKey = event.key.toLowerCase();
      if (promptState || busyMessage !== null || strokeActive) {
        if (modifier && ["o", "s", "w", "z", "y"].includes(commandKey)) {
          event.preventDefault();
        }
        return;
      }
      if (modifier && commandKey === "o") {
        event.preventDefault();
        openVolume();
      } else if (modifier && commandKey === "s") {
        event.preventDefault();
        if (event.shiftKey) saveAs(); else save();
      } else if (modifier && commandKey === "w") {
        event.preventDefault();
        closeVolume();
      } else if (modifier && commandKey === "z") {
        event.preventDefault();
        if (event.shiftKey) redo(); else undo();
      } else if (modifier && commandKey === "y") {
        event.preventDefault();
        redo();
      } else if (!modifier && !editingControl && ["i", "b", "e"].includes(commandKey)) {
        const shortcuts: Record<string, EditorMode> = { i: "inspect", b: "brush", e: "eraser" };
        const selected = shortcuts[commandKey];
        if (selected) setMode(selected);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [busyMessage, closeVolume, openVolume, promptState, redo, save, saveAs, setMode, strokeActive, undo]);

  const cursorText = cursor ? cursor.map((value) => value + 1).join(", ") : "—";
  const dimensionsText = document ? document.dimensions.join(" × ") : "No volume";
  const occupiedText = document ? document.occupiedCount.toLocaleString() : "0";

  return (
    <main className="voxel-editor" data-testid="voxel-editor" aria-busy={busyMessage !== null}>
      <h1 className="sr-only">VoxelInk binary voxel editor</h1>
      <header className="app-toolbar" aria-label="Editor toolbar">
        <div className="app-identity" aria-hidden="true">
          <img className="app-mark" src={appIcon} alt="" />
          <strong>VoxelInk</strong>
        </div>
        <div className="toolbar-group" aria-label="File actions">
          <button type="button" className="button-primary" onClick={openVolume} disabled={interactionLocked} aria-keyshortcuts="Control+O Meta+O">Open volume</button>
          <button type="button" onClick={save} disabled={!document || !dirty || interactionLocked} aria-keyshortcuts="Control+S Meta+S">Save</button>
          <button type="button" onClick={saveAs} disabled={!document || interactionLocked} aria-keyshortcuts="Control+Shift+S Meta+Shift+S">Save As</button>
          <button type="button" onClick={closeVolume} disabled={!document || interactionLocked} aria-keyshortcuts="Control+W Meta+W">Close volume</button>
        </div>
        <div className="toolbar-group" aria-label="Edit history">
          <button type="button" className="icon-button" aria-label="Undo" title="Undo" onClick={undo} disabled={undoCount === 0 || interactionLocked}>↶</button>
          <button type="button" className="icon-button" aria-label="Redo" title="Redo" onClick={redo} disabled={redoCount === 0 || interactionLocked}>↷</button>
        </div>
        <div className="toolbar-group" role="group" aria-label="Editing mode">
          {(["inspect", "brush", "eraser"] as const).map((tool) => (
            <button
              key={tool}
              type="button"
              className="tool-button"
              aria-pressed={mode === tool}
              disabled={interactionLocked}
              aria-keyshortcuts={tool === "inspect" ? "I" : tool === "brush" ? "B" : "E"}
              onClick={() => setMode(tool)}
            >
              {tool[0]?.toUpperCase()}{tool.slice(1)}
            </button>
          ))}
          <label className="brush-control">
            <span>Diameter</span>
            <input
              type="range"
              min="1"
              max="31"
              step="2"
              value={brushDiameter}
              disabled={mode === "inspect" || interactionLocked}
              onChange={(event) => setBrushDiameter(event.currentTarget.valueAsNumber)}
            />
            <span className="brush-value">{brushDiameter}</span>
          </label>
        </div>
        <div className="toolbar-spacer" />
        <span className="document-name" title={source?.name}>{source?.name ?? "Local files only"}</span>
      </header>

      <div className="workspace">
        <SliceView axis={2} label="Axial" shortLabel="Z" />
        <SliceView axis={1} label="Coronal" shortLabel="Y" />
        <SliceView axis={0} label="Sagittal" shortLabel="X" />
        <VolumeView3D />
      </div>

      <footer className="status-bar" aria-label="Document status">
        <span className="status-primary">{dimensionsText}</span>
        <span>Cursor {cursorText}</span>
        <span>{occupiedText} filled</span>
        <span>{document?.units ?? "unknown units"}</span>
        <span className="status-spacer" />
        {normalizationPending && <span className="dirty-indicator">Normalized import</span>}
        <span className={dirty ? "dirty-indicator" : undefined}>{dirty ? "Unsaved" : document ? "Saved" : "Ready"}</span>
      </footer>

      {noticeMessage && (
        <div className="notice-banner" role="status">
          {noticeMessage}
          <button type="button" className="banner-close" aria-label="Dismiss notice" onClick={() => setNotice(null)}>×</button>
        </div>
      )}
      {errorMessageState && (
        <div className="error-banner" role="alert">
          {errorMessageState}
          <button type="button" className="banner-close" aria-label="Dismiss error" onClick={() => setError(null)}>×</button>
        </div>
      )}
      {busyMessage && (
        <div className="busy-overlay" role="status" aria-live="polite">
          <div className="busy-card"><span className="spinner" aria-hidden="true" />{busyMessage}</div>
        </div>
      )}
      {promptState && (
        <Modal
          title={promptState.title}
          onCancel={() => choosePrompt("cancel")}
          actions={promptState.options.map((option) => ({
            ...option,
            onSelect: () => choosePrompt(option.id),
          }))}
        >
          <p>{promptState.message}</p>
        </Modal>
      )}
    </main>
  );
}

export default App;
