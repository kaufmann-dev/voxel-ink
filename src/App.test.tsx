import axe from "axe-core";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { IDENTITY_AFFINE } from "./core/affine";
import { useEditorStore } from "./editor/store";
import {
  cancelNativeExit,
  chooseAndReadVolume,
  chooseSaveDestination,
  confirmNativeExit,
  releaseSource,
  writeSavedVolume,
} from "./native/files";

const codecMocks = vi.hoisted(() => ({
  decode: vi.fn(),
  dispose: vi.fn(),
  encode: vi.fn(),
}));

const nativeMocks = vi.hoisted(() => ({
  closeHandler: null as (() => void) | null,
}));

vi.mock("./codecs/client", () => ({
  CodecWorkerClient: class {
    readonly decode = codecMocks.decode;
    readonly dispose = codecMocks.dispose;
    readonly encode = codecMocks.encode;
  },
}));

vi.mock("./components/SliceView", () => ({
  SliceView: ({ label }: { label: string }) => <div role="application" aria-label={`${label} slice`} />,
}));

vi.mock("./components/VolumeView3D", () => ({
  VolumeView3D: () => <div role="img" aria-label="3D volume view" />,
}));

vi.mock("./native/files", () => ({
  cancelNativeExit: vi.fn(() => Promise.resolve(undefined)),
  chooseAndReadVolume: vi.fn(() => Promise.resolve(null)),
  chooseSaveDestination: vi.fn(() => Promise.resolve(null)),
  confirmNativeExit: vi.fn(() => Promise.resolve(undefined)),
  errorMessage: (error: unknown) => error instanceof Error ? error.message : String(error),
  onNativeCloseRequested: vi.fn((handler: () => void) => {
    nativeMocks.closeHandler = handler;
    return Promise.resolve(() => {
      if (nativeMocks.closeHandler === handler) nativeMocks.closeHandler = null;
    });
  }),
  releaseSource: vi.fn(() => Promise.resolve(undefined)),
  saveTicketVariant: vi.fn(() => ".binvox"),
  writeSavedVolume: vi.fn(),
}));

function installFixture(convertedNonBinary = false): void {
  useEditorStore.getState().installDocument({
    dimensions: [2, 2, 2],
    occupancy: new Uint8Array(8),
    occupiedCount: 0,
    geometry: {
      indexToWorld: IDENTITY_AFFINE,
      worldToIndex: IDENTITY_AFFINE,
      spacing: [1, 1, 1],
      units: ["mm", "mm", "mm"],
    },
    sourceFormat: "binvox",
    sourceVariant: ".binvox",
    saveState: {
      kind: "binvox",
      variant: ".binvox",
      version: 1,
      translation: [0, 0, 0],
      scale: 1,
      dimensions: [2, 2, 2],
    },
    normalization: { convertedNonBinary },
  }, {
    sourceId: "00000000-0000-4000-8000-000000000001",
    name: "fixture.binvox",
    extension: ".binvox",
    size: 32,
  });
}

function makeFixtureDirty(): void {
  const store = useEditorStore.getState();
  store.beginStroke(1);
  store.extendStroke([0]);
  store.finishStroke();
}

function relativeLuminance(hex: string): number {
  const channels = hex.match(/[\da-f]{2}/gi)?.map((value) => Number.parseInt(value, 16) / 255) ?? [];
  const [red = 0, green = 0, blue = 0] = channels.map((value) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background));
  const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

beforeEach(() => {
  codecMocks.decode.mockReset();
  codecMocks.dispose.mockReset();
  codecMocks.encode.mockReset();
  codecMocks.encode.mockResolvedValue(Uint8Array.of(1));
  nativeMocks.closeHandler = null;
  vi.mocked(cancelNativeExit).mockClear();
  vi.mocked(chooseAndReadVolume).mockResolvedValue(null);
  vi.mocked(chooseSaveDestination).mockResolvedValue(null);
  vi.mocked(confirmNativeExit).mockClear();
  vi.mocked(writeSavedVolume).mockReset();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  useEditorStore.getState().clearDocument();
  useEditorStore.getState().setMode("inspect");
  useEditorStore.getState().setError(null);
  useEditorStore.getState().setNotice(null);
});

afterEach(() => vi.clearAllMocks());

describe("VoxelInk editor shell", () => {
  it("has accessible names, focus order, and no automatic axe violations", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    expect(screen.getByRole("heading", { name: /VoxelInk binary voxel editor/i })).toBeInTheDocument();
    expect(screen.getByRole("application", { name: "Axial slice" })).toBeInTheDocument();
    await user.tab();
    expect(screen.getByRole("button", { name: "Open volume" })).toHaveFocus();
    const results = await axe.run(container);
    expect(results.violations).toEqual([]);
  });

  it("supports keyboard mode shortcuts", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.keyboard("b");
    expect(screen.getByRole("button", { name: "Brush" })).toHaveAttribute("aria-pressed", "true");
    await user.keyboard("e");
    expect(screen.getByRole("button", { name: "Eraser" })).toHaveAttribute("aria-pressed", "true");
  });

  it("enables undo as soon as a rendered stroke is committed", () => {
    installFixture();
    render(<App />);
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    act(() => {
      const store = useEditorStore.getState();
      store.beginStroke(1);
      store.extendStroke([0]);
    });
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    act(() => {
      const store = useEditorStore.getState();
      store.finishStroke();
    });
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
  });

  it("guards a dirty close with Save, Discard, and Cancel choices", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Close volume" }));
    const dialog = await screen.findByRole("dialog", { name: "Save before closing the volume?" });
    expect(dialog).toHaveTextContent("unsaved voxel changes");
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Discard" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toHaveFocus();
    await user.click(within(dialog).getByRole("button", { name: "Discard" }));
    expect(useEditorStore.getState().document).toBeNull();
  });

  it("guards opening another volume and leaves the current document intact on cancel", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    const original = useEditorStore.getState().document;
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Open volume" }));
    const dialog = await screen.findByRole("dialog", { name: "Save before opening another volume?" });
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(chooseAndReadVolume).not.toHaveBeenCalled();
    expect(useEditorStore.getState().document).toBe(original);
  });

  it("guards a native exit and sends cancellation back to the host", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    render(<App />);
    await waitFor(() => expect(nativeMocks.closeHandler).not.toBeNull());
    act(() => nativeMocks.closeHandler?.());
    const dialog = await screen.findByRole("dialog", { name: "Save before exiting VoxelInk?" });
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelNativeExit).toHaveBeenCalledOnce());
    expect(confirmNativeExit).not.toHaveBeenCalled();
  });

  it("cancels a native exit until the active pointer stroke finishes", async () => {
    installFixture();
    render(<App />);
    await waitFor(() => expect(nativeMocks.closeHandler).not.toBeNull());
    act(() => {
      useEditorStore.getState().beginStroke(1);
      nativeMocks.closeHandler?.();
    });
    expect(await screen.findByRole("status")).toHaveTextContent("Finish the active brush stroke");
    expect(cancelNativeExit).toHaveBeenCalledOnce();
    expect(useEditorStore.getState().strokeActive).toBe(true);
    act(() => useEditorStore.getState().cancelStroke());
  });

  it("announces normalization and requires a first-overwrite decision", async () => {
    const user = userEvent.setup();
    installFixture(true);
    render(<App />);
    expect(screen.getByRole("status")).toHaveTextContent("Non-binary source values were normalized");
    await user.click(screen.getByRole("button", { name: "Save" }));
    const dialog = await screen.findByRole("dialog", { name: "Overwrite normalized import?" });
    expect(within(dialog).getByRole("button", { name: "Save a Copy" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Overwrite" })).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(useEditorStore.getState().document?.dirty).toBe(true);
  });

  it("keeps the document open and exposes native file errors accessibly", async () => {
    const user = userEvent.setup();
    installFixture();
    vi.mocked(releaseSource).mockRejectedValueOnce(new Error("Source authority could not be released."));
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Close volume" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Source authority could not be released");
    expect(useEditorStore.getState().document).not.toBeNull();
  });

  it("locks file and history commands until an atomic save finishes", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    let finishEncoding: ((bytes: Uint8Array) => void) | undefined;
    codecMocks.encode.mockReturnValueOnce(new Promise<Uint8Array>((resolve) => {
      finishEncoding = resolve;
    }));
    vi.mocked(chooseSaveDestination).mockResolvedValueOnce({
      ticket: "save-ticket",
      mode: "overwriteSource",
      displayName: "fixture.binvox",
      destinationVariant: "binvox",
    });
    vi.mocked(writeSavedVolume).mockResolvedValueOnce({
      mode: "overwriteSource",
      adopted: true,
      sourceId: "00000000-0000-4000-8000-000000000001",
      displayName: "fixture.binvox",
      sourceVariant: "binvox",
      size: 1,
    });
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Encoding binary volume…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open volume" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    await user.keyboard("{Control>}z{/Control}");
    expect(useEditorStore.getState().document?.getIndex(0)).toBe(1);
    act(() => { finishEncoding?.(Uint8Array.of(1)); });
    await waitFor(() => expect(useEditorStore.getState().document?.dirty).toBe(false));
    expect(screen.getByRole("button", { name: "Open volume" })).toBeEnabled();
  });

  it("leaves dirty state unchanged when saving is cancelled", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByTestId("voxel-editor")).toHaveAttribute("aria-busy", "false"));
    expect(useEditorStore.getState().document?.dirty).toBe(true);
    expect(writeSavedVolume).not.toHaveBeenCalled();
  });

  it("traps modal focus and restores it after cancellation", async () => {
    const user = userEvent.setup();
    installFixture();
    makeFixtureDirty();
    render(<App />);
    const close = screen.getByRole("button", { name: "Close volume" });
    await user.click(close);
    const dialog = await screen.findByRole("dialog");
    const cancel = within(dialog).getByRole("button", { name: "Cancel" });
    expect(cancel).toHaveFocus();
    await user.tab();
    expect(within(dialog).getByRole("button", { name: "Save" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(close).toHaveFocus();
    expect(useEditorStore.getState().document).not.toBeNull();
  });

  it("defines forced-color and reduced-motion behavior with WCAG AA palette contrast", () => {
    const styles = readFileSync(resolve(process.cwd(), "src/App.css"), "utf8");
    expect(styles).toContain("@media (forced-colors: active)");
    expect(styles).not.toContain("forced-color-adjust: none");
    for (const systemColor of ["CanvasText", "Canvas", "ButtonText", "ButtonFace", "HighlightText", "Highlight"]) {
      expect(styles).toContain(systemColor);
    }
    expect(styles).toContain("@media (prefers-reduced-motion: reduce)");
    for (const [foreground, background] of [
      ["#ecf3f5", "#0b1114"],
      ["#aab9bf", "#10191d"],
      ["#071512", "#4ed3b1"],
      ["#fff3f3", "#72343a"],
      ["#fff5e7", "#49371f"],
      ["#fff0f0", "#542b30"],
      ["#152329", "#e8eef0"],
      ["#4f656e", "#f7fafb"],
      ["#ffffff", "#087f68"],
    ] as const) {
      expect(contrastRatio(foreground, background)).toBeGreaterThanOrEqual(4.5);
    }
  });
});
