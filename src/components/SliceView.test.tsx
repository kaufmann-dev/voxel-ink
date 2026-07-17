import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { IDENTITY_AFFINE } from "../core/affine";
import { useEditorStore } from "../editor/store";
import { SliceView } from "./SliceView";

const context = {
  beginPath: vi.fn(),
  clearRect: vi.fn(),
  createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4), width, height }),
  drawImage: vi.fn(),
  fillRect: vi.fn(),
  fillText: vi.fn(),
  lineTo: vi.fn(),
  moveTo: vi.fn(),
  putImageData: vi.fn(),
  stroke: vi.fn(),
  fillStyle: "",
  font: "",
  imageSmoothingEnabled: false,
  lineWidth: 1,
  strokeStyle: "",
  textAlign: "start",
  textBaseline: "alphabetic",
};

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D);
  useEditorStore.getState().clearDocument();
  useEditorStore.getState().installDocument({
    dimensions: [3, 3, 1],
    occupancy: new Uint8Array(9),
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
      dimensions: [3, 3, 1],
    },
    normalization: { convertedNonBinary: false },
  }, {
    sourceId: "00000000-0000-4000-8000-000000000002",
    name: "keyboard.binvox",
    extension: ".binvox",
    size: 12,
  });
  useEditorStore.getState().setMode("brush");
});

describe("slice keyboard editing", () => {
  it("moves the shared cursor and applies a brush without a pointer", () => {
    render(<SliceView axis={2} label="Axial" shortLabel="Z" />);
    const canvas = screen.getByRole("application", { name: "Axial slice" });
    fireEvent.keyDown(canvas, { key: "ArrowRight" });
    expect(useEditorStore.getState().document?.cursor).toEqual([2, 1, 0]);
    fireEvent.keyDown(canvas, { key: " " });
    expect(useEditorStore.getState().document?.get([2, 1, 0])).toBe(1);
    expect(useEditorStore.getState().history?.undoCount).toBe(1);
  });

  it("rerenders the other panes when the shared cursor moves", () => {
    render(
      <>
        <SliceView axis={1} label="Coronal" shortLabel="Y" />
        <SliceView axis={0} label="Sagittal" shortLabel="X" />
      </>,
    );
    expect(screen.getByText("X 2 / 3")).toBeInTheDocument();
    act(() => {
      useEditorStore.getState().setCursor([2, 1, 0]);
    });
    expect(screen.getByText("X 3 / 3")).toBeInTheDocument();
  });

  it("does not merge keyboard input from another pane into an active pointer stroke", () => {
    render(<SliceView axis={2} label="Axial" shortLabel="Z" />);
    const store = useEditorStore.getState();
    act(() => {
      store.beginStroke(1);
      store.extendStroke([0]);
    });
    fireEvent.keyDown(screen.getByRole("application", { name: "Axial slice" }), { key: " " });
    expect(useEditorStore.getState().document?.occupiedCount).toBe(1);
    expect(useEditorStore.getState().history?.undoCount).toBe(0);
    act(() => useEditorStore.getState().cancelStroke());
  });
});
