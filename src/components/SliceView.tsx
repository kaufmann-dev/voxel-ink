import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { brushStrokeIndices, diskBrushIndices } from "../core/brush";
import type { GridCoordinate, SliceAxis, VolumeDocument } from "../core/volume";
import { subscribeVoxelChanges, useEditorStore } from "../editor/store";
import {
  createSliceRaster,
  sliceShape,
  updateSliceRaster,
  type SliceRaster,
} from "../render/slice";

interface SliceViewProps {
  readonly axis: SliceAxis;
  readonly label: string;
  readonly shortLabel: string;
}

interface Layout {
  readonly scale: number;
  readonly offsetX: number;
  readonly offsetY: number;
  readonly width: number;
  readonly height: number;
}

function layoutFor(canvas: HTMLCanvasElement, document: VolumeDocument, axis: SliceAxis): Layout {
  const [width, height] = sliceShape(document, axis);
  const scale = Math.min(canvas.width / width, canvas.height / height);
  return {
    scale,
    offsetX: (canvas.width - width * scale) / 2,
    offsetY: (canvas.height - height * scale) / 2,
    width,
    height,
  };
}

function planeToCoordinate(
  axis: SliceAxis,
  slice: number,
  u: number,
  v: number,
): GridCoordinate {
  if (axis === 0) return [slice, u, v];
  if (axis === 1) return [u, slice, v];
  return [u, v, slice];
}

function coordinateToPlane(axis: SliceAxis, coordinate: GridCoordinate): readonly [number, number] {
  if (axis === 0) return [coordinate[1], coordinate[2]];
  if (axis === 1) return [coordinate[0], coordinate[2]];
  return [coordinate[0], coordinate[1]];
}

export function SliceView({ axis, label, shortLabel }: SliceViewProps) {
  const document = useEditorStore((state) => state.document);
  const cursor = useEditorStore((state) => state.document?.cursor ?? null);
  const busy = useEditorStore((state) => state.busyMessage !== null);
  const strokeActive = useEditorStore((state) => state.strokeActive);
  const mode = useEditorStore((state) => state.mode);
  const brushDiameter = useEditorStore((state) => state.brushDiameter);
  const setCursor = useEditorStore((state) => state.setCursor);
  const setActiveAxis = useEditorStore((state) => state.setActiveAxis);
  const beginStroke = useEditorStore((state) => state.beginStroke);
  const extendStroke = useEditorStore((state) => state.extendStroke);
  const finishStroke = useEditorStore((state) => state.finishStroke);
  const cancelStroke = useEditorStore((state) => state.cancelStroke);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceImageRef = useRef<ImageData | null>(null);
  const rasterRef = useRef<SliceRaster | null>(null);
  const cursorRef = useRef(cursor);
  const activePointer = useRef<number | null>(null);
  const lastAppliedCoordinate = useRef<GridCoordinate | null>(null);
  const pendingCoordinate = useRef<GridCoordinate | null>(null);
  const pointerFrame = useRef<number | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 320, height: 240 });

  useEffect(() => {
    cursorRef.current = cursor;
  }, [cursor]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const resize = () => {
      const bounds = container.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      setCanvasSize({
        width: Math.max(1, Math.round(bounds.width * ratio)),
        height: Math.max(1, Math.round(bounds.height * ratio)),
      });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    return () => observer.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#0d1418";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const currentCursor = cursorRef.current;
    if (!document || !currentCursor) {
      context.fillStyle = "#829198";
      context.font = `${Math.max(12, canvas.width / 26)}px system-ui`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("Open a volume", canvas.width / 2, canvas.height / 2);
      return;
    }

    const source = sourceCanvasRef.current;
    if (!source) return;
    const layout = layoutFor(canvas, document, axis);
    context.imageSmoothingEnabled = false;
    context.drawImage(
      source,
      layout.offsetX,
      layout.offsetY,
      layout.width * layout.scale,
      layout.height * layout.scale,
    );

    if (layout.scale >= 10) {
      context.strokeStyle = "rgba(255,255,255,0.08)";
      context.lineWidth = 1;
      context.beginPath();
      for (let u = 0; u <= layout.width; u += 1) {
        const x = layout.offsetX + u * layout.scale;
        context.moveTo(x, layout.offsetY);
        context.lineTo(x, layout.offsetY + layout.height * layout.scale);
      }
      for (let v = 0; v <= layout.height; v += 1) {
        const y = layout.offsetY + v * layout.scale;
        context.moveTo(layout.offsetX, y);
        context.lineTo(layout.offsetX + layout.width * layout.scale, y);
      }
      context.stroke();
    }

    const [cursorU, cursorV] = coordinateToPlane(axis, currentCursor);
    const crosshairX = layout.offsetX + (cursorU + 0.5) * layout.scale;
    const crosshairY = layout.offsetY + (layout.height - cursorV - 0.5) * layout.scale;
    context.strokeStyle = "#ffb454";
    context.lineWidth = Math.max(1.5, canvas.width / 640);
    context.beginPath();
    context.moveTo(layout.offsetX, crosshairY);
    context.lineTo(layout.offsetX + layout.width * layout.scale, crosshairY);
    context.moveTo(crosshairX, layout.offsetY);
    context.lineTo(crosshairX, layout.offsetY + layout.height * layout.scale);
    context.stroke();
  }, [axis, document]);

  const sliceNumber = cursor?.[axis] ?? -1;
  useEffect(() => {
    if (!document || sliceNumber < 0) {
      sourceCanvasRef.current = null;
      sourceImageRef.current = null;
      rasterRef.current = null;
      draw();
      return;
    }
    const created = createSliceRaster(document, axis, sliceNumber);
    const source = window.document.createElement("canvas");
    source.width = created.width;
    source.height = created.height;
    const sourceContext = source.getContext("2d");
    if (!sourceContext) return;
    const image = sourceContext.createImageData(created.width, created.height);
    image.data.set(created.pixels);
    sourceContext.putImageData(image, 0, 0);
    sourceCanvasRef.current = source;
    sourceImageRef.current = image;
    rasterRef.current = { ...created, pixels: image.data };
    draw();
  }, [axis, document, draw, sliceNumber]);

  useEffect(() => {
    draw();
  }, [canvasSize, cursor, draw]);

  useEffect(() => {
    if (!document) return;
    const pending: Array<{ indices: Uint32Array; value: 0 | 1 }> = [];
    let frame: number | null = null;
    const flush = () => {
      frame = null;
      const raster = rasterRef.current;
      const source = sourceCanvasRef.current;
      const image = sourceImageRef.current;
      const sourceContext = source?.getContext("2d");
      if (!raster || !image || !sourceContext) {
        pending.length = 0;
        return;
      }
      let minX = raster.width;
      let minY = raster.height;
      let maxX = -1;
      let maxY = -1;
      for (const change of pending.splice(0)) {
        const dirty = updateSliceRaster(raster, document, change.indices, change.value);
        if (!dirty) continue;
        minX = Math.min(minX, dirty.x);
        minY = Math.min(minY, dirty.y);
        maxX = Math.max(maxX, dirty.x + dirty.width - 1);
        maxY = Math.max(maxY, dirty.y + dirty.height - 1);
      }
      if (maxX >= minX) {
        sourceContext.putImageData(
          image,
          0,
          0,
          minX,
          minY,
          maxX - minX + 1,
          maxY - minY + 1,
        );
        draw();
      }
    };
    const unsubscribe = subscribeVoxelChanges((change) => {
      if (change.document !== document) return;
      pending.push({ indices: change.indices, value: change.value });
      if (frame === null) frame = requestAnimationFrame(flush);
    });
    return () => {
      unsubscribe();
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [document, draw]);

  const coordinateFromClient = useCallback((clientX: number, clientY: number): GridCoordinate | null => {
    const canvas = canvasRef.current;
    if (!canvas || !document || !cursor) return null;
    const bounds = canvas.getBoundingClientRect();
    const x = (clientX - bounds.left) * canvas.width / bounds.width;
    const y = (clientY - bounds.top) * canvas.height / bounds.height;
    const layout = layoutFor(canvas, document, axis);
    const u = Math.floor((x - layout.offsetX) / layout.scale);
    const displayV = Math.floor((y - layout.offsetY) / layout.scale);
    if (u < 0 || displayV < 0 || u >= layout.width || displayV >= layout.height) return null;
    return planeToCoordinate(axis, cursor[axis], u, layout.height - 1 - displayV);
  }, [axis, cursor, document]);

  const coordinateFromPointer = useCallback(
    (event: PointerEvent<HTMLCanvasElement>): GridCoordinate | null =>
      coordinateFromClient(event.clientX, event.clientY),
    [coordinateFromClient],
  );

  const applyAt = useCallback((coordinate: GridCoordinate, previous?: GridCoordinate) => {
    if (!document || mode === "inspect" || busy) return;
    const indices = previous
      ? brushStrokeIndices(document.dimensions, previous, coordinate, brushDiameter, axis)
      : diskBrushIndices(document.dimensions, coordinate, axis, brushDiameter);
    extendStroke(indices);
  }, [axis, brushDiameter, busy, document, extendStroke, mode]);

  const applyPendingCoordinate = useCallback(() => {
    pointerFrame.current = null;
    const coordinate = pendingCoordinate.current;
    pendingCoordinate.current = null;
    if (!coordinate) return;
    const previous = lastAppliedCoordinate.current;
    setCursor(coordinate);
    applyAt(coordinate, previous ?? undefined);
    lastAppliedCoordinate.current = coordinate;
  }, [applyAt, setCursor]);

  const queueCoordinate = useCallback((coordinate: GridCoordinate) => {
    pendingCoordinate.current = coordinate;
    if (pointerFrame.current === null) {
      pointerFrame.current = requestAnimationFrame(applyPendingCoordinate);
    }
  }, [applyPendingCoordinate]);

  const flushCoordinate = useCallback(() => {
    if (pointerFrame.current !== null) {
      cancelAnimationFrame(pointerFrame.current);
      pointerFrame.current = null;
    }
    applyPendingCoordinate();
  }, [applyPendingCoordinate]);

  const clearQueuedCoordinate = useCallback(() => {
    if (pointerFrame.current !== null) cancelAnimationFrame(pointerFrame.current);
    pointerFrame.current = null;
    pendingCoordinate.current = null;
    lastAppliedCoordinate.current = null;
  }, []);

  const finishPointerStroke = useCallback((pointerId: number, coordinate?: GridCoordinate | null) => {
    if (activePointer.current !== pointerId) return;
    if (coordinate) pendingCoordinate.current = coordinate;
    flushCoordinate();
    activePointer.current = null;
    lastAppliedCoordinate.current = null;
    finishStroke();
  }, [finishStroke, flushCoordinate]);

  const cancelPointerStroke = useCallback((pointerId: number) => {
    if (activePointer.current !== pointerId) return;
    activePointer.current = null;
    clearQueuedCoordinate();
    cancelStroke();
  }, [cancelStroke, clearQueuedCoordinate]);

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (busy || strokeActive) return;
    const coordinate = coordinateFromPointer(event);
    if (!coordinate || !document) return;
    event.currentTarget.focus();
    setActiveAxis(axis);
    setCursor(coordinate);
    if (mode === "inspect") return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    activePointer.current = event.pointerId;
    lastAppliedCoordinate.current = coordinate;
    pendingCoordinate.current = null;
    beginStroke(mode === "brush" ? 1 : 0);
    applyAt(coordinate);
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    if (busy) return;
    if (activePointer.current !== event.pointerId) return;
    const coordinate = coordinateFromPointer(event);
    if (!coordinate) return;
    queueCoordinate(coordinate);
  };

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    if (activePointer.current !== event.pointerId) return;
    finishPointerStroke(event.pointerId, coordinateFromPointer(event));
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const handlePointerCancel = (event: PointerEvent<HTMLCanvasElement>) => {
    if (activePointer.current !== event.pointerId) return;
    cancelPointerStroke(event.pointerId);
  };

  useEffect(() => {
    const finishWindowStroke = (event: globalThis.PointerEvent) => {
      finishPointerStroke(event.pointerId, coordinateFromClient(event.clientX, event.clientY));
    };
    const cancelWindowStroke = (event: globalThis.PointerEvent) => {
      cancelPointerStroke(event.pointerId);
    };
    const cancelInterruptedStroke = () => {
      if (activePointer.current === null) return;
      cancelPointerStroke(activePointer.current);
    };
    const handleVisibilityChange = () => {
      if (window.document.visibilityState !== "visible") cancelInterruptedStroke();
    };

    window.addEventListener("pointerup", finishWindowStroke, true);
    window.addEventListener("pointercancel", cancelWindowStroke, true);
    window.addEventListener("blur", cancelInterruptedStroke);
    window.document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pointerup", finishWindowStroke, true);
      window.removeEventListener("pointercancel", cancelWindowStroke, true);
      window.removeEventListener("blur", cancelInterruptedStroke);
      window.document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [cancelPointerStroke, coordinateFromClient, finishPointerStroke]);

  const handleKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    if (busy || strokeActive) {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown", " ", "Enter"].includes(event.key)) {
        event.preventDefault();
      }
      return;
    }
    if (!document || !cursor) return;
    const [u, v] = coordinateToPlane(axis, cursor);
    const [width, height] = sliceShape(document, axis);
    let nextU = u;
    let nextV = v;
    let nextSlice = cursor[axis];
    if (event.key === "ArrowLeft") nextU -= 1;
    else if (event.key === "ArrowRight") nextU += 1;
    else if (event.key === "ArrowUp") nextV += 1;
    else if (event.key === "ArrowDown") nextV -= 1;
    else if (event.key === "PageUp") nextSlice += 1;
    else if (event.key === "PageDown") nextSlice -= 1;
    else if ((event.key === " " || event.key === "Enter") && mode !== "inspect") {
      event.preventDefault();
      beginStroke(mode === "brush" ? 1 : 0);
      applyAt(cursor);
      finishStroke();
      return;
    } else return;
    event.preventDefault();
    nextU = Math.max(0, Math.min(width - 1, nextU));
    nextV = Math.max(0, Math.min(height - 1, nextV));
    nextSlice = Math.max(0, Math.min(document.dimensions[axis] - 1, nextSlice));
    setActiveAxis(axis);
    setCursor(planeToCoordinate(axis, nextSlice, nextU, nextV));
  };

  return (
    <section className="pane slice-pane" aria-labelledby={`slice-${axis}-title`}>
      <header className="pane-header">
        <h2 id={`slice-${axis}-title`}>{label}</h2>
        <span>{document && cursor ? `${shortLabel} ${cursor[axis] + 1} / ${document.dimensions[axis]}` : shortLabel}</span>
      </header>
      <div ref={containerRef} className="slice-canvas-wrap">
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          className="slice-canvas"
          role="application"
          tabIndex={0}
          aria-disabled={busy || strokeActive}
          aria-label={`${label} slice`}
          aria-description="Use arrow keys to move, Page Up and Page Down to change slice, and Space to apply the selected tool."
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onLostPointerCapture={handlePointerCancel}
          onKeyDown={handleKeyDown}
        />
      </div>
    </section>
  );
}
