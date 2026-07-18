import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { brushStrokeIndices, sphereBrushIndices } from "../core/brush";
import { adjacentCoordinate, firstOccupiedVoxel, intersectActiveSlice } from "../core/dda";
import { indexRayFromWorld, transformPoint } from "../core/affine";
import type { GridCoordinate } from "../core/volume";
import { useEditorStore } from "../editor/store";
import {
  chunkKey,
  chunksAffectedByIndices,
  chunksForDimensions,
  type ChunkCoordinate,
  type MeshPayload,
} from "../render/mesh";

interface WorkerMeshMessage {
  readonly type: "mesh";
  readonly chunk: ChunkCoordinate;
  readonly revision: number;
  readonly generation: number;
  readonly payload: MeshPayload;
}

interface WorkerReadyMessage {
  readonly type: "ready";
  readonly revision: number;
  readonly generation: number;
}

type MeshWorkerMessage = WorkerMeshMessage | WorkerReadyMessage;

const GEOMETRY_BUDGET = 256 * 1024 * 1024;
const MAX_WORKER_DELTA_INDICES = 262_144;

export function VolumeView3D() {
  const document = useEditorStore((state) => state.document);
  const revision = useEditorStore((state) => state.revision);
  const pendingMeshChanges = useEditorStore((state) => state.pendingMeshChanges);
  const mode = useEditorStore((state) => state.mode);
  const busy = useEditorStore((state) => state.busyMessage !== null);
  const strokeActive = useEditorStore((state) => state.strokeActive);
  const brushDiameter = useEditorStore((state) => state.brushDiameter);
  const activeAxis = useEditorStore((state) => state.activeAxis);
  const setCursor = useEditorStore((state) => state.setCursor);
  const beginStroke = useEditorStore((state) => state.beginStroke);
  const extendStroke = useEditorStore((state) => state.extendStroke);
  const finishStroke = useEditorStore((state) => state.finishStroke);
  const cancelStroke = useEditorStore((state) => state.cancelStroke);
  const acknowledgeMeshChanges = useEditorStore((state) => state.acknowledgeMeshChanges);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const generationRef = useRef(0);
  const revisionRef = useRef(revision);
  const handledSequenceRef = useRef(-1);
  const readyRef = useRef(false);
  const inFlightRef = useRef(false);
  const queueRef = useRef<ChunkCoordinate[]>([]);
  const queuedKeysRef = useRef(new Set<string>());
  const meshesRef = useRef(new Map<string, THREE.Mesh>());
  const geometryBytesRef = useRef(0);
  const requestRenderRef = useRef<() => void>(() => undefined);
  const activePointer = useRef<number | null>(null);
  const lastAppliedCoordinate = useRef<GridCoordinate | null>(null);
  const pendingCoordinate = useRef<GridCoordinate | null>(null);
  const pointerFrame = useRef<number | null>(null);
  const [status, setStatus] = useState("3D preview idle");

  useEffect(() => {
    revisionRef.current = revision;
  }, [revision]);

  const enqueue = useCallback((chunks: readonly ChunkCoordinate[]) => {
    for (const chunk of chunks) {
      const key = chunkKey(chunk);
      if (!queuedKeysRef.current.has(key)) {
        queuedKeysRef.current.add(key);
        queueRef.current.push(chunk);
      }
    }
  }, []);

  const pump = useCallback(() => {
    const worker = workerRef.current;
    if (!worker || !readyRef.current || inFlightRef.current) return;
    const chunk = queueRef.current.shift();
    if (!chunk) {
      setStatus(useEditorStore.getState().strokeActive
        ? "3D preview paused while painting"
        : "3D preview ready");
      requestRenderRef.current();
      return;
    }
    queuedKeysRef.current.delete(chunkKey(chunk));
    inFlightRef.current = true;
    worker.postMessage({
      type: "mesh",
      chunk,
      revision: revisionRef.current,
      generation: generationRef.current,
    });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    } catch {
      queueMicrotask(() => { setStatus("3D preview is unavailable on this graphics driver"); });
      return;
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0d1418, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1_000_000);
    cameraRef.current = camera;
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false;
    controls.screenSpacePanning = true;
    controlsRef.current = controls;
    scene.add(new THREE.HemisphereLight(0xd8fff5, 0x26343c, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(1, -1, 2);
    scene.add(keyLight);

    let requested = false;
    const render = () => {
      requested = false;
      renderer.render(scene, camera);
      if (import.meta.env.VITE_WDIO) {
        canvas.dispatchEvent(new Event("voxelink:frame-rendered"));
      }
    };
    const requestRender = () => {
      if (requested) return;
      requested = true;
      requestAnimationFrame(render);
    };
    requestRenderRef.current = requestRender;
    controls.addEventListener("change", requestRender);
    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) return;
      renderer.setSize(bounds.width, bounds.height, false);
      camera.aspect = bounds.width / bounds.height;
      camera.updateProjectionMatrix();
      requestRender();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    return () => {
      observer.disconnect();
      controls.removeEventListener("change", requestRender);
      controls.dispose();
      renderer.dispose();
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    workerRef.current?.terminate();
    workerRef.current = null;
    readyRef.current = false;
    inFlightRef.current = false;
    queueRef.current = [];
    queuedKeysRef.current.clear();
    geometryBytesRef.current = 0;
    const meshes = meshesRef.current;
    for (const mesh of meshes.values()) mesh.geometry.dispose();
    meshes.clear();
    if (groupRef.current) scene.remove(groupRef.current);
    materialRef.current?.dispose();
    materialRef.current = null;
    groupRef.current = null;

    if (!document) {
      requestRenderRef.current();
      return;
    }

    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    group.matrix.set(...document.indexToWorld);
    groupRef.current = group;
    scene.add(group);
    const material = new THREE.MeshStandardMaterial({ color: 0x4ed3b1, roughness: 0.72, metalness: 0.02 });
    materialRef.current = material;

    const [nx, ny, nz] = document.dimensions;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(nx, ny, nz)),
      new THREE.LineBasicMaterial({ color: 0x50646d, transparent: true, opacity: 0.55 }),
    );
    edges.position.set((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2);
    group.add(edges);

    const center = transformPoint(document.indexToWorld, [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2]);
    const extent = Math.max(nx * document.spacing[0], ny * document.spacing[1], nz * document.spacing[2], 1);
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (camera && controls) {
      camera.position.set(center[0] + extent * 1.35, center[1] - extent * 1.35, center[2] + extent * 1.15);
      camera.near = Math.max(extent / 10_000, 0.001);
      camera.far = extent * 20;
      camera.updateProjectionMatrix();
      controls.target.set(...center);
      controls.update();
    }

    const startWorker = () => {
      workerRef.current?.terminate();
      readyRef.current = false;
      inFlightRef.current = false;
      generationRef.current += 1;
      const generation = generationRef.current;
      queueRef.current = [];
      queuedKeysRef.current.clear();
      const centerChunk = document.cursor.map((value) => Math.floor(value / 32)) as [number, number, number];
      const chunks = chunksForDimensions(document.dimensions).sort((left, right) => {
        const leftDistance = Math.abs(left.x - centerChunk[0]) + Math.abs(left.y - centerChunk[1]) + Math.abs(left.z - centerChunk[2]);
        const rightDistance = Math.abs(right.x - centerChunk[0]) + Math.abs(right.y - centerChunk[1]) + Math.abs(right.z - centerChunk[2]);
        return leftDistance - rightDistance;
      });
      enqueue(chunks);
      setStatus("Preparing 3D preview…");
      const worker = new Worker(new URL("../render/mesh.worker.ts", import.meta.url), { type: "module" });
      workerRef.current = worker;
      worker.onmessage = (event: MessageEvent<MeshWorkerMessage>) => {
        const message = event.data;
        if (message.generation !== generationRef.current) return;
        if (message.type === "ready") {
          readyRef.current = true;
          pump();
          return;
        }
        inFlightRef.current = false;
        const key = chunkKey(message.chunk);
        const superseded = message.revision !== revisionRef.current
          && queuedKeysRef.current.has(key);
        if (!superseded) {
          const previous = meshesRef.current.get(key);
          if (previous) {
            geometryBytesRef.current -= previous.geometry.getAttribute("position").array.byteLength;
            geometryBytesRef.current -= previous.geometry.getAttribute("normal").array.byteLength;
            geometryBytesRef.current -= previous.geometry.index?.array.byteLength ?? 0;
            previous.geometry.dispose();
            group.remove(previous);
            meshesRef.current.delete(key);
          }
          if (message.payload.quadCount > 0) {
            const payloadBytes = message.payload.positions.byteLength + message.payload.normals.byteLength + message.payload.indices.byteLength;
            if (geometryBytesRef.current + payloadBytes > GEOMETRY_BUDGET) {
              queueRef.current = [];
              queuedKeysRef.current.clear();
              setStatus("3D preview paused at the 256 MiB geometry limit; slice editing remains available");
              requestRenderRef.current();
              return;
            }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", new THREE.BufferAttribute(message.payload.positions, 3));
            geometry.setAttribute("normal", new THREE.BufferAttribute(message.payload.normals, 3));
            geometry.setIndex(new THREE.BufferAttribute(message.payload.indices, 1));
            const [dimensionX, dimensionY, dimensionZ] = document.dimensions;
            const startX = message.chunk.x * 32;
            const startY = message.chunk.y * 32;
            const startZ = message.chunk.z * 32;
            const sizeX = Math.min(32, dimensionX - startX);
            const sizeY = Math.min(32, dimensionY - startY);
            const sizeZ = Math.min(32, dimensionZ - startZ);
            geometry.boundingSphere = new THREE.Sphere(
              new THREE.Vector3(
                startX + (sizeX - 1) / 2,
                startY + (sizeY - 1) / 2,
                startZ + (sizeZ - 1) / 2,
              ),
              Math.hypot(sizeX, sizeY, sizeZ) / 2,
            );
            const mesh = new THREE.Mesh(geometry, material);
            mesh.frustumCulled = true;
            meshesRef.current.set(key, mesh);
            group.add(mesh);
            geometryBytesRef.current += payloadBytes;
          }
        }
        pump();
      };
      worker.onerror = () => {
        readyRef.current = false;
        setStatus("3D preparation failed; slice editing remains available");
      };
      const snapshot = document.copyOccupancy();
      worker.postMessage({
        type: "init",
        dimensions: document.dimensions,
        occupancy: snapshot.buffer,
        revision: document.revision,
        generation,
      }, [snapshot.buffer]);
      const latestPending = useEditorStore.getState().pendingMeshChanges.at(-1);
      if (latestPending) {
        handledSequenceRef.current = latestPending.sequence;
        useEditorStore.getState().acknowledgeMeshChanges(latestPending.sequence);
      }
    };
    startWorker();
    requestRenderRef.current();

    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      edges.geometry.dispose();
      (edges.material as THREE.Material).dispose();
      for (const mesh of meshes.values()) mesh.geometry.dispose();
      meshes.clear();
      material.dispose();
      scene.remove(group);
    };
  }, [document, enqueue, pump]);

  useEffect(() => {
    if (!document || pendingMeshChanges.length === 0) return;
    const changes = pendingMeshChanges.filter(
      (change) => change.sequence > handledSequenceRef.current,
    );
    if (changes.length === 0) return;
    const throughSequence = changes.at(-1)!.sequence;
    const worker = workerRef.current;
    if (!worker) return;
    const affectedChunks = new Map<string, ChunkCoordinate>();
    for (const change of changes) {
      for (const chunk of chunksAffectedByIndices(change.indices, document.dimensions)) {
        affectedChunks.set(chunkKey(chunk), chunk);
      }
      for (let offset = 0; offset < change.indices.length; offset += MAX_WORKER_DELTA_INDICES) {
        const segment = change.indices.slice(
          offset,
          Math.min(change.indices.length, offset + MAX_WORKER_DELTA_INDICES),
        );
        worker.postMessage({
          type: "apply",
          indices: segment.buffer,
          value: change.value,
          revision: change.revision,
          generation: generationRef.current,
        }, [segment.buffer]);
      }
    }
    enqueue([...affectedChunks.values()]);
    setStatus(strokeActive ? "3D preview paused while painting" : "Updating 3D preview…");
    pump();
    handledSequenceRef.current = throughSequence;
    acknowledgeMeshChanges(throughSequence);
  }, [acknowledgeMeshChanges, document, enqueue, pendingMeshChanges, pump, strokeActive]);

  useEffect(() => {
    if (controlsRef.current) controlsRef.current.enabled = mode === "inspect" && !busy;
  }, [busy, mode]);

  const coordinateFromClient = useCallback((clientX: number, clientY: number): GridCoordinate | null => {
    if (!document || !cameraRef.current || !canvasRef.current) return null;
    const bounds = canvasRef.current.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - bounds.left) / bounds.width) * 2 - 1,
      -((clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, cameraRef.current);
    const { origin, direction } = raycaster.ray;
    const ray = indexRayFromWorld(
      document.worldToIndex,
      [origin.x, origin.y, origin.z],
      [direction.x, direction.y, direction.z],
    );
    const hit = firstOccupiedVoxel(document.occupancy, document.dimensions, ray);
    if (mode === "eraser") return hit?.coordinate ?? null;
    if (hit) return adjacentCoordinate(hit, document.dimensions);
    return intersectActiveSlice(ray, document.dimensions, activeAxis, document.cursor[activeAxis]);
  }, [activeAxis, document, mode]);

  const coordinateFromPointer = useCallback(
    (event: PointerEvent<HTMLCanvasElement>): GridCoordinate | null =>
      coordinateFromClient(event.clientX, event.clientY),
    [coordinateFromClient],
  );

  const applyAt = useCallback((coordinate: GridCoordinate, previous?: GridCoordinate) => {
    if (!document || mode === "inspect" || busy) return;
    const indices = previous
      ? brushStrokeIndices(document.dimensions, previous, coordinate, brushDiameter)
      : sphereBrushIndices(document.dimensions, coordinate, brushDiameter);
    extendStroke(indices);
  }, [brushDiameter, busy, document, extendStroke, mode]);

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
    if (mode === "inspect" || busy || strokeActive) return;
    const coordinate = coordinateFromPointer(event);
    if (!coordinate) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    activePointer.current = event.pointerId;
    lastAppliedCoordinate.current = coordinate;
    pendingCoordinate.current = null;
    setCursor(coordinate);
    beginStroke(mode === "brush" ? 1 : 0);
    applyAt(coordinate);
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
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

  const displayedStatus = strokeActive ? "3D preview paused while painting" : status;

  return (
    <section className="pane volume-pane" aria-labelledby="volume-title">
      <header className="pane-header">
        <h2 id="volume-title">3D volume</h2>
        <span aria-live="polite">{document ? displayedStatus : "Open a volume to prepare 3D"}</span>
      </header>
      <div className="volume-canvas-wrap">
        <canvas
          ref={canvasRef}
          className="volume-canvas"
          aria-label="3D volume view"
          aria-disabled={busy || strokeActive}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onLostPointerCapture={handlePointerCancel}
        />
      </div>
    </section>
  );
}
