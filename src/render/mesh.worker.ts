import {
  meshChunk,
  shouldApplyMeshChanges,
  type ChunkCoordinate,
  type Dimensions,
} from "./mesh";

interface InitMessage {
  readonly type: "init";
  readonly dimensions: Dimensions;
  readonly occupancy: ArrayBuffer;
  readonly revision: number;
  readonly generation: number;
}

interface ApplyMessage {
  readonly type: "apply";
  readonly indices: ArrayBuffer;
  readonly value: 0 | 1;
  readonly revision: number;
  readonly generation: number;
}

interface MeshMessage {
  readonly type: "mesh";
  readonly chunk: ChunkCoordinate;
  readonly revision: number;
  readonly generation: number;
}

type WorkerMessage = InitMessage | ApplyMessage | MeshMessage;

let dimensions: Dimensions = [0, 0, 0];
let occupancy = new Uint8Array();
let revision = -1;
let generation = -1;

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === "init") {
    dimensions = message.dimensions;
    occupancy = new Uint8Array(message.occupancy);
    revision = message.revision;
    generation = message.generation;
    self.postMessage({ type: "ready", revision, generation });
    return;
  }
  if (message.type === "apply") {
    if (!shouldApplyMeshChanges(message, { revision, generation })) return;
    const indices = new Uint32Array(message.indices);
    for (let offset = 0; offset < indices.length; offset += 1) {
      occupancy[indices[offset]!] = message.value;
    }
    revision = message.revision;
    return;
  }
  if (message.generation !== generation || message.revision !== revision) return;
  const payload = meshChunk(occupancy, dimensions, message.chunk);
  self.postMessage(
    { type: "mesh", chunk: message.chunk, revision, generation, payload },
    { transfer: [payload.positions.buffer, payload.normals.buffer, payload.indices.buffer] },
  );
};

export {};
