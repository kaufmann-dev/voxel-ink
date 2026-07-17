export type Dimensions = readonly [number, number, number];

export interface ChunkCoordinate {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface MeshPayload {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  readonly faceCount: number;
}

export const CHUNK_SIZE = 32;
export const MAX_INCREMENTAL_MESH_CHANGES = 65_536;

export interface RevisionTag {
  readonly revision: number;
  readonly generation: number;
}

export function isCurrentMeshResult(result: RevisionTag, current: RevisionTag): boolean {
  return result.revision === current.revision && result.generation === current.generation;
}

export function shouldApplyMeshChanges(incoming: RevisionTag, current: RevisionTag): boolean {
  return incoming.generation === current.generation;
}

export function requiresFullMeshRestart(changeCount: number, full: boolean): boolean {
  return full || changeCount > MAX_INCREMENTAL_MESH_CHANGES;
}

const FACE_DEFINITIONS = [
  { neighbor: [-1, 0, 0], normal: [-1, 0, 0], corners: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]] },
  { neighbor: [1, 0, 0], normal: [1, 0, 0], corners: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]] },
  { neighbor: [0, -1, 0], normal: [0, -1, 0], corners: [[-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5]] },
  { neighbor: [0, 1, 0], normal: [0, 1, 0], corners: [[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]] },
  { neighbor: [0, 0, -1], normal: [0, 0, -1], corners: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]] },
  { neighbor: [0, 0, 1], normal: [0, 0, 1], corners: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
] as const;

export function chunkKey(chunk: ChunkCoordinate): string {
  return `${chunk.x}:${chunk.y}:${chunk.z}`;
}

export function chunksForDimensions(dimensions: Dimensions, chunkSize = CHUNK_SIZE): ChunkCoordinate[] {
  const chunks: ChunkCoordinate[] = [];
  for (let z = 0; z < Math.ceil(dimensions[2] / chunkSize); z += 1) {
    for (let y = 0; y < Math.ceil(dimensions[1] / chunkSize); y += 1) {
      for (let x = 0; x < Math.ceil(dimensions[0] / chunkSize); x += 1) chunks.push({ x, y, z });
    }
  }
  return chunks;
}

export function chunksAffectedByIndices(
  indices: Iterable<number>,
  dimensions: Dimensions,
  chunkSize = CHUNK_SIZE,
): ChunkCoordinate[] {
  const [nx, ny, nz] = dimensions;
  const cxMax = Math.ceil(nx / chunkSize);
  const cyMax = Math.ceil(ny / chunkSize);
  const czMax = Math.ceil(nz / chunkSize);
  const result = new Map<string, ChunkCoordinate>();
  for (const index of indices) {
    const x = index % nx;
    const y = Math.floor(index / nx) % ny;
    const z = Math.floor(index / (nx * ny));
    const base = { x: Math.floor(x / chunkSize), y: Math.floor(y / chunkSize), z: Math.floor(z / chunkSize) };
    const candidates = [base];
    if (x % chunkSize === 0) candidates.push({ ...base, x: base.x - 1 });
    if (x % chunkSize === chunkSize - 1) candidates.push({ ...base, x: base.x + 1 });
    if (y % chunkSize === 0) candidates.push({ ...base, y: base.y - 1 });
    if (y % chunkSize === chunkSize - 1) candidates.push({ ...base, y: base.y + 1 });
    if (z % chunkSize === 0) candidates.push({ ...base, z: base.z - 1 });
    if (z % chunkSize === chunkSize - 1) candidates.push({ ...base, z: base.z + 1 });
    for (const candidate of candidates) {
      if (candidate.x < 0 || candidate.y < 0 || candidate.z < 0 || candidate.x >= cxMax || candidate.y >= cyMax || candidate.z >= czMax) continue;
      result.set(chunkKey(candidate), candidate);
    }
  }
  return [...result.values()];
}

export function meshChunk(
  occupancy: Uint8Array,
  dimensions: Dimensions,
  chunk: ChunkCoordinate,
  chunkSize = CHUNK_SIZE,
): MeshPayload {
  const [nx, ny, nz] = dimensions;
  const x0 = chunk.x * chunkSize;
  const y0 = chunk.y * chunkSize;
  const z0 = chunk.z * chunkSize;
  const x1 = Math.min(x0 + chunkSize, nx);
  const y1 = Math.min(y0 + chunkSize, ny);
  const z1 = Math.min(z0 + chunkSize, nz);
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  let faceCount = 0;

  const filled = (x: number, y: number, z: number): boolean =>
    x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && occupancy[x + nx * (y + ny * z)] === 1;

  for (let z = z0; z < z1; z += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        if (!filled(x, y, z)) continue;
        for (const face of FACE_DEFINITIONS) {
          if (filled(x + face.neighbor[0], y + face.neighbor[1], z + face.neighbor[2])) continue;
          const vertex = positions.length / 3;
          for (const corner of face.corners) {
            positions.push(x + corner[0], y + corner[1], z + corner[2]);
            normals.push(face.normal[0], face.normal[1], face.normal[2]);
          }
          indices.push(vertex, vertex + 1, vertex + 2, vertex, vertex + 2, vertex + 3);
          faceCount += 1;
        }
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    faceCount,
  };
}
