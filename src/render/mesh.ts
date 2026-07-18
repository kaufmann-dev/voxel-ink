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
  readonly quadCount: number;
  readonly unitFaceCount: number;
}

export const CHUNK_SIZE = 32;

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

const FACE_AXES = [
  { axis: 0, uAxis: 1, vAxis: 2 },
  { axis: 1, uAxis: 2, vAxis: 0 },
  { axis: 2, uAxis: 0, vAxis: 1 },
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
  const add = (x: number, y: number, z: number): void => {
    if (x < 0 || y < 0 || z < 0 || x >= cxMax || y >= cyMax || z >= czMax) return;
    const key = `${x}:${y}:${z}`;
    if (!result.has(key)) result.set(key, { x, y, z });
  };
  for (const index of indices) {
    const x = index % nx;
    const y = Math.floor(index / nx) % ny;
    const z = Math.floor(index / (nx * ny));
    const chunkX = Math.floor(x / chunkSize);
    const chunkY = Math.floor(y / chunkSize);
    const chunkZ = Math.floor(z / chunkSize);
    add(chunkX, chunkY, chunkZ);
    if (x % chunkSize === 0) add(chunkX - 1, chunkY, chunkZ);
    if (x % chunkSize === chunkSize - 1) add(chunkX + 1, chunkY, chunkZ);
    if (y % chunkSize === 0) add(chunkX, chunkY - 1, chunkZ);
    if (y % chunkSize === chunkSize - 1) add(chunkX, chunkY + 1, chunkZ);
    if (z % chunkSize === 0) add(chunkX, chunkY, chunkZ - 1);
    if (z % chunkSize === chunkSize - 1) add(chunkX, chunkY, chunkZ + 1);
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
  let quadCount = 0;
  let unitFaceCount = 0;
  const starts = [x0, y0, z0] as const;
  const ends = [x1, y1, z1] as const;
  const occupiedStarts: [number, number, number] = [x1, y1, z1];
  const occupiedEnds: [number, number, number] = [x0, y0, z0];

  const filled = (x: number, y: number, z: number): boolean =>
    x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && occupancy[x + nx * (y + ny * z)] === 1;

  for (let z = z0; z < z1; z += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        if (!filled(x, y, z)) continue;
        occupiedStarts[0] = Math.min(occupiedStarts[0], x);
        occupiedStarts[1] = Math.min(occupiedStarts[1], y);
        occupiedStarts[2] = Math.min(occupiedStarts[2], z);
        occupiedEnds[0] = Math.max(occupiedEnds[0], x + 1);
        occupiedEnds[1] = Math.max(occupiedEnds[1], y + 1);
        occupiedEnds[2] = Math.max(occupiedEnds[2], z + 1);
      }
    }
  }

  if (occupiedEnds[0] === x0 && occupiedEnds[1] === y0 && occupiedEnds[2] === z0) {
    return {
      positions: new Float32Array(),
      normals: new Float32Array(),
      indices: new Uint32Array(),
      quadCount: 0,
      unitFaceCount: 0,
    };
  }

  for (const face of FACE_AXES) {
    const uStart = occupiedStarts[face.uAxis];
    const vStart = occupiedStarts[face.vAxis];
    const uLength = occupiedEnds[face.uAxis] - uStart;
    const vLength = occupiedEnds[face.vAxis] - vStart;
    const mask = new Int8Array(uLength * vLength);

    for (
      let plane = occupiedStarts[face.axis];
      plane <= occupiedEnds[face.axis];
      plane += 1
    ) {
      mask.fill(0);
      for (let v = 0; v < vLength; v += 1) {
        for (let u = 0; u < uLength; u += 1) {
          const lower: [number, number, number] = [0, 0, 0];
          lower[face.axis] = plane - 1;
          lower[face.uAxis] = uStart + u;
          lower[face.vAxis] = vStart + v;
          const upper: [number, number, number] = [...lower];
          upper[face.axis] = plane;
          const lowerFilled = filled(lower[0], lower[1], lower[2]);
          const upperFilled = filled(upper[0], upper[1], upper[2]);
          const lowerOwned = plane > starts[face.axis];
          const upperOwned = plane < ends[face.axis];
          if (lowerOwned && lowerFilled && !upperFilled) mask[u + uLength * v] = 1;
          else if (upperOwned && upperFilled && !lowerFilled) mask[u + uLength * v] = -1;
        }
      }

      for (let v = 0; v < vLength; v += 1) {
        for (let u = 0; u < uLength;) {
          const sign = mask[u + uLength * v]!;
          if (sign === 0) {
            u += 1;
            continue;
          }
          let width = 1;
          while (u + width < uLength && mask[u + width + uLength * v] === sign) width += 1;
          let height = 1;
          heightLoop: while (v + height < vLength) {
            for (let column = 0; column < width; column += 1) {
              if (mask[u + column + uLength * (v + height)] !== sign) break heightLoop;
            }
            height += 1;
          }

          for (let row = 0; row < height; row += 1) {
            mask.fill(0, u + uLength * (v + row), u + width + uLength * (v + row));
          }

          const origin: [number, number, number] = [0, 0, 0];
          origin[face.axis] = plane - 0.5;
          origin[face.uAxis] = uStart + u - 0.5;
          origin[face.vAxis] = vStart + v - 0.5;
          const alongU: [number, number, number] = [...origin];
          alongU[face.uAxis] = alongU[face.uAxis] + width;
          const diagonal: [number, number, number] = [...alongU];
          diagonal[face.vAxis] = diagonal[face.vAxis] + height;
          const alongV: [number, number, number] = [...origin];
          alongV[face.vAxis] = alongV[face.vAxis] + height;
          const corners = sign > 0
            ? [origin, alongU, diagonal, alongV]
            : [origin, alongV, diagonal, alongU];
          const vertex = positions.length / 3;
          for (const corner of corners) {
            positions.push(corner[0], corner[1], corner[2]);
            const normal: [number, number, number] = [0, 0, 0];
            normal[face.axis] = sign;
            normals.push(normal[0], normal[1], normal[2]);
          }
          indices.push(vertex, vertex + 1, vertex + 2, vertex, vertex + 2, vertex + 3);
          quadCount += 1;
          unitFaceCount += width * height;
          u += width;
        }
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    quadCount,
    unitFaceCount,
  };
}
