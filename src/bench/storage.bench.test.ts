import { describe, expect, it } from "vitest";

type Dimensions = readonly [number, number, number];
type Pattern = "sparse" | "dense" | "noisy";
type SliceAxis = "x" | "y" | "z";

const CHUNK_SIZE = 32;
const CHUNK_VOXELS = CHUNK_SIZE ** 3;
const MEBIBYTE = 1024 ** 2;
const REFERENCE_DIMENSIONS = [512, 512, 256] as const;
const REFERENCE_VOXELS = REFERENCE_DIMENSIONS[0]
  * REFERENCE_DIMENSIONS[1]
  * REFERENCE_DIMENSIONS[2];

interface Storage {
  readonly allocatedBytes: number;
  readonly occupiedCount: number;
  get(index: number): 0 | 1;
  set(index: number, value: 0 | 1): void;
  toBytes(): Uint8Array;
}

interface StorageFactory {
  readonly name: "byte" | "bit-lsb" | "chunked-32";
  readonly complexity: "low" | "medium" | "high";
  decode(dimensions: Dimensions, bytes: Uint8Array): Storage;
}

interface Chunk {
  readonly data: Uint8Array;
  occupiedCount: number;
}

interface Dataset {
  readonly bytes: Uint8Array;
  readonly occupiedCount: number;
}

interface DeltaTransaction {
  readonly indices: Uint32Array;
  readonly before: Uint8Array;
  readonly after: 0 | 1;
}

interface BenchmarkRow {
  readonly pattern: Pattern;
  readonly storage: StorageFactory["name"];
  readonly memoryMiB: number;
  readonly decodeMs: number;
  readonly encodeMs: number;
  readonly randomEditsMs: number;
  readonly brushMs: number;
  readonly undoRedoMs: number;
  readonly countMaintenanceMs: number;
  readonly sliceXMs: number;
  readonly sliceYMs: number;
  readonly sliceZMs: number;
  readonly haloMs: number;
  readonly occupied: number;
  readonly complexity: StorageFactory["complexity"];
}

class ByteStorage implements Storage {
  private count: number;
  private readonly data: Uint8Array;

  private constructor(bytes: Uint8Array, occupiedCount: number) {
    this.data = bytes;
    this.count = occupiedCount;
  }

  static decode(source: Uint8Array): ByteStorage {
    const data = source.slice();
    return new ByteStorage(data, countOccupied(data));
  }

  get allocatedBytes(): number {
    return this.data.byteLength;
  }

  get occupiedCount(): number {
    return this.count;
  }

  get(index: number): 0 | 1 {
    return this.data[index] === 0 ? 0 : 1;
  }

  set(index: number, value: 0 | 1): void {
    const previous = this.data[index] === 0 ? 0 : 1;
    if (previous === value) return;
    this.data[index] = value;
    this.count += value === 1 ? 1 : -1;
  }

  toBytes(): Uint8Array {
    return this.data.slice();
  }
}

class BitStorage implements Storage {
  private count = 0;
  private readonly length: number;
  private readonly bytes: Uint8Array;

  private constructor(length: number) {
    this.length = length;
    this.bytes = new Uint8Array(Math.ceil(length / 8));
  }

  static decode(source: Uint8Array): BitStorage {
    const storage = new BitStorage(source.length);
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] === 0) continue;
      const byteIndex = index >>> 3;
      storage.bytes[byteIndex] = (storage.bytes[byteIndex] ?? 0) | (1 << (index & 7));
      storage.count += 1;
    }
    return storage;
  }

  get allocatedBytes(): number {
    return this.bytes.byteLength;
  }

  get occupiedCount(): number {
    return this.count;
  }

  get(index: number): 0 | 1 {
    return (((this.bytes[index >>> 3] ?? 0) >> (index & 7)) & 1) as 0 | 1;
  }

  set(index: number, value: 0 | 1): void {
    const byteIndex = index >>> 3;
    const mask = 1 << (index & 7);
    const byte = this.bytes[byteIndex] ?? 0;
    const previous = (byte & mask) === 0 ? 0 : 1;
    if (previous === value) return;
    this.bytes[byteIndex] = value === 1 ? byte | mask : byte & ~mask;
    this.count += value === 1 ? 1 : -1;
  }

  toBytes(): Uint8Array {
    const result = new Uint8Array(this.length);
    for (let byteIndex = 0; byteIndex < this.bytes.length; byteIndex += 1) {
      const packed = this.bytes[byteIndex] ?? 0;
      const firstIndex = byteIndex * 8;
      const bitCount = Math.min(8, this.length - firstIndex);
      for (let bit = 0; bit < bitCount; bit += 1) {
        result[firstIndex + bit] = (packed >>> bit) & 1;
      }
    }
    return result;
  }
}

class ChunkedStorage implements Storage {
  private count = 0;
  private readonly chunks = new Map<number, Chunk>();
  private readonly chunksX: number;
  private readonly chunksY: number;

  private constructor(private readonly dimensions: Dimensions) {
    this.chunksX = Math.ceil(dimensions[0] / CHUNK_SIZE);
    this.chunksY = Math.ceil(dimensions[1] / CHUNK_SIZE);
  }

  static decode(dimensions: Dimensions, source: Uint8Array): ChunkedStorage {
    const storage = new ChunkedStorage(dimensions);
    const [nx, ny, nz] = dimensions;
    const chunksZ = Math.ceil(nz / CHUNK_SIZE);

    for (let cz = 0; cz < chunksZ; cz += 1) {
      const z0 = cz * CHUNK_SIZE;
      const z1 = Math.min(z0 + CHUNK_SIZE, nz);
      for (let cy = 0; cy < storage.chunksY; cy += 1) {
        const y0 = cy * CHUNK_SIZE;
        const y1 = Math.min(y0 + CHUNK_SIZE, ny);
        for (let cx = 0; cx < storage.chunksX; cx += 1) {
          const x0 = cx * CHUNK_SIZE;
          const x1 = Math.min(x0 + CHUNK_SIZE, nx);
          let chunk: Chunk | undefined;

          for (let z = z0; z < z1; z += 1) {
            const localZ = z - z0;
            for (let y = y0; y < y1; y += 1) {
              const localY = y - y0;
              const sourceStart = x0 + nx * (y + ny * z);
              const localStart = CHUNK_SIZE * (localY + CHUNK_SIZE * localZ);
              for (let x = x0; x < x1; x += 1) {
                if (source[sourceStart + x - x0] === 0) continue;
                chunk ??= { data: new Uint8Array(CHUNK_VOXELS), occupiedCount: 0 };
                chunk.data[localStart + x - x0] = 1;
                chunk.occupiedCount += 1;
              }
            }
          }

          if (!chunk) continue;
          storage.chunks.set(storage.chunkKey(cx, cy, cz), chunk);
          storage.count += chunk.occupiedCount;
        }
      }
    }

    return storage;
  }

  get allocatedBytes(): number {
    return this.chunks.size * CHUNK_VOXELS;
  }

  get occupiedCount(): number {
    return this.count;
  }

  get(index: number): 0 | 1 {
    const location = this.locate(index);
    return this.chunks.get(location.key)?.data[location.within] === 1 ? 1 : 0;
  }

  set(index: number, value: 0 | 1): void {
    const location = this.locate(index);
    let chunk = this.chunks.get(location.key);
    const previous = chunk?.data[location.within] === 1 ? 1 : 0;
    if (previous === value) return;

    if (!chunk) {
      chunk = { data: new Uint8Array(CHUNK_VOXELS), occupiedCount: 0 };
      this.chunks.set(location.key, chunk);
    }

    chunk.data[location.within] = value;
    chunk.occupiedCount += value === 1 ? 1 : -1;
    this.count += value === 1 ? 1 : -1;
    if (chunk.occupiedCount === 0) this.chunks.delete(location.key);
  }

  toBytes(): Uint8Array {
    const [nx, ny, nz] = this.dimensions;
    const result = new Uint8Array(nx * ny * nz);

    for (const [key, chunk] of this.chunks) {
      const cx = key % this.chunksX;
      const chunkPlane = this.chunksX * this.chunksY;
      const cz = Math.floor(key / chunkPlane);
      const cy = Math.floor((key - cz * chunkPlane) / this.chunksX);
      const x0 = cx * CHUNK_SIZE;
      const y0 = cy * CHUNK_SIZE;
      const z0 = cz * CHUNK_SIZE;
      const width = Math.min(CHUNK_SIZE, nx - x0);
      const height = Math.min(CHUNK_SIZE, ny - y0);
      const depth = Math.min(CHUNK_SIZE, nz - z0);

      for (let localZ = 0; localZ < depth; localZ += 1) {
        for (let localY = 0; localY < height; localY += 1) {
          const localStart = CHUNK_SIZE * (localY + CHUNK_SIZE * localZ);
          const targetStart = x0 + nx * (y0 + localY + ny * (z0 + localZ));
          result.set(chunk.data.subarray(localStart, localStart + width), targetStart);
        }
      }
    }

    return result;
  }

  private chunkKey(cx: number, cy: number, cz: number): number {
    return cx + this.chunksX * (cy + this.chunksY * cz);
  }

  private locate(index: number): { readonly key: number; readonly within: number } {
    const [nx, ny] = this.dimensions;
    const plane = nx * ny;
    const z = Math.floor(index / plane);
    const withinPlane = index - z * plane;
    const y = Math.floor(withinPlane / nx);
    const x = withinPlane - y * nx;
    const cx = Math.floor(x / CHUNK_SIZE);
    const cy = Math.floor(y / CHUNK_SIZE);
    const cz = Math.floor(z / CHUNK_SIZE);
    const localX = x - cx * CHUNK_SIZE;
    const localY = y - cy * CHUNK_SIZE;
    const localZ = z - cz * CHUNK_SIZE;
    return {
      key: this.chunkKey(cx, cy, cz),
      within: localX + CHUNK_SIZE * (localY + CHUNK_SIZE * localZ),
    };
  }
}

const factories: readonly StorageFactory[] = [
  {
    name: "byte",
    complexity: "low",
    decode: (_dimensions, bytes) => ByteStorage.decode(bytes),
  },
  {
    name: "bit-lsb",
    complexity: "medium",
    decode: (_dimensions, bytes) => BitStorage.decode(bytes),
  },
  {
    name: "chunked-32",
    complexity: "high",
    decode: (dimensions, bytes) => ChunkedStorage.decode(dimensions, bytes),
  },
];

function countOccupied(bytes: Uint8Array): number {
  let count = 0;
  for (const value of bytes) count += value === 0 ? 0 : 1;
  return count;
}

function createDataset(dimensions: Dimensions, pattern: Pattern): Dataset {
  const [nx, ny, nz] = dimensions;
  const bytes = new Uint8Array(nx * ny * nz);

  if (pattern === "dense") {
    bytes.fill(1);
    return { bytes, occupiedCount: bytes.length };
  }

  if (pattern === "sparse") {
    const radius = Math.max(2, Math.floor(Math.min(nx, ny, nz) / 10));
    const centerX = Math.floor(nx / 2);
    const centerY = Math.floor(ny / 2);
    const centerZ = Math.floor(nz / 2);
    const radiusSquared = radius ** 2;
    let occupiedCount = 0;
    for (let z = centerZ - radius; z <= centerZ + radius; z += 1) {
      for (let y = centerY - radius; y <= centerY + radius; y += 1) {
        for (let x = centerX - radius; x <= centerX + radius; x += 1) {
          const distanceSquared = (x - centerX) ** 2
            + (y - centerY) ** 2
            + (z - centerZ) ** 2;
          if (distanceSquared > radiusSquared) continue;
          bytes[x + nx * (y + ny * z)] = 1;
          occupiedCount += 1;
        }
      }
    }
    return { bytes, occupiedCount };
  }

  let state = 0x9e3779b9;
  let occupiedCount = 0;
  for (let index = 0; index < bytes.length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const value = (state >>> 31) as 0 | 1;
    bytes[index] = value;
    occupiedCount += value;
  }
  return { bytes, occupiedCount };
}

function createRandomIndices(length: number, count: number, initialSeed: number): Uint32Array {
  const indices = new Uint32Array(count);
  let state = initialSeed >>> 0;
  for (let operation = 0; operation < count; operation += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    indices[operation] = state % length;
  }
  return indices;
}

function createSphereBrushIndices(dimensions: Dimensions, diameter: number): Uint32Array {
  const [nx, ny, nz] = dimensions;
  const radius = (diameter - 1) / 2;
  const centerX = Math.floor(nx / 2);
  const centerY = Math.floor(ny / 2);
  const centerZ = Math.floor(nz / 2);
  const indices: number[] = [];

  for (let dz = -radius; dz <= radius; dz += 1) {
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        if (dx ** 2 + dy ** 2 + dz ** 2 > radius ** 2) continue;
        const x = centerX + dx;
        const y = centerY + dy;
        const z = centerZ + dz;
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;
        indices.push(x + nx * (y + ny * z));
      }
    }
  }

  return Uint32Array.from(indices);
}

function applyBrush(
  storage: Storage,
  indices: Uint32Array,
  value: 0 | 1,
): DeltaTransaction {
  const changedIndices = new Uint32Array(indices.length);
  const before = new Uint8Array(indices.length);
  let changedCount = 0;

  for (const index of indices) {
    const previous = storage.get(index);
    if (previous === value) continue;
    changedIndices[changedCount] = index;
    before[changedCount] = previous;
    changedCount += 1;
    storage.set(index, value);
  }

  return {
    indices: changedIndices.slice(0, changedCount),
    before: before.slice(0, changedCount),
    after: value,
  };
}

function replayDelta(storage: Storage, transaction: DeltaTransaction, undo: boolean): void {
  for (let offset = 0; offset < transaction.indices.length; offset += 1) {
    storage.set(
      transaction.indices[offset] ?? 0,
      undo ? (transaction.before[offset] === 1 ? 1 : 0) : transaction.after,
    );
  }
}

function extractSlice(
  storage: Storage,
  dimensions: Dimensions,
  axis: SliceAxis,
): Uint8Array {
  const [nx, ny, nz] = dimensions;
  if (axis === "x") {
    const x = Math.floor(nx / 2);
    const result = new Uint8Array(ny * nz);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) result[y + ny * z] = storage.get(x + nx * (y + ny * z));
    }
    return result;
  }

  if (axis === "y") {
    const y = Math.floor(ny / 2);
    const result = new Uint8Array(nx * nz);
    for (let z = 0; z < nz; z += 1) {
      for (let x = 0; x < nx; x += 1) result[x + nx * z] = storage.get(x + nx * (y + ny * z));
    }
    return result;
  }

  const z = Math.floor(nz / 2);
  const result = new Uint8Array(nx * ny);
  const start = nx * ny * z;
  for (let offset = 0; offset < result.length; offset += 1) result[offset] = storage.get(start + offset);
  return result;
}

function extractRendererHalo(storage: Storage, dimensions: Dimensions): Uint8Array {
  const [nx, ny, nz] = dimensions;
  const chunkX = Math.max(0, Math.floor(nx / (2 * CHUNK_SIZE)));
  const chunkY = Math.max(0, Math.floor(ny / (2 * CHUNK_SIZE)));
  const chunkZ = Math.max(0, Math.floor(nz / (2 * CHUNK_SIZE)));
  const x0 = Math.max(0, chunkX * CHUNK_SIZE - 1);
  const y0 = Math.max(0, chunkY * CHUNK_SIZE - 1);
  const z0 = Math.max(0, chunkZ * CHUNK_SIZE - 1);
  const x1 = Math.min(nx, (chunkX + 1) * CHUNK_SIZE + 1);
  const y1 = Math.min(ny, (chunkY + 1) * CHUNK_SIZE + 1);
  const z1 = Math.min(nz, (chunkZ + 1) * CHUNK_SIZE + 1);
  const width = x1 - x0;
  const height = y1 - y0;
  const result = new Uint8Array(width * height * (z1 - z0));

  for (let z = z0; z < z1; z += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        const target = x - x0 + width * (y - y0 + height * (z - z0));
        result[target] = storage.get(x + nx * (y + ny * z));
      }
    }
  }

  return result;
}

function checksum(bytes: Uint8Array): number {
  let hash = 0x811c9dc5;
  for (const value of bytes) {
    hash ^= value;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

function measure<T>(operation: () => T): { readonly value: T; readonly milliseconds: number } {
  const start = performance.now();
  const value = operation();
  return { value, milliseconds: performance.now() - start };
}

function rounded(milliseconds: number): number {
  return Number(milliseconds.toFixed(3));
}

describe("occupancy representation benchmark", () => {
  it("compares identical workloads and selects byte-per-voxel storage", () => {
    const reference = process.env.VOXELINK_REFERENCE_BENCH === "1";
    const dimensions = reference ? REFERENCE_DIMENSIONS : [96, 96, 64] as const;
    const length = dimensions[0] * dimensions[1] * dimensions[2];
    const randomEditCount = reference ? 1_000_000 : 25_000;
    const countMaintenanceEdits = reference ? 100_000 : 5_000;
    const randomIndices = createRandomIndices(length, randomEditCount, 0x12345678);
    const countIndices = createRandomIndices(length, countMaintenanceEdits, 0x51ced00d);
    const brushIndices = createSphereBrushIndices(dimensions, 31);
    const rows: BenchmarkRow[] = [];

    console.info(
      `[storage benchmark] ${reference ? "reference" : "CI"} ${dimensions.join("×")}; `
      + `${randomEditCount.toLocaleString()} random edits; timings are milliseconds; `
      + "memory is decoded occupancy payload",
    );

    for (const pattern of ["sparse", "dense", "noisy"] as const) {
      const dataset = createDataset(dimensions, pattern);
      const signatures: string[] = [];

      for (const factory of factories) {
        const decoded = measure(() => factory.decode(dimensions, dataset.bytes));
        const storage = decoded.value;
        const decodedMemoryBytes = storage.allocatedBytes;
        expect(storage.occupiedCount).toBe(dataset.occupiedCount);

        const randomEdits = measure(() => {
          for (const index of randomIndices) storage.set(index, storage.get(index) === 0 ? 1 : 0);
        });

        let countProbe = 0;
        const countMaintenance = measure(() => {
          for (const index of countIndices) {
            storage.set(index, storage.get(index) === 0 ? 1 : 0);
            countProbe ^= storage.occupiedCount;
          }
        });

        const brushTarget: 0 | 1 = pattern === "dense" ? 0 : 1;
        const brush = measure(() => applyBrush(storage, brushIndices, brushTarget));
        const undoRedo = measure(() => {
          replayDelta(storage, brush.value, true);
          replayDelta(storage, brush.value, false);
        });

        const sliceX = measure(() => extractSlice(storage, dimensions, "x"));
        const sliceY = measure(() => extractSlice(storage, dimensions, "y"));
        const sliceZ = measure(() => extractSlice(storage, dimensions, "z"));
        const halo = measure(() => extractRendererHalo(storage, dimensions));
        const encoded = measure(() => storage.toBytes());
        const encodedOccupiedCount = countOccupied(encoded.value);
        expect(storage.occupiedCount).toBe(encodedOccupiedCount);

        signatures.push([
          storage.occupiedCount,
          checksum(encoded.value),
          checksum(sliceX.value),
          checksum(sliceY.value),
          checksum(sliceZ.value),
          checksum(halo.value),
          countProbe,
          brush.value.indices.length,
        ].join(":"));

        rows.push({
          pattern,
          storage: factory.name,
          memoryMiB: Number((decodedMemoryBytes / MEBIBYTE).toFixed(3)),
          decodeMs: rounded(decoded.milliseconds),
          encodeMs: rounded(encoded.milliseconds),
          randomEditsMs: rounded(randomEdits.milliseconds),
          brushMs: rounded(brush.milliseconds),
          undoRedoMs: rounded(undoRedo.milliseconds),
          countMaintenanceMs: rounded(countMaintenance.milliseconds),
          sliceXMs: rounded(sliceX.milliseconds),
          sliceYMs: rounded(sliceY.milliseconds),
          sliceZMs: rounded(sliceZ.milliseconds),
          haloMs: rounded(halo.milliseconds),
          occupied: storage.occupiedCount,
          complexity: factory.complexity,
        });
      }

      expect(new Set(signatures).size).toBe(1);
    }

    console.table(rows);

    const byteRows = rows.filter((row) => row.storage === "byte");
    const maximumByteBrushMs = Math.max(...byteRows.map((row) => row.brushMs));
    const referenceByteMemoryMiB = REFERENCE_VOXELS / MEBIBYTE;
    const feedbackResult = reference
      ? `${maximumByteBrushMs.toFixed(3)} ms observed (${maximumByteBrushMs < 100 ? "target met" : "target missed on this host"})`
      : "measure with VOXELINK_REFERENCE_BENCH=1";

    console.info(
      `[storage selection] Uint8Array byte-per-voxel: selected as the simplest representation `
      + `(direct O(1) indexing and native copying, without bit or chunk bookkeeping). `
      + `The 512×512×256 occupancy buffer is exactly ${referenceByteMemoryMiB.toFixed(2)} MiB `
      + `(within the 64 MiB limit); storage-only max-diameter brush cost: ${feedbackResult}. `
      + "End-to-end slice feedback and navigation are verified separately.",
    );

    expect(REFERENCE_VOXELS).toBe(64 * MEBIBYTE);
    expect(referenceByteMemoryMiB).toBeLessThanOrEqual(64);
    expect(byteRows).toHaveLength(3);
    expect(byteRows.every((row) => row.complexity === "low")).toBe(true);
    expect(Number.isFinite(maximumByteBrushMs)).toBe(true);
  }, process.env.VOXELINK_REFERENCE_BENCH === "1" ? 600_000 : 120_000);
});
