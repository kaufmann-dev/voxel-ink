import {
  createMatrix4,
  IDENTITY_AFFINE,
  invertAffine,
  spacingFromAffine,
  type Matrix4,
  type Vec3,
} from "./affine";

export type Dimensions = readonly [number, number, number];
export type GridCoordinate = readonly [number, number, number];
export type BinaryVoxel = 0 | 1;
export type SliceAxis = 0 | 1 | 2;
export type NormalizationState = "binary" | "normalized-non-binary";

export const MAX_VOXEL_COUNT = 0x7fffffff;

export interface VolumeDocumentOptions<TSaveState = unknown> {
  dimensions: Dimensions;
  occupancy?: Uint8Array;
  /** Count supplied only for occupancy already normalized by a trusted codec worker. */
  occupiedCount?: number;
  indexToWorld?: Matrix4 | readonly number[];
  units?: string;
  sourceFormat?: string;
  saveState?: TSaveState;
  cursor?: GridCoordinate;
  normalizationState?: NormalizationState;
}

export interface VoxelWriteResult {
  readonly indices: Uint32Array;
  readonly beforeRevision: number;
  readonly afterRevision: number;
}

export interface VolumeSlice {
  readonly axis: SliceAxis;
  readonly slice: number;
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;
}

export function voxelCount(dimensions: Dimensions): number {
  let count = 1;
  for (const dimension of dimensions) {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new Error("Volume dimensions must be positive safe integers");
    }
    count *= dimension;
    if (!Number.isSafeInteger(count) || count > MAX_VOXEL_COUNT) {
      throw new Error(`Volume exceeds the ${MAX_VOXEL_COUNT} voxel limit`);
    }
  }
  return count;
}

export function isCoordinateInBounds(
  dimensions: Dimensions,
  coordinate: GridCoordinate,
): boolean {
  const [nx, ny, nz] = dimensions;
  const [x, y, z] = coordinate;
  return (
    Number.isInteger(x) &&
    Number.isInteger(y) &&
    Number.isInteger(z) &&
    x >= 0 &&
    y >= 0 &&
    z >= 0 &&
    x < nx &&
    y < ny &&
    z < nz
  );
}

export function linearIndex(
  dimensions: Dimensions,
  coordinate: GridCoordinate,
): number {
  if (!isCoordinateInBounds(dimensions, coordinate)) {
    throw new RangeError("Voxel coordinate is outside the volume");
  }
  const [nx, ny] = dimensions;
  const [x, y, z] = coordinate;
  return x + nx * (y + ny * z);
}

export function coordinateFromIndex(
  dimensions: Dimensions,
  index: number,
): GridCoordinate {
  const count = voxelCount(dimensions);
  if (!Number.isInteger(index) || index < 0 || index >= count) {
    throw new RangeError("Voxel index is outside the volume");
  }
  const [nx, ny] = dimensions;
  const plane = nx * ny;
  const z = Math.floor(index / plane);
  const inPlane = index - z * plane;
  const y = Math.floor(inPlane / nx);
  return [inPlane - y * nx, y, z];
}

function validateBinaryOccupancy(
  data: Uint8Array,
  expectedLength: number,
  trustedOccupiedCount?: number,
): number {
  if (data.length !== expectedLength) {
    throw new Error(
      `Occupancy has ${data.length} voxels; expected ${expectedLength}`,
    );
  }

  if (trustedOccupiedCount !== undefined) {
    if (
      !Number.isSafeInteger(trustedOccupiedCount) ||
      trustedOccupiedCount < 0 ||
      trustedOccupiedCount > expectedLength
    ) {
      throw new Error("Trusted occupied count is outside the occupancy bounds");
    }
    return trustedOccupiedCount;
  }

  let occupiedCount = 0;
  for (const value of data) {
    if (value !== 0 && value !== 1) {
      throw new Error("Occupancy must contain only binary values 0 and 1");
    }
    occupiedCount += value;
  }
  return occupiedCount;
}

function validateCursor(
  dimensions: Dimensions,
  cursor: GridCoordinate | undefined,
): GridCoordinate {
  const resolved =
    cursor ??
    dimensions.map((dimension) => Math.floor((dimension - 1) / 2)) as unknown as GridCoordinate;
  if (!isCoordinateInBounds(dimensions, resolved)) {
    throw new RangeError("Cursor is outside the volume");
  }
  return [...resolved] as GridCoordinate;
}

export class VolumeDocument<TSaveState = unknown> {
  readonly dimensions: Dimensions;
  readonly occupancy: Uint8Array;
  readonly indexToWorld: Matrix4;
  readonly worldToIndex: Matrix4;
  readonly spacing: Vec3;
  readonly units: string;
  readonly sourceFormat: string;
  readonly saveState: TSaveState | undefined;
  readonly normalizationState: NormalizationState;

  cursor: GridCoordinate;
  occupiedCount: number;
  revision: number;
  savedRevision: number;

  private nextRevision: number;

  constructor(options: VolumeDocumentOptions<TSaveState>) {
    voxelCount(options.dimensions);
    this.dimensions = Object.freeze([...options.dimensions]) as unknown as Dimensions;

    const data = options.occupancy ?? new Uint8Array(voxelCount(this.dimensions));
    this.occupiedCount = validateBinaryOccupancy(
      data,
      voxelCount(this.dimensions),
      options.occupiedCount,
    );
    this.occupancy = data;

    this.indexToWorld = options.indexToWorld
      ? createMatrix4(options.indexToWorld)
      : IDENTITY_AFFINE;
    this.worldToIndex = invertAffine(this.indexToWorld);
    this.spacing = spacingFromAffine(this.indexToWorld);
    this.units = options.units ?? "unknown";
    this.sourceFormat = options.sourceFormat ?? "unknown";
    this.saveState = options.saveState;
    this.cursor = validateCursor(this.dimensions, options.cursor);
    this.normalizationState = options.normalizationState ?? "binary";

    this.revision = this.normalizationState === "normalized-non-binary" ? 1 : 0;
    this.savedRevision = 0;
    this.nextRevision = this.revision + 1;
  }

  get dirty(): boolean {
    return this.revision !== this.savedRevision;
  }

  get(coordinate: GridCoordinate): BinaryVoxel {
    return this.getIndex(linearIndex(this.dimensions, coordinate));
  }

  getIndex(index: number): BinaryVoxel {
    this.assertIndex(index);
    return this.occupancy[index] as BinaryVoxel;
  }

  setCursor(cursor: GridCoordinate): void {
    this.cursor = validateCursor(this.dimensions, cursor);
  }

  writeIndices(
    indices: Iterable<number>,
    value: BinaryVoxel,
  ): VoxelWriteResult {
    if (value !== 0 && value !== 1) {
      throw new Error("Voxel value must be 0 or 1");
    }

    const candidates = Array.from(indices);
    candidates.forEach((index) => this.assertIndex(index));
    const changed: number[] = [];
    const beforeRevision = this.revision;

    for (const index of candidates) {
      if (this.occupancy[index] === value) {
        continue;
      }
      this.occupancy[index] = value;
      this.occupiedCount += value === 1 ? 1 : -1;
      changed.push(index);
    }

    if (changed.length > 0) {
      this.revision = this.allocateRevision();
    }

    return {
      indices: Uint32Array.from(changed),
      beforeRevision,
      afterRevision: this.revision,
    };
  }

  applyHistoryState(
    indices: Uint32Array,
    value: BinaryVoxel,
    revision: number,
  ): void {
    indices.forEach((index) => this.assertIndex(index));
    if (!Number.isSafeInteger(revision) || revision < 0) {
      throw new Error("Revision must be a non-negative safe integer");
    }

    for (const index of indices) {
      const previous = this.occupancy[index] as BinaryVoxel;
      if (previous === value) {
        continue;
      }
      this.occupancy[index] = value;
      this.occupiedCount += value === 1 ? 1 : -1;
    }
    this.revision = revision;
    this.nextRevision = Math.max(this.nextRevision, revision + 1);
  }

  markSaved(revision: number = this.revision): void {
    if (!Number.isSafeInteger(revision) || revision < 0 || revision >= this.nextRevision) {
      throw new Error("Saved revision is not a known document revision");
    }
    this.savedRevision = revision;
  }

  extractSlice(axis: SliceAxis, slice: number): VolumeSlice {
    if (!Number.isInteger(axis) || axis < 0 || axis > 2) {
      throw new RangeError("Slice axis must be 0, 1, or 2");
    }
    if (!Number.isInteger(slice) || slice < 0 || slice >= this.dimensions[axis]) {
      throw new RangeError("Slice is outside the volume");
    }

    const [nx, ny, nz] = this.dimensions;
    const width = axis === 0 ? ny : nx;
    const height = axis === 2 ? ny : nz;
    const data = new Uint8Array(width * height);

    let output = 0;
    if (axis === 0) {
      for (let z = 0; z < nz; z += 1) {
        for (let y = 0; y < ny; y += 1) {
          data[output] = this.occupancy[slice + nx * (y + ny * z)]!;
          output += 1;
        }
      }
    } else if (axis === 1) {
      for (let z = 0; z < nz; z += 1) {
        const rowStart = nx * (slice + ny * z);
        data.set(this.occupancy.subarray(rowStart, rowStart + nx), output);
        output += nx;
      }
    } else {
      data.set(this.occupancy.subarray(nx * ny * slice, nx * ny * (slice + 1)));
    }

    return { axis, slice, width, height, data };
  }

  copyOccupancy(): Uint8Array {
    return this.occupancy.slice();
  }

  private allocateRevision(): number {
    const revision = this.nextRevision;
    this.nextRevision += 1;
    return revision;
  }

  private assertIndex(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.occupancy.length) {
      throw new RangeError("Voxel index is outside the volume");
    }
  }
}
