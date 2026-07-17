import { CodecError, type VolumeDimensions } from "./types";

export const MAX_VOXELS = 512 * 512 * 512;
export const MAX_AXIS_LENGTH = 8_192;
export const MAX_SLICE_VOXELS = 4_096 * 4_096;
export const VALIDATED_CHUNK_EDGE = 32;
export const MAX_RENDER_CHUNKS = 16_384;

export const checkedVoxelCount = (dimensions: VolumeDimensions): number => {
  let count = 1;
  for (const [axis, dimension] of dimensions.entries()) {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new CodecError(
        "invalid-dimensions",
        "Volume dimensions must be positive safe integers.",
      );
    }
    if (dimension > MAX_AXIS_LENGTH) {
      throw new CodecError(
        "size-limit",
        `Volume axis ${"xyz"[axis]} exceeds the ${MAX_AXIS_LENGTH.toLocaleString()} voxel axis limit.`,
      );
    }
    count *= dimension;
    if (!Number.isSafeInteger(count) || count > MAX_VOXELS) {
      throw new CodecError(
        "size-limit",
        `Volume exceeds the ${MAX_VOXELS.toLocaleString()} voxel limit.`,
      );
    }
  }

  const [nx, ny, nz] = dimensions;
  const sliceAreas = [
    ["XY", nx * ny],
    ["XZ", nx * nz],
    ["YZ", ny * nz],
  ] as const;
  const oversizedSlice = sliceAreas.find(
    ([, area]) => area > MAX_SLICE_VOXELS,
  );
  if (oversizedSlice) {
    throw new CodecError(
      "size-limit",
      `Volume ${oversizedSlice[0]} slices exceed the ${MAX_SLICE_VOXELS.toLocaleString()} voxel slice limit.`,
    );
  }

  const chunkCount = dimensions.reduce(
    (product, dimension) =>
      product * Math.ceil(dimension / VALIDATED_CHUNK_EDGE),
    1,
  );
  if (chunkCount > MAX_RENDER_CHUNKS) {
    throw new CodecError(
      "size-limit",
      `Volume requires ${chunkCount.toLocaleString()} render chunks, exceeding the ${MAX_RENDER_CHUNKS.toLocaleString()} chunk limit.`,
    );
  }

  return count;
};

export const toUint8Array = (
  value: Uint8Array | ArrayBuffer,
): Uint8Array =>
  value instanceof Uint8Array ? value : new Uint8Array(value);

export const copyUint8Array = (value: Uint8Array): Uint8Array =>
  new Uint8Array(value);

export const sliceBytes = (
  bytes: Uint8Array,
  start: number,
  end = bytes.length,
): Uint8Array => bytes.slice(start, end);

export const dataView = (bytes: Uint8Array): DataView =>
  new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

export const concatBytes = (...parts: readonly Uint8Array[]): Uint8Array => {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
};

export const decodeFixedString = (
  bytes: Uint8Array,
  offset: number,
  length: number,
): string => {
  let end = offset;
  const maximum = Math.min(bytes.length, offset + length);
  while (end < maximum && bytes[end] !== 0) {
    end += 1;
  }
  return new TextDecoder("latin1").decode(bytes.subarray(offset, end));
};

export const extensionOf = (name: string): string => {
  const normalized = name.trim().toLowerCase();
  return normalized.endsWith(".nii.gz")
    ? ".nii.gz"
    : normalized.slice(normalized.lastIndexOf("."));
};

export const requireBytes = (
  bytes: Uint8Array,
  offset: number,
  length: number,
  context: string,
): void => {
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(length) ||
    offset < 0 ||
    length < 0 ||
    offset + length > bytes.length
  ) {
    throw new CodecError(
      "invalid-payload",
      `${context} extends beyond the end of the file.`,
    );
  }
};

export const isAllZero = (bytes: Uint8Array): boolean => {
  for (const value of bytes) {
    if (value !== 0) return false;
  }
  return true;
};
