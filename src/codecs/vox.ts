import { createMatrix4 } from "../core/affine";
import { geometryFromAffine } from "./affine";
import {
  checkedVoxelCount,
  concatBytes,
  dataView,
  requireBytes,
} from "./binary";
import {
  CodecError,
  occupancyValueAt,
  type DecodedVolume,
  type EncodableVolume,
  type SourceVariant,
  type VolumeCodec,
  type VolumeDimensions,
  type VoxSaveState,
} from "./types";

const VOX_VERSION = 150;
const MAX_VOX_MODEL_VOXELS = 256 * 256 * 256;
const textDecoder = new TextDecoder("ascii");
const textEncoder = new TextEncoder();

interface ChunkBounds {
  readonly id: string;
  readonly contentStart: number;
  readonly contentEnd: number;
  readonly childrenStart: number;
  readonly childrenEnd: number;
}

const readChunk = (
  bytes: Uint8Array,
  offset: number,
  containingEnd: number,
): ChunkBounds => {
  requireBytes(bytes, offset, 12, "The VOX chunk header");
  const view = dataView(bytes);
  const id = textDecoder.decode(bytes.subarray(offset, offset + 4));
  const contentSize = view.getInt32(offset + 4, true);
  const childrenSize = view.getInt32(offset + 8, true);
  if (contentSize < 0 || childrenSize < 0) {
    throw new CodecError("invalid-payload", `VOX chunk '${id}' has a negative size.`);
  }
  const contentStart = offset + 12;
  const contentEnd = contentStart + contentSize;
  const childrenStart = contentEnd;
  const childrenEnd = childrenStart + childrenSize;
  if (
    !Number.isSafeInteger(childrenEnd) ||
    childrenEnd > containingEnd ||
    childrenEnd > bytes.length
  ) {
    throw new CodecError("invalid-payload", `VOX chunk '${id}' exceeds its container.`);
  }
  return { id, contentStart, contentEnd, childrenStart, childrenEnd };
};

interface VoxModelData {
  dimensions: VolumeDimensions | null;
  voxels: Uint8Array | null;
  packCount: number | null;
  sawPack: boolean;
  sizeCount: number;
  voxelCount: number;
}

const inspectChunk = (
  bytes: Uint8Array,
  chunk: ChunkBounds,
  model: VoxModelData,
): void => {
  const view = dataView(bytes);
  if (chunk.id === "PACK") {
    if (model.sawPack || chunk.contentEnd - chunk.contentStart !== 4) {
      throw new CodecError("invalid-payload", "The VOX PACK chunk is invalid.");
    }
    model.sawPack = true;
    model.packCount = view.getInt32(chunk.contentStart, true);
    if (model.packCount !== 1) {
      throw new CodecError(
        "unsupported-multi-model",
        "Only single-model VOX files are supported.",
      );
    }
  } else if (chunk.id === "SIZE") {
    model.sizeCount += 1;
    if (model.sizeCount > 1) {
      throw new CodecError(
        "unsupported-multi-model",
        "Only single-model VOX files are supported.",
      );
    }
    if (chunk.contentEnd - chunk.contentStart !== 12) {
      throw new CodecError("invalid-payload", "The VOX SIZE chunk is invalid.");
    }
    const dimensions = [
      view.getInt32(chunk.contentStart, true),
      view.getInt32(chunk.contentStart + 4, true),
      view.getInt32(chunk.contentStart + 8, true),
    ] as const;
    if (
      dimensions.some(
        (dimension) =>
          !Number.isSafeInteger(dimension) || dimension <= 0 || dimension > 256,
      )
    ) {
      throw new CodecError(
        "invalid-dimensions",
        "VOX model dimensions must be between 1 and 256.",
      );
    }
    checkedVoxelCount(dimensions);
    model.dimensions = dimensions;
  } else if (chunk.id === "XYZI") {
    model.voxelCount += 1;
    if (model.voxelCount > 1) {
      throw new CodecError(
        "unsupported-multi-model",
        "Only single-model VOX files are supported.",
      );
    }
    if (chunk.contentEnd - chunk.contentStart < 4) {
      throw new CodecError("invalid-payload", "The VOX XYZI chunk is truncated.");
    }
    const count = view.getUint32(chunk.contentStart, true);
    if (count > MAX_VOX_MODEL_VOXELS) {
      throw new CodecError(
        "size-limit",
        "The VOX model contains more voxel entries than a 256³ model can hold.",
      );
    }
    const expectedSize = 4 + count * 4;
    if (
      !Number.isSafeInteger(expectedSize) ||
      chunk.contentEnd - chunk.contentStart !== expectedSize
    ) {
      throw new CodecError("invalid-payload", "The VOX XYZI chunk size is invalid.");
    }
    model.voxels = bytes.subarray(chunk.contentStart + 4, chunk.contentEnd);
  }
};

export const decodeVox = (bytes: Uint8Array): DecodedVolume => {
  requireBytes(bytes, 0, 20, "The VOX file header");
  if (textDecoder.decode(bytes.subarray(0, 4)) !== "VOX ") {
    throw new CodecError("invalid-format", "The VOX signature is invalid.");
  }
  const view = dataView(bytes);
  const version = view.getInt32(4, true);
  if (version !== VOX_VERSION) {
    throw new CodecError(
      "unsupported-variant",
      `VOX version ${version} is not supported; expected version ${VOX_VERSION}.`,
    );
  }
  const main = readChunk(bytes, 8, bytes.length);
  if (
    main.id !== "MAIN" ||
    main.contentStart !== main.contentEnd ||
    main.childrenEnd !== bytes.length
  ) {
    throw new CodecError("invalid-format", "The VOX MAIN chunk is invalid.");
  }

  const model: VoxModelData = {
    dimensions: null,
    voxels: null,
    packCount: null,
    sawPack: false,
    sizeCount: 0,
    voxelCount: 0,
  };
  const containers: { cursor: number; end: number; depth: number }[] = [
    { cursor: main.childrenStart, end: main.childrenEnd, depth: 0 },
  ];
  while (containers.length > 0) {
    const container = containers[containers.length - 1]!;
    if (container.cursor === container.end) {
      containers.pop();
      continue;
    }
    if (container.cursor > container.end) {
      throw new CodecError("invalid-payload", "VOX child chunks overlap their container.");
    }
    const chunk = readChunk(bytes, container.cursor, container.end);
    container.cursor = chunk.childrenEnd;
    inspectChunk(bytes, chunk, model);
    if (chunk.childrenStart < chunk.childrenEnd) {
      if (container.depth >= 63) {
        throw new CodecError("invalid-payload", "VOX chunk nesting is too deep.");
      }
      containers.push({
        cursor: chunk.childrenStart,
        end: chunk.childrenEnd,
        depth: container.depth + 1,
      });
    }
  }

  if (model.dimensions === null || model.voxels === null) {
    throw new CodecError(
      "invalid-payload",
      "A single-model VOX file requires one SIZE and one XYZI chunk.",
    );
  }
  const dimensions = model.dimensions;
  const voxelBytes: Uint8Array = model.voxels;
  const [nx, ny, nz] = dimensions;
  const count = checkedVoxelCount(dimensions);
  if (voxelBytes.length / 4 > count) {
    throw new CodecError(
      "invalid-payload",
      "The VOX XYZI chunk has more entries than the declared model can contain.",
    );
  }
  const occupancy = new Uint8Array(count);
  let converted = false;
  for (let offset = 0; offset < voxelBytes.length; offset += 4) {
    const x: number = voxelBytes[offset]!;
    const y: number = voxelBytes[offset + 1]!;
    const z: number = voxelBytes[offset + 2]!;
    const color: number = voxelBytes[offset + 3]!;
    if (x >= nx || y >= ny || z >= nz) {
      throw new CodecError(
        "invalid-payload",
        `VOX coordinate (${x}, ${y}, ${z}) is outside SIZE.`,
      );
    }
    if (color === 0) {
      throw new CodecError("invalid-payload", "VOX palette index 0 is invalid.");
    }
    const index = x + nx * (y + ny * z);
    if (occupancy[index] !== 0) {
      throw new CodecError("invalid-payload", "The VOX XYZI chunk contains duplicate coordinates.");
    }
    occupancy[index] = 1;
    if (color !== 1) converted = true;
  }
  const saveState: VoxSaveState = {
    kind: "vox",
    variant: ".vox",
    version,
    dimensions,
  };
  return {
    dimensions,
    occupancy,
    occupiedCount: voxelBytes.length / 4,
    geometry: geometryFromAffine(
      createMatrix4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      [null, null, null],
    ),
    sourceFormat: "vox",
    sourceVariant: ".vox",
    saveState,
    normalization: { convertedNonBinary: converted },
  };
};

const makeChunk = (
  id: string,
  content: Uint8Array,
  children: Uint8Array = new Uint8Array(0),
): Uint8Array => {
  if (id.length !== 4) throw new Error("VOX chunk ids must contain four bytes.");
  const bytes = new Uint8Array(12 + content.length + children.length);
  bytes.set(textEncoder.encode(id), 0);
  const view = dataView(bytes);
  view.setInt32(4, content.length, true);
  view.setInt32(8, children.length, true);
  bytes.set(content, 12);
  bytes.set(children, 12 + content.length);
  return bytes;
};

export const encodeVox = (document: EncodableVolume): Uint8Array => {
  if (document.saveState.kind !== "vox") {
    throw new CodecError("invalid-format", "A VOX save requires VOX source metadata.");
  }
  const state = document.saveState;
  if (document.dimensions.some((value, axis) => value !== state.dimensions[axis])) {
    throw new CodecError(
      "invalid-dimensions",
      "VoxelInk does not resize VOX models during save.",
    );
  }
  if (document.dimensions.some((dimension) => dimension <= 0 || dimension > 256)) {
    throw new CodecError("invalid-dimensions", "VOX dimensions must be between 1 and 256.");
  }
  const count = checkedVoxelCount(document.dimensions);
  if (document.occupancy.length !== count) {
    throw new CodecError(
      "invalid-payload",
      "The occupancy length does not match the VOX dimensions.",
    );
  }
  const [nx, ny, nz] = document.dimensions;
  let occupiedCount = 0;
  for (let index = 0; index < count; index += 1) {
    occupiedCount += occupancyValueAt(document.occupancy, index);
  }
  const sizeContent = new Uint8Array(12);
  const sizeView = dataView(sizeContent);
  sizeView.setInt32(0, nx, true);
  sizeView.setInt32(4, ny, true);
  sizeView.setInt32(8, nz, true);
  const xyziContent = new Uint8Array(4 + occupiedCount * 4);
  dataView(xyziContent).setUint32(0, occupiedCount, true);
  let output = 4;
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const index = x + nx * (y + ny * z);
        if (occupancyValueAt(document.occupancy, index) === 0) continue;
        xyziContent[output] = x;
        xyziContent[output + 1] = y;
        xyziContent[output + 2] = z;
        xyziContent[output + 3] = 1;
        output += 4;
      }
    }
  }
  const children = concatBytes(
    makeChunk("SIZE", sizeContent),
    makeChunk("XYZI", xyziContent),
  );
  const header = new Uint8Array(8);
  header.set(textEncoder.encode("VOX "), 0);
  dataView(header).setInt32(4, state.version, true);
  return concatBytes(header, makeChunk("MAIN", new Uint8Array(), children));
};

export const voxCodec: VolumeCodec = {
  format: "vox",
  extensions: [".vox"],
  decode(bytes, extension) {
    if (extension !== ".vox") {
      throw new CodecError("unsupported-variant", `VOX cannot decode ${extension}.`);
    }
    return decodeVox(bytes);
  },
  encode(document, targetVariant?: SourceVariant) {
    if (targetVariant !== undefined && targetVariant !== ".vox") {
      throw new CodecError("unsupported-variant", `VOX cannot encode ${targetVariant}.`);
    }
    return encodeVox(document);
  },
};
