import { createMatrix4 } from "../core/affine";
import { geometryFromAffine } from "./affine";
import { checkedVoxelCount, concatBytes } from "./binary";
import {
  CodecError,
  occupancyValueAt,
  type BinvoxSaveState,
  type DecodedVolume,
  type EncodableVolume,
  type SourceVariant,
  type VolumeCodec,
  type VolumeDimensions,
} from "./types";

const MAX_HEADER_BYTES = 1024 * 1024;

interface BinvoxHeader {
  readonly version: 1 | 2;
  readonly dimensions: VolumeDimensions;
  readonly translation: readonly [number, number, number];
  readonly scale: number;
  readonly payloadOffset: number;
}

const parseFiniteNumbers = (
  tokens: readonly string[],
  count: number,
  field: string,
): number[] => {
  if (tokens.length !== count) {
    throw new CodecError(
      "invalid-header",
      `BINVOX ${field} requires ${count} values.`,
    );
  }
  const values = tokens.map(Number);
  if (!values.every(Number.isFinite)) {
    throw new CodecError(
      "invalid-header",
      `BINVOX ${field} contains a non-finite value.`,
    );
  }
  return values;
};

const parseHeader = (bytes: Uint8Array): BinvoxHeader => {
  let lineStart = 0;
  let payloadOffset = -1;
  const lines: string[] = [];
  const limit = Math.min(bytes.length, MAX_HEADER_BYTES);
  for (let cursor = 0; cursor < limit; cursor += 1) {
    if (bytes[cursor] !== 10) continue;
    const contentEnd = cursor > lineStart && bytes[cursor - 1] === 13 ? cursor - 1 : cursor;
    const lineBytes = bytes.subarray(lineStart, contentEnd);
    if (lineBytes.some((value) => value > 0x7f)) {
      throw new CodecError("invalid-header", "The BINVOX header must be ASCII.");
    }
    const line = new TextDecoder("ascii").decode(lineBytes).trim();
    if (line === "data") {
      payloadOffset = cursor + 1;
      break;
    }
    lines.push(line);
    lineStart = cursor + 1;
  }
  if (payloadOffset < 0) {
    throw new CodecError(
      "invalid-header",
      "The BINVOX header is missing its data line.",
    );
  }
  const signature = lines.shift()?.match(/^#binvox\s+([12])$/);
  if (signature === null || signature === undefined) {
    throw new CodecError("invalid-format", "The BINVOX signature is invalid.");
  }
  const version = Number(signature[1]) as 1 | 2;
  let dimensions: VolumeDimensions | null = null;
  let translation: readonly [number, number, number] = [0, 0, 0];
  let scale = 1;
  let sawTranslation = false;
  let sawScale = false;
  for (const line of lines) {
    if (line.length === 0) continue;
    const [field, ...tokens] = line.split(/\s+/);
    if (field === "dim") {
      if (dimensions !== null) {
        throw new CodecError("invalid-header", "BINVOX dim appears more than once.");
      }
      const values = parseFiniteNumbers(tokens, 3, "dim");
      if (!values.every((value) => Number.isSafeInteger(value) && value > 0)) {
        throw new CodecError(
          "invalid-dimensions",
          "BINVOX dimensions must be positive safe integers.",
        );
      }
      dimensions = [values[0]!, values[1]!, values[2]!];
    } else if (field === "translate") {
      if (sawTranslation) {
        throw new CodecError(
          "invalid-header",
          "BINVOX translate appears more than once.",
        );
      }
      const values = parseFiniteNumbers(tokens, 3, "translate");
      translation = [values[0]!, values[1]!, values[2]!];
      sawTranslation = true;
    } else if (field === "scale") {
      if (sawScale) {
        throw new CodecError("invalid-header", "BINVOX scale appears more than once.");
      }
      const values = parseFiniteNumbers(tokens, 1, "scale");
      if (values[0]! <= 0) {
        throw new CodecError("invalid-header", "BINVOX scale must be positive.");
      }
      scale = values[0]!;
      sawScale = true;
    } else {
      throw new CodecError(
        "invalid-header",
        `Unsupported BINVOX header field '${field ?? ""}'.`,
      );
    }
  }
  if (dimensions === null) {
    throw new CodecError("invalid-header", "The BINVOX dim field is required.");
  }
  checkedVoxelCount(dimensions);
  return { version, dimensions, translation, scale, payloadOffset };
};

export const decodeBinvox = (bytes: Uint8Array): DecodedVolume => {
  const header = parseHeader(bytes);
  const count = checkedVoxelCount(header.dimensions);
  const sourceOrder = new Uint8Array(count);
  let sourceIndex = 0;
  let cursor = header.payloadOffset;
  let converted = false;
  while (sourceIndex < count) {
    if (cursor + 2 > bytes.length) {
      throw new CodecError("invalid-payload", "The BINVOX RLE payload is truncated.");
    }
    const value = bytes[cursor]!;
    const runLength = bytes[cursor + 1]!;
    cursor += 2;
    if (runLength === 0 || sourceIndex + runLength > count) {
      throw new CodecError("invalid-payload", "The BINVOX RLE run length is invalid.");
    }
    if (header.version === 1 && value !== 0 && value !== 1) {
      throw new CodecError(
        "invalid-payload",
        "BINVOX version 1 permits only values 0 and 1.",
      );
    }
    sourceOrder.fill(value === 0 ? 0 : 1, sourceIndex, sourceIndex + runLength);
    if (value !== 0 && value !== 1) converted = true;
    sourceIndex += runLength;
  }
  if (cursor !== bytes.length) {
    throw new CodecError("invalid-payload", "The BINVOX file has trailing RLE data.");
  }

  const [nx, ny, nz] = header.dimensions;
  const occupancy = new Uint8Array(count);
  let occupiedCount = 0;
  for (let x = 0; x < nx; x += 1) {
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        const source = x * ny * nz + z * ny + y;
        const target = x + nx * (y + ny * z);
        const value = sourceOrder[source]!;
        occupancy[target] = value;
        occupiedCount += value;
      }
    }
  }
  const stepX = header.scale / nx;
  const stepY = header.scale / ny;
  const stepZ = header.scale / nz;
  const affine = createMatrix4([
    stepX, 0, 0, header.translation[0] + stepX / 2,
    0, stepY, 0, header.translation[1] + stepY / 2,
    0, 0, stepZ, header.translation[2] + stepZ / 2,
    0, 0, 0, 1,
  ]);
  const saveState: BinvoxSaveState = {
    kind: "binvox",
    variant: ".binvox",
    version: header.version,
    translation: header.translation,
    scale: header.scale,
    dimensions: header.dimensions,
  };
  return {
    dimensions: header.dimensions,
    occupancy,
    occupiedCount,
    geometry: geometryFromAffine(affine, [null, null, null]),
    sourceFormat: "binvox",
    sourceVariant: ".binvox",
    saveState,
    normalization: { convertedNonBinary: converted },
  };
};

export const encodeBinvox = (document: EncodableVolume): Uint8Array => {
  if (document.saveState.kind !== "binvox") {
    throw new CodecError(
      "invalid-format",
      "A BINVOX save requires BINVOX source metadata.",
    );
  }
  const state = document.saveState;
  if (document.dimensions.some((value, axis) => value !== state.dimensions[axis])) {
    throw new CodecError(
      "invalid-dimensions",
      "VoxelInk does not resize BINVOX volumes during save.",
    );
  }
  const count = checkedVoxelCount(document.dimensions);
  if (document.occupancy.length !== count) {
    throw new CodecError(
      "invalid-payload",
      "The occupancy length does not match the BINVOX dimensions.",
    );
  }
  const [nx, ny, nz] = document.dimensions;
  const rle = new Uint8Array(count * 2);
  let rleLength = 0;
  let runValue = -1;
  let runLength = 0;
  const flush = (): void => {
    if (runLength > 0) {
      rle[rleLength] = runValue;
      rle[rleLength + 1] = runLength;
      rleLength += 2;
    }
  };
  for (let x = 0; x < nx; x += 1) {
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        const index = x + nx * (y + ny * z);
        const value = occupancyValueAt(document.occupancy, index);
        if (value !== runValue || runLength === 255) {
          flush();
          runValue = value;
          runLength = 1;
        } else {
          runLength += 1;
        }
      }
    }
  }
  flush();
  const header = new TextEncoder().encode(
    [
      `#binvox ${state.version}`,
      `dim ${nx} ${ny} ${nz}`,
      `translate ${state.translation.join(" ")}`,
      `scale ${state.scale}`,
      "data",
      "",
    ].join("\n"),
  );
  return concatBytes(header, rle.subarray(0, rleLength));
};

export const binvoxCodec: VolumeCodec = {
  format: "binvox",
  extensions: [".binvox"],
  decode(bytes, extension) {
    if (extension !== ".binvox") {
      throw new CodecError("unsupported-variant", `BINVOX cannot decode ${extension}.`);
    }
    return decodeBinvox(bytes);
  },
  encode(document, targetVariant?: SourceVariant) {
    if (targetVariant !== undefined && targetVariant !== ".binvox") {
      throw new CodecError("unsupported-variant", `BINVOX cannot encode ${targetVariant}.`);
    }
    return encodeBinvox(document);
  },
};
