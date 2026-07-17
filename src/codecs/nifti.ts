import { Gunzip, gzipSync } from "fflate";
import { createMatrix4, type Matrix4 } from "../core/affine";
import { affineFromRows, geometryFromAffine, isValidAffine } from "./affine";
import {
  checkedVoxelCount,
  concatBytes,
  copyUint8Array,
  dataView,
  decodeFixedString,
  isAllZero,
  requireBytes,
} from "./binary";
import {
  CodecError,
  occupancyValueAt,
  type DecodedVolume,
  type EncodableVolume,
  type Endianness,
  type NiftiExtensionInfo,
  type NiftiMetadata,
  type NiftiSaveState,
  type SourceVariant,
  type VolumeCodec,
  type VolumeDimensions,
} from "./types";

const NIFTI1_HEADER_SIZE = 348;
const NIFTI2_HEADER_SIZE = 540;
const NIFTI1_PREFIX_SIZE = 352;
const NIFTI2_PREFIX_SIZE = 544;
const MAX_DECOMPRESSED_BYTES = 1_200_000_000;

const DATATYPE = {
  binary: 1,
  uint8: 2,
  int16: 4,
  int32: 8,
  float32: 16,
  complex64: 32,
  float64: 64,
  rgb24: 128,
  int8: 256,
  uint16: 512,
  uint32: 768,
  int64: 1024,
  uint64: 1280,
  float128: 1536,
  complex128: 1792,
  complex256: 2048,
  rgba32: 2304,
} as const;

type SupportedDatatype =
  | typeof DATATYPE.uint8
  | typeof DATATYPE.int8
  | typeof DATATYPE.int16
  | typeof DATATYPE.uint16
  | typeof DATATYPE.int32
  | typeof DATATYPE.uint32
  | typeof DATATYPE.int64
  | typeof DATATYPE.uint64
  | typeof DATATYPE.float32
  | typeof DATATYPE.float64;

interface ParsedHeader {
  readonly version: 1 | 2;
  readonly littleEndian: boolean;
  readonly dimensions: VolumeDimensions;
  readonly allDimensions: readonly number[];
  readonly pixelDimensions: readonly number[];
  readonly datatype: SupportedDatatype;
  readonly bitsPerVoxel: number;
  readonly bytesPerVoxel: number;
  readonly voxelOffset: number;
  readonly scaleSlope: number;
  readonly scaleIntercept: number;
  readonly qformCode: number;
  readonly sformCode: number;
  readonly qform: Matrix4 | null;
  readonly sform: Matrix4 | null;
  readonly unitsCode: number;
  readonly intentCode: number;
  readonly intentName: string;
  readonly description: string;
  readonly auxiliaryFile: string;
  readonly extensions: readonly NiftiExtensionInfo[];
}

const isGzip = (bytes: Uint8Array): boolean =>
  bytes.length >= 3 && bytes[0] === 0x1f && bytes[1] === 0x8b && bytes[2] === 8;

const gunzipWithLimit = (bytes: Uint8Array, limit: number): Uint8Array => {
  const chunks: Uint8Array[] = [];
  let outputLength = 0;
  let finished = false;
  const stream = new Gunzip((chunk, final) => {
    if (outputLength + chunk.length > limit) {
      throw new CodecError(
        "size-limit",
        "The decompressed NIfTI file exceeds the local processing limit.",
      );
    }
    chunks.push(chunk);
    outputLength += chunk.length;
    finished = final;
  });
  const inputChunkSize = 64 * 1024;
  for (let offset = 0; offset < bytes.length; offset += inputChunkSize) {
    const end = Math.min(offset + inputChunkSize, bytes.length);
    stream.push(bytes.subarray(offset, end), end === bytes.length);
  }
  if (!finished) {
    throw new CodecError("invalid-format", "The NIfTI gzip stream is truncated.");
  }
  const output = new Uint8Array(outputLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
};

const readInt64AsSafeNumber = (
  view: DataView,
  offset: number,
  littleEndian: boolean,
  field: string,
): number => {
  const value = view.getBigInt64(offset, littleEndian);
  if (
    value < BigInt(Number.MIN_SAFE_INTEGER) ||
    value > BigInt(Number.MAX_SAFE_INTEGER)
  ) {
    throw new CodecError(
      "size-limit",
      `${field} is outside JavaScript's safe integer range.`,
    );
  }
  return Number(value);
};

const readFloatArray = (
  view: DataView,
  offset: number,
  count: number,
  bytesPerValue: 4 | 8,
  littleEndian: boolean,
): number[] => {
  const values: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const location = offset + index * bytesPerValue;
    values.push(
      bytesPerValue === 4
        ? view.getFloat32(location, littleEndian)
        : view.getFloat64(location, littleEndian),
    );
  }
  return values;
};

const quaternionAffine = (
  bInput: number,
  cInput: number,
  dInput: number,
  offsets: readonly [number, number, number],
  pixelDimensions: readonly number[],
): Matrix4 | null => {
  let b = bInput;
  let c = cInput;
  let d = dInput;
  if (![b, c, d, ...offsets].every(Number.isFinite)) return null;

  let squaredA = 1 - (b * b + c * c + d * d);
  let a: number;
  if (squaredA < 1e-7) {
    const length = Math.hypot(b, c, d);
    if (!(length > 0) || !Number.isFinite(length)) return null;
    b /= length;
    c /= length;
    d /= length;
    a = 0;
    squaredA = 0;
  } else {
    a = Math.sqrt(squaredA);
  }

  const dx =
    Number.isFinite(pixelDimensions[1]) && pixelDimensions[1]! !== 0
      ? Math.abs(pixelDimensions[1]!)
      : 1;
  const dy =
    Number.isFinite(pixelDimensions[2]) && pixelDimensions[2]! !== 0
      ? Math.abs(pixelDimensions[2]!)
      : 1;
  const dz =
    Number.isFinite(pixelDimensions[3]) && pixelDimensions[3]! !== 0
      ? Math.abs(pixelDimensions[3]!)
      : 1;
  const qfac = pixelDimensions[0]! < 0 ? -1 : 1;

  const aa = squaredA;
  const bb = b * b;
  const cc = c * c;
  const dd = d * d;
  return affineFromRows([
    [
      (aa + bb - cc - dd) * dx,
      2 * (b * c - a * d) * dy,
      2 * (b * d + a * c) * dz * qfac,
      offsets[0],
    ],
    [
      2 * (b * c + a * d) * dx,
      (aa + cc - bb - dd) * dy,
      2 * (c * d - a * b) * dz * qfac,
      offsets[1],
    ],
    [
      2 * (b * d - a * c) * dx,
      2 * (c * d + a * b) * dy,
      (aa + dd - cc - bb) * dz * qfac,
      offsets[2],
    ],
  ]);
};

const fallbackAffine = (pixelDimensions: readonly number[]): Matrix4 => {
  const spacing = [1, 2, 3].map((index) => {
    const value = Math.abs(pixelDimensions[index] ?? 1);
    return Number.isFinite(value) && value > 0 ? value : 1;
  });
  return createMatrix4([
    spacing[0]!,
    0,
    0,
    0,
    0,
    spacing[1]!,
    0,
    0,
    0,
    0,
    spacing[2]!,
    0,
    0,
    0,
    0,
    1,
  ]);
};

const parseExtensions = (
  bytes: Uint8Array,
  view: DataView,
  start: number,
  voxelOffset: number,
  littleEndian: boolean,
): NiftiExtensionInfo[] => {
  if (bytes[start - 4] === 0) return [];
  const extensions: NiftiExtensionInfo[] = [];
  let cursor = start;
  while (cursor < voxelOffset) {
    const remaining = bytes.subarray(cursor, voxelOffset);
    if (isAllZero(remaining)) break;
    if (remaining.length < 8) {
      throw new CodecError(
        "invalid-extension",
        "The NIfTI extension header is truncated.",
      );
    }
    const size = view.getInt32(cursor, littleEndian);
    const code = view.getInt32(cursor + 4, littleEndian);
    if (size < 16 || size % 16 !== 0 || cursor + size > voxelOffset) {
      throw new CodecError(
        "invalid-extension",
        "A NIfTI extension has an invalid size.",
      );
    }
    if (code < 0) {
      throw new CodecError(
        "invalid-extension",
        "A NIfTI extension has a negative code.",
      );
    }
    extensions.push({ size, code, offset: cursor });
    cursor += size;
  }
  return extensions;
};

const datatypeDetails = (
  datatype: number,
  bitsPerVoxel: number,
): { datatype: SupportedDatatype; bytesPerVoxel: number } => {
  const supported = new Map<number, number>([
    [DATATYPE.uint8, 8],
    [DATATYPE.int8, 8],
    [DATATYPE.int16, 16],
    [DATATYPE.uint16, 16],
    [DATATYPE.int32, 32],
    [DATATYPE.uint32, 32],
    [DATATYPE.int64, 64],
    [DATATYPE.uint64, 64],
    [DATATYPE.float32, 32],
    [DATATYPE.float64, 64],
  ]);
  const expectedBits = supported.get(datatype);
  if (expectedBits === undefined) {
    const scalarButUnsupported = new Set<number>([
      DATATYPE.binary,
      DATATYPE.float128,
    ]);
    const nonScalar = new Set<number>([
      DATATYPE.complex64,
      DATATYPE.complex128,
      DATATYPE.complex256,
      DATATYPE.rgb24,
      DATATYPE.rgba32,
    ]);
    const qualifier = nonScalar.has(datatype)
      ? "vector, complex, RGB, and RGBA values"
      : scalarButUnsupported.has(datatype)
        ? "this scalar datatype"
        : `datatype code ${datatype}`;
    throw new CodecError(
      "unsupported-datatype",
      `NIfTI ${qualifier} are not supported.`,
    );
  }
  if (bitsPerVoxel !== expectedBits) {
    throw new CodecError(
      "invalid-datatype",
      `NIfTI datatype ${datatype} requires bitpix ${expectedBits}, not ${bitsPerVoxel}.`,
    );
  }
  return {
    datatype: datatype as SupportedDatatype,
    bytesPerVoxel: expectedBits / 8,
  };
};

const parseHeader = (bytes: Uint8Array): ParsedHeader => {
  if (bytes.length < NIFTI1_PREFIX_SIZE) {
    throw new CodecError("invalid-header", "The NIfTI header is truncated.");
  }
  const view = dataView(bytes);
  const littleSize = view.getInt32(0, true);
  const bigSize = view.getInt32(0, false);
  let version: 1 | 2;
  let littleEndian: boolean;
  if (littleSize === NIFTI1_HEADER_SIZE || littleSize === NIFTI2_HEADER_SIZE) {
    version = littleSize === NIFTI1_HEADER_SIZE ? 1 : 2;
    littleEndian = true;
  } else if (bigSize === NIFTI1_HEADER_SIZE || bigSize === NIFTI2_HEADER_SIZE) {
    version = bigSize === NIFTI1_HEADER_SIZE ? 1 : 2;
    littleEndian = false;
  } else {
    throw new CodecError(
      "invalid-header",
      "The file is not a NIfTI-1 or NIfTI-2 image.",
    );
  }

  const minimumPrefix = version === 1 ? NIFTI1_PREFIX_SIZE : NIFTI2_PREFIX_SIZE;
  if (bytes.length < minimumPrefix) {
    throw new CodecError("invalid-header", "The NIfTI header is truncated.");
  }
  const magic =
    version === 1
      ? bytes.subarray(344, 348)
      : bytes.subarray(4, 12);
  const expected =
    version === 1
      ? new Uint8Array([0x6e, 0x2b, 0x31, 0])
      : new Uint8Array([0x6e, 0x2b, 0x32, 0, 13, 10, 26, 10]);
  if (!magic.every((value, index) => value === expected[index])) {
    const pairMagic =
      magic[0] === 0x6e && magic[1] === 0x69 && magic[2] === (version === 1 ? 0x31 : 0x32);
    throw new CodecError(
      pairMagic ? "unsupported-detached-data" : "unsupported-variant",
      pairMagic
        ? "Detached NIfTI .hdr/.img pairs are not supported."
        : "The NIfTI single-file magic is invalid.",
    );
  }

  const allDimensions: number[] = [];
  if (version === 1) {
    for (let index = 0; index < 8; index += 1) {
      allDimensions.push(view.getInt16(40 + index * 2, littleEndian));
    }
  } else {
    for (let index = 0; index < 8; index += 1) {
      allDimensions.push(
        readInt64AsSafeNumber(
          view,
          16 + index * 8,
          littleEndian,
          `NIfTI dimension ${index}`,
        ),
      );
    }
  }
  const dimensionCount = allDimensions[0]!;
  if (!Number.isInteger(dimensionCount) || dimensionCount < 1 || dimensionCount > 7) {
    throw new CodecError(
      "invalid-dimensions",
      "NIfTI dim[0] must be between 1 and 7.",
    );
  }
  for (let axis = 1; axis <= dimensionCount; axis += 1) {
    if (!Number.isSafeInteger(allDimensions[axis]) || allDimensions[axis]! <= 0) {
      throw new CodecError(
        "invalid-dimensions",
        `NIfTI dimension ${axis} must be positive.`,
      );
    }
  }
  for (let axis = 4; axis <= dimensionCount; axis += 1) {
    if (allDimensions[axis] !== 1) {
      throw new CodecError(
        "unsupported-multidimensional-data",
        "Only a single 3D NIfTI scalar volume is supported.",
      );
    }
  }
  const dimensions = [
    dimensionCount >= 1 ? allDimensions[1]! : 1,
    dimensionCount >= 2 ? allDimensions[2]! : 1,
    dimensionCount >= 3 ? allDimensions[3]! : 1,
  ] as const;
  checkedVoxelCount(dimensions);

  const datatype = view.getInt16(version === 1 ? 70 : 12, littleEndian);
  const bitsPerVoxel = view.getInt16(version === 1 ? 72 : 14, littleEndian);
  const datatypeInfo = datatypeDetails(datatype, bitsPerVoxel);
  const pixelDimensions = readFloatArray(
    view,
    version === 1 ? 76 : 104,
    8,
    version === 1 ? 4 : 8,
    littleEndian,
  );

  const rawVoxelOffset =
    version === 1
      ? view.getFloat32(108, littleEndian)
      : readInt64AsSafeNumber(view, 168, littleEndian, "NIfTI vox_offset");
  const voxelOffset = rawVoxelOffset < minimumPrefix ? minimumPrefix : rawVoxelOffset;
  if (
    !Number.isSafeInteger(voxelOffset) ||
    voxelOffset < minimumPrefix ||
    voxelOffset > bytes.length
  ) {
    throw new CodecError(
      "invalid-header",
      "NIfTI vox_offset is invalid or outside the file.",
    );
  }

  const scaleSlope =
    version === 1
      ? view.getFloat32(112, littleEndian)
      : view.getFloat64(176, littleEndian);
  const scaleIntercept =
    version === 1
      ? view.getFloat32(116, littleEndian)
      : view.getFloat64(184, littleEndian);
  if (!Number.isFinite(scaleSlope)) {
    throw new CodecError(
      "invalid-scaling",
      "NIfTI scl_slope must be finite.",
    );
  }
  if (scaleSlope !== 0 && !Number.isFinite(scaleIntercept)) {
    throw new CodecError(
      "invalid-scaling",
      "NIfTI scl_inter must be finite when scaling is active.",
    );
  }

  const qformCode =
    version === 1
      ? view.getInt16(252, littleEndian)
      : view.getInt32(344, littleEndian);
  const sformCode =
    version === 1
      ? view.getInt16(254, littleEndian)
      : view.getInt32(348, littleEndian);
  const scalarSize = version === 1 ? 4 : 8;
  const readScalar = (offset: number): number =>
    scalarSize === 4
      ? view.getFloat32(offset, littleEndian)
      : view.getFloat64(offset, littleEndian);
  const quaternionOffset = version === 1 ? 256 : 352;
  const translationOffset = version === 1 ? 268 : 376;
  const qform =
    qformCode > 0
      ? quaternionAffine(
          readScalar(quaternionOffset),
          readScalar(quaternionOffset + scalarSize),
          readScalar(quaternionOffset + scalarSize * 2),
          [
            readScalar(translationOffset),
            readScalar(translationOffset + scalarSize),
            readScalar(translationOffset + scalarSize * 2),
          ],
          pixelDimensions,
        )
      : null;
  const sformRows: [number, number, number, number][] = [];
  const sformOffset = version === 1 ? 280 : 400;
  for (let row = 0; row < 3; row += 1) {
    sformRows.push([
      readScalar(sformOffset + (row * 4 + 0) * scalarSize),
      readScalar(sformOffset + (row * 4 + 1) * scalarSize),
      readScalar(sformOffset + (row * 4 + 2) * scalarSize),
      readScalar(sformOffset + (row * 4 + 3) * scalarSize),
    ]);
  }
  const sform =
    sformCode > 0 && sformRows.flat().every(Number.isFinite)
      ? affineFromRows(sformRows)
      : null;

  const unitsCode =
    version === 1 ? view.getUint8(123) : view.getInt32(500, littleEndian);
  const intentCode =
    version === 1
      ? view.getInt16(68, littleEndian)
      : view.getInt32(504, littleEndian);
  const extensions = parseExtensions(
    bytes,
    view,
    minimumPrefix,
    voxelOffset,
    littleEndian,
  );

  return {
    version,
    littleEndian,
    dimensions,
    allDimensions,
    pixelDimensions,
    datatype: datatypeInfo.datatype,
    bitsPerVoxel,
    bytesPerVoxel: datatypeInfo.bytesPerVoxel,
    voxelOffset,
    scaleSlope,
    scaleIntercept,
    qformCode,
    sformCode,
    qform,
    sform,
    unitsCode,
    intentCode,
    intentName: decodeFixedString(
      bytes,
      version === 1 ? 328 : 508,
      16,
    ),
    description: decodeFixedString(
      bytes,
      version === 1 ? 148 : 240,
      80,
    ),
    auxiliaryFile: decodeFixedString(
      bytes,
      version === 1 ? 228 : 320,
      24,
    ),
    extensions,
  };
};

type ScalarReader = (offset: number) => number | bigint;

const scalarReader = (
  view: DataView,
  datatype: SupportedDatatype,
  littleEndian: boolean,
): ScalarReader => {
  switch (datatype) {
    case DATATYPE.uint8:
      return (offset) => view.getUint8(offset);
    case DATATYPE.int8:
      return (offset) => view.getInt8(offset);
    case DATATYPE.int16:
      return (offset) => view.getInt16(offset, littleEndian);
    case DATATYPE.uint16:
      return (offset) => view.getUint16(offset, littleEndian);
    case DATATYPE.int32:
      return (offset) => view.getInt32(offset, littleEndian);
    case DATATYPE.uint32:
      return (offset) => view.getUint32(offset, littleEndian);
    case DATATYPE.int64:
      return (offset) => view.getBigInt64(offset, littleEndian);
    case DATATYPE.uint64:
      return (offset) => view.getBigUint64(offset, littleEndian);
    case DATATYPE.float32:
      return (offset) => view.getFloat32(offset, littleEndian);
    case DATATYPE.float64:
      return (offset) => view.getFloat64(offset, littleEndian);
  }
};

const normalizePayload = (
  bytes: Uint8Array,
  header: ParsedHeader,
): { occupancy: Uint8Array; occupiedCount: number; converted: boolean } => {
  const count = checkedVoxelCount(header.dimensions);
  const payloadBytes = count * header.bytesPerVoxel;
  requireBytes(bytes, header.voxelOffset, payloadBytes, "The NIfTI voxel payload");
  const view = dataView(bytes);
  const read = scalarReader(view, header.datatype, header.littleEndian);
  const occupancy = new Uint8Array(count);
  let occupiedCount = 0;
  let converted = false;
  for (let index = 0; index < count; index += 1) {
    const raw = read(header.voxelOffset + index * header.bytesPerVoxel);
    const numeric = typeof raw === "bigint" ? Number(raw) : raw;
    const effective =
      header.scaleSlope === 0
        ? numeric
        : numeric * header.scaleSlope + header.scaleIntercept;
    const filled = effective === 0 ? 0 : 1;
    occupancy[index] = filled;
    occupiedCount += filled;
    if (effective !== 0 && effective !== 1) converted = true;
  }
  return { occupancy, occupiedCount, converted };
};

const unitsForCode = (
  unitsCode: number,
): readonly [string | null, string | null, string | null] => {
  const spatialCode = unitsCode & 0x07;
  const unit =
    spatialCode === 1
      ? "m"
      : spatialCode === 2
        ? "mm"
        : spatialCode === 3
          ? "µm"
          : null;
  return [unit, unit, unit];
};

export const decodeNifti = (
  sourceBytes: Uint8Array,
  extension: ".nii" | ".nii.gz",
): DecodedVolume => {
  const compressed = isGzip(sourceBytes);
  if ((extension === ".nii.gz") !== compressed) {
    throw new CodecError(
      "invalid-format",
      extension === ".nii.gz"
        ? "A .nii.gz file must contain a gzip stream."
        : "A .nii file must not be gzip-compressed.",
    );
  }
  let bytes: Uint8Array;
  try {
    bytes = compressed
      ? gunzipWithLimit(sourceBytes, MAX_DECOMPRESSED_BYTES)
      : copyUint8Array(sourceBytes);
  } catch (error) {
    if (error instanceof CodecError) throw error;
    throw new CodecError(
      "invalid-format",
      `The NIfTI gzip stream could not be decompressed: ${error instanceof Error ? error.message : "unknown error"}`,
    );
  }
  if (bytes.length > MAX_DECOMPRESSED_BYTES) {
    throw new CodecError(
      "size-limit",
      "The decompressed NIfTI file exceeds the local processing limit.",
    );
  }

  const header = parseHeader(bytes);
  const normalized = normalizePayload(bytes, header);
  const effectiveAffine =
    header.sform !== null && isValidAffine(header.sform)
      ? header.sform
      : header.qform !== null && isValidAffine(header.qform)
        ? header.qform
        : fallbackAffine(header.pixelDimensions);
  const endianness: Endianness = header.littleEndian ? "little" : "big";
  const metadata: NiftiMetadata = {
    version: header.version,
    endianness,
    dimensions: header.allDimensions,
    pixelDimensions: header.pixelDimensions,
    datatype: header.datatype,
    bitsPerVoxel: header.bitsPerVoxel,
    scaleSlope: header.scaleSlope,
    scaleIntercept: header.scaleIntercept,
    qformCode: header.qformCode,
    sformCode: header.sformCode,
    qform: header.qform,
    sform: header.sform,
    unitsCode: header.unitsCode,
    intentCode: header.intentCode,
    intentName: header.intentName,
    description: header.description,
    auxiliaryFile: header.auxiliaryFile,
    extensions: header.extensions,
  };
  const saveState: NiftiSaveState = {
    kind: "nifti",
    variant: extension,
    prefix: bytes.slice(0, header.voxelOffset),
    voxelOffset: header.voxelOffset,
    metadata,
  };
  return {
    dimensions: header.dimensions,
    occupancy: normalized.occupancy,
    occupiedCount: normalized.occupiedCount,
    geometry: geometryFromAffine(effectiveAffine, unitsForCode(header.unitsCode)),
    sourceFormat: "nifti",
    sourceVariant: extension,
    saveState,
    normalization: { convertedNonBinary: normalized.converted },
  };
};

const validateNiftiDocument = (
  document: EncodableVolume,
): { state: NiftiSaveState; voxelCount: number } => {
  if (document.saveState.kind !== "nifti") {
    throw new CodecError(
      "invalid-format",
      "A NIfTI save requires NIfTI source metadata.",
    );
  }
  const voxelCount = checkedVoxelCount(document.dimensions);
  if (document.occupancy.length !== voxelCount) {
    throw new CodecError(
      "invalid-payload",
      "The occupancy length does not match the volume dimensions.",
    );
  }
  const original = document.saveState.metadata.dimensions;
  const originalDimensionCount = original[0] ?? 0;
  if (
    document.dimensions[0] !== (originalDimensionCount >= 1 ? original[1] : 1) ||
    document.dimensions[1] !== (originalDimensionCount >= 2 ? original[2] : 1) ||
    document.dimensions[2] !== (originalDimensionCount >= 3 ? original[3] : 1)
  ) {
    throw new CodecError(
      "invalid-dimensions",
      "VoxelInk does not resize NIfTI volumes during save.",
    );
  }
  return { state: document.saveState, voxelCount };
};

export const encodeNifti = (
  document: EncodableVolume,
  targetVariant?: ".nii" | ".nii.gz",
): Uint8Array => {
  const { state, voxelCount } = validateNiftiDocument(document);
  const variant = targetVariant ?? state.variant;
  const prefix = copyUint8Array(state.prefix);
  if (prefix.length !== state.voxelOffset) {
    throw new CodecError(
      "invalid-header",
      "The preserved NIfTI header prefix is inconsistent.",
    );
  }
  const view = dataView(prefix);
  const littleEndian = state.metadata.endianness === "little";
  if (state.metadata.version === 1) {
    view.setInt16(70, DATATYPE.uint8, littleEndian);
    view.setInt16(72, 8, littleEndian);
    view.setFloat32(112, 1, littleEndian);
    view.setFloat32(116, 0, littleEndian);
    view.setFloat32(124, 1, littleEndian);
    view.setFloat32(128, 0, littleEndian);
    view.setInt32(140, 1, littleEndian);
    view.setInt32(144, 0, littleEndian);
    prefix.set([0x6e, 0x2b, 0x31, 0], 344);
  } else {
    view.setInt16(12, DATATYPE.uint8, littleEndian);
    view.setInt16(14, 8, littleEndian);
    view.setFloat64(176, 1, littleEndian);
    view.setFloat64(184, 0, littleEndian);
    view.setFloat64(192, 1, littleEndian);
    view.setFloat64(200, 0, littleEndian);
    prefix.set([0x6e, 0x2b, 0x32, 0, 13, 10, 26, 10], 4);
  }
  const payload = new Uint8Array(voxelCount);
  for (let index = 0; index < voxelCount; index += 1) {
    payload[index] = occupancyValueAt(document.occupancy, index);
  }
  const raw = concatBytes(prefix, payload);
  return variant === ".nii.gz"
    ? gzipSync(raw, { level: 6, mtime: 0 })
    : raw;
};

export const niftiCodec: VolumeCodec = {
  format: "nifti",
  extensions: [".nii", ".nii.gz"],
  decode(bytes, extension) {
    if (extension !== ".nii" && extension !== ".nii.gz") {
      throw new CodecError(
        "unsupported-variant",
        `NIfTI cannot decode ${extension}.`,
      );
    }
    return decodeNifti(bytes, extension);
  },
  encode(document, targetVariant?: SourceVariant) {
    if (
      targetVariant !== undefined &&
      targetVariant !== ".nii" &&
      targetVariant !== ".nii.gz"
    ) {
      throw new CodecError(
        "unsupported-variant",
        `NIfTI cannot encode ${targetVariant}.`,
      );
    }
    return encodeNifti(document, targetVariant);
  },
};
