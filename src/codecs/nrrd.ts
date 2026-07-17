import { Gunzip, gzipSync } from "fflate";
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
  type Endianness,
  type NrrdHeaderNode,
  type NrrdSaveState,
  type SourceVariant,
  type VolumeCodec,
} from "./types";

type NrrdEncoding = "raw" | "gzip" | "ascii";
type Scalar = number | bigint;
type SpatialAxes = readonly [number, number, number];
const MAX_NRRD_HEADER_BYTES = 1024 * 1024;
const MAX_IGNORED_GZIP_TRAILING_BYTES = 64 * 1024 * 1024;

interface NrrdType {
  readonly canonical: string;
  readonly bytes: 1 | 2 | 4 | 8;
  readonly integer: boolean;
  readonly signed: boolean;
  readonly bigint: boolean;
}

const TYPES = new Map<string, NrrdType>();

const registerType = (aliases: readonly string[], type: NrrdType): void => {
  for (const alias of aliases) TYPES.set(alias, type);
};

registerType(["int8", "int8_t", "signed char"], {
  canonical: "int8",
  bytes: 1,
  integer: true,
  signed: true,
  bigint: false,
});
registerType(["uint8", "uint8_t", "uchar", "unsigned char"], {
  canonical: "uint8",
  bytes: 1,
  integer: true,
  signed: false,
  bigint: false,
});
registerType(["int16", "int16_t", "short", "short int", "signed short", "signed short int"], {
  canonical: "int16",
  bytes: 2,
  integer: true,
  signed: true,
  bigint: false,
});
registerType(["uint16", "uint16_t", "ushort", "unsigned short", "unsigned short int"], {
  canonical: "uint16",
  bytes: 2,
  integer: true,
  signed: false,
  bigint: false,
});
registerType(["int32", "int32_t", "int", "signed int"], {
  canonical: "int32",
  bytes: 4,
  integer: true,
  signed: true,
  bigint: false,
});
registerType(["uint32", "uint32_t", "uint", "unsigned int"], {
  canonical: "uint32",
  bytes: 4,
  integer: true,
  signed: false,
  bigint: false,
});
registerType(["int64", "int64_t", "longlong", "long long", "long long int", "signed long long", "signed long long int"], {
  canonical: "int64",
  bytes: 8,
  integer: true,
  signed: true,
  bigint: true,
});
registerType(["uint64", "uint64_t", "ulonglong", "unsigned long long", "unsigned long long int"], {
  canonical: "uint64",
  bytes: 8,
  integer: true,
  signed: false,
  bigint: true,
});
registerType(["float"], {
  canonical: "float",
  bytes: 4,
  integer: false,
  signed: true,
  bigint: false,
});
registerType(["double"], {
  canonical: "double",
  bytes: 8,
  integer: false,
  signed: true,
  bigint: false,
});

const FIELD_ALIASES = new Map<string, string>([
  ["datafile", "data file"],
  ["lineskip", "line skip"],
  ["byteskip", "byte skip"],
  ["oldmin", "old min"],
  ["oldmax", "old max"],
  ["blocksize", "block size"],
  ["spacedimension", "space dimension"],
  ["spacedirections", "space directions"],
  ["spaceorigin", "space origin"],
  ["spaceunits", "space units"],
  ["sampleunits", "sample units"],
]);

const normalizeFieldName = (name: string): string => {
  const normalized = name.trim().toLowerCase().replace(/\s+/g, " ");
  return FIELD_ALIASES.get(normalized) ?? normalized;
};

const parseHeaderBoundary = (
  bytes: Uint8Array,
): { text: string; payloadOffset: number; lineEnding: "\n" | "\r\n" } => {
  let lineStart = 0;
  let firstLineEnding: "\n" | "\r\n" = "\n";
  const scanLength = Math.min(bytes.length, MAX_NRRD_HEADER_BYTES);
  for (let cursor = 0; cursor < scanLength; cursor += 1) {
    if (bytes[cursor] !== 10) continue;
    const contentEnd = cursor > lineStart && bytes[cursor - 1] === 13 ? cursor - 1 : cursor;
    if (lineStart === 0) {
      firstLineEnding = contentEnd === cursor - 1 ? "\r\n" : "\n";
    }
    if (contentEnd === lineStart) {
      const headerBytes = bytes.subarray(0, lineStart);
      if (headerBytes.some((value) => value > 0x7f)) {
        throw new CodecError(
          "invalid-header",
          "A NRRD header must contain only ASCII text.",
        );
      }
      return {
        text: new TextDecoder("ascii").decode(headerBytes),
        payloadOffset: cursor + 1,
        lineEnding: firstLineEnding,
      };
    }
    lineStart = cursor + 1;
  }
  if (bytes.length > MAX_NRRD_HEADER_BYTES) {
    throw new CodecError(
      "size-limit",
      "The NRRD header exceeds the 1 MiB processing limit.",
    );
  }
  throw new CodecError(
    "invalid-header",
    "An attached NRRD header must end with a blank line.",
  );
};

const parseHeaderNodes = (
  text: string,
): { magic: string; nodes: NrrdHeaderNode[]; fields: Map<string, string> } => {
  const lines = text.split(/\r?\n/);
  const magic = lines.shift() ?? "";
  if (!/^NRRD000[1-5]$/.test(magic)) {
    throw new CodecError("invalid-format", "The NRRD magic is invalid.");
  }
  const nodes: NrrdHeaderNode[] = [];
  const fields = new Map<string, string>();
  for (const rawLine of lines) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line.length === 0) continue;
    if (line.startsWith("#")) {
      nodes.push({ kind: "comment", raw: line });
      continue;
    }
    const customIndex = line.indexOf(":=");
    if (customIndex >= 0) {
      const key = line.slice(0, customIndex);
      if (key.length === 0) {
        throw new CodecError("invalid-header", "A NRRD custom key is empty.");
      }
      nodes.push({
        kind: "custom",
        key,
        value: line.slice(customIndex + 2),
      });
      continue;
    }
    const separator = line.indexOf(":");
    if (separator <= 0) {
      throw new CodecError(
        "invalid-header",
        `Invalid NRRD header line: ${line}`,
      );
    }
    const name = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trimStart();
    const normalizedName = normalizeFieldName(name);
    if (fields.has(normalizedName)) {
      throw new CodecError(
        "invalid-header",
        `NRRD field '${name}' appears more than once.`,
      );
    }
    fields.set(normalizedName, value);
    nodes.push({ kind: "field", name, normalizedName, value });
  }
  return { magic, nodes, fields };
};

const requiredField = (fields: Map<string, string>, name: string): string => {
  const value = fields.get(name);
  if (value === undefined || value.length === 0) {
    throw new CodecError(
      "invalid-header",
      `The NRRD '${name}' field is required.`,
    );
  }
  return value;
};

const integerTokens = (value: string, field: string): number[] => {
  const tokens = value.trim().split(/\s+/);
  return tokens.map((token) => {
    if (!/^[+]?[0-9]+$/.test(token)) {
      throw new CodecError(
        "invalid-header",
        `NRRD '${field}' contains a non-integer value.`,
      );
    }
    const parsed = Number(token);
    if (!Number.isSafeInteger(parsed)) {
      throw new CodecError(
        "size-limit",
        `NRRD '${field}' exceeds the safe integer range.`,
      );
    }
    return parsed;
  });
};

const parseEncoding = (value: string): NrrdEncoding => {
  const normalized = value.trim().toLowerCase();
  if (normalized === "raw") return "raw";
  if (normalized === "gzip" || normalized === "gz") return "gzip";
  if (normalized === "ascii" || normalized === "text" || normalized === "txt") {
    return "ascii";
  }
  throw new CodecError(
    "unsupported-encoding",
    `NRRD encoding '${value}' is not supported.`,
  );
};

const parseEndianness = (
  fields: Map<string, string>,
  type: NrrdType,
  encoding: NrrdEncoding,
): Endianness => {
  const raw = fields.get("endian")?.trim().toLowerCase();
  if (encoding === "ascii" || type.bytes === 1) {
    if (raw !== undefined && raw !== "little" && raw !== "big") {
      throw new CodecError("invalid-header", "NRRD endian must be little or big.");
    }
    return raw === "big" ? "big" : "little";
  }
  if (raw !== "little" && raw !== "big") {
    throw new CodecError(
      "invalid-header",
      "Binary NRRD data wider than one byte requires an endian field.",
    );
  }
  return raw;
};

const parseTuple = (value: string, field: string): number[] => {
  const trimmed = value.trim();
  if (!trimmed.startsWith("(") || !trimmed.endsWith(")")) {
    throw new CodecError(
      "invalid-header",
      `NRRD '${field}' must be a parenthesized vector.`,
    );
  }
  const tokens = trimmed.slice(1, -1).split(",");
  if (tokens.some((token) => token.trim().length === 0)) {
    throw new CodecError(
      "invalid-header",
      `NRRD '${field}' contains an empty vector component.`,
    );
  }
  const result = tokens.map((token) => Number(token.trim()));
  if (result.length === 0 || !result.every(Number.isFinite)) {
    throw new CodecError(
      "invalid-header",
      `NRRD '${field}' contains an invalid vector.`,
    );
  }
  return result;
};

const parseDirectionTokens = (value: string): (number[] | null)[] => {
  const result: (number[] | null)[] = [];
  let cursor = 0;
  const pattern = /none|\([^)]*\)/gi;
  for (const match of value.matchAll(pattern)) {
    const index = match.index;
    if (value.slice(cursor, index).trim().length !== 0) {
      throw new CodecError(
        "invalid-header",
        "NRRD space directions contain invalid syntax.",
      );
    }
    const token = match[0];
    result.push(token.toLowerCase() === "none" ? null : parseTuple(token, "space directions"));
    cursor = index + token.length;
  }
  if (value.slice(cursor).trim().length !== 0) {
    throw new CodecError(
      "invalid-header",
      "NRRD space directions contain invalid syntax.",
    );
  }
  return result;
};

const selectSpatialAxes = (
  fields: Map<string, string>,
  dimension: number,
): SpatialAxes => {
  const directionValue = fields.get("space directions");
  if (directionValue !== undefined) {
    const directions = parseDirectionTokens(directionValue);
    if (directions.length !== dimension) {
      throw new CodecError(
        "invalid-header",
        "NRRD space directions must contain one value per axis.",
      );
    }
    const axes = directions.flatMap((direction, axis) => direction === null ? [] : [axis]);
    if (axes.length !== 3) {
      throw new CodecError(
        "invalid-affine",
        "A scalar 3D NRRD requires exactly three spatial directions.",
      );
    }
    return [axes[0]!, axes[1]!, axes[2]!];
  }

  const kinds = fields.get("kinds")?.trim().toLowerCase().split(/\s+/);
  if (kinds !== undefined) {
    const explicitSpace = kinds.flatMap((kind, axis) => kind === "space" ? [axis] : []);
    if (explicitSpace.length === 3) {
      return [explicitSpace[0]!, explicitSpace[1]!, explicitSpace[2]!];
    }
    const domainAxes = kinds.flatMap((kind, axis) =>
      kind === "space" || kind === "domain" ? [axis] : [],
    );
    if (domainAxes.length === 3) {
      return [domainAxes[0]!, domainAxes[1]!, domainAxes[2]!];
    }
  }
  return [0, 1, 2];
};

const parseQuotedUnits = (value: string): (string | null)[] => {
  const quoted = [...value.matchAll(/"((?:\\.|[^"\\])*)"/g)];
  if (quoted.length > 0) {
    return quoted.map((match) =>
      match[1]!.replace(/\\"/g, '"').replace(/\\\\/g, "\\"),
    );
  }
  return value
    .trim()
    .split(/\s+/)
    .map((unit) => (unit.toLowerCase() === "none" ? null : unit));
};

const parseGeometry = (
  fields: Map<string, string>,
  dimension: number,
  spatialAxes: SpatialAxes,
): ReturnType<typeof geometryFromAffine> => {
  const declaredSpaceDimension = fields.get("space dimension");
  if (
    declaredSpaceDimension !== undefined &&
    declaredSpaceDimension.trim() !== "3"
  ) {
    throw new CodecError(
      "invalid-affine",
      "VoxelInk supports only three-dimensional NRRD world spaces.",
    );
  }
  const directionValue = fields.get("space directions");
  let directions: readonly [number[], number[], number[]] | null = null;
  if (directionValue !== undefined) {
    const parsed = parseDirectionTokens(directionValue);
    if (parsed.length !== dimension) {
      throw new CodecError(
        "invalid-header",
        "NRRD space directions must contain one value per axis.",
      );
    }
    const selected = spatialAxes.map((axis) => parsed[axis]);
    if (selected.some((direction) => direction === undefined || direction === null || direction.length !== 3)) {
      throw new CodecError(
        "invalid-affine",
        "The selected NRRD spatial axes require 3D space directions.",
      );
    }
    const spatialSet = new Set<number>(spatialAxes);
    if (parsed.some((direction, axis) => !spatialSet.has(axis) && direction !== null)) {
      throw new CodecError(
        "unsupported-multidimensional-data",
        "Singleton non-spatial NRRD axes must use a 'none' space direction.",
      );
    }
    directions = selected as [number[], number[], number[]];
  }

  const spacingTokens = fields.get("spacings")?.trim().split(/\s+/) ?? [];
  const spacings = spacingTokens.map((token) =>
    token.toLowerCase() === "nan" ? Number.NaN : Number(token),
  );
  if (spacings.length > 0 && spacings.length !== dimension) {
    throw new CodecError(
      "invalid-header",
      "NRRD spacings must contain one value per axis.",
    );
  }
  if (
    spacings.some((spacing, index) =>
      spacing === 0 ||
      (!Number.isFinite(spacing) && spacingTokens[index]?.toLowerCase() !== "nan"),
    )
  ) {
    throw new CodecError(
      "invalid-affine",
      "NRRD spacings must be finite and non-zero, or explicitly 'nan'.",
    );
  }
  const originValue = fields.get("space origin");
  const origin = originValue === undefined ? [0, 0, 0] : parseTuple(originValue, "space origin");
  if (origin.length !== 3) {
    throw new CodecError("invalid-affine", "NRRD space origin must be 3D.");
  }

  const affine = directions
    ? createMatrix4([
        directions[0][0]!, directions[1][0]!, directions[2][0]!, origin[0]!,
        directions[0][1]!, directions[1][1]!, directions[2][1]!, origin[1]!,
        directions[0][2]!, directions[1][2]!, directions[2][2]!, origin[2]!,
        0, 0, 0, 1,
      ])
    : createMatrix4([
        Number.isFinite(spacings[spatialAxes[0]]) ? spacings[spatialAxes[0]]! : 1, 0, 0, origin[0]!,
        0, Number.isFinite(spacings[spatialAxes[1]]) ? spacings[spatialAxes[1]]! : 1, 0, origin[1]!,
        0, 0, Number.isFinite(spacings[spatialAxes[2]]) ? spacings[spatialAxes[2]]! : 1, origin[2]!,
        0, 0, 0, 1,
      ]);
  const spaceUnits = fields.get("space units");
  const units = parseQuotedUnits(spaceUnits ?? fields.get("units") ?? "");
  const selectedUnits = spaceUnits === undefined
    ? spatialAxes.map((axis) => units[axis] ?? null)
    : [units[0] ?? null, units[1] ?? null, units[2] ?? null];
  return geometryFromAffine(affine, [selectedUnits[0]!, selectedUnits[1]!, selectedUnits[2]!]);
};

const binaryReader = (
  view: DataView,
  type: NrrdType,
  littleEndian: boolean,
): ((offset: number) => Scalar) => {
  switch (type.canonical) {
    case "int8": return (offset) => view.getInt8(offset);
    case "uint8": return (offset) => view.getUint8(offset);
    case "int16": return (offset) => view.getInt16(offset, littleEndian);
    case "uint16": return (offset) => view.getUint16(offset, littleEndian);
    case "int32": return (offset) => view.getInt32(offset, littleEndian);
    case "uint32": return (offset) => view.getUint32(offset, littleEndian);
    case "int64": return (offset) => view.getBigInt64(offset, littleEndian);
    case "uint64": return (offset) => view.getBigUint64(offset, littleEndian);
    case "float": return (offset) => view.getFloat32(offset, littleEndian);
    case "double": return (offset) => view.getFloat64(offset, littleEndian);
    default:
      throw new CodecError("unsupported-datatype", `Unsupported NRRD type '${type.canonical}'.`);
  }
};

const FLOAT_TOKEN = /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)$/i;
const INTEGER_TOKEN = /^[+-]?\d+$/;

const parseAsciiValue = (token: string, type: NrrdType): Scalar => {
  if (type.integer) {
    if (!INTEGER_TOKEN.test(token)) {
      throw new CodecError("invalid-payload", `Invalid integer NRRD sample '${token}'.`);
    }
    if (type.bigint) {
      const value = BigInt(token);
      const minimum = type.signed ? -(1n << 63n) : 0n;
      const maximum = type.signed ? (1n << 63n) - 1n : (1n << 64n) - 1n;
      if (value < minimum || value > maximum) {
        throw new CodecError("invalid-payload", `NRRD sample '${token}' is out of range.`);
      }
      return value;
    }
    const value = Number(token);
    const bits = type.bytes * 8;
    const minimum = type.signed ? -(2 ** (bits - 1)) : 0;
    const maximum = type.signed ? 2 ** (bits - 1) - 1 : 2 ** bits - 1;
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      throw new CodecError("invalid-payload", `NRRD sample '${token}' is out of range.`);
    }
    return value;
  }
  if (!FLOAT_TOKEN.test(token)) {
    throw new CodecError("invalid-payload", `Invalid floating NRRD sample '${token}'.`);
  }
  const normalized = token.toLowerCase();
  if (normalized.includes("nan")) return Number.NaN;
  if (normalized.includes("inf")) return normalized.startsWith("-") ? -Infinity : Infinity;
  const value = Number(token);
  return type.canonical === "float" ? Math.fround(value) : value;
};

const gunzipExpectedPayload = (
  bytes: Uint8Array,
  expectedBytes: number,
): Uint8Array => {
  const output = new Uint8Array(expectedBytes);
  let produced = 0;
  let finished = false;
  const stream = new Gunzip((chunk, final) => {
    if (
      produced + chunk.length >
      expectedBytes + MAX_IGNORED_GZIP_TRAILING_BYTES
    ) {
      throw new CodecError(
        "size-limit",
        "The NRRD gzip payload exceeds the declared array by more than 64 MiB.",
      );
    }
    if (produced < expectedBytes) {
      output.set(chunk.subarray(0, expectedBytes - produced), produced);
    }
    produced += chunk.length;
    finished = final;
  });
  const inputChunkSize = 64 * 1024;
  for (let offset = 0; offset < bytes.length; offset += inputChunkSize) {
    const end = Math.min(offset + inputChunkSize, bytes.length);
    stream.push(bytes.subarray(offset, end), end === bytes.length);
  }
  if (!finished || produced < expectedBytes) {
    throw new CodecError(
      "invalid-payload",
      "The NRRD gzip payload is truncated.",
    );
  }
  return output;
};

const isAsciiWhitespace = (value: number): boolean =>
  value === 32 || (value >= 9 && value <= 13);

function* asciiScalars(payload: Uint8Array, type: NrrdType): Generator<Scalar> {
  const decoder = new TextDecoder("ascii");
  let tokenStart = -1;
  for (let offset = 0; offset <= payload.length; offset += 1) {
    const value = offset === payload.length ? 32 : payload[offset]!;
    if (value > 0x7f) {
      throw new CodecError("invalid-payload", "The NRRD ASCII payload contains a non-ASCII byte.");
    }
    if (!isAsciiWhitespace(value)) {
      if (tokenStart < 0) tokenStart = offset;
      if (offset - tokenStart >= 128) {
        throw new CodecError("invalid-payload", "A NRRD ASCII sample is unreasonably long.");
      }
      continue;
    }
    if (tokenStart >= 0) {
      yield parseAsciiValue(decoder.decode(payload.subarray(tokenStart, offset)), type);
      tokenStart = -1;
    }
  }
}

const normalizeScalars = (
  values: Iterable<Scalar>,
  count: number,
): { occupancy: Uint8Array; occupiedCount: number; converted: boolean } => {
  const occupancy = new Uint8Array(count);
  let index = 0;
  let occupiedCount = 0;
  let converted = false;
  for (const value of values) {
    const zero = typeof value === "bigint" ? value === 0n : value === 0;
    const one = typeof value === "bigint" ? value === 1n : value === 1;
    const filled = zero ? 0 : 1;
    occupancy[index] = filled;
    occupiedCount += filled;
    if (!zero && !one) converted = true;
    index += 1;
    if (index === count) {
      return { occupancy, occupiedCount, converted };
    }
  }
  if (index !== count) {
    throw new CodecError("invalid-payload", "The NRRD payload has too few samples.");
  }
  return { occupancy, occupiedCount, converted };
};

export const decodeNrrd = (bytes: Uint8Array): DecodedVolume => {
  const boundary = parseHeaderBoundary(bytes);
  const parsed = parseHeaderNodes(boundary.text);
  const version = Number(parsed.magic.slice(-1));
  if (version < 2 && parsed.nodes.some((node) => node.kind === "custom")) {
    throw new CodecError("invalid-header", "NRRD key/value pairs require NRRD0002 or newer.");
  }
  if (version < 3 && parsed.fields.has("kinds")) {
    throw new CodecError("invalid-header", "The NRRD kinds field requires NRRD0003 or newer.");
  }
  const orientationFields = [
    "space", "space dimension", "space directions", "space origin", "space units",
  ];
  if (version < 4 && orientationFields.some((field) => parsed.fields.has(field))) {
    throw new CodecError("invalid-header", "NRRD spatial orientation fields require NRRD0004 or newer.");
  }
  if (version < 5 && parsed.fields.has("measurement frame")) {
    throw new CodecError("invalid-header", "The NRRD measurement frame requires NRRD0005.");
  }
  if (parsed.fields.has("data file")) {
    throw new CodecError(
      "unsupported-detached-data",
      "Detached NRRD data files are not supported.",
    );
  }
  for (const skipField of ["line skip", "byte skip"] as const) {
    const value = parsed.fields.get(skipField);
    if (value !== undefined && value.trim() !== "0") {
      throw new CodecError(
        "unsupported-encoding",
        `NRRD ${skipField} is not supported.`,
      );
    }
  }

  const typeName = requiredField(parsed.fields, "type").trim().toLowerCase().replace(/\s+/g, " ");
  if (typeName === "block") {
    throw new CodecError("unsupported-datatype", "NRRD block data is not scalar.");
  }
  const type = TYPES.get(typeName);
  if (type === undefined) {
    throw new CodecError("unsupported-datatype", `NRRD type '${typeName}' is not supported.`);
  }
  const dimensionValues = integerTokens(requiredField(parsed.fields, "dimension"), "dimension");
  if (dimensionValues.length !== 1 || dimensionValues[0]! < 3) {
    throw new CodecError(
      "invalid-dimensions",
      "NRRD scalar volumes require at least three dimensions.",
    );
  }
  const dimension = dimensionValues[0]!;
  const sizes = integerTokens(requiredField(parsed.fields, "sizes"), "sizes");
  if (sizes.length !== dimension || sizes.some((size) => size <= 0)) {
    throw new CodecError(
      "invalid-dimensions",
      "NRRD sizes must contain one positive size per dimension.",
    );
  }
  const spatialAxes = selectSpatialAxes(parsed.fields, dimension);
  const spatialSet = new Set<number>(spatialAxes);
  if (sizes.some((size, axis) => !spatialSet.has(axis) && size !== 1)) {
    throw new CodecError(
      "unsupported-multidimensional-data",
      "Only a single 3D NRRD scalar volume is supported.",
    );
  }
  const kinds = parsed.fields.get("kinds")?.trim().toLowerCase().split(/\s+/);
  if (kinds !== undefined) {
    if (kinds.length !== dimension) {
      throw new CodecError(
        "invalid-header",
        "NRRD kinds must contain one value per axis.",
      );
    }
    const selectedSpatialKinds = spatialAxes.map((axis) => kinds[axis]);
    const nonScalarKind = (kind: string): boolean =>
      /(?:vector|color|complex|matrix|quaternion|normal|list|point)/.test(kind);
    if (
      selectedSpatialKinds.some(
        (kind) => kind !== "domain" && kind !== "space" && kind !== "none",
      ) ||
      kinds.some(nonScalarKind)
    ) {
      throw new CodecError(
        "unsupported-vector-data",
        "NRRD vector, list, color, complex, and other non-scalar axis kinds are not supported.",
      );
    }
  }

  const dimensions = [
    sizes[spatialAxes[0]]!,
    sizes[spatialAxes[1]]!,
    sizes[spatialAxes[2]]!,
  ] as const;
  const count = checkedVoxelCount(dimensions);
  const expectedBytes = count * type.bytes;
  const encoding = parseEncoding(requiredField(parsed.fields, "encoding"));
  const endianness = parseEndianness(parsed.fields, type, encoding);
  let payload = bytes.subarray(boundary.payloadOffset);
  if (encoding === "gzip") {
    try {
      payload = gunzipExpectedPayload(payload, expectedBytes);
    } catch (error) {
      if (error instanceof CodecError) throw error;
      throw new CodecError(
        "invalid-payload",
        `The NRRD gzip payload is invalid: ${error instanceof Error ? error.message : "unknown error"}`,
      );
    }
  }

  let normalized: ReturnType<typeof normalizeScalars>;
  if (encoding === "ascii") {
    normalized = normalizeScalars(asciiScalars(payload, type), count);
  } else {
    const bytesPerSample = type.bytes;
    requireBytes(payload, 0, expectedBytes, "The NRRD voxel payload");
    const view = dataView(payload);
    const read = binaryReader(view, type, endianness === "little");
    function* values(): Generator<Scalar> {
      for (let index = 0; index < count; index += 1) {
        yield read(index * bytesPerSample);
      }
    }
    normalized = normalizeScalars(values(), count);
  }

  const saveState: NrrdSaveState = {
    kind: "nrrd",
    variant: ".nrrd",
    magic: parsed.magic,
    lineEnding: boundary.lineEnding,
    header: parsed.nodes,
    encoding,
    dimensions,
  };
  return {
    dimensions,
    occupancy: normalized.occupancy,
    occupiedCount: normalized.occupiedCount,
    geometry: parseGeometry(parsed.fields, dimension, spatialAxes),
    sourceFormat: "nrrd",
    sourceVariant: ".nrrd",
    saveState,
    normalization: { convertedNonBinary: normalized.converted },
  };
};

const serializeHeader = (state: NrrdSaveState): Uint8Array => {
  const lines = [state.magic];
  let sawType = false;
  let sawEncoding = false;
  for (const node of state.header) {
    if (node.kind === "comment") {
      lines.push(node.raw);
      continue;
    }
    if (node.kind === "custom") {
      lines.push(`${node.key}:=${node.value}`);
      continue;
    }
    switch (node.normalizedName) {
      case "type":
        lines.push(`${node.name}: uint8`);
        sawType = true;
        break;
      case "encoding":
        lines.push(`${node.name}: ${state.encoding}`);
        sawEncoding = true;
        break;
      case "endian":
      case "old min":
      case "old max":
      case "line skip":
      case "byte skip":
        break;
      case "min":
        lines.push(`${node.name}: 0`);
        break;
      case "max":
        lines.push(`${node.name}: 1`);
        break;
      default:
        lines.push(`${node.name}: ${node.value}`);
    }
  }
  if (!sawType) lines.push("type: uint8");
  if (!sawEncoding) lines.push(`encoding: ${state.encoding}`);
  return new TextEncoder().encode(`${lines.join(state.lineEnding)}${state.lineEnding}${state.lineEnding}`);
};

const encodeAsciiPayload = (
  binary: Uint8Array,
  lineEnding: "\n" | "\r\n",
): Uint8Array => {
  const lineCount = Math.ceil(binary.length / 32);
  const spaceCount = binary.length - lineCount;
  const output = new Uint8Array(
    binary.length + spaceCount + lineCount * lineEnding.length,
  );
  let cursor = 0;
  for (let index = 0; index < binary.length; index += 1) {
    output[cursor] = binary[index] === 0 ? 48 : 49;
    cursor += 1;
    const lineEnd = index % 32 === 31 || index === binary.length - 1;
    if (!lineEnd) {
      output[cursor] = 32;
      cursor += 1;
    } else {
      if (lineEnding === "\r\n") {
        output[cursor] = 13;
        cursor += 1;
      }
      output[cursor] = 10;
      cursor += 1;
    }
  }
  return output;
};

export const encodeNrrd = (document: EncodableVolume): Uint8Array => {
  if (document.saveState.kind !== "nrrd") {
    throw new CodecError("invalid-format", "An NRRD save requires NRRD source metadata.");
  }
  const state = document.saveState;
  if (document.dimensions.some((value, axis) => value !== state.dimensions[axis])) {
    throw new CodecError("invalid-dimensions", "VoxelInk does not resize NRRD volumes during save.");
  }
  const count = checkedVoxelCount(document.dimensions);
  if (document.occupancy.length !== count) {
    throw new CodecError("invalid-payload", "The occupancy length does not match the NRRD dimensions.");
  }
  const binary = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    binary[index] = occupancyValueAt(document.occupancy, index);
  }
  const payload =
    state.encoding === "gzip"
      ? gzipSync(binary, { level: 6, mtime: 0 })
      : state.encoding === "ascii"
        ? encodeAsciiPayload(binary, state.lineEnding)
        : binary;
  return concatBytes(serializeHeader(state), payload);
};

export const nrrdCodec: VolumeCodec = {
  format: "nrrd",
  extensions: [".nrrd"],
  decode(bytes, extension) {
    if (extension !== ".nrrd") {
      throw new CodecError("unsupported-variant", `NRRD cannot decode ${extension}.`);
    }
    return decodeNrrd(bytes);
  },
  encode(document, targetVariant?: SourceVariant) {
    if (targetVariant !== undefined && targetVariant !== ".nrrd") {
      throw new CodecError("unsupported-variant", `NRRD cannot encode ${targetVariant}.`);
    }
    return encodeNrrd(document);
  },
};
