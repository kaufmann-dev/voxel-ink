import type { Matrix4, Vec3 } from "../core/affine";

export type VolumeDimensions = readonly [number, number, number];

export type SourceFormat = "nifti" | "nrrd" | "binvox" | "vox";

export type SourceVariant =
  | ".nii"
  | ".nii.gz"
  | ".nrrd"
  | ".binvox"
  | ".vox";

export type Endianness = "little" | "big";

export interface VolumeGeometry {
  /** Row-major affine mapping voxel-center indices to world coordinates. */
  readonly indexToWorld: Matrix4;
  /** Row-major inverse of indexToWorld. */
  readonly worldToIndex: Matrix4;
  readonly spacing: Vec3;
  readonly units: readonly [string | null, string | null, string | null];
}

export interface NormalizationState {
  /** True when at least one effective source sample was outside exact {0, 1}. */
  readonly convertedNonBinary: boolean;
}

export interface NiftiExtensionInfo {
  readonly size: number;
  readonly code: number;
  readonly offset: number;
}

export interface NiftiMetadata {
  readonly version: 1 | 2;
  readonly endianness: Endianness;
  readonly dimensions: readonly number[];
  readonly pixelDimensions: readonly number[];
  readonly datatype: number;
  readonly bitsPerVoxel: number;
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

export interface NiftiSaveState {
  readonly kind: "nifti";
  readonly variant: ".nii" | ".nii.gz";
  readonly prefix: Uint8Array;
  readonly voxelOffset: number;
  readonly metadata: NiftiMetadata;
}

export type NrrdHeaderNode =
  | { readonly kind: "comment"; readonly raw: string }
  | {
      readonly kind: "field";
      readonly name: string;
      readonly normalizedName: string;
      readonly value: string;
    }
  | {
      readonly kind: "custom";
      readonly key: string;
      readonly value: string;
    };

export interface NrrdSaveState {
  readonly kind: "nrrd";
  readonly variant: ".nrrd";
  readonly magic: string;
  readonly lineEnding: "\n" | "\r\n";
  readonly header: readonly NrrdHeaderNode[];
  readonly encoding: "raw" | "gzip" | "ascii";
  readonly dimensions: VolumeDimensions;
}

export interface BinvoxSaveState {
  readonly kind: "binvox";
  readonly variant: ".binvox";
  readonly version: 1 | 2;
  readonly translation: readonly [number, number, number];
  readonly scale: number;
  readonly dimensions: VolumeDimensions;
}

export interface VoxSaveState {
  readonly kind: "vox";
  readonly variant: ".vox";
  readonly version: number;
  readonly dimensions: VolumeDimensions;
}

export type CodecSaveState =
  | NiftiSaveState
  | NrrdSaveState
  | BinvoxSaveState
  | VoxSaveState;

export interface DecodedVolume {
  readonly dimensions: VolumeDimensions;
  readonly occupancy: Uint8Array;
  readonly occupiedCount: number;
  readonly geometry: VolumeGeometry;
  readonly sourceFormat: SourceFormat;
  readonly sourceVariant: SourceVariant;
  readonly saveState: CodecSaveState;
  readonly normalization: NormalizationState;
}

export interface OccupancyAccessor {
  readonly length: number;
  get(index: number): number;
}

export type OccupancySource = Uint8Array | OccupancyAccessor;

export interface EncodableVolume {
  readonly dimensions: VolumeDimensions;
  readonly occupancy: OccupancySource;
  readonly saveState: CodecSaveState;
}

export interface VolumeCodec {
  readonly format: SourceFormat;
  readonly extensions: readonly SourceVariant[];
  decode(
    bytes: Uint8Array,
    extension: SourceVariant,
  ): DecodedVolume | Promise<DecodedVolume>;
  encode(
    document: EncodableVolume,
    targetVariant?: SourceVariant,
  ): Uint8Array | Promise<Uint8Array>;
}

export type CodecErrorCode =
  | "invalid-format"
  | "invalid-header"
  | "invalid-dimensions"
  | "invalid-datatype"
  | "invalid-scaling"
  | "invalid-affine"
  | "invalid-payload"
  | "invalid-extension"
  | "unsupported-variant"
  | "unsupported-datatype"
  | "unsupported-encoding"
  | "unsupported-detached-data"
  | "unsupported-multidimensional-data"
  | "unsupported-vector-data"
  | "unsupported-multi-model"
  | "size-limit";

export class CodecError extends Error {
  readonly code: CodecErrorCode;

  constructor(code: CodecErrorCode, message: string) {
    super(message);
    this.name = "CodecError";
    this.code = code;
  }
}

export const occupancyValueAt = (
  occupancy: OccupancySource,
  index: number,
): 0 | 1 => {
  const value =
    occupancy instanceof Uint8Array ? occupancy[index] : occupancy.get(index);
  return value === 0 ? 0 : 1;
};
