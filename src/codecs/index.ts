import { binvoxCodec } from "./binvox";
import { niftiCodec } from "./nifti";
import { nrrdCodec } from "./nrrd";
import { codecForExtension, supportedVariants } from "./registry";
import {
  CodecError,
  type DecodedVolume,
  type EncodableVolume,
  type SourceVariant,
  type VolumeCodec,
} from "./types";
import { voxCodec } from "./vox";

export * from "./types";
export { binvoxCodec, niftiCodec, nrrdCodec, voxCodec };
export { codecForExtension, supportedVariants };

export const codecs: readonly VolumeCodec[] = Object.freeze([
  niftiCodec,
  nrrdCodec,
  binvoxCodec,
  voxCodec,
]);

export const variantFromName = (nameOrExtension: string): SourceVariant => {
  const normalized = nameOrExtension.trim().toLowerCase();
  const variant = normalized.endsWith(".nii.gz")
    ? ".nii.gz"
    : normalized.slice(normalized.lastIndexOf("."));
  if (!(supportedVariants as readonly string[]).includes(variant)) {
    throw new CodecError(
      "unsupported-variant",
      `No volume codec is registered for '${nameOrExtension}'.`,
    );
  }
  return variant as SourceVariant;
};

export const decodeVolume = async (
  bytes: Uint8Array,
  nameOrExtension: string,
): Promise<DecodedVolume> => {
  const variant = variantFromName(nameOrExtension);
  return codecForExtension(variant).decode(bytes, variant);
};

export const encodeVolume = async (
  document: EncodableVolume,
  targetVariant: SourceVariant = document.saveState.variant,
): Promise<Uint8Array> =>
  codecForExtension(targetVariant).encode(document, targetVariant);
