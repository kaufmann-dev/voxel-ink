import { binvoxCodec } from "./binvox";
import { niftiCodec } from "./nifti";
import { nrrdCodec } from "./nrrd";
import type { SourceVariant, VolumeCodec } from "./types";
import { voxCodec } from "./vox";

const codecs: readonly VolumeCodec[] = [niftiCodec, nrrdCodec, binvoxCodec, voxCodec];

export function codecForExtension(extension: SourceVariant): VolumeCodec {
  const codec = codecs.find((candidate) => candidate.extensions.includes(extension));
  if (!codec) throw new Error(`No volume codec is registered for ${extension}.`);
  return codec;
}

export const supportedVariants = codecs.flatMap((codec) => codec.extensions);
