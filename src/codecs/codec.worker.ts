import { codecForExtension } from "./registry";
import type { CodecSaveState, DecodedVolume, EncodableVolume, SourceVariant, VolumeDimensions } from "./types";

interface DecodeRequest {
  readonly id: number;
  readonly type: "decode";
  readonly bytes: ArrayBuffer;
  readonly extension: SourceVariant;
}

interface EncodeRequest {
  readonly id: number;
  readonly type: "encode";
  readonly dimensions: VolumeDimensions;
  readonly occupancy: ArrayBuffer;
  readonly saveState: CodecSaveState;
  readonly targetVariant: SourceVariant;
}

type CodecRequest = DecodeRequest | EncodeRequest;

function transferDecoded(decoded: DecodedVolume): Transferable[] {
  const transfers: Transferable[] = [
    decoded.occupancy.buffer,
  ];
  if (decoded.saveState.kind === "nifti") transfers.push(decoded.saveState.prefix.buffer);
  return transfers;
}

self.onmessage = async (event: MessageEvent<CodecRequest>) => {
  const request = event.data;
  try {
    const codec = codecForExtension(request.type === "decode" ? request.extension : request.targetVariant);
    if (request.type === "decode") {
      const result = await codec.decode(new Uint8Array(request.bytes), request.extension);
      self.postMessage({ id: request.id, ok: true, result }, { transfer: transferDecoded(result) });
      return;
    }
    const document: EncodableVolume = {
      dimensions: request.dimensions,
      occupancy: new Uint8Array(request.occupancy),
      saveState: request.saveState,
    };
    const result = await codec.encode(document, request.targetVariant);
    self.postMessage({ id: request.id, ok: true, result }, { transfer: [result.buffer] });
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error && typeof error.code === "string"
      ? error.code
      : "codec-error";
    const message = error instanceof Error ? error.message : "The volume codec failed.";
    self.postMessage({ id: request.id, ok: false, error: { code, message } });
  }
};

export {};
