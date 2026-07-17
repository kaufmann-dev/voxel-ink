import type { VolumeDocument } from "../core/volume";
import type { CodecSaveState, DecodedVolume, SourceVariant } from "./types";

interface CodecSuccess<T> {
  readonly id: number;
  readonly ok: true;
  readonly result: T;
}

interface CodecFailure {
  readonly id: number;
  readonly ok: false;
  readonly error: { readonly code: string; readonly message: string };
}

type CodecResponse = CodecSuccess<DecodedVolume | Uint8Array> | CodecFailure;
type CodecResult = DecodedVolume | Uint8Array;

interface PendingRequest<T> {
  readonly resolve: (value: T) => void;
  readonly reject: (reason: Error) => void;
}

export class CodecWorkerClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest<CodecResult>>();

  decode(bytes: Uint8Array, extension: SourceVariant): Promise<DecodedVolume> {
    const transferable = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes
      : bytes.slice();
    return this.request<DecodedVolume>({
      type: "decode",
      bytes: transferable.buffer,
      extension,
    }, [transferable.buffer]);
  }

  encode(
    document: VolumeDocument<CodecSaveState>,
    targetVariant: SourceVariant,
  ): Promise<Uint8Array> {
    const occupancy = document.copyOccupancy();
    if (!document.saveState) return Promise.reject(new Error("The document has no codec save state."));
    return this.request<Uint8Array>({
      type: "encode",
      dimensions: document.dimensions,
      occupancy: occupancy.buffer,
      saveState: document.saveState,
      targetVariant,
    }, [occupancy.buffer]);
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const pending of this.pending.values()) pending.reject(new Error("The codec worker was stopped."));
    this.pending.clear();
  }

  private request<T extends CodecResult>(body: object, transfers: Transferable[]): Promise<T> {
    const worker = this.getWorker();
    const id = this.nextId;
    this.nextId += 1;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => { resolve(value as T); },
        reject,
      });
      worker.postMessage({ id, ...body }, transfers);
    });
  }

  private getWorker(): Worker {
    if (this.worker) return this.worker;
    if (typeof Worker === "undefined") throw new Error("This platform cannot start the required volume codec worker.");
    const worker = new Worker(new URL("./codec.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<CodecResponse>) => {
      const response = event.data;
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      if (response.ok) pending.resolve(response.result);
      else pending.reject(Object.assign(new Error(response.error.message), { code: response.error.code }));
    };
    worker.onerror = () => {
      for (const pending of this.pending.values()) pending.reject(new Error("The volume codec worker stopped unexpectedly."));
      this.pending.clear();
      worker.terminate();
      this.worker = null;
    };
    this.worker = worker;
    return worker;
  }
}
