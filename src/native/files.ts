import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { SourceVariant } from "../codecs/types";
import type { SourceHandle } from "../editor/store";

const SAVE_TICKET_HEADER = "x-voxelink-ticket";

export type SaveMode = "overwriteSource" | "saveAsAdopt" | "saveCopy";

interface NativeOpenTicket {
  readonly ticket: string;
  readonly sourceId: string;
  readonly displayName: string;
  readonly sourceVariant: string;
  readonly size: number;
}

export interface SaveTicket {
  readonly ticket: string;
  readonly mode: SaveMode;
  readonly displayName: string;
  readonly destinationVariant: string;
}

export interface WriteSaveResult {
  readonly mode: SaveMode;
  readonly adopted: boolean;
  readonly sourceId: string;
  readonly displayName: string;
  readonly sourceVariant: string;
  readonly size: number;
}

export interface OpenedBytes {
  readonly bytes: Uint8Array;
  readonly source: SourceHandle;
}

const variants = new Set<SourceVariant>([".nii", ".nii.gz", ".nrrd", ".binvox", ".vox"]);

function extensionOf(value: string): SourceVariant {
  const extension = `.${value.replace(/^\./, "").toLowerCase()}` as SourceVariant;
  if (!variants.has(extension)) throw new Error("The native host returned an unsupported file variant.");
  return extension;
}

function requireNative(): void {
  if (!isTauri()) throw new Error("Native file commands are available only in the VoxelInk desktop application.");
}

export function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  if (error && typeof error === "object" && "message" in error && typeof error.message === "string") return error.message;
  return "The native file operation failed.";
}

export async function chooseAndReadVolume(): Promise<OpenedBytes | null> {
  requireNative();
  const selected = await invoke<NativeOpenTicket | null>("request_open_ticket");
  if (!selected) return null;
  const body = await invoke<ArrayBuffer>("read_open_ticket", { ticket: selected.ticket });
  return {
    bytes: new Uint8Array(body),
    source: {
      sourceId: selected.sourceId,
      name: selected.displayName,
      extension: extensionOf(selected.sourceVariant),
      size: selected.size,
    },
  };
}

export async function chooseSaveDestination(
  source: SourceHandle,
  mode: SaveMode,
): Promise<SaveTicket | null> {
  requireNative();
  const allowedExtensions = source.extension === ".nii" || source.extension === ".nii.gz"
    ? ["nii", "nii.gz"]
    : [source.extension.slice(1)];
  return invoke<SaveTicket | null>("request_save_ticket", {
    request: {
      mode,
      sourceId: source.sourceId,
      suggestedName: source.name,
      allowedExtensions,
    },
  });
}

export async function writeSavedVolume(ticket: SaveTicket, bytes: Uint8Array): Promise<WriteSaveResult> {
  requireNative();
  return invoke<WriteSaveResult>("write_save_ticket", bytes, {
    headers: { [SAVE_TICKET_HEADER]: ticket.ticket },
  });
}

export async function onNativeCloseRequested(handler: () => void): Promise<UnlistenFn> {
  if (!isTauri()) return () => undefined;
  return listen("voxelink://close-requested", handler);
}

export async function confirmNativeExit(): Promise<void> {
  requireNative();
  await invoke("confirm_exit");
}

export async function cancelNativeExit(): Promise<void> {
  if (!isTauri()) return;
  await invoke("cancel_exit");
}

export async function releaseSource(sourceId: string): Promise<void> {
  requireNative();
  await invoke("release_source", { sourceId });
}

export const saveTicketVariant = (ticket: SaveTicket): SourceVariant =>
  extensionOf(ticket.destinationVariant);
