import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MeshPayload } from "./mesh";

interface WorkerOutput {
  readonly type: "ready" | "mesh";
  readonly revision: number;
  readonly generation: number;
  readonly payload?: MeshPayload;
}

describe("mesh worker protocol", () => {
  const posted = vi.fn<(message: WorkerOutput) => void>();

  beforeEach(() => {
    vi.resetModules();
    posted.mockReset();
    vi.stubGlobal("postMessage", posted);
  });

  afterEach(() => {
    self.onmessage = null;
    vi.unstubAllGlobals();
  });

  it("cancels old generations and applies revision-decreasing undo deltas", async () => {
    await import("./mesh.worker");
    const dispatch = (data: unknown): void => {
      self.onmessage?.call(self, new MessageEvent("message", { data }));
    };

    dispatch({
      type: "init",
      dimensions: [1, 1, 1],
      occupancy: Uint8Array.of(1).buffer,
      revision: 9,
      generation: 1,
    });
    dispatch({
      type: "init",
      dimensions: [1, 1, 1],
      occupancy: Uint8Array.of(1).buffer,
      revision: 10,
      generation: 2,
    });
    const beforeStaleRequest = posted.mock.calls.length;
    dispatch({
      type: "mesh",
      chunk: { x: 0, y: 0, z: 0 },
      revision: 9,
      generation: 1,
    });
    expect(posted).toHaveBeenCalledTimes(beforeStaleRequest);

    dispatch({
      type: "apply",
      indices: Uint32Array.of(0).buffer,
      values: Uint8Array.of(0).buffer,
      revision: 4,
      generation: 2,
    });
    dispatch({
      type: "mesh",
      chunk: { x: 0, y: 0, z: 0 },
      revision: 4,
      generation: 2,
    });

    const result = posted.mock.calls.map(([message]) => message).find(
      (message) => message.type === "mesh" && message.revision === 4,
    );
    expect(result?.payload?.faceCount).toBe(0);
  });
});
