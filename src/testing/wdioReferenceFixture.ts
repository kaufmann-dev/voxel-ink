import { IDENTITY_AFFINE } from "../core/affine";
import { useEditorStore } from "../editor/store";

const REFERENCE_DIMENSIONS = [512, 512, 256] as const;

export function installWdioReferenceFixture(): void {
  const [nx, ny, nz] = REFERENCE_DIMENSIONS;
  const occupancy = new Uint8Array(nx * ny * nz);
  const center = [nx * 0.33, ny * 0.5, nz * 0.5] as const;
  const radii = [nx * 0.2, ny * 0.31, nz * 0.34] as const;
  let occupiedCount = 0;

  for (let z = 0; z < nz; z += 1) {
    const normalizedZ = (z - center[2]) / radii[2];
    const zSquared = normalizedZ * normalizedZ;
    if (zSquared > 1) continue;
    for (let y = 0; y < ny; y += 1) {
      const normalizedY = (y - center[1]) / radii[1];
      const remaining = 1 - zSquared - normalizedY * normalizedY;
      if (remaining < 0) continue;
      const halfWidth = radii[0] * Math.sqrt(remaining);
      const startX = Math.max(0, Math.ceil(center[0] - halfWidth));
      const endX = Math.min(nx - 1, Math.floor(center[0] + halfWidth));
      if (startX > endX) continue;
      const rowStart = nx * (y + ny * z);
      occupancy.fill(1, rowStart + startX, rowStart + endX + 1);
      occupiedCount += endX - startX + 1;
    }
  }

  useEditorStore.getState().installDocument({
    dimensions: REFERENCE_DIMENSIONS,
    occupancy,
    occupiedCount,
    geometry: {
      indexToWorld: IDENTITY_AFFINE,
      worldToIndex: IDENTITY_AFFINE,
      spacing: [1, 1, 1],
      units: ["mm", "mm", "mm"],
    },
    sourceFormat: "binvox",
    sourceVariant: ".binvox",
    saveState: {
      kind: "binvox",
      variant: ".binvox",
      version: 1,
      translation: [0, 0, 0],
      scale: 1,
      dimensions: REFERENCE_DIMENSIONS,
    },
    normalization: { convertedNonBinary: false },
  }, {
    sourceId: "00000000-0000-4000-8000-000000000512",
    name: "wdio-reference-512x512x256.binvox",
    extension: ".binvox",
    size: occupancy.byteLength,
  });
  useEditorStore.getState().setCursor([450, 450, 127]);
}
