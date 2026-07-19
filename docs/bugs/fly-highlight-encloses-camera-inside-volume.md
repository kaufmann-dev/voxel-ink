# Fly highlight encloses the camera inside occupied data

## Fix context

- Timestamp: 2026-07-19 12:57:57 CEST (+0200)
- Pre-fix commit: `a32be471d8f2fabbce8ce87999de391b2b09a4a7`

## Symptom

In Fly mode, an Erase or other selected-tool preview could surround the camera and cover most of the viewport instead of remaining a compact footprint at the center reticle. The same brush highlighted correctly in Orbit. This was a picking defect triggered only when the camera occupied solid data. It was not the close-range rendering defect that broke otherwise-correct highlight faces into triangular or partial regions.

## Confirmed root cause

No-clip Fly can place the camera inside an occupied voxel. Ray picking returned the ray's first occupied voxel unconditionally, which in this case was the voxel containing the camera. A 3D brush centered there surrounded the viewpoint, causing its boundary geometry to fill the screen. Orbit cameras remain outside the volume and therefore did not trigger this path.

A native software-Vulkan reproduction with a solid volume produced the oversized, room-like highlight after flying through the outer surface. The same reproduction showed a compact footprint on the forward wall after correcting the picker. The previous preview-generation cleanup remains valid for its separate transient stale-frame defect. Coplanar highlight depth fighting is documented separately in `fly-highlight-depth-fighting.md`.

## Changes

- Changed affine-aware surface traversal so a ray beginning in occupied data selects that occupied run's forward exit surface.
- Preserved existing outside-to-inside nearest-surface behavior and returned an in-bounds face-adjacent exterior voxel when one exists.
- Added regression coverage for forward and reverse exit surfaces, painting and erasing, full-volume boundary exits, and a nontrivial affine.
- Increased initial Fly speed from `0.10` to `0.15` volume radii per second and minimum speed from `0.001` to `0.002` radii per second.
