# Fly highlight depth fighting at close range

## Fix context

- Timestamp: 2026-07-19 14:16:04 CEST (+0200)
- Pre-fix commit: `eba74bacdddf1fb8ee32518e88f2ce69295cdc54`

## Symptom

With Erase selected, a brush near 30 mm, and the Fly camera very close to `craniofacial_structures_combined.nii.gz`, correctly selected voxel faces broke into gray triangular or shard-like regions. Some faces looked partial or sheared. Orbit rendering remained stable, and the underlying gray voxel mesh retained intact rectangular faces.

This was distinct from a stale asynchronous preview during motion and from choosing the camera's containing voxel when the camera was inside occupied data. Those behaviors have independent reproductions and regression tests.

## Confirmed root cause

The highlight pass draws translucent unit faces coplanar with the opaque greedy volume faces. The original `LessEqual` depth state used a constant bias of `-2` and slope bias of `-0.5`. That offset was too weak at close Fly projections on the native Vulkan driver, allowing parts of the opaque surface to win the depth test. Different triangulation between unit highlight faces and larger greedy quads made the failures appear as triangles and shards even though neither mesh was corrupt.

The supplied NIfTI was rendered natively with the same camera, brush, affine, clipping, and four-sample MSAA while changing only the highlight depth state. Zero bias reproduced incomplete faces. `Always` comparison removed the holes but incorrectly revealed occluded highlight faces. A stronger negative bias retained `LessEqual` occlusion and produced intact faces. Because clipping and MSAA were unchanged during the successful A/B, neither was the primary cause.

## Changes

- Kept `LessEqual`, disabled highlight depth writes, replaced the weak mixed bias with a constant `-64` depth-unit bias, and removed slope bias.
- Left greedy chunk meshes, highlighted voxel selection, brush semantics, and affine geometry unchanged.
- Removed the speculative exposed-surface masking experiment; it addressed a different diagnosis and was not needed.
- Added software-Vulkan pixel readback coverage for a close, oblique, four-sample render using the supplied NIfTI's anisotropic affine and index scale.
- Added exact depth-state assertions so the driver-specific weak-bias configuration cannot return unnoticed.
