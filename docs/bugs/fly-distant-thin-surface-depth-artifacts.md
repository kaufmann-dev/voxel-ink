# Fly depth artifacts on distant thin surfaces

## Fix context

- Timestamp: 2026-07-23 11:52:01 CEST (+0200)
- Pre-fix commit: `0183b782fa85251eac58185f8f2855f06f4364fa`

## Symptom

In Fly mode, distant parts of thin voxel sheets could develop rectangular or
triangular patches. The wrong patch was dark in normal rendering and could have
a completely different leak-light color. Moving closer removed the artifact,
Orbit did not reproduce it, and disabling voxel outlines had no effect.

## Confirmed root cause

Fly permits the camera to remain inside the volume bounds, where its
conventional perspective projection used a near plane as small as `0.0001`
scene radii. With `Depth24Plus`, one-voxel-separated front and rear faces at the
far side of a large volume could quantize to the same depth. Both face
directions are intentionally rendered, and `LessEqual` allowed the later rear
face to replace the nearer face when their stored depths matched.

A software-Vulkan reproduction placed two differently shaded,
one-unit-separated faces 4,096 units from a Fly camera and drew the farther face
last. Reversed-Z with a floating-point depth attachment kept their depths
distinct in both normal and leak-light rendering.

## Changes

- Switched the application and volume pipelines to `Depth32Float`.
- Replaced finite conventional projections with infinite reversed-Z
  projections for Orbit and Fly.
- Reset egui's shared depth attachment to the reversed-Z far value inside the
  volume pane before drawing.
- Mirrored opaque, leak-light, and highlight depth comparisons and highlight
  bias for reversed-Z.
- Updated camera-ray reconstruction for the reversed depth convention.
- Added projection, pipeline-state, distant thin-sheet, and existing
  close-highlight software-Vulkan coverage.
