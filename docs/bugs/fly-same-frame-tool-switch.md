# Fly edit uses the previous tool after a same-frame shortcut

## Fix context

- Timestamp: 2026-07-19 14:59:51 CEST (+0200)
- Pre-fix commit: `bf462de386cd2ae13c584e426c3cb5daa83b0114`

## Symptom

In captured Fly mode, Paint could remove voxels or Erase could add voxels immediately after switching tools with `B` or `E`. The selected button and preview used the new tool, making the completed edit appear to have inverted semantics.

## Confirmed root cause

The application processed pane pointer input before keyboard tool shortcuts. When a tool shortcut and primary click arrived in the same egui frame, the 3D pane began the stroke with the previous `Tool`. At the end of that frame, shortcut handling changed the selected tool shown by the interface, but the engine correctly continued the already-submitted stroke with its original value. The engine's Paint-to-one and Erase-to-zero mappings were not inverted.

A UI reproduction selected Erase, captured Fly, then submitted `B` and a center-ray click in one frame. Before the fix, the interface selected Paint while the occupied count decreased. The reverse transition could fail the same way.

## Changes

- Processed keyboard tool selection before rendering and handling pane interactions.
- Kept file, history, cursor, and capture shortcut handling in its existing post-layout path.
- Added a Fly UI regression that switches from Erase to Paint and Paint to Erase in the same frame as a click, verifying the occupied count changes in the selected tool's direction.
