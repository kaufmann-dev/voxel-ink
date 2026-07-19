# Oversized bridge preview opens a modal before release

## Fix context

- Timestamp: 2026-07-19 16:11:41 CEST (+0200)
- Pre-fix commit: `cdd8fc3407bf1cf691f4fd09773529dd7432040f`

## Symptom

Dragging a large Bridge could open a modal error before the mouse button was released. The message incorrectly described valid volume dimensions as invalid, discarded the bridge anchor, and interrupted Fly capture even though the user had not asked VoxelInk to commit the operation yet.

## Confirmed root cause

Bridge previews and committed bridges used different footprint builders and different safety behavior. The preview builder treated its 2,000,000 candidate-visit rendering guard as an `InvalidDimensions` failure, while the commit builder accumulated an unbounded `HashSet`. The application handled every preview failure as fatal, so an expected visualization limit cleared the gesture and opened the global error dialog. The limit also counted duplicate and already-occupied candidates rather than the voxels Bridge would actually add.

## Changes

- Replaced the duplicate bridge builders with one cancellable streaming traversal that preserves affine-aware brush geometry.
- Added separate 16,000,000-cell preview and 64,000,000-cell commit work budgets, plus the renderer-backed 2,000,000 changed-voxel cap.
- Filtered occupied voxels before deduplication and fully collected a bounded footprint before opening undo history, keeping rejected or cancelled commits atomic.
- Added typed preview-limit and commit-rejection engine events instead of routing expected limits through invalid-dimension errors.
- Kept the anchor and endpoint visible with a yellow nonmodal notice when an exact preview is unavailable, and restored the exact preview automatically after shortening the bridge.
- Made Escape, focus loss, mode changes, and tool changes cancel in-flight bridge collection. Expected final rejections leave Fly capture intact and report that no voxels changed.
