# Fly controls and view settings

## Fix context

- Timestamp: 2026-07-19 12:03:11 CEST (+0200)
- Pre-fix commit: `c48da91a5872a63fb57386f4d659b1f32783ef75`

## Symptom

While the Fly cursor was captured, undo and redo shortcuts did nothing. The Orbit selector label wrapped in a narrow 3D pane, selected Orbit/Fly labels did not reliably render black, the default flight speed was too high, and voxel outlines were forced on only in Fly mode with no user control.

## Confirmed root cause

Captured input explicitly consumed every undo and redo chord before returning from shortcut handling. The Orbit button had only 48 points of width despite 20 points of horizontal theme padding and inherited egui's wrapping behavior. Selected labels depended on theme fallback colors. Fly initialized at half the scene radius per second, and the renderer derived its seam uniform directly from `VolumeViewMode::Fly`.

## Changes

- Kept captured file shortcuts suppressed while allowing `Ctrl+Z`, `Ctrl+Shift+Z`, and `Ctrl+Y` through the existing history command path.
- Widened the view selector buttons, forced single-line labels, and explicitly colored selected labels black.
- Reduced initial Fly speed from `0.5` to `0.1` scene radii per second and narrowed its bounded range.
- Added a logarithmic flight-speed slider to the repair panel.
- Replaced the Fly-only seam condition with a session-level voxel-outline setting shared by Orbit and Fly, exposed through a repair-panel checkbox.
- Added renderer and UI regressions for shortcut behavior, speed bounds, shared outline uniforms, selector layout/color, and panel containment.
