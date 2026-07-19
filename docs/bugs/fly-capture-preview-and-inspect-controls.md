# Fly capture, preview, and Inspect controls

## Fix context

- Timestamp: 2026-07-19 12:35:36 CEST (+0200)
- Pre-fix commit: `053e56e2d029334f6eba30e49e843fbe528e5b8c`

## Symptom

Dragging flight speed to its minimum could jump it to the maximum when the viewport captured the cursor. Inspect showed disabled Shape, Depth, and Size controls that belong only to Paint, Erase, and Bridge. In Fly mode, a selected-tool voxel highlight could appear displaced or much larger than the center-reticle target. Brush diameter was also audited for the capture-time jump.

## Confirmed root cause

Fly began accepting raw wheel events on the pointer-button release frame even though the initial capture click was still suppressed. A platform pointer-lock wheel spike could overflow the exponential speed adjustment and clamp directly to the maximum. Brush diameter had no capture-time mutation path and remained stable in both voxel and millimetre modes.

The Inspect regression used `add_enabled_ui`, which renders children in a disabled state, where the intended behavior was to omit brush-only controls entirely.

Hover previews are computed asynchronously. When the Fly camera moved, the previously accepted world-space brush footprint stayed visible until its replacement arrived, so the old highlight was rendered from the new camera and appeared offset or oversized. Affine center-ray and nearest-surface picking remained correct.

## Changes

- Kept look, movement, and wheel input inert until the Fly capture click is fully released.
- Bounded raw wheel adjustment per event and per frame so platform spikes cannot saturate flight speed.
- Added capture regressions for minimum flight speed and both brush-diameter unit modes.
- Hid Shape, Depth, and Size for Inspect while retaining them for Paint, Erase, and Bridge.
- Cleared an accepted hover footprint immediately when submitting a different preview request, and rejected superseded preview generations before they can repopulate the highlight.
- Added UI, layout, input, and preview-lifecycle regression coverage.
