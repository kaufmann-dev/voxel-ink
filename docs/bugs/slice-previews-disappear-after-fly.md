# Slice previews disappear after Fly

## Fix context

- Timestamp: 2026-07-23 13:16:23 CEST (+0200)
- Pre-fix commit: `e957366db4c5a177a8a696d304f063e79e0bf039`

## Symptom

After selecting Fly in the 3D pane and restoring the four-view workspace, Paint,
Erase, and Bridge previews in the 2D slices could remain absent or flicker.
The previews normally worked before entering Fly.

## Confirmed root cause

The Fly pane submitted a center-ray preview whenever Fly was the selected view
mode, even when its cursor was not captured and the pointer was hovering a 2D
slice. Because the 3D pane is processed after the slice panes and both use one
latest-only asynchronous preview channel, the 3D request superseded the slice
request every frame. Workspace maximize and restore changes also did not
invalidate in-flight preview work, and a moving slice pointer cleared each
accepted slice result before its replacement arrived.

A UI reproduction selected Fly, maximized and restored the 3D pane, then hovered
the axial slice with Paint selected. The test timed out with preview geometry
only in one slice intersection from the 3D footprint. The same reproduction
passed for Paint, Erase, and Bridge after the fix.

## Changes

- Limited persistent center-ray preview requests to captured Fly sessions;
  uncaptured Fly now previews only while the 3D pane is actually hovered.
- Invalidated preview generations when maximizing or restoring a workspace
  pane, and stopped the outgoing pane from submitting more input in that frame.
- Retained the last valid slice preview while a newer slice request is
  calculated, without changing the immediate stale-preview clearing required
  by moving 3D cameras.
- Added preview lifecycle and rendered UI regressions for the Fly-to-slice
  transition and same-slice replacement behavior.
