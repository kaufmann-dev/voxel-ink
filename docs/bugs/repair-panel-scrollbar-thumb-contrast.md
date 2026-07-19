# Repair panel scrollbar thumb contrast

## Fix context

- Timestamp: 2026-07-19 14:47:47 CEST (+0200)
- Pre-fix commit: `a0a2c112023df0481d55d2f0edd63dccd27d4ffe`

## Symptom

The repair panel's scrollbar thumb was invisible while idle and only barely visible on hover. Its position became clear only while the scrollbar was actively dragged, when the thumb changed to cyan.

## Confirmed root cause

egui's solid scrollbar paints its rail with `extreme_bg_color` and its thumb with the current widget state's `bg_fill`. VoxelInk configured both the rail and inactive widget fill as gray 24, making the idle thumb identical to the rail. The hovered widget fill was gray 36, which provided only a slight change. The active fill was cyan and already had sufficient contrast.

## Changes

- Added repair-scrollbar-specific idle gray 104 and hovered gray 166 thumb colors while preserving the active cyan state.
- Restored the normal application style inside the scroll area's content so buttons and other controls do not inherit the scrollbar colors.
- Kept forced-colors mode at a white thumb on a black rail.
- Added a regression test for distinct idle, hovered, and active states and for isolation from the base content style.
