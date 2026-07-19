# Inconsistent button hover feedback

## Fix context

- Timestamp: 2026-07-19 17:47:17 CEST (+0200)
- Pre-fix commit: `ba6a7674a5ba1b20e054a5306e59c0764d6d6a39`

## Symptom

The selected Fly button appeared to gain a one-pixel border when hovered. The full-width Fill enclosed cavities action did not visibly react to hover, and the tool, shape, depth, and brush-unit selection buttons also lacked consistent hover feedback.

## Confirmed root cause

The three button families used different egui styling paths. Normal and view-mode buttons inherited egui's interaction-state visuals, including its default hovered expansion and cyan stroke. Custom selection buttons and the cavity action instead set `Button::fill` and `Button::stroke` directly, which override egui's hovered fill and stroke. As a result, the Fly button expanded and changed outline while the explicitly painted buttons stayed visually static.

## Changes

- Added shared state-aware button palettes for selection controls, selected view modes, and accent actions.
- Made every enabled button communicate hover through a visible fill change.
- Kept button border widths and painted bounds stable across idle, hovered, and pressed states.
- Preserved tool-specific accents, selected-state colors, disabled behavior, and forced-colors contrast.
- Added rendered UI coverage for normal, selected, unselected, view-mode, and full-width action buttons, plus palette-level coverage for stable border widths.
