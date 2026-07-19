# Uneven 3D header button spacing

## Fix context

- Timestamp: 2026-07-19 17:57:42 CEST (+0200)
- Pre-fix commit: `47ee99e139ae10734f5f288ddde3a19e48e36969`

## Symptom

The Orbit and Fly buttons in the 3D pane header were visibly closer together than Fly, Reset, and Expand.

## Confirmed root cause

The header positioned its controls manually. Orbit–Fly used a dedicated hard-coded two-point gap, while Fly–Reset and Reset–Expand each used five points. A rendered layout reproduction measured the three gaps as 2, 5, and 5 points.

## Changes

- Replaced the separate literals with one five-point 3D header control-gap constant.
- Applied the shared gap to Orbit–Fly, Fly–Reset, and Reset–Expand.
- Extended the existing rendered selector test to require all three measured gaps to match.
