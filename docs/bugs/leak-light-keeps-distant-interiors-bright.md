# Leak light keeps distant interiors bright

## Fix context

- Timestamp: 2026-07-22 22:30:23 CEST (+0200)
- Pre-fix commit: `97c75197d635e769840c41d69e81a958d76a0440`

## Symptom

After leak distances were made reach-independent, surfaces deep inside an
exterior-connected path stayed almost fully bright regardless of their physical
distance from the opening. Sealed cavities still used their separate dark
sentinel, but the distance gradient for every connected path was effectively
lost.

## Confirmed root cause

Leak-distance steps are stored in a `Snorm16x4` vertex attribute. The GPU
normalizes connected values from `0..32767` to `0.0..1.0`, but the fragment
shader multiplied that normalized value directly by the per-step physical
scale. It therefore treated the farthest representable connected point as one
tiny fractional step away instead of 32,767 steps away. The original GPU test
validated the packed bits before vertex conversion and did not exercise the
rendered intensity.

## Changes

- Restored the 32,767-step magnitude after signed-normalized vertex decoding and
  before converting the distance to physical units.
- Added a shader contract regression for the signed-normalized range conversion.
- Extended the software-Vulkan test through final pixel readback and required a
  maximally distant connected surface to render dark.
