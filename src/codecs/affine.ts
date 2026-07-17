import {
  affineReciprocalCondition,
  createMatrix4,
  IDENTITY_AFFINE,
  invertAffine,
  MIN_AFFINE_RECIPROCAL_CONDITION,
  spacingFromAffine,
  type Matrix4,
} from "../core/affine";
import { CodecError, type VolumeGeometry } from "./types";

export const identityAffine = (): Matrix4 => IDENTITY_AFFINE;

export const affineFromRows = (
  rows: readonly (readonly [number, number, number, number])[],
): Matrix4 => {
  const row0 = rows[0] ?? [1, 0, 0, 0];
  const row1 = rows[1] ?? [0, 1, 0, 0];
  const row2 = rows[2] ?? [0, 0, 1, 0];
  return createMatrix4([
    ...row0,
    ...row1,
    ...row2,
    0,
    0,
    0,
    1,
  ]);
};

export const isValidAffine = (matrix: readonly number[]): boolean => {
  if (matrix.length !== 16 || !matrix.every(Number.isFinite)) return false;
  return (
    affineReciprocalCondition(matrix) >=
    MIN_AFFINE_RECIPROCAL_CONDITION
  );
};

export const geometryFromAffine = (
  indexToWorld: Matrix4,
  units: readonly [string | null, string | null, string | null],
): VolumeGeometry => {
  if (!isValidAffine(indexToWorld)) {
    throw new CodecError("invalid-affine", "The volume affine is invalid.");
  }
  return {
    indexToWorld,
    worldToIndex: invertAffine(indexToWorld),
    spacing: spacingFromAffine(indexToWorld),
    units,
  };
};
