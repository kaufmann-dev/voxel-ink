export type Vec3 = readonly [number, number, number];

/** Row-major matrix mapping [x, y, z, 1] from index space to world space. */
export type Matrix4 = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export const IDENTITY_AFFINE: Matrix4 = Object.freeze([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

const AFFINE_EPSILON = 1e-12;
export const MIN_AFFINE_RECIPROCAL_CONDITION = 1e-12;

interface LinearAnalysis {
  readonly scale: number;
  readonly cofactors: readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  readonly determinant: number;
  readonly reciprocalCondition: number;
}

function analyzeLinearPart(matrix: readonly number[]): LinearAnalysis | null {
  if (matrix.length !== 16 || !matrix.every(Number.isFinite)) return null;

  const linear = [
    matrix[0]!,
    matrix[1]!,
    matrix[2]!,
    matrix[4]!,
    matrix[5]!,
    matrix[6]!,
    matrix[8]!,
    matrix[9]!,
    matrix[10]!,
  ] as const;
  const scale = Math.max(...linear.map(Math.abs));
  if (!(scale > 0) || !Number.isFinite(scale)) return null;

  const [a00, a01, a02, a10, a11, a12, a20, a21, a22] = linear.map(
    (value) => value / scale,
  );
  const c00 = a11! * a22! - a12! * a21!;
  const c01 = a02! * a21! - a01! * a22!;
  const c02 = a01! * a12! - a02! * a11!;
  const c10 = a12! * a20! - a10! * a22!;
  const c11 = a00! * a22! - a02! * a20!;
  const c12 = a02! * a10! - a00! * a12!;
  const c20 = a10! * a21! - a11! * a20!;
  const c21 = a01! * a20! - a00! * a21!;
  const c22 = a00! * a11! - a01! * a10!;
  const cofactors = [c00, c01, c02, c10, c11, c12, c20, c21, c22] as const;
  const determinant = a00! * c00 + a01! * c10 + a02! * c20;
  if (determinant === 0 || !Number.isFinite(determinant)) return null;

  const matrixNorm = Math.max(
    Math.abs(a00!) + Math.abs(a01!) + Math.abs(a02!),
    Math.abs(a10!) + Math.abs(a11!) + Math.abs(a12!),
    Math.abs(a20!) + Math.abs(a21!) + Math.abs(a22!),
  );
  const adjugateNorm = Math.max(
    Math.abs(c00) + Math.abs(c01) + Math.abs(c02),
    Math.abs(c10) + Math.abs(c11) + Math.abs(c12),
    Math.abs(c20) + Math.abs(c21) + Math.abs(c22),
  );
  const denominator = matrixNorm * adjugateNorm;
  const reciprocalCondition = Math.abs(determinant) / denominator;
  if (!(reciprocalCondition > 0) || !Number.isFinite(reciprocalCondition)) {
    return null;
  }

  return { scale, cofactors, determinant, reciprocalCondition };
}

export function affineReciprocalCondition(matrix: readonly number[]): number {
  return analyzeLinearPart(matrix)?.reciprocalCondition ?? 0;
}

function assertFiniteVector(vector: Vec3, name: string): void {
  if (!vector.every(Number.isFinite)) {
    throw new Error(`${name} must contain only finite values`);
  }
}

export function createMatrix4(values: readonly number[]): Matrix4 {
  if (values.length !== 16 || !values.every(Number.isFinite)) {
    throw new Error("An affine matrix must contain exactly 16 finite values");
  }

  if (
    Math.abs(values[12]!) > AFFINE_EPSILON ||
    Math.abs(values[13]!) > AFFINE_EPSILON ||
    Math.abs(values[14]!) > AFFINE_EPSILON ||
    Math.abs(values[15]! - 1) > AFFINE_EPSILON
  ) {
    throw new Error("The matrix must be affine with a final row of [0, 0, 0, 1]");
  }

  return Object.freeze([...values]) as unknown as Matrix4;
}

export function transformPoint(matrix: Matrix4, point: Vec3): Vec3 {
  assertFiniteVector(point, "Point");
  const [x, y, z] = point;
  return [
    matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
    matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
    matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
  ];
}

export function transformDirection(matrix: Matrix4, direction: Vec3): Vec3 {
  assertFiniteVector(direction, "Direction");
  const [x, y, z] = direction;
  return [
    matrix[0] * x + matrix[1] * y + matrix[2] * z,
    matrix[4] * x + matrix[5] * y + matrix[6] * z,
    matrix[8] * x + matrix[9] * y + matrix[10] * z,
  ];
}

export function invertAffine(matrix: Matrix4): Matrix4 {
  const analysis = analyzeLinearPart(matrix);
  if (
    analysis === null ||
    analysis.reciprocalCondition < MIN_AFFINE_RECIPROCAL_CONDITION
  ) {
    throw new Error("The affine matrix is singular or ill-conditioned");
  }

  const [c00, c01, c02, c10, c11, c12, c20, c21, c22] =
    analysis.cofactors;
  const inverseDeterminant = 1 / analysis.determinant / analysis.scale;
  const i00 = c00 * inverseDeterminant;
  const i01 = c01 * inverseDeterminant;
  const i02 = c02 * inverseDeterminant;
  const i10 = c10 * inverseDeterminant;
  const i11 = c11 * inverseDeterminant;
  const i12 = c12 * inverseDeterminant;
  const i20 = c20 * inverseDeterminant;
  const i21 = c21 * inverseDeterminant;
  const i22 = c22 * inverseDeterminant;

  const tx = matrix[3];
  const ty = matrix[7];
  const tz = matrix[11];

  const inverseValues = [
    i00,
    i01,
    i02,
    -(i00 * tx + i01 * ty + i02 * tz),
    i10,
    i11,
    i12,
    -(i10 * tx + i11 * ty + i12 * tz),
    i20,
    i21,
    i22,
    -(i20 * tx + i21 * ty + i22 * tz),
    0,
    0,
    0,
    1,
  ];
  if (!inverseValues.every(Number.isFinite)) {
    throw new Error("The affine matrix is singular or ill-conditioned");
  }

  return createMatrix4(inverseValues);
}

export function spacingFromAffine(matrix: Matrix4): Vec3 {
  const spacing: Vec3 = [
    Math.hypot(matrix[0], matrix[4], matrix[8]),
    Math.hypot(matrix[1], matrix[5], matrix[9]),
    Math.hypot(matrix[2], matrix[6], matrix[10]),
  ];

  if (!spacing.every((value) => value > 0 && Number.isFinite(value))) {
    throw new Error("Every affine axis must have finite, non-zero spacing");
  }

  return spacing;
}

export function multiplyAffine(left: Matrix4, right: Matrix4): Matrix4 {
  const result = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let inner = 0; inner < 4; inner += 1) {
        const outputIndex = row * 4 + column;
        result[outputIndex] =
          result[outputIndex]! +
          left[row * 4 + inner]! * right[inner * 4 + column]!;
      }
    }
  }
  return createMatrix4(result);
}

export function indexRayFromWorld(
  worldToIndex: Matrix4,
  origin: Vec3,
  direction: Vec3,
): { origin: Vec3; direction: Vec3 } {
  const indexDirection = transformDirection(worldToIndex, direction);
  if (indexDirection.every((component) => component === 0)) {
    throw new Error("Ray direction becomes zero in index space");
  }
  return {
    origin: transformPoint(worldToIndex, origin),
    direction: indexDirection,
  };
}
