import type { Vec3 } from "./affine";
import {
  isCoordinateInBounds,
  linearIndex,
  voxelCount,
  type Dimensions,
  type GridCoordinate,
  type SliceAxis,
} from "./volume";

export interface IndexRay {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface DdaHit {
  readonly coordinate: GridCoordinate;
  readonly index: number;
  /** Outward normal of the filled face through which the ray entered. */
  readonly normal: GridCoordinate;
  /** Ray parameter in origin + direction * distance; direction need not be normalized. */
  readonly distance: number;
}

const DDA_EPSILON = 1e-9;

function validateRay(ray: IndexRay): void {
  if (!ray.origin.every(Number.isFinite) || !ray.direction.every(Number.isFinite)) {
    throw new Error("Ray origin and direction must contain only finite values");
  }
  if (ray.direction.every((component) => component === 0)) {
    throw new Error("Ray direction must be non-zero");
  }
}

interface BoundsIntersection {
  readonly enter: number;
  readonly exit: number;
  readonly normal: GridCoordinate;
}

function intersectVolumeBounds(
  ray: IndexRay,
  dimensions: Dimensions,
): BoundsIntersection | null {
  let enter = Number.NEGATIVE_INFINITY;
  let exit = Number.POSITIVE_INFINITY;
  let entryAxis: SliceAxis | null = null;
  let entrySign = 0;

  for (const axis of [0, 1, 2] as const) {
    const minimum = -0.5;
    const maximum = dimensions[axis] - 0.5;
    const origin = ray.origin[axis];
    const direction = ray.direction[axis];
    if (direction === 0) {
      if (origin < minimum || origin > maximum) {
        return null;
      }
      continue;
    }

    const minimumT = (minimum - origin) / direction;
    const maximumT = (maximum - origin) / direction;
    const nearT = Math.min(minimumT, maximumT);
    const farT = Math.max(minimumT, maximumT);
    if (nearT > enter) {
      enter = nearT;
      entryAxis = axis;
      entrySign = direction > 0 ? -1 : 1;
    }
    exit = Math.min(exit, farT);
    if (enter > exit) {
      return null;
    }
  }

  if (exit < 0) {
    return null;
  }

  const normal = [0, 0, 0] as [number, number, number];
  if (enter >= 0 && entryAxis !== null) {
    normal[entryAxis] = entrySign;
  }
  return { enter: Math.max(enter, 0), exit, normal };
}

function pointAlongRay(ray: IndexRay, distance: number): Vec3 {
  return [
    ray.origin[0] + ray.direction[0] * distance,
    ray.origin[1] + ray.direction[1] * distance,
    ray.origin[2] + ray.direction[2] * distance,
  ];
}

function cellContainingPoint(
  point: Vec3,
  dimensions: Dimensions,
): [number, number, number] {
  return [
    Math.min(dimensions[0] - 1, Math.max(0, Math.floor(point[0] + 0.5))),
    Math.min(dimensions[1] - 1, Math.max(0, Math.floor(point[1] + 0.5))),
    Math.min(dimensions[2] - 1, Math.max(0, Math.floor(point[2] + 0.5))),
  ];
}

export function firstOccupiedVoxel(
  occupancy: Uint8Array,
  dimensions: Dimensions,
  ray: IndexRay,
): DdaHit | null {
  validateRay(ray);
  const expectedLength = voxelCount(dimensions);
  if (occupancy.length !== expectedLength) {
    throw new Error("Occupancy length does not match dimensions");
  }
  const bounds = intersectVolumeBounds(ray, dimensions);
  if (!bounds) {
    return null;
  }

  const directionScale = Math.max(...ray.direction.map(Math.abs));
  const startDistance = Math.min(
    bounds.exit,
    bounds.enter + DDA_EPSILON / directionScale,
  );
  const coordinate = cellContainingPoint(
    pointAlongRay(ray, startDistance),
    dimensions,
  );
  const step = ray.direction.map((component) =>
    component > 0 ? 1 : component < 0 ? -1 : 0,
  ) as [number, number, number];
  const nextBoundaryDistance: [number, number, number] = [0, 0, 0];
  const deltaDistance: [number, number, number] = [0, 0, 0];

  for (const axis of [0, 1, 2] as const) {
    if (step[axis] === 0) {
      nextBoundaryDistance[axis] = Number.POSITIVE_INFINITY;
      deltaDistance[axis] = Number.POSITIVE_INFINITY;
      continue;
    }
    const boundary = coordinate[axis] + (step[axis] > 0 ? 0.5 : -0.5);
    nextBoundaryDistance[axis] =
      (boundary - ray.origin[axis]) / ray.direction[axis];
    deltaDistance[axis] = 1 / Math.abs(ray.direction[axis]);
  }

  let distance = bounds.enter;
  let normal = [...bounds.normal] as [number, number, number];
  while (isCoordinateInBounds(dimensions, coordinate)) {
    const index = linearIndex(dimensions, coordinate);
    if (occupancy[index] !== 0) {
      return {
        coordinate: [...coordinate] as GridCoordinate,
        index,
        normal: [...normal] as GridCoordinate,
        distance,
      };
    }

    const nextDistance = Math.min(...nextBoundaryDistance);
    if (nextDistance > bounds.exit + DDA_EPSILON) {
      break;
    }

    const crossedAxes: SliceAxis[] = [];
    for (const axis of [0, 1, 2] as const) {
      if (Math.abs(nextBoundaryDistance[axis] - nextDistance) <= DDA_EPSILON) {
        crossedAxes.push(axis);
      }
    }
    if (crossedAxes.length === 0) {
      break;
    }

    normal = [0, 0, 0];
    let faceAxis = crossedAxes[0]!;
    for (const axis of crossedAxes.slice(1)) {
      if (Math.abs(ray.direction[axis]) > Math.abs(ray.direction[faceAxis])) {
        faceAxis = axis;
      }
    }
    normal[faceAxis] = -step[faceAxis];
    for (const axis of crossedAxes) {
      coordinate[axis] += step[axis];
      nextBoundaryDistance[axis] += deltaDistance[axis];
    }
    distance = nextDistance;
  }

  return null;
}

export function adjacentCoordinate(
  hit: DdaHit,
  dimensions: Dimensions,
): GridCoordinate | null {
  if (hit.normal[0] === 0 && hit.normal[1] === 0 && hit.normal[2] === 0) {
    return null;
  }
  const adjacent: GridCoordinate = [
    hit.coordinate[0] + hit.normal[0],
    hit.coordinate[1] + hit.normal[1],
    hit.coordinate[2] + hit.normal[2],
  ];
  return isCoordinateInBounds(dimensions, adjacent) ? adjacent : null;
}

export function intersectActiveSlice(
  ray: IndexRay,
  dimensions: Dimensions,
  axis: SliceAxis,
  slice: number,
): GridCoordinate | null {
  validateRay(ray);
  if (!Number.isInteger(axis) || axis < 0 || axis > 2) {
    throw new RangeError("Slice axis must be 0, 1, or 2");
  }
  if (!Number.isInteger(slice) || slice < 0 || slice >= dimensions[axis]) {
    throw new RangeError("Slice is outside the volume");
  }

  const axisDirection = ray.direction[axis];
  if (axisDirection === 0) {
    return null;
  }
  const distance = (slice - ray.origin[axis]) / axisDirection;
  if (distance < 0) {
    return null;
  }

  const point = pointAlongRay(ray, distance);
  const coordinate = point.map((component, componentAxis) =>
    componentAxis === axis ? slice : Math.floor(component + 0.5),
  ) as [number, number, number];
  return isCoordinateInBounds(dimensions, coordinate) ? coordinate : null;
}
