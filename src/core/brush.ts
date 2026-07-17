import {
  isCoordinateInBounds,
  linearIndex,
  type Dimensions,
  type GridCoordinate,
  type SliceAxis,
} from "./volume";

export const MIN_BRUSH_DIAMETER = 1;
export const MAX_BRUSH_DIAMETER = 31;

export function validateBrushDiameter(diameter: number): number {
  if (
    !Number.isInteger(diameter) ||
    diameter < MIN_BRUSH_DIAMETER ||
    diameter > MAX_BRUSH_DIAMETER ||
    diameter % 2 === 0
  ) {
    throw new RangeError("Brush diameter must be an odd integer from 1 through 31");
  }
  return diameter;
}

function validateCoordinate(coordinate: GridCoordinate, name: string): void {
  if (!coordinate.every(Number.isFinite)) {
    throw new Error(`${name} must contain only finite values`);
  }
}

function integerCenter(center: GridCoordinate): GridCoordinate {
  validateCoordinate(center, "Brush center");
  if (!center.every(Number.isInteger)) {
    throw new Error("Brush center must use integer voxel coordinates");
  }
  return center;
}

export function diskOffsets(
  sliceAxis: SliceAxis,
  diameter: number,
): readonly GridCoordinate[] {
  validateBrushDiameter(diameter);
  if (!Number.isInteger(sliceAxis) || sliceAxis < 0 || sliceAxis > 2) {
    throw new RangeError("Slice axis must be 0, 1, or 2");
  }
  const radius = (diameter - 1) / 2;
  const offsets: GridCoordinate[] = [];

  for (let first = -radius; first <= radius; first += 1) {
    for (let second = -radius; second <= radius; second += 1) {
      if (first * first + second * second > radius * radius) {
        continue;
      }
      const offset = [0, 0, 0] as [number, number, number];
      const normalizedFirst = first || 0;
      const normalizedSecond = second || 0;
      if (sliceAxis === 0) {
        offset[1] = normalizedFirst;
        offset[2] = normalizedSecond;
      } else if (sliceAxis === 1) {
        offset[0] = normalizedFirst;
        offset[2] = normalizedSecond;
      } else {
        offset[0] = normalizedFirst;
        offset[1] = normalizedSecond;
      }
      offsets.push(offset);
    }
  }
  return offsets;
}

export function sphereOffsets(diameter: number): readonly GridCoordinate[] {
  validateBrushDiameter(diameter);
  const radius = (diameter - 1) / 2;
  const offsets: GridCoordinate[] = [];

  for (let z = -radius; z <= radius; z += 1) {
    for (let y = -radius; y <= radius; y += 1) {
      for (let x = -radius; x <= radius; x += 1) {
        if (x * x + y * y + z * z <= radius * radius) {
          offsets.push([x || 0, y || 0, z || 0]);
        }
      }
    }
  }
  return offsets;
}

function indicesForOffsets(
  dimensions: Dimensions,
  center: GridCoordinate,
  offsets: readonly GridCoordinate[],
): Uint32Array {
  integerCenter(center);
  const indices: number[] = [];
  for (const offset of offsets) {
    const coordinate: GridCoordinate = [
      center[0] + offset[0],
      center[1] + offset[1],
      center[2] + offset[2],
    ];
    if (isCoordinateInBounds(dimensions, coordinate)) {
      indices.push(linearIndex(dimensions, coordinate));
    }
  }
  return Uint32Array.from(indices);
}

export function diskBrushIndices(
  dimensions: Dimensions,
  center: GridCoordinate,
  sliceAxis: SliceAxis,
  diameter: number,
): Uint32Array {
  return indicesForOffsets(
    dimensions,
    center,
    diskOffsets(sliceAxis, diameter),
  );
}

export function sphereBrushIndices(
  dimensions: Dimensions,
  center: GridCoordinate,
  diameter: number,
): Uint32Array {
  return indicesForOffsets(dimensions, center, sphereOffsets(diameter));
}

function clipSegmentToVolume(
  start: GridCoordinate,
  end: GridCoordinate,
  dimensions: Dimensions,
): readonly [GridCoordinate, GridCoordinate] | null {
  validateCoordinate(start, "Stroke start");
  validateCoordinate(end, "Stroke end");
  let minimumT = 0;
  let maximumT = 1;

  for (const axis of [0, 1, 2] as const) {
    const delta = end[axis] - start[axis];
    const minimum = 0;
    const maximum = dimensions[axis] - 1;
    if (delta === 0) {
      if (start[axis] < minimum || start[axis] > maximum) {
        return null;
      }
      continue;
    }

    let nearT = (minimum - start[axis]) / delta;
    let farT = (maximum - start[axis]) / delta;
    if (nearT > farT) {
      [nearT, farT] = [farT, nearT];
    }
    minimumT = Math.max(minimumT, nearT);
    maximumT = Math.min(maximumT, farT);
    if (minimumT > maximumT) {
      return null;
    }
  }

  const interpolate = (t: number): GridCoordinate => [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
    start[2] + (end[2] - start[2]) * t,
  ];
  return [interpolate(minimumT), interpolate(maximumT)];
}

export function interpolateGridLine(
  start: GridCoordinate,
  end: GridCoordinate,
  dimensions: Dimensions,
): readonly GridCoordinate[] {
  const clipped = clipSegmentToVolume(start, end, dimensions);
  if (!clipped) {
    return [];
  }

  const [clippedStart, clippedEnd] = clipped;
  const maximumDelta = Math.max(
    Math.abs(clippedEnd[0] - clippedStart[0]),
    Math.abs(clippedEnd[1] - clippedStart[1]),
    Math.abs(clippedEnd[2] - clippedStart[2]),
  );
  const steps = Math.max(1, Math.ceil(maximumDelta));
  const coordinates: GridCoordinate[] = [];
  let previousIndex = -1;

  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const coordinate: GridCoordinate = [
      Math.round(clippedStart[0] + (clippedEnd[0] - clippedStart[0]) * t),
      Math.round(clippedStart[1] + (clippedEnd[1] - clippedStart[1]) * t),
      Math.round(clippedStart[2] + (clippedEnd[2] - clippedStart[2]) * t),
    ];
    if (!isCoordinateInBounds(dimensions, coordinate)) {
      continue;
    }
    const index = linearIndex(dimensions, coordinate);
    if (index !== previousIndex) {
      coordinates.push(coordinate);
      previousIndex = index;
    }
  }
  return coordinates;
}

export function brushStrokeIndices(
  dimensions: Dimensions,
  start: GridCoordinate,
  end: GridCoordinate,
  diameter: number,
  sliceAxis?: SliceAxis,
): Uint32Array {
  validateBrushDiameter(diameter);
  const indices = new Set<number>();
  for (const center of interpolateGridLine(start, end, dimensions)) {
    const dab =
      sliceAxis === undefined
        ? sphereBrushIndices(dimensions, center, diameter)
        : diskBrushIndices(dimensions, center, sliceAxis, diameter);
    for (const index of dab) {
      indices.add(index);
    }
  }
  return Uint32Array.from(indices);
}
