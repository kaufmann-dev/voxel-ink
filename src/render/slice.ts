import type { BinaryVoxel, SliceAxis, VolumeDocument } from "../core/volume";

export interface SliceRaster {
  readonly axis: SliceAxis;
  readonly slice: number;
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint8ClampedArray;
}

export interface DirtyRectangle {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

const EMPTY = [22, 33, 39] as const;
const FILLED = [78, 211, 177] as const;

export function sliceShape(
  document: VolumeDocument,
  axis: SliceAxis,
): readonly [number, number] {
  const [nx, ny, nz] = document.dimensions;
  return axis === 0 ? [ny, nz] : axis === 1 ? [nx, nz] : [nx, ny];
}

function setPixel(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  u: number,
  v: number,
  value: BinaryVoxel,
): void {
  const output = 4 * (u + width * (height - 1 - v));
  const color = value === 1 ? FILLED : EMPTY;
  pixels[output] = color[0];
  pixels[output + 1] = color[1];
  pixels[output + 2] = color[2];
  pixels[output + 3] = 255;
}

export function createSliceRaster(
  document: VolumeDocument,
  axis: SliceAxis,
  slice: number,
): SliceRaster {
  const [width, height] = sliceShape(document, axis);
  const pixels = new Uint8ClampedArray(width * height * 4);
  const [nx, ny] = document.dimensions;
  for (let v = 0; v < height; v += 1) {
    for (let u = 0; u < width; u += 1) {
      const index = axis === 0
        ? slice + nx * (u + ny * v)
        : axis === 1
          ? u + nx * (slice + ny * v)
          : u + nx * (v + ny * slice);
      setPixel(pixels, width, height, u, v, document.occupancy[index] as BinaryVoxel);
    }
  }
  return { axis, slice, width, height, pixels };
}

export function updateSliceRaster(
  raster: SliceRaster,
  document: VolumeDocument,
  indices: Iterable<number>,
  value: BinaryVoxel,
): DirtyRectangle | null {
  const [nx, ny] = document.dimensions;
  const planeSize = nx * ny;
  let minX = raster.width;
  let minY = raster.height;
  let maxX = -1;
  let maxY = -1;

  for (const index of indices) {
    const z = Math.floor(index / planeSize);
    const inPlane = index - z * planeSize;
    const y = Math.floor(inPlane / nx);
    const x = inPlane - y * nx;
    const coordinate = [x, y, z] as const;
    if (coordinate[raster.axis] !== raster.slice) continue;
    const u = raster.axis === 0 ? y : x;
    const v = raster.axis === 2 ? y : z;
    const imageY = raster.height - 1 - v;
    setPixel(raster.pixels, raster.width, raster.height, u, v, value);
    minX = Math.min(minX, u);
    minY = Math.min(minY, imageY);
    maxX = Math.max(maxX, u);
    maxY = Math.max(maxY, imageY);
  }

  return maxX < minX
    ? null
    : { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}
