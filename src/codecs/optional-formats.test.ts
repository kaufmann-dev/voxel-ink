import { gzipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { decodeBinvox, encodeBinvox } from "./binvox";
import { concatBytes, dataView } from "./binary";
import { decodeNrrd, encodeNrrd } from "./nrrd";
import type { EncodableVolume } from "./types";
import { decodeVox, encodeVox } from "./vox";

const encoder = new TextEncoder();

const makeNrrd = (
  encoding: "raw" | "gzip" | "ascii",
  lineEnding: "\n" | "\r\n" = "\n",
): Uint8Array => {
  const fields = [
    "NRRD0005",
    "# retained comment",
    "voxel-ink-test:=custom value",
    encoding === "ascii" ? "type: double" : "type: short",
    "dimension: 3",
    "sizes: 2 2 1",
    `encoding: ${encoding}`,
    ...(encoding === "ascii" ? [] : ["endian: big"]),
    "space: right-anterior-superior",
    "space directions: (2,0,0) (0,3,0) (0,0,-4)",
    "space origin: (10,20,30)",
    'space units: "mm" "mm" "mm"',
    'labels: "x" "y" "z"',
    "kinds: space space space",
    "measurement frame: (1,0,0) (0,1,0) (0,0,1)",
    "min: -1",
    "max: 256",
    "",
    "",
  ];
  const header = encoder.encode(fields.join(lineEnding));
  if (encoding === "ascii") {
    return concatBytes(header, encoder.encode("0 256 -1 1\n"));
  }
  const payload = new Uint8Array(8);
  const view = dataView(payload);
  [0, 256, -1, 1].forEach((value, index) =>
    view.setInt16(index * 2, value, false),
  );
  return concatBytes(
    header,
    encoding === "gzip" ? gzipSync(payload, { mtime: 0 }) : payload,
  );
};

describe.each([
  { encoding: "raw" as const, lineEnding: "\n" as const },
  { encoding: "gzip" as const, lineEnding: "\r\n" as const },
  { encoding: "ascii" as const, lineEnding: "\n" as const },
])("attached NRRD $encoding", ({ encoding, lineEnding }) => {
  it("decodes, retains semantic metadata, and saves in the source encoding", () => {
    const decoded = decodeNrrd(makeNrrd(encoding, lineEnding));
    expect(decoded.dimensions).toEqual([2, 2, 1]);
    expect([...decoded.occupancy]).toEqual([0, 1, 1, 1]);
    expect(decoded.normalization.convertedNonBinary).toBe(true);
    expect(decoded.geometry.indexToWorld).toEqual([
      2, 0, 0, 10,
      0, 3, 0, 20,
      0, 0, -4, 30,
      0, 0, 0, 1,
    ]);
    expect(decoded.geometry.units).toEqual(["mm", "mm", "mm"]);

    const document: EncodableVolume = {
      dimensions: decoded.dimensions,
      occupancy: new Uint8Array([1, 0, 1, 0]),
      saveState: decoded.saveState,
    };
    const encoded = encodeNrrd(document);
    const headerText = new TextDecoder().decode(encoded.subarray(0, Math.min(encoded.length, 600)));
    expect(headerText).toContain("# retained comment");
    expect(headerText).toContain("voxel-ink-test:=custom value");
    expect(headerText).toContain("measurement frame:");
    expect(headerText).toContain("type: uint8");
    expect(headerText).not.toContain("endian:");
    expect(headerText).toContain("min: 0");
    expect(headerText).toContain("max: 1");
    const reopened = decodeNrrd(encoded);
    expect([...reopened.occupancy]).toEqual([1, 0, 1, 0]);
    expect(reopened.geometry.indexToWorld).toEqual(decoded.geometry.indexToWorld);
    expect(reopened.normalization.convertedNonBinary).toBe(false);
  });
});

describe("NRRD validation", () => {
  const makeInvalid = (extra: string): Uint8Array =>
    encoder.encode(
      `NRRD0005\ntype: uint8\ndimension: 3\nsizes: 1 1 1\nencoding: raw\n${extra}\n\n\0`,
    );

  it("rejects detached data, vectors, and unsupported encodings", () => {
    expect(() => decodeNrrd(makeInvalid("data file: volume.raw"))).toThrow(/Detached/);
    expect(() => decodeNrrd(makeInvalid("kinds: space vector space"))).toThrow(/vector/);
    expect(() =>
      decodeNrrd(
        encoder.encode(
          "NRRD0005\ntype: uint8\ndimension: 3\nsizes: 1 1 1\nencoding: hex\n\n00",
        ),
      ),
    ).toThrow(/not supported/);
  });

  it("supports singleton non-spatial axes and signed spacing orientation", () => {
    const singletonAxis = concatBytes(
      encoder.encode([
        "NRRD0005",
        "type: uint8",
        "dimension: 4",
        "sizes: 1 2 2 1",
        "encoding: raw",
        "kinds: time space space space",
        "space directions: none (2,0,0) (0,3,0) (0,0,-4)",
        "space origin: (10,20,30)",
        "",
        "",
      ].join("\n")),
      Uint8Array.from([0, 1, 1, 0]),
    );
    const decoded = decodeNrrd(singletonAxis);
    expect(decoded.dimensions).toEqual([2, 2, 1]);
    expect([...decoded.occupancy]).toEqual([0, 1, 1, 0]);
    expect(decoded.geometry.indexToWorld).toEqual([
      2, 0, 0, 10,
      0, 3, 0, 20,
      0, 0, -4, 30,
      0, 0, 0, 1,
    ]);
    expect(decodeNrrd(encodeNrrd({
      dimensions: decoded.dimensions,
      occupancy: decoded.occupancy,
      saveState: decoded.saveState,
    })).geometry.indexToWorld).toEqual(decoded.geometry.indexToWorld);

    const signedSpacing = concatBytes(
      encoder.encode(
        "NRRD0003\ntype: uint8\ndimension: 3\nsizes: 1 1 1\nencoding: raw\nkinds: none none none\nspacings: -2 3 4\n\n",
      ),
      Uint8Array.of(1),
    );
    expect(decodeNrrd(signedSpacing).geometry.indexToWorld).toEqual([
      -2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      0, 0, 0, 1,
    ]);
  });

  it("rejects malformed vectors and oversized headers while ignoring sanctioned trailing data", () => {
    expect(() => decodeNrrd(makeInvalid("space directions: (1,,0) (0,1,0) (0,0,1)"))).toThrow(/empty vector/);
    const oversized = new Uint8Array(1024 * 1024 + 1);
    oversized.fill(65);
    expect(() => decodeNrrd(oversized)).toThrow(/1 MiB/);
    const withTrailing = concatBytes(
      encoder.encode("NRRD0005\ntype: uint8\ndimension: 3\nsizes: 1 1 1\nencoding: raw\n\n"),
      Uint8Array.from([0, 9, 9]),
    );
    expect([...decodeNrrd(withTrailing).occupancy]).toEqual([0]);
  });
});

const rle = (values: readonly number[]): Uint8Array => {
  const output: number[] = [];
  let value = values[0] ?? 0;
  let count = 0;
  for (const next of values) {
    if (next !== value || count === 255) {
      if (count > 0) output.push(value, count);
      value = next;
      count = 1;
    } else {
      count += 1;
    }
  }
  if (count > 0) output.push(value, count);
  return Uint8Array.from(output);
};

const makeBinvox = (version: 1 | 2, payloadValues: readonly number[]): Uint8Array =>
  concatBytes(
    encoder.encode(
      `#binvox ${version}\ndim 2 3 2\ntranslate 10 20 30\nscale 6\ndata\n`,
    ),
    rle(payloadValues),
  );

describe("BINVOX", () => {
  it("reorders y-fastest RLE, retains transforms, and normalizes v2 labels", () => {
    const values = [0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    const decoded = decodeBinvox(makeBinvox(2, values));
    expect(decoded.dimensions).toEqual([2, 3, 2]);
    expect([...decoded.occupancy]).toEqual([
      0, 0,
      1, 0,
      0, 1,
      0, 0,
      0, 0,
      0, 0,
    ]);
    expect(decoded.normalization.convertedNonBinary).toBe(true);
    expect(decoded.geometry.indexToWorld).toEqual([
      3, 0, 0, 11.5,
      0, 2, 0, 21,
      0, 0, 3, 31.5,
      0, 0, 0, 1,
    ]);
    const encoded = encodeBinvox({
      dimensions: decoded.dimensions,
      occupancy: decoded.occupancy,
      saveState: decoded.saveState,
    });
    const reopened = decodeBinvox(encoded);
    expect([...reopened.occupancy]).toEqual([...decoded.occupancy]);
    expect(reopened.normalization.convertedNonBinary).toBe(false);
    if (reopened.saveState.kind !== "binvox") throw new Error("Expected BINVOX state");
    expect(reopened.saveState.version).toBe(2);
    expect(reopened.saveState.translation).toEqual([10, 20, 30]);
    expect(reopened.saveState.scale).toBe(6);
    expect(reopened.saveState.dimensions).toEqual([2, 3, 2]);
    expect(reopened.geometry).toEqual(decoded.geometry);
    expect(reopened.geometry.spacing).toEqual([3, 2, 3]);
  });

  it("splits canonical RLE runs at 255", () => {
    const original = decodeBinvox(
      concatBytes(
        encoder.encode("#binvox 1\ndim 300 1 1\ndata\n"),
        Uint8Array.from([0, 255, 0, 45]),
      ),
    );
    const encoded = encodeBinvox({
      dimensions: original.dimensions,
      occupancy: original.occupancy,
      saveState: original.saveState,
    });
    expect(decodeBinvox(encoded).occupancy).toEqual(original.occupancy);
  });

  it("rejects invalid version-1 labels and RLE overruns", () => {
    expect(() =>
      decodeBinvox(makeBinvox(1, [2, ...Array<number>(11).fill(0)])),
    ).toThrow(/version 1/);
    expect(() =>
      decodeBinvox(
        concatBytes(
          encoder.encode("#binvox 1\ndim 1 1 1\ndata\n"),
          Uint8Array.from([1, 2]),
        ),
      ),
    ).toThrow(/run length/);
  });
});

const voxChunk = (
  id: string,
  content: Uint8Array,
  children: Uint8Array = new Uint8Array(0),
): Uint8Array => {
  const result = new Uint8Array(12 + content.length + children.length);
  result.set(encoder.encode(id));
  const view = dataView(result);
  view.setInt32(4, content.length, true);
  view.setInt32(8, children.length, true);
  result.set(content, 12);
  result.set(children, 12 + content.length);
  return result;
};

const makeVox = (packCount = 1, duplicate = false): Uint8Array => {
  const pack = new Uint8Array(4);
  dataView(pack).setInt32(0, packCount, true);
  const size = new Uint8Array(12);
  const sizeView = dataView(size);
  sizeView.setInt32(0, 3, true);
  sizeView.setInt32(4, 2, true);
  sizeView.setInt32(8, 2, true);
  const voxels = duplicate
    ? Uint8Array.from([2, 0, 0, 2, 2, 0, 0, 1])
    : Uint8Array.from([2, 0, 0, 2, 0, 1, 1, 1]);
  const xyzi = new Uint8Array(4 + voxels.length);
  dataView(xyzi).setUint32(0, voxels.length / 4, true);
  xyzi.set(voxels, 4);
  const children = concatBytes(
    voxChunk("PACK", pack),
    voxChunk("SIZE", size),
    voxChunk("TEST", Uint8Array.from([1, 2, 3])),
    voxChunk("XYZI", xyzi),
  );
  const header = new Uint8Array(8);
  header.set(encoder.encode("VOX "));
  dataView(header).setInt32(4, 150, true);
  return concatBytes(header, voxChunk("MAIN", new Uint8Array(0), children));
};

const makeVoxWithTwoModelsAndNoPack = (): Uint8Array => {
  const size = new Uint8Array(12);
  const sizeView = dataView(size);
  sizeView.setInt32(0, 1, true);
  sizeView.setInt32(4, 1, true);
  sizeView.setInt32(8, 1, true);
  const xyzi = Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 1]);
  const children = concatBytes(
    voxChunk("SIZE", size),
    voxChunk("XYZI", xyzi),
    voxChunk("SIZE", size),
    voxChunk("XYZI", xyzi),
  );
  const header = new Uint8Array(8);
  header.set(encoder.encode("VOX "));
  dataView(header).setInt32(4, 150, true);
  return concatBytes(header, voxChunk("MAIN", new Uint8Array(0), children));
};

describe("MagicaVoxel VOX", () => {
  it("loads one model, skips unknown chunks, and writes palette index 1", () => {
    const decoded = decodeVox(makeVox());
    expect(decoded.dimensions).toEqual([3, 2, 2]);
    expect(decoded.occupiedCount).toBe(2);
    expect(decoded.occupancy[2]).toBe(1);
    expect(decoded.occupancy[9]).toBe(1);
    expect(decoded.normalization.convertedNonBinary).toBe(true);
    const encoded = encodeVox({
      dimensions: decoded.dimensions,
      occupancy: decoded.occupancy,
      saveState: decoded.saveState,
    });
    const reopened = decodeVox(encoded);
    expect(reopened.occupancy).toEqual(decoded.occupancy);
    expect(reopened.normalization.convertedNonBinary).toBe(false);
  });

  it("rejects multi-model packs and duplicate coordinates", () => {
    expect(() => decodeVox(makeVox(2))).toThrow(/single-model/);
    expect(() => decodeVox(makeVoxWithTwoModelsAndNoPack())).toThrowError(
      expect.objectContaining({ code: "unsupported-multi-model" }),
    );
    expect(() => decodeVox(makeVox(1, true))).toThrow(/duplicate/);
  });
});
