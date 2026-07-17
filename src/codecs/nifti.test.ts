import { gzipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { dataView } from "./binary";
import { decodeNifti, encodeNifti } from "./nifti";
import { CodecError, type EncodableVolume } from "./types";

type FixtureScalar = number | bigint;

interface NiftiFixtureOptions {
  readonly version: 1 | 2;
  readonly littleEndian: boolean;
  readonly compressed: boolean;
  readonly datatype?: number;
  readonly bitpix?: number;
  readonly values?: readonly FixtureScalar[];
  readonly slope?: number;
  readonly intercept?: number;
  readonly fourthDimension?: number;
}

const writePayloadValue = (
  view: DataView,
  offset: number,
  datatype: number,
  value: FixtureScalar,
  littleEndian: boolean,
): void => {
  switch (datatype) {
    case 2: view.setUint8(offset, Number(value)); break;
    case 4: view.setInt16(offset, Number(value), littleEndian); break;
    case 8: view.setInt32(offset, Number(value), littleEndian); break;
    case 16: view.setFloat32(offset, Number(value), littleEndian); break;
    case 64: view.setFloat64(offset, Number(value), littleEndian); break;
    case 256: view.setInt8(offset, Number(value)); break;
    case 512: view.setUint16(offset, Number(value), littleEndian); break;
    case 768: view.setUint32(offset, Number(value), littleEndian); break;
    case 1024: view.setBigInt64(offset, BigInt(value), littleEndian); break;
    case 1280: view.setBigUint64(offset, BigInt(value), littleEndian); break;
    default: break;
  }
};

const makeNifti = (options: NiftiFixtureOptions): Uint8Array => {
  const datatype = options.datatype ?? 4;
  const bitpix = options.bitpix ?? 16;
  const values = options.values ?? [0, 2, 256, -1];
  const bytesPerVoxel = Math.max(1, bitpix / 8);
  const headerSize = options.version === 1 ? 348 : 540;
  const minimumPrefix = headerSize + 4;
  const voxelOffset = minimumPrefix + 16;
  const raw = new Uint8Array(voxelOffset + values.length * bytesPerVoxel);
  const view = dataView(raw);
  view.setInt32(0, headerSize, options.littleEndian);
  if (options.version === 1) {
    raw.set([0x6e, 0x2b, 0x31, 0], 344);
  } else {
    raw.set([0x6e, 0x2b, 0x32, 0, 13, 10, 26, 10], 4);
  }

  const dimensions = [
    options.fourthDimension === undefined ? 3 : 4,
    values.length,
    1,
    1,
    options.fourthDimension ?? 1,
    1,
    1,
    1,
  ];
  if (options.version === 1) {
    dimensions.forEach((value, index) =>
      view.setInt16(40 + index * 2, value, options.littleEndian),
    );
  } else {
    dimensions.forEach((value, index) =>
      view.setBigInt64(16 + index * 8, BigInt(value), options.littleEndian),
    );
  }
  const floatSize = options.version === 1 ? 4 : 8;
  const setFloat = (offset: number, value: number): void => {
    if (floatSize === 4) view.setFloat32(offset, value, options.littleEndian);
    else view.setFloat64(offset, value, options.littleEndian);
  };
  const pixelOffset = options.version === 1 ? 76 : 104;
  [-1, 2, 3, 4, 1, 1, 1, 1].forEach((value, index) =>
    setFloat(pixelOffset + index * floatSize, value),
  );
  view.setInt16(options.version === 1 ? 70 : 12, datatype, options.littleEndian);
  view.setInt16(options.version === 1 ? 72 : 14, bitpix, options.littleEndian);
  if (options.version === 1) view.setFloat32(108, voxelOffset, options.littleEndian);
  else view.setBigInt64(168, BigInt(voxelOffset), options.littleEndian);
  setFloat(options.version === 1 ? 112 : 176, options.slope ?? 2);
  setFloat(options.version === 1 ? 116 : 184, options.intercept ?? -4);
  raw.set(new TextEncoder().encode("fixture description"), options.version === 1 ? 148 : 240);
  raw.set(new TextEncoder().encode("fixture aux"), options.version === 1 ? 228 : 320);
  raw.set(new TextEncoder().encode("fixture intent"), options.version === 1 ? 328 : 508);
  if (options.version === 1) {
    view.setInt16(68, 0, options.littleEndian);
    view.setInt16(252, 1, options.littleEndian);
    view.setInt16(254, 2, options.littleEndian);
    raw[123] = 2;
  } else {
    view.setInt32(344, 1, options.littleEndian);
    view.setInt32(348, 2, options.littleEndian);
    view.setInt32(500, 2, options.littleEndian);
  }
  const quaternionOffset = options.version === 1 ? 256 : 352;
  const translationOffset = options.version === 1 ? 268 : 376;
  setFloat(quaternionOffset, 0.1);
  setFloat(quaternionOffset + floatSize, 0.2);
  setFloat(quaternionOffset + floatSize * 2, 0.3);
  setFloat(translationOffset, 4);
  setFloat(translationOffset + floatSize, 5);
  setFloat(translationOffset + floatSize * 2, 6);
  const rows = [
    [1, 0.25, 0, 10],
    [0, 2, 0.5, 20],
    [0, 0, 3, 30],
  ];
  const sformOffset = options.version === 1 ? 280 : 400;
  rows.flat().forEach((value, index) =>
    setFloat(sformOffset + index * floatSize, value),
  );
  raw[headerSize] = 1;
  view.setInt32(minimumPrefix, 16, options.littleEndian);
  view.setInt32(minimumPrefix + 4, 6, options.littleEndian);
  raw.set([65, 66, 67, 68, 0, 0, 0, 0], minimumPrefix + 8);
  values.forEach((value, index) =>
    writePayloadValue(
      view,
      voxelOffset + index * bytesPerVoxel,
      datatype,
      value,
      options.littleEndian,
    ),
  );
  return options.compressed ? gzipSync(raw, { mtime: 0 }) : raw;
};

const niftiState = (decoded: ReturnType<typeof decodeNifti>) => {
  if (decoded.saveState.kind !== "nifti") {
    throw new Error("Expected NIfTI state");
  }
  return decoded.saveState;
};

const expectPreservedNiftiSemantics = (
  original: ReturnType<typeof decodeNifti>,
  reopened: ReturnType<typeof decodeNifti>,
): void => {
  const before = niftiState(original);
  const after = niftiState(reopened);

  expect(reopened.dimensions).toEqual(original.dimensions);
  expect(reopened.geometry).toEqual(original.geometry);
  expect({
    version: after.metadata.version,
    endianness: after.metadata.endianness,
    dimensions: after.metadata.dimensions,
    pixelDimensions: after.metadata.pixelDimensions,
    qformCode: after.metadata.qformCode,
    sformCode: after.metadata.sformCode,
    qform: after.metadata.qform,
    sform: after.metadata.sform,
    unitsCode: after.metadata.unitsCode,
    intentCode: after.metadata.intentCode,
    intentName: after.metadata.intentName,
    description: after.metadata.description,
    auxiliaryFile: after.metadata.auxiliaryFile,
    extensions: after.metadata.extensions,
  }).toEqual({
    version: before.metadata.version,
    endianness: before.metadata.endianness,
    dimensions: before.metadata.dimensions,
    pixelDimensions: before.metadata.pixelDimensions,
    qformCode: before.metadata.qformCode,
    sformCode: before.metadata.sformCode,
    qform: before.metadata.qform,
    sform: before.metadata.sform,
    unitsCode: before.metadata.unitsCode,
    intentCode: before.metadata.intentCode,
    intentName: before.metadata.intentName,
    description: before.metadata.description,
    auxiliaryFile: before.metadata.auxiliaryFile,
    extensions: before.metadata.extensions,
  });
  for (const extension of before.metadata.extensions) {
    expect(
      after.prefix.slice(extension.offset, extension.offset + extension.size),
    ).toEqual(
      before.prefix.slice(extension.offset, extension.offset + extension.size),
    );
  }
};

describe.each([
  { version: 1 as const, littleEndian: true, compressed: false },
  { version: 1 as const, littleEndian: false, compressed: true },
  { version: 2 as const, littleEndian: true, compressed: true },
  { version: 2 as const, littleEndian: false, compressed: false },
])("NIfTI-$version $littleEndian compressed=$compressed", (variant) => {
  it("decodes scaling, extensions, qfac, and sform-first geometry", () => {
    const extension = variant.compressed ? ".nii.gz" : ".nii";
    const decoded = decodeNifti(makeNifti(variant), extension);
    expect(decoded.dimensions).toEqual([4, 1, 1]);
    expect([...decoded.occupancy]).toEqual([1, 0, 1, 1]);
    expect(decoded.occupiedCount).toBe(3);
    expect(decoded.normalization.convertedNonBinary).toBe(true);
    expect(decoded.geometry.indexToWorld).toEqual([
      1, 0.25, 0, 10,
      0, 2, 0.5, 20,
      0, 0, 3, 30,
      0, 0, 0, 1,
    ]);
    expect(decoded.geometry.units).toEqual(["mm", "mm", "mm"]);
    expect(decoded.saveState.kind).toBe("nifti");
    if (decoded.saveState.kind !== "nifti") throw new Error("Expected NIfTI state");
    expect(decoded.saveState.metadata.version).toBe(variant.version);
    expect(decoded.saveState.metadata.endianness).toBe(
      variant.littleEndian ? "little" : "big",
    );
    expect(decoded.saveState.metadata.extensions).toEqual([
      { size: 16, code: 6, offset: variant.version === 1 ? 352 : 544 },
    ]);
    expect(decoded.saveState.metadata.qform).not.toEqual(
      decoded.saveState.metadata.sform,
    );
    const qform = decoded.saveState.metadata.qform;
    if (qform === null) throw new Error("Expected qform metadata");
    const determinant =
      qform[0] * (qform[5] * qform[10] - qform[6] * qform[9]) -
      qform[1] * (qform[4] * qform[10] - qform[6] * qform[8]) +
      qform[2] * (qform[4] * qform[9] - qform[5] * qform[8]);
    expect(determinant).toBeLessThan(0);
  });

  it("writes canonical uint8 data while retaining geometry and extensions", () => {
    const extension = variant.compressed ? ".nii.gz" : ".nii";
    const decoded = decodeNifti(makeNifti(variant), extension);
    const document: EncodableVolume = {
      dimensions: decoded.dimensions,
      occupancy: new Uint8Array([0, 1, 0, 1]),
      saveState: decoded.saveState,
    };
    const encoded = encodeNifti(document, extension);
    const reopened = decodeNifti(encoded, extension);
    expect([...reopened.occupancy]).toEqual([0, 1, 0, 1]);
    expect(reopened.normalization.convertedNonBinary).toBe(false);
    if (reopened.saveState.kind !== "nifti") throw new Error("Expected NIfTI state");
    expect(reopened.saveState.metadata.datatype).toBe(2);
    expect(reopened.saveState.metadata.bitsPerVoxel).toBe(8);
    expect(reopened.saveState.metadata.scaleSlope).toBe(1);
    expect(reopened.saveState.metadata.scaleIntercept).toBe(0);
    expect(reopened.saveState.metadata.sform).toEqual(
      decoded.geometry.indexToWorld,
    );
    expect(reopened.saveState.metadata.extensions[0]?.code).toBe(6);
    expectPreservedNiftiSemantics(decoded, reopened);
  });
});

describe.each([
  { version: 1 as const, littleEndian: true },
  { version: 2 as const, littleEndian: false },
])("NIfTI-$version geometry selection", ({ version, littleEndian }) => {
  it("uses qform when sform is absent and pixel spacing when both forms are absent", () => {
    const qformOnlyBytes = makeNifti({
      version,
      littleEndian,
      compressed: false,
    });
    const view = dataView(qformOnlyBytes);
    if (version === 1) {
      view.setInt16(254, 0, littleEndian);
    } else {
      view.setInt32(348, 0, littleEndian);
    }
    const qformOnly = decodeNifti(qformOnlyBytes, ".nii");
    const qformOnlyState = niftiState(qformOnly);
    expect(qformOnlyState.metadata.sform).toBeNull();
    expect(qformOnlyState.metadata.qform).not.toBeNull();
    expect(qformOnly.geometry.indexToWorld).toEqual(
      qformOnlyState.metadata.qform,
    );

    const fallbackBytes = qformOnlyBytes.slice();
    const fallbackView = dataView(fallbackBytes);
    if (version === 1) {
      fallbackView.setInt16(252, 0, littleEndian);
    } else {
      fallbackView.setInt32(344, 0, littleEndian);
    }
    const fallback = decodeNifti(fallbackBytes, ".nii");
    const fallbackState = niftiState(fallback);
    expect(fallbackState.metadata.qform).toBeNull();
    expect(fallbackState.metadata.sform).toBeNull();
    expect(fallback.geometry.indexToWorld).toEqual([
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      0, 0, 0, 1,
    ]);
    expect(fallback.geometry.spacing).toEqual([2, 3, 4]);
  });
});

describe.each([
  { version: 1 as const, littleEndian: false },
  { version: 2 as const, littleEndian: true },
])("NIfTI-$version compression switching", ({ version, littleEndian }) => {
  it("switches between .nii and .nii.gz without changing volume semantics", () => {
    const original = decodeNifti(
      makeNifti({ version, littleEndian, compressed: false }),
      ".nii",
    );
    const document: EncodableVolume = {
      dimensions: original.dimensions,
      occupancy: new Uint8Array([1, 0, 0, 1]),
      saveState: original.saveState,
    };

    const compressed = encodeNifti(document, ".nii.gz");
    expect([...compressed.subarray(0, 3)]).toEqual([0x1f, 0x8b, 8]);
    const reopenedCompressed = decodeNifti(compressed, ".nii.gz");
    expect(reopenedCompressed.sourceVariant).toBe(".nii.gz");
    expect([...reopenedCompressed.occupancy]).toEqual([1, 0, 0, 1]);
    expectPreservedNiftiSemantics(original, reopenedCompressed);

    const uncompressed = encodeNifti(
      {
        dimensions: reopenedCompressed.dimensions,
        occupancy: reopenedCompressed.occupancy,
        saveState: reopenedCompressed.saveState,
      },
      ".nii",
    );
    expect([...uncompressed.subarray(0, 3)]).not.toEqual([0x1f, 0x8b, 8]);
    const reopenedUncompressed = decodeNifti(uncompressed, ".nii");
    expect(reopenedUncompressed.sourceVariant).toBe(".nii");
    expect([...reopenedUncompressed.occupancy]).toEqual([1, 0, 0, 1]);
    expectPreservedNiftiSemantics(reopenedCompressed, reopenedUncompressed);
  });
});

describe.each([
  { datatype: 512, bitpix: 16, values: [0, 256], expected: [0, 1] },
  { datatype: 8, bitpix: 32, values: [-1, 0], expected: [1, 0] },
  { datatype: 1280, bitpix: 64, values: [0n, 9007199254740993n], expected: [0, 1] },
  { datatype: 64, bitpix: 64, values: [-0, Number.NaN], expected: [0, 1] },
])("NIfTI scalar datatype $datatype", ({ datatype, bitpix, values, expected }) => {
  it("normalizes wide, signed, and floating labels before narrowing", () => {
    const decoded = decodeNifti(
      makeNifti({
        version: 2,
        littleEndian: false,
        compressed: true,
        datatype,
        bitpix,
        values,
        slope: 0,
        intercept: 99,
      }),
      ".nii.gz",
    );
    expect([...decoded.occupancy]).toEqual(expected);
  });
});

describe("NIfTI validation", () => {
  it("rejects non-singleton higher dimensions", () => {
    expect(() =>
      decodeNifti(
        makeNifti({
          version: 1,
          littleEndian: true,
          compressed: false,
          fourthDimension: 2,
        }),
        ".nii",
      ),
    ).toThrowError(CodecError);
  });

  it("rejects complex data and detached-pair magic", () => {
    expect(() =>
      decodeNifti(
        makeNifti({
          version: 1,
          littleEndian: true,
          compressed: false,
          datatype: 32,
          bitpix: 64,
        }),
        ".nii",
      ),
    ).toThrow(/complex/);
    const pair = makeNifti({
      version: 1,
      littleEndian: true,
      compressed: false,
    });
    pair.set([0x6e, 0x69, 0x31, 0], 344);
    try {
      decodeNifti(pair, ".nii");
      throw new Error("Expected detached NIfTI rejection");
    } catch (error) {
      expect(error).toMatchObject({ code: "unsupported-detached-data" });
      expect(error).toHaveProperty("message", expect.stringMatching(/Detached/));
    }
  });

  it("rejects scalar encodings without interoperable JavaScript decoding", () => {
    for (const [datatype, bitpix] of [[1, 1], [1536, 128]] as const) {
      try {
        decodeNifti(
          makeNifti({
            version: 1,
            littleEndian: true,
            compressed: false,
            datatype,
            bitpix,
          }),
          ".nii",
        );
        throw new Error("Expected unsupported scalar datatype rejection");
      } catch (error) {
        expect(error).toMatchObject({ code: "unsupported-datatype" });
      }
    }
  });

  it("requires extension and compression to agree", () => {
    const raw = makeNifti({
      version: 1,
      littleEndian: true,
      compressed: false,
    });
    expect(() => decodeNifti(raw, ".nii.gz")).toThrow(/gzip/);
  });
});
