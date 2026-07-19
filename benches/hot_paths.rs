use std::{hint::black_box, io::Cursor};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use voxel_ink::{
    brush::{BrushDepth, BrushShape, BrushSize, BrushSpec, brush_indices},
    cancel::CancellationToken,
    codecs::{
        BinvoxMetadata, CodecSaveState, DecodeOptions, EncodeOptions, SourceVariant, VolumeCodec,
        binvox::BINVOX_CODEC,
    },
    mesh::{ChunkCoord, ChunkInput, MeshJobTag, mesh_chunk},
    volume::{SpatialUnits, VolumeGeometry, VolumeView},
};

const VOLUME_DIMENSIONS: [u32; 3] = [96, 96, 96];
const CHUNK: ChunkCoord = ChunkCoord::new(1, 1, 1);

fn occupancy(workload: &str) -> Vec<u8> {
    let voxel_count = VOLUME_DIMENSIONS
        .into_iter()
        .map(|extent| extent as usize)
        .product();
    let mut state = 0x9e37_79b9_u32;
    (0..voxel_count)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match workload {
                "empty" => 0,
                "dense" => u8::from(state % 100 < 98),
                "noisy" => u8::from((state.rotate_left(index as u32 & 31) >> 31) != 0),
                _ => unreachable!(),
            }
        })
        .collect()
}

fn tag() -> MeshJobTag {
    MeshJobTag {
        document_epoch: 1,
        revision: 1,
        chunk_generation: 1,
        edit_generation: 0,
        chunk: CHUNK,
    }
}

fn identity_geometry(dimensions: [u32; 3]) -> VolumeGeometry {
    VolumeGeometry::new(
        dimensions,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Millimeter,
    )
    .unwrap()
}

fn hot_paths(c: &mut Criterion) {
    let cancel = CancellationToken::default();
    let empty = occupancy("empty");
    let dense = occupancy("dense");
    let noisy = occupancy("noisy");

    let mut copy_group = c.benchmark_group("chunk_halo_copy");
    copy_group.throughput(Throughput::Bytes(34_u64.pow(3)));
    for (name, volume) in [("empty", &empty), ("dense", &dense), ("noisy", &noisy)] {
        copy_group.bench_with_input(BenchmarkId::from_parameter(name), volume, |b, volume| {
            b.iter(|| {
                ChunkInput::copy_from_volume(VOLUME_DIMENSIONS, black_box(volume), CHUNK, &cancel)
                    .unwrap()
            });
        });
    }
    copy_group.finish();

    let empty_chunk =
        ChunkInput::copy_from_volume(VOLUME_DIMENSIONS, &empty, CHUNK, &cancel).unwrap();
    let dense_chunk =
        ChunkInput::copy_from_volume(VOLUME_DIMENSIONS, &dense, CHUNK, &cancel).unwrap();
    let noisy_chunk =
        ChunkInput::copy_from_volume(VOLUME_DIMENSIONS, &noisy, CHUNK, &cancel).unwrap();
    let mut mesh_group = c.benchmark_group("greedy_mesh_chunk");
    mesh_group.throughput(Throughput::Elements(32_u64.pow(3)));
    for (name, input) in [
        ("empty", &empty_chunk),
        ("dense", &dense_chunk),
        ("noisy", &noisy_chunk),
    ] {
        mesh_group.bench_with_input(BenchmarkId::from_parameter(name), input, |b, input| {
            b.iter(|| mesh_chunk(black_box(input), tag(), &cancel).unwrap());
        });
    }
    mesh_group.finish();

    let geometry = identity_geometry([129, 129, 129]);
    let mut brush_group = c.benchmark_group("maximum_volume_brush");
    for shape in [BrushShape::Round, BrushShape::Square] {
        brush_group.bench_with_input(
            BenchmarkId::from_parameter(match shape {
                BrushShape::Round => "round",
                BrushShape::Square => "square",
            }),
            &shape,
            |b, &shape| {
                b.iter(|| {
                    brush_indices(
                        &geometry,
                        [64; 3],
                        2,
                        BrushSpec {
                            shape,
                            depth: BrushDepth::Volume,
                            size: BrushSize::voxels(101).unwrap(),
                        },
                    )
                    .unwrap()
                });
            },
        );
    }
    brush_group.finish();

    let mut common_brush_group = c.benchmark_group("common_slice_brush");
    for shape in [BrushShape::Round, BrushShape::Square] {
        common_brush_group.bench_with_input(
            BenchmarkId::from_parameter(match shape {
                BrushShape::Round => "round_5",
                BrushShape::Square => "square_5",
            }),
            &shape,
            |b, &shape| {
                b.iter(|| {
                    brush_indices(
                        &geometry,
                        [64; 3],
                        2,
                        BrushSpec {
                            shape,
                            depth: BrushDepth::Slice,
                            size: BrushSize::voxels(5).unwrap(),
                        },
                    )
                    .unwrap()
                });
            },
        );
    }
    common_brush_group.finish();

    let codec_dimensions = [64, 64, 64];
    let codec_geometry = identity_geometry(codec_dimensions);
    let codec_occupancy = (0..codec_dimensions[2])
        .flat_map(|z| {
            (0..codec_dimensions[1]).flat_map(move |y| {
                (0..codec_dimensions[0]).map(move |x| u8::from((x + y + z) % 2 == 0))
            })
        })
        .collect::<Vec<_>>();
    let codec_metadata = BinvoxMetadata {
        version: 1,
        dimensions: codec_dimensions,
        translation: [0.0; 3],
        scale: 1.0,
    };
    let codec_state = CodecSaveState::Binvox(codec_metadata);
    let codec_variant = SourceVariant::Binvox { version: 1 };
    let codec_volume = VolumeView {
        dimensions: codec_dimensions,
        occupancy: &codec_occupancy,
        geometry: &codec_geometry,
    };
    let mut encoded = Vec::new();
    BINVOX_CODEC
        .encode(
            codec_volume,
            &codec_state,
            &mut encoded,
            EncodeOptions {
                destination_variant: codec_variant,
            },
            &cancel,
        )
        .unwrap();
    let mut codec_group = c.benchmark_group("binvox_checkerboard");
    codec_group.throughput(Throughput::Elements(codec_occupancy.len() as u64));
    codec_group.bench_function("encode", |b| {
        b.iter(|| {
            let mut output = Vec::with_capacity(encoded.len());
            BINVOX_CODEC
                .encode(
                    codec_volume,
                    &codec_state,
                    &mut output,
                    EncodeOptions {
                        destination_variant: codec_variant,
                    },
                    &cancel,
                )
                .unwrap();
            black_box(output)
        });
    });
    codec_group.bench_function("decode", |b| {
        b.iter(|| {
            BINVOX_CODEC
                .decode(
                    &mut Cursor::new(black_box(encoded.as_slice())),
                    DecodeOptions {
                        source_variant: codec_variant,
                    },
                    &cancel,
                )
                .unwrap()
        });
    });
    codec_group.finish();
}

criterion_group!(benches, hot_paths);
criterion_main!(benches);
