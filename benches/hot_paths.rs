use std::{hint::black_box, io::Cursor, path::PathBuf, time::SystemTime};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use voxel_ink::{
    VolumeDocument,
    brush::{BrushDepth, BrushShape, BrushSize, BrushSpec, brush_indices},
    cancel::CancellationToken,
    codecs::{
        BinvoxMetadata, CodecSaveState, DecodeOptions, EncodeOptions, SourceVariant, VolumeCodec,
        binvox::BINVOX_CODEC,
    },
    mesh::{ChunkCoord, ChunkInput, MeshJobTag, mesh_chunk, pack_leak_distances},
    volume::{SourceIdentity, SpatialUnits, VolumeGeometry, VolumeView},
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

fn leak_distance_document(occupancy: Vec<u8>) -> VolumeDocument {
    VolumeDocument::new(
        identity_geometry(VOLUME_DIMENSIONS),
        occupancy,
        SourceIdentity {
            canonical_path: PathBuf::from("/tmp/voxel-ink-leak-distance-benchmark.binvox"),
            size: 0,
            modified: SystemTime::UNIX_EPOCH,
            sha256: [0; 32],
        },
        SourceVariant::Binvox { version: 1 },
        CodecSaveState::Binvox(BinvoxMetadata {
            version: 1,
            dimensions: VOLUME_DIMENSIONS,
            translation: [0.0; 3],
            scale: 1.0,
        }),
        false,
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

    let leak_quads = mesh_chunk(&noisy_chunk, tag(), &cancel).unwrap().mesh.quads;
    let leak_halo_dimensions = noisy_chunk.halo_dimensions();
    let leak_halo = vec![1_024_u16; 34_usize.pow(3)];
    let mut leak_pack_group = c.benchmark_group("leak_distance_quad_pack");
    leak_pack_group.throughput(Throughput::Elements(leak_quads.len() as u64));
    leak_pack_group.bench_function("noisy", |b| {
        b.iter(|| {
            pack_leak_distances(
                black_box(&leak_quads),
                CHUNK.origin().unwrap(),
                leak_halo_dimensions,
                black_box(&leak_halo),
            )
            .unwrap()
        });
    });
    leak_pack_group.finish();

    let leak_documents = [
        ("empty", leak_distance_document(empty)),
        ("dense", leak_distance_document(dense)),
        ("noisy", leak_distance_document(noisy)),
    ];
    let mut leak_group = c.benchmark_group("leak_distance_field");
    leak_group.throughput(Throughput::Elements(
        VOLUME_DIMENSIONS.into_iter().map(u64::from).product(),
    ));
    for (name, document) in &leak_documents {
        leak_group.bench_with_input(
            BenchmarkId::from_parameter(name),
            document,
            |b, document| {
                b.iter(|| black_box(document.leak_distance_field(&cancel).unwrap()));
            },
        );
    }
    leak_group.finish();

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
