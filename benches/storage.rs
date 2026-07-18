use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const CHUNK_VOXELS: usize = 32 * 32 * 32;

fn workload(name: &str) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..CHUNK_VOXELS)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match name {
                "sparse" => u8::from(state % 100 < 2),
                "dense" => u8::from(state % 100 < 98),
                "noisy" => u8::from((state.rotate_left(index as u32 & 31) >> 31) != 0),
                _ => unreachable!(),
            }
        })
        .collect()
}

fn pack_lsb(bytes: &[u8]) -> Vec<u8> {
    let mut packed = vec![0_u8; bytes.len().div_ceil(8)];
    for (index, &value) in bytes.iter().enumerate() {
        packed[index / 8] |= value << (index % 8);
    }
    packed
}

fn sparse_indices(bytes: &[u8]) -> Vec<u32> {
    bytes
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value != 0).then_some(index as u32))
        .collect()
}

fn storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_storage_32_cubed");
    group.throughput(Throughput::Elements(CHUNK_VOXELS as u64));
    for name in ["sparse", "dense", "noisy"] {
        let bytes = workload(name);
        let bits = pack_lsb(&bytes);
        let sparse = sparse_indices(&bytes);
        group.bench_with_input(BenchmarkId::new("byte", name), &bytes, |b, input| {
            b.iter(|| input.iter().map(|&value| u64::from(value)).sum::<u64>());
        });
        group.bench_with_input(
            BenchmarkId::new("lsb_bit_packed", name),
            &bits,
            |b, input| {
                b.iter(|| {
                    input
                        .iter()
                        .map(|value| u64::from(value.count_ones()))
                        .sum::<u64>()
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("sparse_u32", name), &sparse, |b, input| {
            b.iter(|| {
                input.iter().fold(0_u64, |occupied, &index| {
                    occupied + u64::from(black_box(index) < CHUNK_VOXELS as u32)
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, storage);
criterion_main!(benches);
