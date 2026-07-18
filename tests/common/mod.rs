#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use voxel_ink::{
    VolumeDocument,
    cancel::CancellationToken,
    codecs::{BinvoxMetadata, CodecSaveState, SourceVariant},
    io::hash_path,
    volume::{SourceIdentity, SpatialUnits, VolumeGeometry},
};

pub fn identity_geometry(dimensions: [u32; 3]) -> VolumeGeometry {
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
    .expect("identity geometry should be valid")
}

pub fn make_document(
    dimensions: [u32; 3],
    occupancy: Vec<u8>,
    normalized_non_binary: bool,
) -> VolumeDocument {
    try_make_document(dimensions, occupancy, normalized_non_binary)
        .expect("test document should be valid")
}

pub fn try_make_document(
    dimensions: [u32; 3],
    occupancy: Vec<u8>,
    normalized_non_binary: bool,
) -> voxel_ink::Result<VolumeDocument> {
    let source_identity = SourceIdentity {
        canonical_path: PathBuf::from("/tmp/voxel-ink-test.binvox"),
        size: 0,
        modified: SystemTime::UNIX_EPOCH,
        sha256: [0; 32],
    };
    VolumeDocument::new(
        identity_geometry(dimensions),
        occupancy,
        source_identity,
        SourceVariant::Binvox { version: 1 },
        CodecSaveState::Binvox(BinvoxMetadata {
            version: 1,
            dimensions,
            translation: [0.0; 3],
            scale: 1.0,
        }),
        normalized_non_binary,
    )
}

pub fn source_identity_for(path: &Path) -> SourceIdentity {
    let canonical_path = path
        .canonicalize()
        .expect("test source should canonicalize");
    let metadata = fs::metadata(&canonical_path).expect("test source metadata should exist");
    SourceIdentity {
        canonical_path: canonical_path.clone(),
        size: metadata.len(),
        modified: metadata
            .modified()
            .expect("test filesystem should report modification time"),
        sha256: hash_path(&canonical_path, &CancellationToken::default())
            .expect("test source should hash"),
    }
}

pub fn document_for_source(
    path: &Path,
    dimensions: [u32; 3],
    occupancy: Vec<u8>,
) -> VolumeDocument {
    VolumeDocument::new(
        identity_geometry(dimensions),
        occupancy,
        source_identity_for(path),
        SourceVariant::Binvox { version: 1 },
        CodecSaveState::Binvox(BinvoxMetadata {
            version: 1,
            dimensions,
            translation: [0.0; 3],
            scale: 1.0,
        }),
        false,
    )
    .expect("test document should be valid")
}

pub fn write_binvox(path: &Path, dimensions: [u32; 3], occupancy: &[u8]) {
    let expected_len = dimensions
        .into_iter()
        .map(|axis| axis as usize)
        .product::<usize>();
    assert_eq!(occupancy.len(), expected_len);
    assert!(occupancy.iter().all(|&value| value <= 1));

    let mut bytes = format!(
        "#binvox 1\ndim {} {} {}\ntranslate 0 0 0\nscale 1\ndata\n",
        dimensions[0], dimensions[1], dimensions[2]
    )
    .into_bytes();
    let [nx, ny, nz] = dimensions.map(|axis| axis as usize);
    let mut run_value = None;
    let mut run_length = 0_u8;
    for file_index in 0..nx * ny * nz {
        let x = file_index / (ny * nz);
        let within_x = file_index % (ny * nz);
        let z = within_x / ny;
        let y = within_x % ny;
        let canonical_index = x + nx * (y + ny * z);
        let value = occupancy[canonical_index];
        if run_value == Some(value) && run_length < u8::MAX {
            run_length += 1;
        } else {
            if let Some(previous) = run_value {
                bytes.extend_from_slice(&[previous, run_length]);
            }
            run_value = Some(value);
            run_length = 1;
        }
    }
    if let Some(value) = run_value {
        bytes.extend_from_slice(&[value, run_length]);
    }
    fs::write(path, bytes).expect("BINVOX fixture should be written");
}
