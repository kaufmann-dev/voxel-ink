mod common;

use std::fs;

use tempfile::tempdir;
use voxel_ink::{
    AppError,
    cancel::CancellationToken,
    io::{SaveMode, hash_path, load_path, save_document, verify_source_unchanged},
};

use common::{document_for_source, source_identity_for};

#[test]
fn hashing_is_streamed_and_observes_cancellation() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("source.binvox");
    fs::write(&path, b"abc").unwrap();
    assert_eq!(
        hash_path(&path, &CancellationToken::default()).unwrap(),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );

    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        hash_path(&path, &cancellation),
        Err(AppError::Cancelled)
    ));
}

#[test]
fn source_change_detection_catches_content_changes_and_deletion() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("source.binvox");
    fs::write(&path, b"same-length-a").unwrap();
    let identity = source_identity_for(&path);
    verify_source_unchanged(&identity, &CancellationToken::default()).unwrap();

    fs::write(&path, b"same-length-b").unwrap();
    assert!(matches!(
        verify_source_unchanged(&identity, &CancellationToken::default()),
        Err(AppError::SourceChanged(changed)) if changed == identity.canonical_path
    ));

    let replacement_identity = source_identity_for(&path);
    fs::remove_file(&path).unwrap();
    assert!(matches!(
        verify_source_unchanged(&replacement_identity, &CancellationToken::default()),
        Err(AppError::SourceChanged(changed)) if changed == replacement_identity.canonical_path
    ));
}

#[test]
fn changed_source_refuses_overwrite_without_touching_external_content() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("source.binvox");
    fs::write(&path, b"original source").unwrap();
    let document = document_for_source(&path, [2, 1, 1], vec![1, 0]);

    fs::write(&path, b"external change").unwrap();
    let result = save_document(
        &document,
        &path,
        SaveMode::Overwrite,
        &CancellationToken::default(),
    );
    assert!(matches!(result, Err(AppError::SourceChanged(_))));
    assert_eq!(fs::read(&path).unwrap(), b"external change");
}

#[test]
fn successful_save_atomically_replaces_destination_and_leaves_no_temporary_file() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.binvox");
    let destination = directory.path().join("saved.binvox");
    fs::write(&source, b"source identity").unwrap();
    fs::write(&destination, b"old destination").unwrap();
    let occupancy = vec![1, 0, 1, 0, 1, 0];
    let document = document_for_source(&source, [3, 2, 1], occupancy.clone());

    let outcome = save_document(
        &document,
        &destination,
        SaveMode::SaveAs,
        &CancellationToken::default(),
    )
    .unwrap();
    assert_eq!(outcome.mode, SaveMode::SaveAs);
    assert_eq!(
        outcome.identity.canonical_path,
        destination.canonicalize().unwrap()
    );
    assert_eq!(outcome.identity, source_identity_for(&destination));

    let loaded = load_path(&destination, &CancellationToken::default()).unwrap();
    assert_eq!(loaded.decoded.geometry.dimensions, [3, 2, 1]);
    assert_eq!(loaded.decoded.occupancy, occupancy);
    assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".voxel-ink-save-")
    }));
}

#[test]
fn cancelled_or_disallowed_save_preserves_an_existing_destination() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.binvox");
    let destination = directory.path().join("saved.binvox");
    let wrong_family = directory.path().join("saved.vox");
    fs::write(&source, b"source identity").unwrap();
    fs::write(&destination, b"keep this").unwrap();
    fs::write(&wrong_family, b"keep that too").unwrap();
    let document = document_for_source(&source, [2, 1, 1], vec![1, 0]);

    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        save_document(&document, &destination, SaveMode::SaveAs, &cancellation),
        Err(AppError::Cancelled)
    ));
    assert_eq!(fs::read(&destination).unwrap(), b"keep this");

    assert!(matches!(
        save_document(
            &document,
            &wrong_family,
            SaveMode::SaveAs,
            &CancellationToken::default(),
        ),
        Err(AppError::ConversionNotAllowed(_))
    ));
    assert_eq!(fs::read(&wrong_family).unwrap(), b"keep that too");
}

#[test]
fn save_a_copy_does_not_mutate_document_identity_or_dirty_state() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.binvox");
    let copy = directory.path().join("copy.binvox");
    fs::write(&source, b"source identity").unwrap();
    let mut document = document_for_source(&source, [1, 1, 1], vec![0]);
    document.begin_stroke(1).unwrap();
    document.apply_stroke_indices([0]).unwrap();
    document.finish_stroke().unwrap();
    let identity_before = document.source_identity().clone();
    let summary_before = document.summary();

    let outcome = save_document(
        &document,
        &copy,
        SaveMode::Copy,
        &CancellationToken::default(),
    )
    .unwrap();

    assert_eq!(outcome.mode, SaveMode::Copy);
    assert_eq!(document.source_identity(), &identity_before);
    assert_eq!(document.summary(), summary_before);
    assert!(document.summary().is_dirty());
}
