mod common;

use std::{path::PathBuf, time::SystemTime};

use voxel_ink::{
    codecs::SourceVariant,
    volume::{HISTORY_BYTES, HISTORY_TRANSACTIONS, SourceIdentity},
};

use common::{make_document, try_make_document};

#[test]
fn construction_requires_exact_binary_occupancy() {
    assert!(try_make_document([2, 1, 1], vec![0], false).is_err());
    assert!(try_make_document([2, 1, 1], vec![0, 1, 0], false).is_err());
    assert!(try_make_document([2, 1, 1], vec![0, 2], false).is_err());
}

#[test]
fn strokes_update_count_once_and_cancel_restores_every_change() {
    let mut document = make_document([3, 2, 1], vec![0, 1, 0, 0, 0, 0], false);
    assert_eq!(document.occupied_count(), 1);

    document.begin_stroke(1).unwrap();
    assert_eq!(document.apply_stroke_indices([0, 0, 1, 2]).unwrap(), 2);
    assert_eq!(document.occupied_count(), 3);
    assert_eq!(document.revision(), 1, "active strokes are previews");
    document.cancel_stroke();

    assert_eq!(document.view().occupancy, &[0, 1, 0, 0, 0, 0]);
    assert_eq!(document.occupied_count(), 1);
    assert_eq!(document.revision(), 1);
    assert!(!document.summary().can_undo);
}

#[test]
fn undo_redo_branching_uses_monotonic_unique_revisions() {
    let mut document = make_document([4, 1, 1], vec![0; 4], false);

    document.begin_stroke(1).unwrap();
    document.apply_stroke_indices([0]).unwrap();
    assert_eq!(document.finish_stroke(), Some(2));
    document.begin_stroke(1).unwrap();
    document.apply_stroke_indices([1]).unwrap();
    assert_eq!(document.finish_stroke(), Some(3));

    let saved_identity = document.source_identity().clone();
    document.mark_saved(saved_identity, SourceVariant::Binvox { version: 1 }, false);
    assert_eq!(document.saved_revision(), 3);
    assert!(!document.summary().is_dirty());

    assert_eq!(document.undo(), Some(2));
    assert_eq!(document.view().occupancy, &[1, 0, 0, 0]);
    assert!(document.summary().is_dirty());
    assert_eq!(document.redo(), Some(3));
    assert!(!document.summary().is_dirty());

    assert_eq!(document.undo(), Some(2));
    document.begin_stroke(1).unwrap();
    document.apply_stroke_indices([2]).unwrap();
    assert_eq!(document.finish_stroke(), Some(4));
    assert_eq!(document.next_revision(), 5);
    assert_eq!(document.redo(), None, "a new edit clears the redo branch");
    assert_eq!(document.view().occupancy, &[1, 0, 1, 0]);
    assert_eq!(document.saved_revision(), 3);
    assert!(document.summary().is_dirty());
}

#[test]
fn normalized_import_starts_dirty_and_mark_saved_clears_normalization() {
    let mut document = make_document([1, 1, 1], vec![1], true);
    let original_identity = document.source_identity().clone();
    assert_eq!(document.revision(), 1);
    assert_eq!(document.saved_revision(), 0);
    assert!(document.summary().is_dirty());
    assert!(document.summary().normalized_non_binary);

    let ignored_identity = SourceIdentity {
        canonical_path: PathBuf::from("/tmp/a-copy.binvox"),
        size: 99,
        modified: SystemTime::now(),
        sha256: [7; 32],
    };
    document.mark_saved(
        ignored_identity,
        SourceVariant::Binvox { version: 2 },
        false,
    );

    assert_eq!(document.source_identity(), &original_identity);
    assert_eq!(
        document.source_variant(),
        SourceVariant::Binvox { version: 1 }
    );
    assert_eq!(document.saved_revision(), 1);
    assert!(!document.summary().is_dirty());
    assert!(!document.summary().normalized_non_binary);
}

#[test]
fn transaction_history_is_bounded_and_retains_the_newest_edits() {
    assert_eq!(HISTORY_TRANSACTIONS, 256);
    assert_eq!(HISTORY_BYTES, 128 * 1024 * 1024);
    let mut document = make_document([1, 1, 1], vec![0], false);
    for transaction in 0..=HISTORY_TRANSACTIONS {
        let value = u8::from(transaction.is_multiple_of(2));
        document.begin_stroke(value).unwrap();
        assert_eq!(document.apply_stroke_indices([0]).unwrap(), 1);
        assert!(document.finish_stroke().is_some());
    }

    for _ in 0..HISTORY_TRANSACTIONS {
        assert!(document.undo().is_some());
    }
    assert_eq!(document.undo(), None);
    assert_eq!(document.value([0, 0, 0]), Some(1));
    assert_eq!(document.occupied_count(), 1);
}

#[test]
fn invalid_stroke_inputs_do_not_silently_corrupt_the_document() {
    let mut document = make_document([2, 1, 1], vec![0, 0], false);
    assert!(document.begin_stroke(2).is_err());
    assert!(document.apply_stroke_indices([0]).is_err());

    document.begin_stroke(1).unwrap();
    assert!(document.apply_stroke_indices([0, 2]).is_err());
    assert_eq!(document.value([0, 0, 0]), Some(1));
    document.cancel_stroke();
    assert_eq!(document.view().occupancy, &[0, 0]);
    assert_eq!(document.occupied_count(), 0);
}
