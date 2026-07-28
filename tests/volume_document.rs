mod common;

use std::{path::PathBuf, time::SystemTime};

use voxel_ink::{
    VolumeDocument,
    cancel::CancellationToken,
    codecs::SourceVariant,
    volume::{HISTORY_BYTES, HISTORY_TRANSACTIONS, SourceIdentity},
};

use common::{make_document, try_make_document};

fn shared_values(document: &VolumeDocument) -> Vec<u8> {
    let shared = document.shared_occupancy();
    let occupancy = shared.try_read().unwrap();
    let count = document.geometry().dimensions.into_iter().product::<u32>();
    (0..count)
        .map(|index| u8::from(occupancy.occupied(index).unwrap()))
        .collect()
}

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
fn shared_occupancy_stays_coherent_through_preview_cancel_undo_and_redo() {
    let initial = vec![0, 1, 0, 1, 0];
    let mut document = make_document([5, 1, 1], initial.clone(), false);
    let shared = document.shared_occupancy();
    assert_eq!(shared.dimensions(), [5, 1, 1]);
    assert_eq!(shared_values(&document), initial);

    document.begin_stroke(1).unwrap();
    document.apply_stroke_indices([0, 2]).unwrap();
    assert_eq!(shared_values(&document), [1, 1, 1, 1, 0]);
    document.cancel_stroke();
    assert_eq!(shared_values(&document), initial);

    document.begin_stroke(0).unwrap();
    document.apply_stroke_indices([1, 3]).unwrap();
    assert_eq!(document.finish_stroke(), Some(2));
    assert_eq!(shared_values(&document), [0; 5]);
    assert_eq!(document.undo(), Some(1));
    assert_eq!(shared_values(&document), initial);
    assert_eq!(document.redo(), Some(2));
    assert_eq!(shared_values(&document), [0; 5]);

    let retained = shared.try_read().unwrap();
    assert!(
        (0..5).all(|index| retained.occupied(index).is_some_and(|occupied| !occupied)),
        "previously published handles must observe the live document state"
    );
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

fn index(dimensions: [u32; 3], voxel: [u32; 3]) -> usize {
    (voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2])) as usize
}

fn closed_shell() -> ([u32; 3], Vec<u8>) {
    let dimensions = [5, 5, 5];
    let mut occupancy = vec![0; 125];
    for z in 1..=3 {
        for y in 1..=3 {
            for x in 1..=3 {
                if x == 1 || x == 3 || y == 1 || y == 3 || z == 1 || z == 3 {
                    occupancy[index(dimensions, [x, y, z])] = 1;
                }
            }
        }
    }
    (dimensions, occupancy)
}

#[test]
fn enclosed_cavity_is_one_undoable_edit_and_is_idempotent() {
    let (dimensions, occupancy) = closed_shell();
    let original_count = occupancy.iter().map(|&value| u64::from(value)).sum::<u64>();
    let mut document = make_document(dimensions, occupancy.clone(), false);

    assert_eq!(
        document
            .fill_enclosed_cavities(&CancellationToken::default())
            .unwrap(),
        1
    );
    assert_eq!(document.value([2, 2, 2]), Some(1));
    assert_eq!(shared_values(&document)[index(dimensions, [2, 2, 2])], 1);
    assert_eq!(document.value([0, 0, 0]), Some(0));
    assert_eq!(document.revision(), 2);
    assert_eq!(document.occupied_count(), original_count + 1);
    assert!(document.summary().can_undo);

    assert_eq!(document.undo(), Some(1));
    assert_eq!(document.view().occupancy, occupancy);
    assert_eq!(shared_values(&document)[index(dimensions, [2, 2, 2])], 0);
    assert_eq!(document.redo(), Some(2));
    assert_eq!(document.value([2, 2, 2]), Some(1));
    assert_eq!(shared_values(&document)[index(dimensions, [2, 2, 2])], 1);
    assert_eq!(
        document
            .fill_enclosed_cavities(&CancellationToken::default())
            .unwrap(),
        0,
        "a volume without empty cavities must not create an empty revision"
    );
    assert_eq!(document.revision(), 2);
}

#[test]
fn cavity_fill_fast_paths_leave_empty_full_and_thin_volumes_unchanged() {
    for (dimensions, occupancy) in [
        ([64, 64, 64], vec![0; 64 * 64 * 64]),
        ([64, 64, 64], vec![1; 64 * 64 * 64]),
        (
            [5, 5, 2],
            (0..50).map(|index| u8::from(index % 3 == 0)).collect(),
        ),
    ] {
        let mut document = make_document(dimensions, occupancy.clone(), false);
        assert_eq!(
            document
                .fill_enclosed_cavities(&CancellationToken::default())
                .unwrap(),
            0
        );
        assert_eq!(document.view().occupancy, occupancy);
        assert_eq!(document.revision(), 1);
        assert!(!document.summary().can_undo);
    }
}

#[test]
fn any_face_opening_keeps_the_shell_interior_empty() {
    for opening in [
        [1, 2, 2],
        [3, 2, 2],
        [2, 1, 2],
        [2, 3, 2],
        [2, 2, 1],
        [2, 2, 3],
    ] {
        let (dimensions, mut occupancy) = closed_shell();
        occupancy[index(dimensions, opening)] = 0;
        let mut document = make_document(dimensions, occupancy.clone(), false);

        assert_eq!(
            document
                .fill_enclosed_cavities(&CancellationToken::default())
                .unwrap(),
            0,
            "opening {opening:?} must connect the cavity to the outside"
        );
        assert_eq!(document.view().occupancy, occupancy);
        assert_eq!(document.revision(), 1);
        assert!(!document.summary().can_undo);
    }
}

#[test]
fn multiple_disconnected_cavities_are_filled_together() {
    let dimensions = [7, 7, 7];
    let mut occupancy = vec![0; 343];
    for z in 1..=5 {
        for y in 1..=5 {
            for x in 1..=5 {
                occupancy[index(dimensions, [x, y, z])] = 1;
            }
        }
    }
    for cavity in [[2, 2, 2], [4, 4, 4]] {
        occupancy[index(dimensions, cavity)] = 0;
    }
    let mut document = make_document(dimensions, occupancy, false);

    assert_eq!(
        document
            .fill_enclosed_cavities(&CancellationToken::default())
            .unwrap(),
        2
    );
    assert_eq!(document.value([2, 2, 2]), Some(1));
    assert_eq!(document.value([4, 4, 4]), Some(1));
}

#[test]
fn corner_contact_does_not_count_as_a_six_connected_opening() {
    let dimensions = [3, 3, 3];
    let mut occupancy = vec![1; 27];
    occupancy[index(dimensions, [0, 0, 0])] = 0;
    occupancy[index(dimensions, [1, 1, 1])] = 0;
    let mut document = make_document(dimensions, occupancy, false);

    assert_eq!(
        document
            .fill_enclosed_cavities(&CancellationToken::default())
            .unwrap(),
        1
    );
    assert_eq!(document.value([0, 0, 0]), Some(0));
    assert_eq!(document.value([1, 1, 1]), Some(1));
}

#[test]
fn cavity_fill_cancellation_is_atomic() {
    let mut document = make_document([3, 3, 3], vec![0; 27], false);
    let cancel = CancellationToken::default();
    cancel.cancel();

    assert!(matches!(
        document.fill_enclosed_cavities(&cancel),
        Err(voxel_ink::AppError::Cancelled)
    ));
    assert_eq!(document.view().occupancy, &[0; 27]);
    assert_eq!(document.revision(), 1);
    assert!(!document.summary().can_undo);
}
