mod common;

use std::{
    num::NonZeroUsize,
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::TrySendError;
use tempfile::tempdir;
use voxel_ink::{
    EngineCommand, EngineEvent, EngineHandle,
    brush::BrushDiameter,
    engine::{PointerSample, StrokeSpace},
    mesh::{ChunkCoord, MeshJobTag, bounded_mesh_queue, mesh_queue_capacity, mesh_result_is_stale},
    volume::{SliceAxis, Tool},
};

use common::write_binvox;

#[test]
fn mesh_tags_reject_each_stale_identity_dimension() {
    let tag = MeshJobTag {
        document_epoch: 7,
        revision: 11,
        chunk_generation: 3,
        chunk: ChunkCoord::new(1, 2, 3),
    };
    assert!(!mesh_result_is_stale(tag, 7, 11, 3));
    assert!(mesh_result_is_stale(tag, 8, 11, 3));
    assert!(mesh_result_is_stale(tag, 7, 12, 3));
    assert!(mesh_result_is_stale(tag, 7, 11, 4));
}

#[test]
fn mesh_work_queue_is_bounded_to_twice_the_worker_count() {
    let workers = NonZeroUsize::new(3).unwrap();
    assert_eq!(mesh_queue_capacity(workers), 6);
    let (sender, receiver) = bounded_mesh_queue(workers);
    for value in 0..6 {
        sender.try_send(value).unwrap();
    }
    assert!(matches!(sender.try_send(99), Err(TrySendError::Full(99))));
    assert_eq!(
        receiver.iter().take(6).collect::<Vec<_>>(),
        (0..6).collect::<Vec<_>>()
    );
}

#[test]
fn engine_interpolates_samples_as_one_revision_and_supports_history() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("line.binvox");
    write_binvox(&path, [4, 1, 1], &[0; 4]);

    let engine = EngineHandle::spawn(egui::Context::default());
    let job_id = engine.start_load(path).unwrap();
    let replaced = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let (epoch, summary) = match replaced {
        EngineEvent::DocumentReplaced { epoch, summary, .. } => (epoch, summary),
        _ => unreachable!(),
    };
    assert_eq!(summary.occupied_count, 0);
    assert_eq!(summary.revision, 1);
    wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::Busy {
                job_id: event_job,
                active: false,
                ..
            } if *event_job == job_id
        )
    });

    engine
        .send(EngineCommand::BeginStroke {
            tool: Tool::Brush,
            diameter: BrushDiameter::new(1).unwrap(),
            space: StrokeSpace::Slice(SliceAxis::Axial),
        })
        .unwrap();
    engine
        .send(EngineCommand::SliceSample {
            axis: SliceAxis::Axial,
            voxel: [0, 0, 0],
        })
        .unwrap();
    engine
        .send(EngineCommand::SliceSample {
            axis: SliceAxis::Axial,
            voxel: [3, 0, 0],
        })
        .unwrap();
    engine.send(EngineCommand::CompleteStroke).unwrap();

    let committed = wait_for_summary(&engine, epoch, 2);
    assert_eq!(committed.occupied_count, 4);
    assert!(committed.can_undo);
    assert!(!committed.can_redo);
    assert!(committed.is_dirty());

    engine.send(EngineCommand::Undo).unwrap();
    let undone = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::Summary {
                epoch: event_epoch,
                summary,
            } if *event_epoch == epoch && summary.revision == 1 && summary.can_redo
        )
    });
    let EngineEvent::Summary {
        summary: undone, ..
    } = undone
    else {
        unreachable!()
    };
    assert_eq!(undone.occupied_count, 0);

    engine.send(EngineCommand::Redo).unwrap();
    let redone = wait_for_summary(&engine, epoch, 2);
    assert_eq!(redone.occupied_count, 4);
    assert!(redone.can_undo);

    engine.send(EngineCommand::Close).unwrap();
    let closed = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentClosed { .. })
    });
    assert!(matches!(
        closed,
        EngineEvent::DocumentClosed { epoch: closed_epoch } if closed_epoch > epoch
    ));
}

#[test]
fn latest_sample_submission_remains_non_blocking_when_intermediate_samples_are_replaceable() {
    let engine = EngineHandle::spawn(egui::Context::default());
    for coordinate in 0..10_000 {
        engine
            .send_latest_sample(PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [coordinate, 0, 0],
            })
            .unwrap();
    }
}

#[test]
fn engine_make_solid_handles_open_anatomy_and_undo() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("open-interior.binvox");
    let dimensions = [5, 5, 5];
    let mut occupancy = vec![0; 125];
    for voxel in [[1, 2, 2], [3, 2, 2], [2, 1, 2], [2, 3, 2]] {
        let index = voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2]);
        occupancy[index as usize] = 1;
    }
    write_binvox(&path, dimensions, &occupancy);

    let engine = EngineHandle::spawn(egui::Context::default());
    engine.start_load(path).unwrap();
    let replaced = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let EngineEvent::DocumentReplaced { epoch, .. } = replaced else {
        unreachable!()
    };

    let job_id = engine.start_make_solid().unwrap();
    let made_solid = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::MadeSolid {
                epoch: event_epoch,
                revision: 2,
                added_voxels: 1,
            } if *event_epoch == epoch
        )
    });
    assert!(matches!(
        made_solid,
        EngineEvent::MadeSolid {
            added_voxels: 1,
            ..
        }
    ));
    wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::Busy {
                job_id: event_job,
                active: false,
                ..
            } if *event_job == job_id
        )
    });

    engine.send(EngineCommand::Undo).unwrap();
    let undone = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::Summary { epoch: event_epoch, summary }
                if *event_epoch == epoch && summary.revision == 1
        )
    });
    let EngineEvent::Summary { summary, .. } = undone else {
        unreachable!()
    };
    assert_eq!(summary.occupied_count, 4);
    assert!(summary.can_redo);
    wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::MeshReset {
                epoch: event_epoch,
                revision: 1,
                ..
            } if *event_epoch == epoch
        )
    });
}

fn wait_for_summary(
    engine: &EngineHandle,
    expected_epoch: u64,
    expected_revision: u64,
) -> voxel_ink::DocumentSummary {
    let event = wait_for_event(engine, |event| {
        matches!(
            event,
            EngineEvent::Summary { epoch, summary }
                if *epoch == expected_epoch && summary.revision == expected_revision
        )
    });
    let EngineEvent::Summary { summary, .. } = event else {
        unreachable!()
    };
    summary
}

fn wait_for_event(engine: &EngineHandle, predicate: impl Fn(&EngineEvent) -> bool) -> EngineEvent {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(event) = engine.try_event()
            && predicate(&event)
        {
            return event;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for engine event"
        );
        thread::sleep(Duration::from_millis(1));
    }
}
