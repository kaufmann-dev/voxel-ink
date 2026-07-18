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
    brush::{BrushDepth, BrushShape, BrushSize, BrushSpec},
    engine::{BrushPreviewRequest, PointerSample, StrokeSpace},
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
            tool: Tool::Paint,
            spec: one_voxel_brush(),
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
fn brush_preview_is_exact_filtered_and_tagged_without_mutating_the_document() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("preview.binvox");
    write_binvox(&path, [3, 1, 1], &[1, 0, 0]);

    let engine = EngineHandle::spawn(egui::Context::default());
    engine.start_load(path).unwrap();
    let replaced = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let EngineEvent::DocumentReplaced { epoch, summary, .. } = replaced else {
        unreachable!()
    };
    assert_eq!(summary.revision, 1);

    let spec = BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Slice,
        size: BrushSize::voxels(3).unwrap(),
    };
    engine
        .send_latest_preview(BrushPreviewRequest::Brush {
            generation: 42,
            tool: Tool::Paint,
            spec,
            space: StrokeSpace::Slice(SliceAxis::Axial),
            sample: PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [1, 0, 0],
            },
        })
        .unwrap();
    let preview = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::BrushPreview {
                epoch: event_epoch,
                revision: 1,
                generation: 42,
                ..
            } if *event_epoch == epoch
        )
    });
    let EngineEvent::BrushPreview {
        tool,
        target,
        voxels,
        ..
    } = preview
    else {
        unreachable!()
    };
    assert_eq!(tool, Tool::Paint);
    assert_eq!(target, Some([1, 0, 0]));
    assert_eq!(voxels, vec![[1, 0, 0], [2, 0, 0]]);

    engine
        .send_latest_preview(BrushPreviewRequest::Brush {
            generation: 43,
            tool: Tool::Erase,
            spec,
            space: StrokeSpace::Slice(SliceAxis::Axial),
            sample: PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [1, 0, 0],
            },
        })
        .unwrap();
    let erase_preview = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::BrushPreview { generation: 43, .. })
    });
    let EngineEvent::BrushPreview {
        revision, voxels, ..
    } = erase_preview
    else {
        unreachable!()
    };
    assert_eq!(revision, 1);
    assert_eq!(voxels, vec![[0, 0, 0]]);
}

#[test]
fn bridge_preview_and_release_share_one_undoable_voxel_footprint() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("bridge.binvox");
    write_binvox(&path, [7, 3, 1], &{
        let mut occupancy = vec![0; 21];
        occupancy[(1 + 7) as usize] = 1;
        occupancy[(5 + 7) as usize] = 1;
        occupancy
    });

    let engine = EngineHandle::spawn(egui::Context::default());
    engine.start_load(path).unwrap();
    let replaced = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let EngineEvent::DocumentReplaced { epoch, .. } = replaced else {
        unreachable!()
    };
    let sample = PointerSample::Slice {
        axis: SliceAxis::Axial,
        voxel: [5, 1, 0],
    };
    let spec = one_voxel_brush();
    engine
        .send_latest_preview(BrushPreviewRequest::Bridge {
            generation: 8,
            spec,
            space: StrokeSpace::Slice(SliceAxis::Axial),
            anchor: [1, 1, 0],
            plane_normal: [0.0, 0.0, 1.0],
            sample,
        })
        .unwrap();
    let preview = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::BrushPreview {
                generation: 8,
                tool: Tool::Bridge,
                ..
            }
        )
    });
    let EngineEvent::BrushPreview { target, voxels, .. } = preview else {
        unreachable!()
    };
    assert_eq!(target, Some([5, 1, 0]));
    assert_eq!(voxels, vec![[2, 1, 0], [3, 1, 0], [4, 1, 0]]);

    engine
        .send(EngineCommand::CommitBridge {
            anchor: [1, 1, 0],
            plane_normal: [0.0, 0.0, 1.0],
            sample,
            spec,
            space: StrokeSpace::Slice(SliceAxis::Axial),
        })
        .unwrap();
    let committed = wait_for_summary(&engine, epoch, 2);
    assert_eq!(committed.occupied_count, 5);
    assert!(committed.can_undo);

    engine.send(EngineCommand::Undo).unwrap();
    let undone = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::Summary { epoch: event_epoch, summary }
                if *event_epoch == epoch && summary.revision == 1 && summary.can_redo
        )
    });
    let EngineEvent::Summary { summary, .. } = undone else {
        unreachable!()
    };
    assert_eq!(summary.occupied_count, 2);
}

#[test]
fn edits_larger_than_worker_capacity_remain_incremental() {
    let worker_count = NonZeroUsize::new(rayon::current_num_threads()).unwrap();
    let worker_limit = mesh_queue_capacity(worker_count);
    let dirty_chunk_count = u32::try_from(worker_limit + 1).unwrap();
    let total_chunk_count = dirty_chunk_count + 3;
    let dimensions = [total_chunk_count * 32, 1, 1];
    let directory = tempdir().unwrap();
    let path = directory.path().join("wide-edit.binvox");
    write_binvox(&path, dimensions, &vec![0; dimensions[0] as usize]);

    let engine = EngineHandle::spawn(egui::Context::default());
    engine.start_load(path).unwrap();
    let replaced = wait_for_event(&engine, |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let EngineEvent::DocumentReplaced { epoch, .. } = replaced else {
        unreachable!()
    };

    engine
        .send(EngineCommand::BeginStroke {
            tool: Tool::Paint,
            spec: one_voxel_brush(),
            space: StrokeSpace::Slice(SliceAxis::Axial),
        })
        .unwrap();
    engine
        .send(EngineCommand::SliceSample {
            axis: SliceAxis::Axial,
            voxel: [1, 0, 0],
        })
        .unwrap();
    engine
        .send(EngineCommand::SliceSample {
            axis: SliceAxis::Axial,
            voxel: [dirty_chunk_count * 32 - 2, 0, 0],
        })
        .unwrap();
    engine.send(EngineCommand::CompleteStroke).unwrap();

    wait_for_summary(&engine, epoch, 2);
    let mesh_change = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::MeshInvalidated {
                epoch: event_epoch,
                revision: 2,
                ..
            } | EngineEvent::MeshReset {
                epoch: event_epoch,
                revision: 2,
                ..
            } if *event_epoch == epoch
        )
    });
    let EngineEvent::MeshInvalidated { chunks, .. } = mesh_change else {
        panic!("an edit larger than mesh worker capacity triggered a full mesh reset");
    };
    assert_eq!(chunks.len(), dirty_chunk_count as usize);
    assert!(chunks.len() > worker_limit);
    assert!(!chunks.contains(&ChunkCoord::new(total_chunk_count - 1, 0, 0)));
}

#[test]
fn engine_cavity_fill_remeshes_incrementally_and_supports_undo() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("open-interior.binvox");
    let dimensions = [5, 5, 5];
    let mut occupancy = vec![0; 125];
    for voxel in [
        [1, 2, 2],
        [3, 2, 2],
        [2, 1, 2],
        [2, 3, 2],
        [2, 2, 1],
        [2, 2, 3],
    ] {
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

    let job_id = engine.start_fill_enclosed_cavities().unwrap();
    let cavities_filled = wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::CavitiesFilled {
                epoch: event_epoch,
                revision: 2,
                added_voxels: 1,
            } if *event_epoch == epoch
        )
    });
    assert!(matches!(
        cavities_filled,
        EngineEvent::CavitiesFilled {
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
    assert_eq!(summary.occupied_count, 6);
    assert!(summary.can_redo);
    wait_for_event(&engine, |event| {
        matches!(
            event,
            EngineEvent::MeshInvalidated {
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

fn one_voxel_brush() -> BrushSpec {
    BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Slice,
        size: BrushSize::voxels(1).unwrap(),
    }
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
