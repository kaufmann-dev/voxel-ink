use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use tempfile::tempdir;
use voxel_ink::{
    EngineCommand, EngineEvent, EngineHandle, SliceAxis, Tool, VolumeDocument,
    brush::BrushDiameter,
    cancel::CancellationToken,
    engine::{PointerSample, StrokeSpace},
    io::{SaveMode, load_path},
};

const GENERATED_DIMENSIONS: [u32; 3] = [512, 512, 256];
const MAX_FEEDBACK: Duration = Duration::from_millis(100);
const P95_FEEDBACK: Duration = Duration::from_millis(33);

fn enabled() -> bool {
    std::env::var_os("VOXEL_INK_RUN_ACCEPTANCE").is_some()
}

fn write_generated_mask(path: &Path) {
    let mut bytes = b"#binvox 1\ndim 512 512 256\ntranslate 0 0 0\nscale 1\ndata\n".to_vec();
    let mut remaining = GENERATED_DIMENSIONS
        .into_iter()
        .map(u64::from)
        .product::<u64>();
    while remaining != 0 {
        let empty = remaining.min(255) as u8;
        bytes.extend_from_slice(&[0, empty]);
        remaining -= u64::from(empty);
        if remaining != 0 {
            bytes.extend_from_slice(&[1, 1]);
            remaining -= 1;
        }
    }
    fs::write(path, bytes).unwrap();
}

fn wait_for_event(
    engine: &EngineHandle,
    timeout: Duration,
    mut predicate: impl FnMut(&EngineEvent) -> bool,
) -> EngineEvent {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(event) = engine.try_event() {
            if let EngineEvent::Error { message, .. } = &event {
                panic!("engine error during acceptance run: {message}");
            }
            if predicate(&event) {
                return event;
            }
        } else if Instant::now() >= deadline {
            panic!("timed out waiting for an engine event");
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

#[test]
fn generated_mask_meets_slice_feedback_and_event_loop_limits() {
    if !enabled() {
        eprintln!("set VOXEL_INK_RUN_ACCEPTANCE=1 to run the generated-mask acceptance test");
        return;
    }

    let directory = tempdir().unwrap();
    let source = directory.path().join("generated-512x512x256.binvox");
    write_generated_mask(&source);
    let engine = EngineHandle::spawn(egui::Context::default());
    let submission = Instant::now();
    let load_job = engine.start_load(source).unwrap();
    assert!(submission.elapsed() < MAX_FEEDBACK);
    let loaded = wait_for_event(&engine, Duration::from_secs(30), |event| {
        matches!(event, EngineEvent::DocumentReplaced { .. })
    });
    let EngineEvent::DocumentReplaced { summary, .. } = loaded else {
        unreachable!()
    };
    assert_eq!(summary.dimensions, GENERATED_DIMENSIONS);
    while engine.try_event().is_some() {}

    engine
        .send(EngineCommand::BeginStroke {
            tool: Tool::Brush,
            diameter: BrushDiameter::new(31).unwrap(),
            space: StrokeSpace::Slice(SliceAxis::Axial),
        })
        .unwrap();

    let mut latencies = Vec::new();
    let mut preview_generation = 0;
    for x in 64..128 {
        let voxel = [x, 256, 128];
        let started = Instant::now();
        if x == 64 {
            engine
                .send(EngineCommand::SliceSample {
                    axis: SliceAxis::Axial,
                    voxel,
                })
                .unwrap();
        } else {
            engine
                .send_latest_sample(PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel,
                })
                .unwrap();
        }
        let event = wait_for_event(&engine, Duration::from_secs(2), |event| {
            matches!(
                event,
                EngineEvent::SlicePatch {
                    preview_generation: generation,
                    ..
                } if *generation > preview_generation
            )
        });
        let EngineEvent::SlicePatch {
            preview_generation: generation,
            ..
        } = event
        else {
            unreachable!()
        };
        preview_generation = generation;
        latencies.push(started.elapsed());
    }
    engine.send(EngineCommand::CompleteStroke).unwrap();
    let _ = wait_for_event(&engine, Duration::from_secs(5), |event| {
        matches!(
            event,
            EngineEvent::Summary { summary, .. } if summary.revision > 1
        )
    });

    latencies.sort_unstable();
    let p95_index = (latencies.len() * 95).div_ceil(100) - 1;
    let p95 = latencies[p95_index];
    let maximum = *latencies.last().unwrap();
    eprintln!("generated mask slice feedback p95={p95:?}, max={maximum:?}");
    assert!(p95 <= P95_FEEDBACK, "p95 feedback was {p95:?}");
    assert!(maximum <= MAX_FEEDBACK, "maximum feedback was {maximum:?}");

    let copy = directory.path().join("generated-copy.binvox");
    let submission = Instant::now();
    let save_job = engine.start_save(copy.clone(), SaveMode::Copy).unwrap();
    assert!(submission.elapsed() < MAX_FEEDBACK);
    let _ = wait_for_event(
        &engine,
        Duration::from_secs(30),
        |event| matches!(event, EngineEvent::Saved { job_id, .. } if *job_id == save_job),
    );
    assert!(copy.is_file());
    let _ = load_job;
}

#[test]
fn local_craniofacial_nifti_loads_when_available() {
    if !enabled() {
        return;
    }
    let Some(path) = std::env::var_os("VOXEL_INK_CRANIOFACIAL_NIFTI") else {
        eprintln!("VOXEL_INK_CRANIOFACIAL_NIFTI is unset; skipping local dataset");
        return;
    };
    let path = Path::new(&path);
    let started = Instant::now();
    let loaded = load_path(path, &CancellationToken::default()).unwrap();
    eprintln!(
        "local craniofacial NIfTI {:?}: {:?}, {} occupied",
        started.elapsed(),
        loaded.decoded.geometry.dimensions,
        loaded
            .decoded
            .occupancy
            .iter()
            .map(|&value| u64::from(value))
            .sum::<u64>()
    );

    let decoded = loaded.decoded;
    let mut document = VolumeDocument::new(
        decoded.geometry,
        decoded.occupancy,
        loaded.identity,
        decoded.source_variant,
        decoded.save_state,
        decoded.normalized_non_binary,
    )
    .unwrap();
    let started = Instant::now();
    let added = document.make_solid(&CancellationToken::default()).unwrap();
    eprintln!(
        "local craniofacial solid fill {:?}: {added} voxels added",
        started.elapsed()
    );
    assert_eq!(added, 18_725_586);
    assert_eq!(document.occupied_count(), 24_541_095);
    assert!(document.summary().can_undo);
}
