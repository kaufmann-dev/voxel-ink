use std::{
    collections::{HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select};

use crate::{
    AppError, Result,
    brush::{BrushDiameter, Ray, disk_indices, ray_edit_target, sphere_indices, walk_supercover},
    cancel::CancellationToken,
    io::{SaveMode, load_path, save_document},
    mesh::{
        ChunkCoord, ChunkInput, CompactQuad, MeshJobTag, MeshMemoryBudget, TaggedChunkMesh,
        bounded_mesh_queue, chunk_grid, invalidated_chunks_for_indices_bounded, mesh_chunk,
        mesh_queue_capacity,
    },
    volume::{DocumentSummary, Matrix4, SliceAxis, SliceImage, Tool, VolumeDocument, linear_index},
};

const COMMAND_CAPACITY: usize = 64;
const EVENT_CAPACITY: usize = 256;
const PATCH_TILE: u32 = 32;

pub type JobId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BusyKind {
    Loading,
    Saving,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrokeSpace {
    Slice(SliceAxis),
    Volume3D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerSample {
    Slice {
        axis: SliceAxis,
        voxel: [u32; 3],
    },
    Ray {
        ray: Ray,
        active_slice: SliceAxis,
        slice_coordinate: u32,
    },
}

#[derive(Clone, Debug)]
pub enum EngineCommand {
    Load {
        job_id: JobId,
        path: PathBuf,
        cancel: CancellationToken,
    },
    Close,
    SetCursor([u32; 3]),
    BeginStroke {
        tool: Tool,
        diameter: BrushDiameter,
        space: StrokeSpace,
    },
    SliceSample {
        axis: SliceAxis,
        voxel: [u32; 3],
    },
    RaySample {
        ray: Ray,
        active_slice: SliceAxis,
        slice_coordinate: u32,
    },
    CompleteStroke,
    CancelStroke,
    Undo,
    Redo,
    Save {
        job_id: JobId,
        destination: PathBuf,
        mode: SaveMode,
        cancel: CancellationToken,
    },
    CancelJob(JobId),
    RebuildPreview,
    Shutdown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlicePatch {
    pub axis: SliceAxis,
    pub coordinate: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Busy {
        job_id: JobId,
        kind: BusyKind,
        active: bool,
    },
    Error {
        job_id: Option<JobId>,
        message: String,
        source_changed: bool,
    },
    DocumentReplaced {
        epoch: u64,
        summary: DocumentSummary,
        slices: Vec<SliceImage>,
    },
    DocumentClosed {
        epoch: u64,
    },
    Summary {
        epoch: u64,
        summary: DocumentSummary,
    },
    SlicesReplaced {
        epoch: u64,
        revision: u64,
        slices: Vec<SliceImage>,
    },
    SlicePatch {
        epoch: u64,
        revision: u64,
        preview_generation: u64,
        patch: SlicePatch,
    },
    Saved {
        job_id: JobId,
        mode: SaveMode,
        summary: DocumentSummary,
    },
    MeshReset {
        epoch: u64,
        revision: u64,
        dimensions: [u32; 3],
        index_to_world: Matrix4,
    },
    MeshInvalidated {
        epoch: u64,
        revision: u64,
        chunks: Vec<ChunkCoord>,
    },
    MeshChunk {
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
    },
    PreviewTooComplex {
        epoch: u64,
        revision: u64,
    },
}

pub struct EngineHandle {
    commands: Sender<EngineCommand>,
    events: Receiver<EngineEvent>,
    latest_samples: Sender<PointerSample>,
    sample_drain: Receiver<PointerSample>,
    jobs: Arc<Mutex<HashMap<JobId, CancellationToken>>>,
    next_job: AtomicU64,
}

impl EngineHandle {
    #[must_use]
    pub fn spawn(repaint: egui::Context) -> Self {
        let (commands, command_rx) = bounded(COMMAND_CAPACITY);
        let (event_tx, events) = bounded(EVENT_CAPACITY);
        let (latest_samples, sample_rx) = bounded(1);
        let sample_drain = sample_rx.clone();
        let jobs = Arc::new(Mutex::new(HashMap::new()));
        let worker_count =
            NonZeroUsize::new(rayon::current_num_threads()).unwrap_or(NonZeroUsize::MIN);
        let (mesh_tx, mesh_rx) = bounded_mesh_queue(worker_count);
        let state_jobs = Arc::clone(&jobs);
        std::thread::Builder::new()
            .name("voxel-ink-engine".into())
            .spawn(move || {
                EngineState::new(
                    command_rx,
                    sample_rx,
                    event_tx,
                    mesh_tx,
                    mesh_rx,
                    state_jobs,
                    repaint,
                    worker_count,
                )
                .run();
            })
            .expect("failed to start the VoxelInk engine thread");
        Self {
            commands,
            events,
            latest_samples,
            sample_drain,
            jobs,
            next_job: AtomicU64::new(1),
        }
    }

    pub fn send(&self, command: EngineCommand) -> Result<()> {
        if let EngineCommand::CancelJob(job_id) = command {
            self.cancel_job(job_id);
            return Ok(());
        }
        self.commands
            .try_send(command)
            .map_err(|error| match error {
                TrySendError::Full(_) => AppError::EngineBusy,
                TrySendError::Disconnected(_) => AppError::ChannelDisconnected,
            })
    }

    pub fn send_latest_sample(&self, sample: PointerSample) -> Result<()> {
        match self.latest_samples.try_send(sample) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(sample)) => {
                let _ = self.sample_drain.try_recv();
                self.latest_samples
                    .try_send(sample)
                    .map_err(|_| AppError::ChannelDisconnected)
            }
            Err(TrySendError::Disconnected(_)) => Err(AppError::ChannelDisconnected),
        }
    }

    pub fn start_load(&self, path: PathBuf) -> Result<JobId> {
        let (job_id, cancel) = self.new_job();
        match self.commands.try_send(EngineCommand::Load {
            job_id,
            path,
            cancel,
        }) {
            Ok(()) => Ok(job_id),
            Err(error) => {
                self.jobs
                    .lock()
                    .expect("job registry poisoned")
                    .remove(&job_id);
                Err(match error {
                    TrySendError::Full(_) => AppError::EngineBusy,
                    TrySendError::Disconnected(_) => AppError::ChannelDisconnected,
                })
            }
        }
    }

    pub fn start_save(&self, destination: PathBuf, mode: SaveMode) -> Result<JobId> {
        let (job_id, cancel) = self.new_job();
        match self.commands.try_send(EngineCommand::Save {
            job_id,
            destination,
            mode,
            cancel,
        }) {
            Ok(()) => Ok(job_id),
            Err(error) => {
                self.jobs
                    .lock()
                    .expect("job registry poisoned")
                    .remove(&job_id);
                Err(match error {
                    TrySendError::Full(_) => AppError::EngineBusy,
                    TrySendError::Disconnected(_) => AppError::ChannelDisconnected,
                })
            }
        }
    }

    pub fn cancel_job(&self, job_id: JobId) {
        if let Some(token) = self
            .jobs
            .lock()
            .expect("job registry poisoned")
            .get(&job_id)
        {
            token.cancel();
        }
    }

    pub fn try_event(&self) -> Option<EngineEvent> {
        self.events.try_recv().ok()
    }

    fn new_job(&self) -> (JobId, CancellationToken) {
        let job_id = self.next_job.fetch_add(1, Ordering::Relaxed);
        let cancel = CancellationToken::default();
        self.jobs
            .lock()
            .expect("job registry poisoned")
            .insert(job_id, cancel.clone());
        (job_id, cancel)
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        for token in self.jobs.lock().expect("job registry poisoned").values() {
            token.cancel();
        }
        let _ = self.commands.try_send(EngineCommand::Shutdown);
    }
}

#[derive(Debug)]
struct StrokeContext {
    tool: Tool,
    diameter: BrushDiameter,
    space: StrokeSpace,
    last_applied: Option<[u32; 3]>,
}

type MeshResult = (MeshJobTag, Result<TaggedChunkMesh>);

struct EngineState {
    command_rx: Receiver<EngineCommand>,
    sample_rx: Receiver<PointerSample>,
    event_tx: Sender<EngineEvent>,
    repaint: egui::Context,
    jobs: Arc<Mutex<HashMap<JobId, CancellationToken>>>,
    document: Option<VolumeDocument>,
    epoch: u64,
    stroke: Option<StrokeContext>,
    preview_generation: u64,
    mesh_tx: Sender<MeshResult>,
    mesh_rx: Receiver<MeshResult>,
    mesh_capacity: usize,
    mesh_in_flight: usize,
    mesh_in_flight_chunks: HashSet<ChunkCoord>,
    mesh_pending: VecDeque<ChunkCoord>,
    mesh_scan_grid: [u32; 3],
    mesh_scan_next: u64,
    mesh_scan_total: u64,
    mesh_pass_generation: u64,
    mesh_generations: HashMap<ChunkCoord, u64>,
    mesh_budget: MeshMemoryBudget,
    mesh_cancel: CancellationToken,
    preview_too_complex: bool,
}

impl EngineState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        command_rx: Receiver<EngineCommand>,
        sample_rx: Receiver<PointerSample>,
        event_tx: Sender<EngineEvent>,
        mesh_tx: Sender<MeshResult>,
        mesh_rx: Receiver<MeshResult>,
        jobs: Arc<Mutex<HashMap<JobId, CancellationToken>>>,
        repaint: egui::Context,
        worker_count: NonZeroUsize,
    ) -> Self {
        Self {
            command_rx,
            sample_rx,
            event_tx,
            repaint,
            jobs,
            document: None,
            epoch: 0,
            stroke: None,
            preview_generation: 0,
            mesh_tx,
            mesh_rx,
            mesh_capacity: mesh_queue_capacity(worker_count),
            mesh_in_flight: 0,
            mesh_in_flight_chunks: HashSet::new(),
            mesh_pending: VecDeque::new(),
            mesh_scan_grid: [0; 3],
            mesh_scan_next: 0,
            mesh_scan_total: 0,
            mesh_pass_generation: 0,
            mesh_generations: HashMap::new(),
            mesh_budget: MeshMemoryBudget::default(),
            mesh_cancel: CancellationToken::default(),
            preview_too_complex: false,
        }
    }

    fn run(mut self) {
        loop {
            self.dispatch_mesh_jobs();
            select! {
                recv(self.command_rx) -> command => {
                    match command {
                        Ok(EngineCommand::Shutdown) | Err(_) => break,
                        Ok(command) => self.handle_command(command),
                    }
                }
                recv(self.sample_rx) -> sample => {
                    if let Ok(sample) = sample {
                        self.apply_pointer_sample(sample);
                    }
                }
                recv(self.mesh_rx) -> result => {
                    if let Ok(result) = result {
                        self.handle_mesh_result(result);
                    }
                }
                default(Duration::from_millis(25)) => {}
            }
        }
        self.mesh_cancel.cancel();
    }

    fn handle_command(&mut self, command: EngineCommand) {
        match command {
            EngineCommand::Load {
                job_id,
                path,
                cancel,
            } => self.load(job_id, path, cancel),
            EngineCommand::Close => self.close(),
            EngineCommand::SetCursor(cursor) => self.set_cursor(cursor),
            EngineCommand::BeginStroke {
                tool,
                diameter,
                space,
            } => self.begin_stroke(tool, diameter, space),
            EngineCommand::SliceSample { axis, voxel } => {
                self.apply_pointer_sample(PointerSample::Slice { axis, voxel });
            }
            EngineCommand::RaySample {
                ray,
                active_slice,
                slice_coordinate,
            } => self.apply_pointer_sample(PointerSample::Ray {
                ray,
                active_slice,
                slice_coordinate,
            }),
            EngineCommand::CompleteStroke => {
                while let Ok(sample) = self.sample_rx.try_recv() {
                    self.apply_pointer_sample(sample);
                }
                self.complete_stroke();
            }
            EngineCommand::CancelStroke => self.cancel_stroke(),
            EngineCommand::Undo => self.undo(),
            EngineCommand::Redo => self.redo(),
            EngineCommand::Save {
                job_id,
                destination,
                mode,
                cancel,
            } => self.save(job_id, destination, mode, cancel),
            EngineCommand::CancelJob(job_id) => {
                if let Some(token) = self
                    .jobs
                    .lock()
                    .expect("job registry poisoned")
                    .get(&job_id)
                {
                    token.cancel();
                }
            }
            EngineCommand::RebuildPreview => self.reset_mesh_preview(),
            EngineCommand::Shutdown => {}
        }
    }

    fn load(&mut self, job_id: JobId, path: PathBuf, cancel: CancellationToken) {
        self.emit(EngineEvent::Busy {
            job_id,
            kind: BusyKind::Loading,
            active: true,
        });
        let result = load_path(&path, &cancel).and_then(|loaded| {
            let decoded = loaded.decoded;
            VolumeDocument::new(
                decoded.geometry,
                decoded.occupancy,
                loaded.identity,
                decoded.source_variant,
                decoded.save_state,
                decoded.normalized_non_binary,
            )
        });
        match result {
            Ok(document) => {
                let summary = document.summary();
                let slices = match visible_slices(&document) {
                    Ok(slices) => slices,
                    Err(error) => {
                        self.emit_error(Some(job_id), error);
                        self.finish_job(job_id, BusyKind::Loading);
                        return;
                    }
                };
                self.mesh_cancel.cancel();
                self.epoch = self.epoch.saturating_add(1);
                self.document = Some(document);
                self.stroke = None;
                self.mesh_generations.clear();
                self.reset_mesh_preview();
                self.emit(EngineEvent::DocumentReplaced {
                    epoch: self.epoch,
                    summary,
                    slices,
                });
            }
            Err(AppError::Cancelled) => {}
            Err(error) => self.emit_error(Some(job_id), error),
        }
        self.finish_job(job_id, BusyKind::Loading);
    }

    fn save(
        &mut self,
        job_id: JobId,
        destination: PathBuf,
        mode: SaveMode,
        cancel: CancellationToken,
    ) {
        self.emit(EngineEvent::Busy {
            job_id,
            kind: BusyKind::Saving,
            active: true,
        });
        let result = self
            .document
            .as_ref()
            .ok_or(AppError::NoDocument)
            .and_then(|document| save_document(document, &destination, mode, &cancel));
        match result {
            Ok(outcome) => {
                if outcome.mode != SaveMode::Copy
                    && let Some(document) = self.document.as_mut()
                {
                    document.mark_saved(outcome.identity, outcome.variant, true);
                }
                if let Some(document) = &self.document {
                    self.emit(EngineEvent::Saved {
                        job_id,
                        mode: outcome.mode,
                        summary: document.summary(),
                    });
                }
            }
            Err(AppError::Cancelled) => {}
            Err(error) => self.emit_error(Some(job_id), error),
        }
        self.finish_job(job_id, BusyKind::Saving);
    }

    fn finish_job(&self, job_id: JobId, kind: BusyKind) {
        self.jobs
            .lock()
            .expect("job registry poisoned")
            .remove(&job_id);
        self.emit(EngineEvent::Busy {
            job_id,
            kind,
            active: false,
        });
    }

    fn close(&mut self) {
        self.cancel_stroke();
        self.mesh_cancel.cancel();
        self.document = None;
        self.epoch = self.epoch.saturating_add(1);
        self.mesh_pending.clear();
        self.mesh_scan_grid = [0; 3];
        self.mesh_scan_next = 0;
        self.mesh_scan_total = 0;
        self.mesh_generations.clear();
        self.mesh_budget.clear();
        self.emit(EngineEvent::DocumentClosed { epoch: self.epoch });
    }

    fn set_cursor(&mut self, cursor: [u32; 3]) {
        let Some(document) = self.document.as_mut() else {
            return;
        };
        document.set_cursor(cursor);
        let summary = document.summary();
        let revision = document.revision();
        let slices = match visible_slices(document) {
            Ok(slices) => slices,
            Err(error) => {
                self.emit_error(None, error);
                return;
            }
        };
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        self.emit(EngineEvent::SlicesReplaced {
            epoch: self.epoch,
            revision,
            slices,
        });
        self.prioritize_mesh_near_cursor();
    }

    fn begin_stroke(&mut self, tool: Tool, diameter: BrushDiameter, space: StrokeSpace) {
        self.cancel_stroke();
        let Some(document) = self.document.as_mut() else {
            self.emit_error(None, AppError::NoDocument);
            return;
        };
        if tool != Tool::Inspect {
            let value = u8::from(tool == Tool::Brush);
            if let Err(error) = document.begin_stroke(value) {
                self.emit_error(None, error);
                return;
            }
            self.mesh_cancel.cancel();
        }
        self.stroke = Some(StrokeContext {
            tool,
            diameter,
            space,
            last_applied: None,
        });
    }

    fn apply_pointer_sample(&mut self, sample: PointerSample) {
        let Some(stroke) = self.stroke.as_ref() else {
            return;
        };
        let target = match (stroke.space, sample) {
            (StrokeSpace::Slice(expected), PointerSample::Slice { axis, voxel })
                if expected == axis =>
            {
                Some(voxel)
            }
            (
                StrokeSpace::Volume3D,
                PointerSample::Ray {
                    ray,
                    active_slice,
                    slice_coordinate,
                },
            ) => self.document.as_ref().and_then(|document| {
                let view = document.view();
                ray_edit_target(
                    view.dimensions,
                    view.occupancy,
                    view.geometry,
                    ray,
                    stroke.tool,
                    active_slice,
                    slice_coordinate,
                )
            }),
            _ => None,
        };
        let Some(target) = target else {
            return;
        };
        self.apply_target(target);
    }

    fn apply_target(&mut self, target: [u32; 3]) {
        let Some(stroke) = self.stroke.as_mut() else {
            return;
        };
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if linear_index(document.geometry().dimensions, target).is_none() {
            return;
        }
        let previous_cursor = document.cursor();
        if stroke.tool == Tool::Inspect {
            document.set_cursor(target);
            stroke.last_applied = Some(target);
            let summary = document.summary();
            let revision = document.revision();
            let slices = match visible_slices(document) {
                Ok(slices) => slices,
                Err(error) => {
                    self.emit_error(None, error);
                    return;
                }
            };
            self.emit(EngineEvent::Summary {
                epoch: self.epoch,
                summary,
            });
            self.emit(EngineEvent::SlicesReplaced {
                epoch: self.epoch,
                revision,
                slices,
            });
            return;
        }
        let replacement_axes = axes_with_changed_coordinate(previous_cursor, target);
        let replaced_axes = replacement_axes.iter().copied().collect::<HashSet<_>>();
        let mut patch_tiles = HashSet::new();
        let space = stroke.space;
        let diameter = stroke.diameter;
        let start = stroke.last_applied.unwrap_or(target);
        let apply_result = walk_supercover(start, target, |center| {
            let footprint = match space {
                StrokeSpace::Slice(axis) => {
                    disk_indices(document.geometry().dimensions, axis, center, diameter)
                }
                StrokeSpace::Volume3D => {
                    sphere_indices(document.geometry().dimensions, center, diameter)
                }
            };
            document.apply_stroke_indices(footprint.iter().copied())?;
            add_slice_patch_tiles(
                document,
                footprint.iter().copied(),
                &replaced_axes,
                &mut patch_tiles,
            )
        });
        if let Err(error) = apply_result {
            document.cancel_stroke();
            self.stroke = None;
            self.emit_error(None, error);
            self.resume_mesh_after_cancel();
            return;
        }
        document.set_cursor(target);
        stroke.last_applied = Some(target);
        self.preview_generation = self.preview_generation.saturating_add(1);
        let summary = document.summary();
        let revision = document.revision();
        let replacements = replacement_axes
            .iter()
            .map(|&axis| document.try_extract_slice(axis, target[axis.normal_axis()]))
            .collect::<Result<Vec<_>>>();
        let replacements = match replacements {
            Ok(replacements) => replacements,
            Err(error) => {
                document.cancel_stroke();
                document.set_cursor(previous_cursor);
                self.stroke = None;
                self.emit_error(None, error);
                self.resume_mesh_after_cancel();
                return;
            }
        };
        let patches = match slice_patches_from_tiles(document, patch_tiles) {
            Ok(patches) => patches,
            Err(error) => {
                document.cancel_stroke();
                document.set_cursor(previous_cursor);
                self.stroke = None;
                self.emit_error(None, error);
                self.resume_mesh_after_cancel();
                return;
            }
        };
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        if !replacements.is_empty() {
            self.emit(EngineEvent::SlicesReplaced {
                epoch: self.epoch,
                revision,
                slices: replacements,
            });
        }
        for patch in patches {
            self.emit(EngineEvent::SlicePatch {
                epoch: self.epoch,
                revision,
                preview_generation: self.preview_generation,
                patch,
            });
        }
    }

    fn complete_stroke(&mut self) {
        let Some(stroke) = self.stroke.take() else {
            return;
        };
        if stroke.tool == Tool::Inspect {
            return;
        }
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if document.finish_stroke().is_none() {
            self.resume_mesh_after_cancel();
            return;
        }
        let invalidated = document
            .latest_undo_indices()
            .map(|indices| {
                invalidated_chunks_for_indices_bounded(
                    document.geometry().dimensions,
                    indices.iter().copied(),
                    self.mesh_capacity,
                )
            })
            .unwrap_or_else(|| Some(Vec::new()));
        let summary = document.summary();
        let revision = document.revision();
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        if let Some(invalidated) = invalidated {
            self.restart_mesh_after_change(invalidated, revision);
        } else {
            self.reset_mesh_preview();
        }
    }

    fn cancel_stroke(&mut self) {
        let editable = self
            .stroke
            .take()
            .is_some_and(|stroke| stroke.tool != Tool::Inspect);
        if editable {
            if let Some(document) = self.document.as_mut() {
                document.cancel_stroke();
                let revision = document.revision();
                let summary = document.summary();
                let slices = match visible_slices(document) {
                    Ok(slices) => slices,
                    Err(error) => {
                        self.emit_error(None, error);
                        self.resume_mesh_after_cancel();
                        return;
                    }
                };
                self.emit(EngineEvent::Summary {
                    epoch: self.epoch,
                    summary,
                });
                self.emit(EngineEvent::SlicesReplaced {
                    epoch: self.epoch,
                    revision,
                    slices,
                });
            }
            self.resume_mesh_after_cancel();
        }
    }

    fn undo(&mut self) {
        self.cancel_stroke();
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if document.undo().is_none() {
            return;
        }
        let invalidated = document
            .latest_redo_indices()
            .map(|indices| {
                invalidated_chunks_for_indices_bounded(
                    document.geometry().dimensions,
                    indices.iter().copied(),
                    self.mesh_capacity,
                )
            })
            .unwrap_or_else(|| Some(Vec::new()));
        let revision = document.revision();
        let summary = document.summary();
        let slices = match visible_slices(document) {
            Ok(slices) => slices,
            Err(error) => {
                self.emit_error(None, error);
                return;
            }
        };
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        self.emit(EngineEvent::SlicesReplaced {
            epoch: self.epoch,
            revision,
            slices,
        });
        if let Some(invalidated) = invalidated {
            self.restart_mesh_after_change(invalidated, revision);
        } else {
            self.reset_mesh_preview();
        }
    }

    fn redo(&mut self) {
        self.cancel_stroke();
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if document.redo().is_none() {
            return;
        }
        let invalidated = document
            .latest_undo_indices()
            .map(|indices| {
                invalidated_chunks_for_indices_bounded(
                    document.geometry().dimensions,
                    indices.iter().copied(),
                    self.mesh_capacity,
                )
            })
            .unwrap_or_else(|| Some(Vec::new()));
        let revision = document.revision();
        let summary = document.summary();
        let slices = match visible_slices(document) {
            Ok(slices) => slices,
            Err(error) => {
                self.emit_error(None, error);
                return;
            }
        };
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        self.emit(EngineEvent::SlicesReplaced {
            epoch: self.epoch,
            revision,
            slices,
        });
        if let Some(invalidated) = invalidated {
            self.restart_mesh_after_change(invalidated, revision);
        } else {
            self.reset_mesh_preview();
        }
    }

    fn reset_mesh_preview(&mut self) {
        self.mesh_cancel.cancel();
        self.mesh_pending.clear();
        self.mesh_budget.clear();
        self.preview_too_complex = false;
        self.mesh_cancel = CancellationToken::default();
        self.mesh_pass_generation = self.mesh_pass_generation.saturating_add(1);
        self.mesh_generations.clear();
        let Some(document) = &self.document else {
            self.mesh_scan_grid = [0; 3];
            self.mesh_scan_next = 0;
            self.mesh_scan_total = 0;
            return;
        };
        let dimensions = document.geometry().dimensions;
        let revision = document.revision();
        let index_to_world = document.geometry().index_to_world;
        self.mesh_scan_grid = chunk_grid(dimensions);
        self.mesh_scan_next = 0;
        self.mesh_scan_total = self.mesh_scan_grid.into_iter().map(u64::from).product();
        self.prioritize_mesh_near_cursor();
        self.emit(EngineEvent::MeshReset {
            epoch: self.epoch,
            revision,
            dimensions,
            index_to_world,
        });
    }

    fn restart_mesh_after_change(&mut self, invalidated: Vec<ChunkCoord>, revision: u64) {
        self.mesh_cancel.cancel();
        self.mesh_cancel = CancellationToken::default();
        self.preview_too_complex = false;
        if self.document.is_none() {
            return;
        }
        let mut required = self.mesh_pending.clone();
        for chunk in self
            .mesh_in_flight_chunks
            .iter()
            .copied()
            .chain(invalidated.iter().copied())
        {
            if !push_unique_bounded(&mut required, chunk, self.mesh_capacity) {
                self.reset_mesh_preview();
                return;
            }
        }
        self.mesh_pending = required;
        for &chunk in &invalidated {
            self.mesh_budget.remove(chunk);
            let generation = self
                .mesh_generations
                .get(&chunk)
                .copied()
                .unwrap_or(self.mesh_pass_generation)
                .saturating_add(1);
            self.mesh_generations.insert(chunk, generation);
        }
        self.prioritize_mesh_near_cursor();
        self.emit(EngineEvent::MeshInvalidated {
            epoch: self.epoch,
            revision,
            chunks: invalidated,
        });
    }

    fn resume_mesh_after_cancel(&mut self) {
        self.mesh_cancel = CancellationToken::default();
        for chunk in self.mesh_in_flight_chunks.clone() {
            if !push_unique_bounded(&mut self.mesh_pending, chunk, self.mesh_capacity) {
                self.reset_mesh_preview();
                return;
            }
        }
        self.prioritize_mesh_near_cursor();
    }

    fn prioritize_mesh_near_cursor(&mut self) {
        let Some(document) = &self.document else {
            return;
        };
        let cursor_chunk = ChunkCoord::from_voxel(document.cursor());
        let center_chunk =
            ChunkCoord::from_voxel(document.geometry().dimensions.map(|axis| axis / 2));
        promote_bounded(&mut self.mesh_pending, center_chunk, self.mesh_capacity);
        promote_bounded(&mut self.mesh_pending, cursor_chunk, self.mesh_capacity);
        let grid = chunk_grid(document.geometry().dimensions);
        for radius in 1_i64..=2 {
            for z in -radius..=radius {
                for y in -radius..=radius {
                    for x in -radius..=radius {
                        if x.abs() + y.abs() + z.abs() != radius {
                            continue;
                        }
                        let candidate = [
                            i64::from(cursor_chunk.x) + x,
                            i64::from(cursor_chunk.y) + y,
                            i64::from(cursor_chunk.z) + z,
                        ];
                        if candidate.iter().zip(grid).any(|(&coordinate, extent)| {
                            coordinate < 0 || coordinate >= i64::from(extent)
                        }) {
                            continue;
                        }
                        promote_bounded(
                            &mut self.mesh_pending,
                            ChunkCoord::new(
                                candidate[0] as u32,
                                candidate[1] as u32,
                                candidate[2] as u32,
                            ),
                            self.mesh_capacity,
                        );
                    }
                }
            }
        }
    }

    fn dispatch_mesh_jobs(&mut self) {
        if self
            .stroke
            .as_ref()
            .is_some_and(|stroke| stroke.tool != Tool::Inspect)
            || self.preview_too_complex
        {
            return;
        }
        while self.mesh_in_flight < self.mesh_capacity {
            let Some(chunk) = self.next_mesh_chunk() else {
                break;
            };
            let Some(document) = &self.document else {
                break;
            };
            let generation = self
                .mesh_generations
                .get(&chunk)
                .copied()
                .unwrap_or(self.mesh_pass_generation);
            let tag = MeshJobTag {
                document_epoch: self.epoch,
                revision: document.revision(),
                chunk_generation: generation,
                chunk,
            };
            let input = match ChunkInput::copy_from_volume(
                document.geometry().dimensions,
                document.view().occupancy,
                chunk,
                &self.mesh_cancel,
            ) {
                Ok(input) => input,
                Err(AppError::Cancelled) => break,
                Err(error) => {
                    self.emit_error(None, error);
                    continue;
                }
            };
            let sender = self.mesh_tx.clone();
            let cancel = self.mesh_cancel.clone();
            self.mesh_in_flight += 1;
            self.mesh_in_flight_chunks.insert(chunk);
            rayon::spawn(move || {
                let result = mesh_chunk(&input, tag, &cancel);
                let _ = sender.send((tag, result));
            });
        }
    }

    fn next_mesh_chunk(&mut self) -> Option<ChunkCoord> {
        let pending = self.mesh_pending.len();
        for _ in 0..pending {
            let chunk = self.mesh_pending.pop_front()?;
            if self.mesh_in_flight_chunks.contains(&chunk) {
                self.mesh_pending.push_back(chunk);
            } else {
                return Some(chunk);
            }
        }
        while self.mesh_scan_next < self.mesh_scan_total {
            let chunk = chunk_from_linear(self.mesh_scan_next, self.mesh_scan_grid)?;
            self.mesh_scan_next += 1;
            if self.mesh_in_flight_chunks.contains(&chunk) {
                let _ = push_unique_bounded(&mut self.mesh_pending, chunk, self.mesh_capacity);
                continue;
            }
            return Some(chunk);
        }
        None
    }

    fn handle_mesh_result(&mut self, (tag, result): MeshResult) {
        self.mesh_in_flight = self.mesh_in_flight.saturating_sub(1);
        self.mesh_in_flight_chunks.remove(&tag.chunk);
        if self.preview_too_complex {
            return;
        }
        let Some(document) = &self.document else {
            return;
        };
        let generation = self
            .mesh_generations
            .get(&tag.chunk)
            .copied()
            .unwrap_or(self.mesh_pass_generation);
        if tag.is_stale(self.epoch, document.revision(), generation) {
            return;
        }
        let tagged = match result {
            Ok(tagged) => tagged,
            Err(AppError::Cancelled) => return,
            Err(error) => {
                self.emit_error(None, error);
                return;
            }
        };
        if self
            .mesh_budget
            .try_replace_mesh_or_cancel(tag.chunk, &tagged.mesh, &self.mesh_cancel)
            .is_err()
        {
            self.preview_too_complex = true;
            self.mesh_pending.clear();
            self.mesh_scan_next = self.mesh_scan_total;
            self.emit(EngineEvent::PreviewTooComplex {
                epoch: self.epoch,
                revision: document.revision(),
            });
            return;
        }
        self.emit(EngineEvent::MeshChunk {
            tag: tagged.tag,
            quads: tagged.mesh.quads,
        });
    }

    fn emit_error(&self, job_id: Option<JobId>, error: AppError) {
        let source_changed = matches!(error, AppError::SourceChanged(_));
        self.emit(EngineEvent::Error {
            job_id,
            message: error.to_string(),
            source_changed,
        });
    }

    fn emit(&self, event: EngineEvent) {
        if self.event_tx.send(event).is_ok() {
            self.repaint.request_repaint();
        }
    }
}

fn visible_slices(document: &VolumeDocument) -> Result<Vec<SliceImage>> {
    let cursor = document.cursor();
    [SliceAxis::Axial, SliceAxis::Coronal, SliceAxis::Sagittal]
        .into_iter()
        .map(|axis| document.try_extract_slice(axis, cursor[axis.normal_axis()]))
        .collect()
}

fn axes_with_changed_coordinate(before: [u32; 3], after: [u32; 3]) -> Vec<SliceAxis> {
    [SliceAxis::Axial, SliceAxis::Coronal, SliceAxis::Sagittal]
        .into_iter()
        .filter(|axis| before[axis.normal_axis()] != after[axis.normal_axis()])
        .collect()
}

fn add_slice_patch_tiles(
    document: &VolumeDocument,
    indices: impl IntoIterator<Item = u32>,
    replaced_axes: &HashSet<SliceAxis>,
    tiles: &mut HashSet<(SliceAxis, u32, u32)>,
) -> Result<()> {
    let dimensions = document.geometry().dimensions;
    let cursor = document.cursor();
    let plane = u64::from(dimensions[0]) * u64::from(dimensions[1]);
    for index in indices {
        let index = u64::from(index);
        let z = (index / plane) as u32;
        let remainder = index % plane;
        let y = (remainder / u64::from(dimensions[0])) as u32;
        let x = (remainder % u64::from(dimensions[0])) as u32;
        for axis in [SliceAxis::Axial, SliceAxis::Coronal, SliceAxis::Sagittal] {
            if replaced_axes.contains(&axis) {
                continue;
            }
            let (coordinate, u, v) = match axis {
                SliceAxis::Axial => (z, x, y),
                SliceAxis::Coronal => (y, x, z),
                SliceAxis::Sagittal => (x, y, z),
            };
            if coordinate == cursor[axis.normal_axis()] {
                if tiles.len() == tiles.capacity() {
                    tiles.try_reserve(64).map_err(|_| AppError::Allocation {
                        bytes: tiles.len().saturating_add(64).saturating_mul(size_of::<(
                            SliceAxis,
                            u32,
                            u32,
                        )>(
                        )),
                    })?;
                }
                tiles.insert((axis, u / PATCH_TILE, v / PATCH_TILE));
            }
        }
    }
    Ok(())
}

fn slice_patches_from_tiles(
    document: &VolumeDocument,
    tiles: HashSet<(SliceAxis, u32, u32)>,
) -> Result<Vec<SlicePatch>> {
    let dimensions = document.geometry().dimensions;
    let cursor = document.cursor();
    let mut patches = Vec::new();
    patches
        .try_reserve_exact(tiles.len())
        .map_err(|_| AppError::Allocation {
            bytes: tiles.len().saturating_mul(size_of::<SlicePatch>()),
        })?;
    for (axis, tile_x, tile_y) in tiles {
        let (slice_width, slice_height) = match axis {
            SliceAxis::Axial => (dimensions[0], dimensions[1]),
            SliceAxis::Coronal => (dimensions[0], dimensions[2]),
            SliceAxis::Sagittal => (dimensions[1], dimensions[2]),
        };
        let x = tile_x * PATCH_TILE;
        let y = tile_y * PATCH_TILE;
        let width = PATCH_TILE.min(slice_width - x);
        let height = PATCH_TILE.min(slice_height - y);
        let coordinate = cursor[axis.normal_axis()];
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| AppError::Allocation { bytes: pixel_count })?;
        for row in y..y + height {
            for column in x..x + width {
                let voxel = match axis {
                    SliceAxis::Axial => [column, row, coordinate],
                    SliceAxis::Coronal => [column, coordinate, row],
                    SliceAxis::Sagittal => [coordinate, column, row],
                };
                pixels.push(document.value(voxel).ok_or_else(|| {
                    AppError::InvalidDimensions("slice patch voxel is outside the volume".into())
                })?);
            }
        }
        patches.push(SlicePatch {
            axis,
            coordinate,
            x,
            y,
            width,
            height,
            pixels,
        });
    }
    Ok(patches)
}

fn chunk_from_linear(index: u64, grid: [u32; 3]) -> Option<ChunkCoord> {
    if grid.contains(&0) {
        return None;
    }
    let plane = u64::from(grid[0]).checked_mul(u64::from(grid[1]))?;
    let z = index / plane;
    let remainder = index % plane;
    let y = remainder / u64::from(grid[0]);
    let x = remainder % u64::from(grid[0]);
    if z >= u64::from(grid[2]) {
        return None;
    }
    Some(ChunkCoord::new(
        u32::try_from(x).ok()?,
        u32::try_from(y).ok()?,
        u32::try_from(z).ok()?,
    ))
}

fn push_unique_bounded(
    queue: &mut VecDeque<ChunkCoord>,
    chunk: ChunkCoord,
    capacity: usize,
) -> bool {
    if queue.contains(&chunk) {
        return true;
    }
    if queue.len() >= capacity {
        return false;
    }
    queue.push_back(chunk);
    true
}

fn promote_bounded(queue: &mut VecDeque<ChunkCoord>, chunk: ChunkCoord, capacity: usize) {
    if let Some(position) = queue.iter().position(|candidate| *candidate == chunk) {
        queue.remove(position);
        queue.push_front(chunk);
    } else if queue.len() < capacity {
        queue.push_front(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_sample_channel_discards_intermediate_motion() {
        let context = egui::Context::default();
        let engine = EngineHandle::spawn(context);
        engine
            .send_latest_sample(PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [1, 1, 1],
            })
            .unwrap();
        engine
            .send_latest_sample(PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [9, 9, 9],
            })
            .unwrap();
    }

    #[test]
    fn tile_patches_never_exceed_32_squared() {
        assert_eq!(PATCH_TILE, 32);
    }

    #[test]
    fn lazy_chunk_coordinates_cover_long_volumes_without_materializing_a_queue() {
        let grid = chunk_grid([crate::error::MAX_VOXELS as u32, 1, 1]);
        let total = grid.into_iter().map(u64::from).product::<u64>();
        assert_eq!(total, 67_108_864);
        assert_eq!(chunk_from_linear(0, grid), Some(ChunkCoord::new(0, 0, 0)));
        assert_eq!(
            chunk_from_linear(total - 1, grid),
            Some(ChunkCoord::new(grid[0] - 1, 0, 0))
        );
        assert_eq!(chunk_from_linear(total, grid), None);

        let mut pending = VecDeque::new();
        assert!(push_unique_bounded(
            &mut pending,
            ChunkCoord::new(1, 0, 0),
            2
        ));
        assert!(push_unique_bounded(
            &mut pending,
            ChunkCoord::new(2, 0, 0),
            2
        ));
        assert!(!push_unique_bounded(
            &mut pending,
            ChunkCoord::new(3, 0, 0),
            2
        ));
        assert_eq!(pending.len(), 2);
    }
}
