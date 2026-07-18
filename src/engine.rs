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

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, never, select_biased};

use crate::{
    AppError, Result,
    brush::{
        BrushDepth, BrushSpec, Ray, brush_indices, brush_indices_cancellable, ray_plane_voxel,
        surface_edit_target, walk_supercover,
    },
    cancel::CancellationToken,
    io::{SaveMode, load_path, save_document},
    mesh::{
        ChunkCoord, ChunkInput, CompactQuad, MeshJobTag, MeshMemoryBudget, TaggedChunkMesh,
        bounded_mesh_queue, chunk_grid, invalidated_chunks_for_index_spans,
        invalidated_chunks_for_indices, mesh_chunk, mesh_queue_capacity,
    },
    volume::{
        ChangedVoxels, DocumentSummary, Matrix4, SliceAxis, SliceImage, Tool, VolumeDocument,
        linear_index,
    },
};

const COMMAND_CAPACITY: usize = 64;
const EVENT_CAPACITY: usize = 256;
const MESH_EVENT_BACKLOG_LIMIT: usize = 8;
const PATCH_TILE: u32 = 32;
const MAX_PREVIEW_VOXELS: usize = 2_000_000;

pub type JobId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BusyKind {
    Loading,
    Saving,
    FillingCavities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrokeSpace {
    Slice(SliceAxis),
    Volume3D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerSample {
    Slice { axis: SliceAxis, voxel: [u32; 3] },
    Ray { ray: Ray },
}

/// A latest-only request for the exact voxels an edit would change.
///
/// `generation` is supplied by the UI and echoed by [`EngineEvent::BrushPreview`],
/// allowing a newer hover request to supersede work that was already in flight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrushPreviewRequest {
    Brush {
        generation: u64,
        tool: Tool,
        spec: BrushSpec,
        space: StrokeSpace,
        sample: PointerSample,
    },
    Bridge {
        generation: u64,
        spec: BrushSpec,
        space: StrokeSpace,
        anchor: [u32; 3],
        plane_normal: [f64; 3],
        sample: PointerSample,
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
        spec: BrushSpec,
        space: StrokeSpace,
    },
    SliceSample {
        axis: SliceAxis,
        voxel: [u32; 3],
    },
    RaySample {
        ray: Ray,
    },
    CommitBridge {
        anchor: [u32; 3],
        plane_normal: [f64; 3],
        sample: PointerSample,
        spec: BrushSpec,
        space: StrokeSpace,
    },
    CompleteStroke,
    CancelStroke,
    Undo,
    Redo,
    FillEnclosedCavities {
        job_id: JobId,
        cancel: CancellationToken,
    },
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
    BrushPreview {
        epoch: u64,
        revision: u64,
        generation: u64,
        tool: Tool,
        target: Option<[u32; 3]>,
        voxels: Vec<[u32; 3]>,
    },
    BrushPreviewFailed {
        epoch: u64,
        revision: u64,
        generation: u64,
        message: String,
    },
    Saved {
        job_id: JobId,
        mode: SaveMode,
        summary: DocumentSummary,
    },
    CavitiesFilled {
        epoch: u64,
        revision: u64,
        added_voxels: u64,
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
    latest_previews: Sender<CancellableBrushPreview>,
    preview_drain: Receiver<CancellableBrushPreview>,
    active_preview: Mutex<Option<CancellationToken>>,
    jobs: Arc<Mutex<HashMap<JobId, CancellationToken>>>,
    next_job: AtomicU64,
}

#[derive(Clone, Debug)]
struct CancellableBrushPreview {
    request: BrushPreviewRequest,
    cancel: CancellationToken,
}

impl EngineHandle {
    #[must_use]
    pub fn spawn(repaint: egui::Context) -> Self {
        let (commands, command_rx) = bounded(COMMAND_CAPACITY);
        let (event_tx, events) = bounded(EVENT_CAPACITY);
        let (latest_samples, sample_rx) = bounded(1);
        let sample_drain = sample_rx.clone();
        let (latest_previews, preview_rx) = bounded(1);
        let preview_drain = preview_rx.clone();
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
                    preview_rx,
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
            latest_previews,
            preview_drain,
            active_preview: Mutex::new(None),
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

    pub fn send_latest_preview(&self, request: BrushPreviewRequest) -> Result<()> {
        let cancel = CancellationToken::default();
        let mut active = self
            .active_preview
            .lock()
            .expect("preview cancellation state poisoned");
        if let Some(previous) = active.replace(cancel.clone()) {
            previous.cancel();
        }
        if let Ok(queued) = self.preview_drain.try_recv() {
            queued.cancel.cancel();
        }
        let pending = CancellableBrushPreview { request, cancel };
        match self.latest_previews.try_send(pending) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(pending)) => {
                if let Ok(discarded) = self.preview_drain.try_recv() {
                    discarded.cancel.cancel();
                }
                match self.latest_previews.try_send(pending) {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(pending)) => {
                        pending.cancel.cancel();
                        active.take();
                        Err(AppError::EngineBusy)
                    }
                    Err(TrySendError::Disconnected(pending)) => {
                        pending.cancel.cancel();
                        active.take();
                        Err(AppError::ChannelDisconnected)
                    }
                }
            }
            Err(TrySendError::Disconnected(pending)) => {
                pending.cancel.cancel();
                active.take();
                Err(AppError::ChannelDisconnected)
            }
        }
    }

    /// Cancels queued or already-computing brush preview work.
    pub fn cancel_preview(&self) {
        let mut active = self
            .active_preview
            .lock()
            .expect("preview cancellation state poisoned");
        if let Some(cancel) = active.take() {
            cancel.cancel();
        }
        if let Ok(queued) = self.preview_drain.try_recv() {
            queued.cancel.cancel();
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

    pub fn start_fill_enclosed_cavities(&self) -> Result<JobId> {
        let (job_id, cancel) = self.new_job();
        match self
            .commands
            .try_send(EngineCommand::FillEnclosedCavities { job_id, cancel })
        {
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
        self.cancel_preview();
        for token in self.jobs.lock().expect("job registry poisoned").values() {
            token.cancel();
        }
        let _ = self.commands.try_send(EngineCommand::Shutdown);
    }
}

#[derive(Debug)]
struct StrokeContext {
    tool: Tool,
    spec: BrushSpec,
    space: StrokeSpace,
    last_applied: Option<[u32; 3]>,
}

type MeshResult = (MeshJobTag, Result<TaggedChunkMesh>);

#[derive(Debug, Default)]
struct DirtyChunkQueue {
    grid: [u32; 3],
    limit: usize,
    order: VecDeque<ChunkCoord>,
    members: HashSet<ChunkCoord>,
}

impl DirtyChunkQueue {
    fn configure(&mut self, grid: [u32; 3]) {
        self.clear();
        self.grid = grid;
        self.limit = grid
            .into_iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(u64::from(extent)))
            .and_then(|total| usize::try_from(total).ok())
            .unwrap_or(usize::MAX);
    }

    #[must_use]
    fn len(&self) -> usize {
        self.order.len()
    }

    fn clear(&mut self) {
        self.order.clear();
        self.members.clear();
    }

    fn enqueue_back(&mut self, chunk: ChunkCoord) -> bool {
        if self.members.contains(&chunk) {
            return true;
        }
        if !self.contains_coordinate(chunk) || self.members.len() >= self.limit {
            return false;
        }
        self.members.insert(chunk);
        self.order.push_back(chunk);
        true
    }

    fn promote(&mut self, chunk: ChunkCoord) {
        if self.members.remove(&chunk) {
            let position = self
                .order
                .iter()
                .position(|candidate| *candidate == chunk)
                .expect("dirty chunk membership and order diverged");
            self.order.remove(position);
        } else if !self.contains_coordinate(chunk) || self.members.len() >= self.limit {
            return;
        }
        self.members.insert(chunk);
        self.order.push_front(chunk);
    }

    fn pop_front(&mut self) -> Option<ChunkCoord> {
        let chunk = self.order.pop_front()?;
        assert!(
            self.members.remove(&chunk),
            "dirty chunk membership and order diverged"
        );
        Some(chunk)
    }

    fn contains_coordinate(&self, chunk: ChunkCoord) -> bool {
        chunk
            .as_array()
            .into_iter()
            .zip(self.grid)
            .all(|(coordinate, extent)| coordinate < extent)
    }
}

struct EngineState {
    command_rx: Receiver<EngineCommand>,
    sample_rx: Receiver<PointerSample>,
    preview_rx: Receiver<CancellableBrushPreview>,
    event_tx: Sender<EngineEvent>,
    repaint: egui::Context,
    jobs: Arc<Mutex<HashMap<JobId, CancellationToken>>>,
    document: Option<VolumeDocument>,
    epoch: u64,
    stroke: Option<StrokeContext>,
    preview_generation: u64,
    mesh_tx: Sender<MeshResult>,
    mesh_rx: Receiver<MeshResult>,
    mesh_worker_limit: usize,
    mesh_in_flight: usize,
    mesh_in_flight_chunks: HashSet<ChunkCoord>,
    mesh_dirty: DirtyChunkQueue,
    mesh_scan_grid: [u32; 3],
    mesh_scan_next: u64,
    mesh_scan_total: u64,
    mesh_pass_generation: u64,
    mesh_generations: HashMap<ChunkCoord, u64>,
    mesh_budget: MeshMemoryBudget,
    mesh_cancel: CancellationToken,
    preview_too_complex: bool,
    pending_mesh_chunk: Option<(MeshJobTag, Vec<CompactQuad>)>,
}

impl EngineState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        command_rx: Receiver<EngineCommand>,
        sample_rx: Receiver<PointerSample>,
        preview_rx: Receiver<CancellableBrushPreview>,
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
            preview_rx,
            event_tx,
            repaint,
            jobs,
            document: None,
            epoch: 0,
            stroke: None,
            preview_generation: 0,
            mesh_tx,
            mesh_rx,
            mesh_worker_limit: mesh_queue_capacity(worker_count),
            mesh_in_flight: 0,
            mesh_in_flight_chunks: HashSet::new(),
            mesh_dirty: DirtyChunkQueue::default(),
            mesh_scan_grid: [0; 3],
            mesh_scan_next: 0,
            mesh_scan_total: 0,
            mesh_pass_generation: 0,
            mesh_generations: HashMap::new(),
            mesh_budget: MeshMemoryBudget::default(),
            mesh_cancel: CancellationToken::default(),
            preview_too_complex: false,
            pending_mesh_chunk: None,
        }
    }

    fn run(mut self) {
        loop {
            self.flush_pending_mesh_chunk();
            self.dispatch_mesh_jobs();
            let blocked_mesh_rx = never();
            let selectable_mesh_rx = if self.pending_mesh_chunk.is_none() {
                &self.mesh_rx
            } else {
                &blocked_mesh_rx
            };
            let idle_wait = if self.pending_mesh_chunk.is_some() {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(25)
            };
            select_biased! {
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
                recv(self.preview_rx) -> pending => {
                    if let Ok(pending) = pending {
                        self.compute_brush_preview(pending);
                    }
                }
                recv(selectable_mesh_rx) -> result => {
                    if let Ok(result) = result {
                        self.handle_mesh_result(result);
                    }
                }
                default(idle_wait) => {}
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
            EngineCommand::BeginStroke { tool, spec, space } => {
                self.begin_stroke(tool, spec, space)
            }
            EngineCommand::SliceSample { axis, voxel } => {
                self.apply_pointer_sample(PointerSample::Slice { axis, voxel });
            }
            EngineCommand::RaySample { ray } => {
                self.apply_pointer_sample(PointerSample::Ray { ray });
            }
            EngineCommand::CommitBridge {
                anchor,
                plane_normal,
                sample,
                spec,
                space,
            } => self.commit_bridge(anchor, plane_normal, sample, spec, space),
            EngineCommand::CompleteStroke => {
                while let Ok(sample) = self.sample_rx.try_recv() {
                    self.apply_pointer_sample(sample);
                }
                self.complete_stroke();
            }
            EngineCommand::CancelStroke => self.cancel_stroke(),
            EngineCommand::Undo => self.undo(),
            EngineCommand::Redo => self.redo(),
            EngineCommand::FillEnclosedCavities { job_id, cancel } => {
                self.fill_enclosed_cavities(job_id, cancel);
            }
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

    fn fill_enclosed_cavities(&mut self, job_id: JobId, cancel: CancellationToken) {
        self.emit(EngineEvent::Busy {
            job_id,
            kind: BusyKind::FillingCavities,
            active: true,
        });
        self.cancel_stroke();
        self.mesh_cancel.cancel();
        let result = self
            .document
            .as_mut()
            .ok_or(AppError::NoDocument)
            .and_then(|document| document.fill_enclosed_cavities(&cancel));
        match result {
            Ok(added_voxels) => {
                let Some(document) = self.document.as_ref() else {
                    self.finish_job(job_id, BusyKind::FillingCavities);
                    return;
                };
                let summary = document.summary();
                let revision = document.revision();
                let invalidated = invalidated_chunks_for_changes(
                    document.geometry().dimensions,
                    document.latest_undo_changes(),
                );
                let slices = match visible_slices(document) {
                    Ok(slices) => slices,
                    Err(error) => {
                        self.emit_error(Some(job_id), error);
                        if added_voxels == 0 {
                            self.resume_mesh_after_cancel();
                        } else {
                            self.restart_mesh_after_change(invalidated, revision);
                        }
                        self.finish_job(job_id, BusyKind::FillingCavities);
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
                if added_voxels == 0 {
                    self.resume_mesh_after_cancel();
                } else {
                    self.restart_mesh_after_change(invalidated, revision);
                }
                self.emit(EngineEvent::CavitiesFilled {
                    epoch: self.epoch,
                    revision,
                    added_voxels,
                });
            }
            Err(AppError::Cancelled) => self.resume_mesh_after_cancel(),
            Err(error) => {
                self.resume_mesh_after_cancel();
                self.emit_error(Some(job_id), error);
            }
        }
        self.finish_job(job_id, BusyKind::FillingCavities);
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
        self.mesh_dirty.configure([0; 3]);
        self.mesh_scan_grid = [0; 3];
        self.mesh_scan_next = 0;
        self.mesh_scan_total = 0;
        self.mesh_generations.clear();
        self.mesh_budget.clear();
        self.pending_mesh_chunk = None;
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

    fn begin_stroke(&mut self, tool: Tool, spec: BrushSpec, space: StrokeSpace) {
        self.cancel_stroke();
        if tool == Tool::Bridge {
            self.emit_error(
                None,
                AppError::InvalidDimensions(
                    "bridge edits must be committed after both endpoints are chosen".into(),
                ),
            );
            return;
        }
        let Some(document) = self.document.as_mut() else {
            self.emit_error(None, AppError::NoDocument);
            return;
        };
        if let Some(value) = edit_value(tool) {
            if let Err(error) = document.begin_stroke(value) {
                self.emit_error(None, error);
                return;
            }
            self.mesh_cancel.cancel();
        }
        self.stroke = Some(StrokeContext {
            tool,
            spec,
            space,
            last_applied: None,
        });
    }

    fn apply_pointer_sample(&mut self, sample: PointerSample) {
        let Some(stroke) = self.stroke.as_ref() else {
            return;
        };
        let target = self
            .document
            .as_ref()
            .and_then(|document| edit_target(document, stroke.tool, stroke.space, sample));
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
        let spec = stroke.spec;
        let start = stroke.last_applied.unwrap_or(target);
        let apply_result = walk_supercover(start, target, |center| {
            let footprint = footprint_indices(document, center, spec, space)?;
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
        let revision = document.revision();
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
        // The edited planes are the latency-critical feedback. Publish their
        // small texture patches before extracting any newly selected orthogonal
        // slice, which can require a cold strided read through a large volume.
        let _ = document;
        for patch in patches {
            self.emit(EngineEvent::SlicePatch {
                epoch: self.epoch,
                revision,
                preview_generation: self.preview_generation,
                patch,
            });
        }
        let replacements = replacement_axes
            .iter()
            .map(|&axis| {
                self.document
                    .as_ref()
                    .expect("the document remains loaded during a stroke")
                    .try_extract_slice(axis, target[axis.normal_axis()])
            })
            .collect::<Result<Vec<_>>>();
        let replacements = match replacements {
            Ok(replacements) => replacements,
            Err(error) => {
                let restoration = self.document.as_mut().and_then(|document| {
                    document.cancel_stroke();
                    document.set_cursor(previous_cursor);
                    visible_slices(document)
                        .ok()
                        .map(|slices| (document.summary(), document.revision(), slices))
                });
                self.stroke = None;
                if let Some((summary, revision, slices)) = restoration {
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
                self.emit_error(None, error);
                self.resume_mesh_after_cancel();
                return;
            }
        };
        let summary = self
            .document
            .as_ref()
            .expect("the document remains loaded during a stroke")
            .summary();
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
    }

    fn compute_brush_preview(&self, pending: CancellableBrushPreview) {
        let generation = preview_request_generation(pending.request);
        let Some(document) = self.document.as_ref() else {
            return;
        };
        let revision = document.revision();
        match calculate_brush_preview(
            document,
            self.stroke.as_ref(),
            pending.request,
            &pending.cancel,
        ) {
            Ok(preview) => {
                if pending.cancel.check().is_ok() {
                    self.emit(EngineEvent::BrushPreview {
                        epoch: self.epoch,
                        revision,
                        generation: preview.generation,
                        tool: preview.tool,
                        target: preview.target,
                        voxels: preview.voxels,
                    });
                }
            }
            Err(AppError::Cancelled) => {}
            Err(error) => self.emit(EngineEvent::BrushPreviewFailed {
                epoch: self.epoch,
                revision,
                generation,
                message: error.to_string(),
            }),
        }
    }

    fn commit_bridge(
        &mut self,
        anchor: [u32; 3],
        plane_normal: [f64; 3],
        sample: PointerSample,
        spec: BrushSpec,
        space: StrokeSpace,
    ) {
        self.cancel_stroke();
        let (end, candidates) =
            match self
                .document
                .as_ref()
                .ok_or(AppError::NoDocument)
                .and_then(|document| {
                    let end = bridge_endpoint(document, space, anchor, plane_normal, sample)
                        .ok_or_else(|| {
                            AppError::InvalidDimensions(
                                "bridge endpoint does not intersect the volume".into(),
                            )
                        })?;
                    let candidates = bridge_footprint_indices(document, anchor, end, spec, space)?;
                    Ok((end, candidates))
                }) {
                Ok(output) => output,
                Err(error) => {
                    self.emit_error(None, error);
                    return;
                }
            };
        let Some(document) = self.document.as_mut() else {
            return;
        };
        let previous_cursor = document.cursor();
        let replacement_axes = axes_with_changed_coordinate(previous_cursor, end);
        let replaced_axes = replacement_axes.iter().copied().collect::<HashSet<_>>();
        let mut patch_tiles = HashSet::new();
        if let Err(error) = document.begin_stroke(1).and_then(|()| {
            document.apply_stroke_indices(candidates.iter().copied())?;
            add_slice_patch_tiles(
                document,
                candidates.iter().copied(),
                &replaced_axes,
                &mut patch_tiles,
            )
        }) {
            document.cancel_stroke();
            self.emit_error(None, error);
            return;
        }
        self.mesh_cancel.cancel();
        document.set_cursor(end);
        let replacements = replacement_axes
            .iter()
            .map(|&axis| document.try_extract_slice(axis, end[axis.normal_axis()]))
            .collect::<Result<Vec<_>>>();
        let patches = replacements.and_then(|replacements| {
            slice_patches_from_tiles(document, patch_tiles).map(|patches| (replacements, patches))
        });
        let (replacements, patches) = match patches {
            Ok(output) => output,
            Err(error) => {
                document.cancel_stroke();
                document.set_cursor(previous_cursor);
                self.emit_error(None, error);
                self.resume_mesh_after_cancel();
                return;
            }
        };
        let Some(revision) = document.finish_stroke() else {
            let summary = document.summary();
            let revision = document.revision();
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
            self.resume_mesh_after_cancel();
            return;
        };
        self.preview_generation = self.preview_generation.saturating_add(1);
        let invalidated = invalidated_chunks_for_changes(
            document.geometry().dimensions,
            document.latest_undo_changes(),
        );
        let summary = document.summary();
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
        self.restart_mesh_after_change(invalidated, revision);
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
        let invalidated = invalidated_chunks_for_changes(
            document.geometry().dimensions,
            document.latest_undo_changes(),
        );
        let summary = document.summary();
        let revision = document.revision();
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        self.restart_mesh_after_change(invalidated, revision);
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
        let invalidated = invalidated_chunks_for_changes(
            document.geometry().dimensions,
            document.latest_redo_changes(),
        );
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
        self.restart_mesh_after_change(invalidated, revision);
    }

    fn redo(&mut self) {
        self.cancel_stroke();
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if document.redo().is_none() {
            return;
        }
        let invalidated = invalidated_chunks_for_changes(
            document.geometry().dimensions,
            document.latest_undo_changes(),
        );
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
        self.restart_mesh_after_change(invalidated, revision);
    }

    fn reset_mesh_preview(&mut self) {
        self.mesh_cancel.cancel();
        self.mesh_dirty.clear();
        self.mesh_budget.clear();
        self.pending_mesh_chunk = None;
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
        self.mesh_dirty.configure(self.mesh_scan_grid);
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
        if self.document.is_none() {
            return;
        }
        if let Some((tag, _)) = self.pending_mesh_chunk.take() {
            let queued = self.mesh_dirty.enqueue_back(tag.chunk);
            debug_assert!(queued);
        }
        for chunk in self
            .mesh_in_flight_chunks
            .iter()
            .copied()
            .chain(invalidated.iter().copied())
        {
            let queued = self.mesh_dirty.enqueue_back(chunk);
            debug_assert!(queued);
        }
        for &chunk in &invalidated {
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
            let queued = self.mesh_dirty.enqueue_back(chunk);
            debug_assert!(queued);
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
        self.mesh_dirty.promote(center_chunk);
        self.mesh_dirty.promote(cursor_chunk);
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
                        self.mesh_dirty.promote(ChunkCoord::new(
                            candidate[0] as u32,
                            candidate[1] as u32,
                            candidate[2] as u32,
                        ));
                    }
                }
            }
        }
    }

    fn dispatch_mesh_jobs(&mut self) {
        if self.pending_mesh_chunk.is_some()
            || self
                .stroke
                .as_ref()
                .is_some_and(|stroke| stroke.tool != Tool::Inspect)
            || self.preview_too_complex
        {
            return;
        }
        while self.mesh_in_flight < self.mesh_worker_limit {
            if !self.command_rx.is_empty()
                || !self.sample_rx.is_empty()
                || !self.preview_rx.is_empty()
            {
                break;
            }
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
        let pending = self.mesh_dirty.len();
        for _ in 0..pending {
            let chunk = self.mesh_dirty.pop_front()?;
            if self.mesh_in_flight_chunks.contains(&chunk) {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
            } else {
                return Some(chunk);
            }
        }
        while self.mesh_scan_next < self.mesh_scan_total {
            let chunk = chunk_from_linear(self.mesh_scan_next, self.mesh_scan_grid)?;
            self.mesh_scan_next += 1;
            if self.mesh_in_flight_chunks.contains(&chunk) {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
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
            Err(AppError::Cancelled) => {
                // An editable stroke cancels the current mesh pass but leaves
                // the document revision unchanged until the stroke commits.
                // A cancelled result can therefore arrive and be removed from
                // `mesh_in_flight_chunks` before completion has a chance to
                // requeue it. Keep that non-stale chunk dirty immediately so
                // the replacement pass cannot lose work already passed by the
                // lazy scan cursor.
                let queued = self.mesh_dirty.enqueue_back(tag.chunk);
                debug_assert!(queued);
                return;
            }
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
            self.mesh_dirty.clear();
            self.mesh_scan_next = self.mesh_scan_total;
            self.emit(EngineEvent::PreviewTooComplex {
                epoch: self.epoch,
                revision: document.revision(),
            });
            return;
        }
        self.queue_mesh_chunk(tagged.tag, tagged.mesh.quads);
    }

    fn queue_mesh_chunk(&mut self, tag: MeshJobTag, quads: Vec<CompactQuad>) {
        debug_assert!(self.pending_mesh_chunk.is_none());
        if self.event_tx.len() >= MESH_EVENT_BACKLOG_LIMIT {
            self.pending_mesh_chunk = Some((tag, quads));
            return;
        }
        match self
            .event_tx
            .try_send(EngineEvent::MeshChunk { tag, quads })
        {
            Ok(()) => self.repaint.request_repaint(),
            Err(TrySendError::Full(EngineEvent::MeshChunk { tag, quads })) => {
                self.pending_mesh_chunk = Some((tag, quads));
            }
            Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => unreachable!("only mesh chunks are queued here"),
        }
    }

    fn flush_pending_mesh_chunk(&mut self) {
        let Some((tag, quads)) = self.pending_mesh_chunk.take() else {
            return;
        };
        self.queue_mesh_chunk(tag, quads);
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

const fn edit_value(tool: Tool) -> Option<u8> {
    match tool {
        Tool::Inspect => None,
        Tool::Paint | Tool::Bridge => Some(1),
        Tool::Erase => Some(0),
    }
}

fn edit_target(
    document: &VolumeDocument,
    tool: Tool,
    space: StrokeSpace,
    sample: PointerSample,
) -> Option<[u32; 3]> {
    match (space, sample) {
        (StrokeSpace::Slice(expected), PointerSample::Slice { axis, voxel })
            if expected == axis
                && linear_index(document.geometry().dimensions, voxel).is_some() =>
        {
            Some(voxel)
        }
        (StrokeSpace::Volume3D, PointerSample::Ray { ray }) => {
            let view = document.view();
            surface_edit_target(view.dimensions, view.occupancy, view.geometry, ray, tool)
        }
        _ => None,
    }
}

fn bridge_endpoint(
    document: &VolumeDocument,
    space: StrokeSpace,
    anchor: [u32; 3],
    plane_normal: [f64; 3],
    sample: PointerSample,
) -> Option<[u32; 3]> {
    match (space, sample) {
        (StrokeSpace::Slice(expected), PointerSample::Slice { axis, voxel })
            if expected == axis
                && linear_index(document.geometry().dimensions, voxel).is_some() =>
        {
            Some(voxel)
        }
        (StrokeSpace::Volume3D, PointerSample::Ray { ray }) => {
            edit_target(document, Tool::Bridge, space, sample).or_else(|| {
                ray_plane_voxel(
                    document.geometry().dimensions,
                    document.geometry(),
                    ray,
                    anchor,
                    plane_normal,
                )
            })
        }
        _ => None,
    }
}

#[derive(Debug, Eq, PartialEq)]
struct BrushPreviewComputation {
    generation: u64,
    tool: Tool,
    target: Option<[u32; 3]>,
    voxels: Vec<[u32; 3]>,
}

const fn preview_request_generation(request: BrushPreviewRequest) -> u64 {
    match request {
        BrushPreviewRequest::Brush { generation, .. }
        | BrushPreviewRequest::Bridge { generation, .. } => generation,
    }
}

fn calculate_brush_preview(
    document: &VolumeDocument,
    stroke: Option<&StrokeContext>,
    request: BrushPreviewRequest,
    cancel: &CancellationToken,
) -> Result<BrushPreviewComputation> {
    cancel.check()?;
    let generation = preview_request_generation(request);
    let (tool, target, candidates) = match request {
        BrushPreviewRequest::Brush {
            tool,
            spec,
            space,
            sample,
            ..
        } => {
            let target = edit_target(document, tool, space, sample);
            let candidates = match target {
                None => Vec::new(),
                Some(end)
                    if matches!(tool, Tool::Paint | Tool::Erase)
                        && stroke.is_some_and(|stroke| {
                            stroke.tool == tool && stroke.spec == spec && stroke.space == space
                        }) =>
                {
                    let start = stroke.and_then(|stroke| stroke.last_applied).unwrap_or(end);
                    preview_path_footprint_indices(document, start, end, spec, space, cancel)?
                }
                Some(center) => preview_footprint_indices(document, center, spec, space, cancel)?,
            };
            (tool, target, candidates)
        }
        BrushPreviewRequest::Bridge {
            spec,
            space,
            anchor,
            plane_normal,
            sample,
            ..
        } => {
            let target = bridge_endpoint(document, space, anchor, plane_normal, sample);
            let candidates = match target {
                Some(end) => {
                    preview_path_footprint_indices(document, anchor, end, spec, space, cancel)?
                }
                None => Vec::new(),
            };
            (Tool::Bridge, target, candidates)
        }
    };
    let voxels = changed_voxels_cancellable(
        document,
        candidates,
        edit_value(tool),
        cancel,
        Some(MAX_PREVIEW_VOXELS),
    )?;
    cancel.check()?;
    Ok(BrushPreviewComputation {
        generation,
        tool,
        target,
        voxels,
    })
}

fn footprint_indices(
    document: &VolumeDocument,
    center: [u32; 3],
    spec: BrushSpec,
    space: StrokeSpace,
) -> Result<Vec<u32>> {
    let (normal_axis, spec) = match space {
        StrokeSpace::Slice(axis) => (axis.normal_axis(), spec),
        StrokeSpace::Volume3D => (0, spec.with_depth(BrushDepth::Volume)),
    };
    brush_indices(document.geometry(), center, normal_axis, spec)
}

fn preview_footprint_indices(
    document: &VolumeDocument,
    center: [u32; 3],
    spec: BrushSpec,
    space: StrokeSpace,
    cancel: &CancellationToken,
) -> Result<Vec<u32>> {
    let (normal_axis, spec) = match space {
        StrokeSpace::Slice(axis) => (axis.normal_axis(), spec),
        StrokeSpace::Volume3D => (0, spec.with_depth(BrushDepth::Volume)),
    };
    brush_indices_cancellable(
        document.geometry(),
        center,
        normal_axis,
        spec,
        cancel,
        Some(MAX_PREVIEW_VOXELS),
    )
}

fn preview_path_footprint_indices(
    document: &VolumeDocument,
    start: [u32; 3],
    end: [u32; 3],
    spec: BrushSpec,
    space: StrokeSpace,
    cancel: &CancellationToken,
) -> Result<Vec<u32>> {
    if linear_index(document.geometry().dimensions, start).is_none()
        || linear_index(document.geometry().dimensions, end).is_none()
    {
        return Err(AppError::InvalidDimensions(
            "brush path endpoint lies outside the volume".into(),
        ));
    }
    let mut unique = HashSet::new();
    let mut indices = Vec::new();
    let mut candidate_visits = 0_usize;
    walk_supercover(start, end, |center| {
        cancel.check()?;
        for index in preview_footprint_indices(document, center, spec, space, cancel)? {
            if candidate_visits >= MAX_PREVIEW_VOXELS {
                return Err(preview_voxel_limit_error());
            }
            candidate_visits += 1;
            if unique.contains(&index) {
                continue;
            }
            if unique.len() >= MAX_PREVIEW_VOXELS {
                return Err(preview_voxel_limit_error());
            }
            if unique.len() == unique.capacity() {
                unique
                    .try_reserve(1_024.min(MAX_PREVIEW_VOXELS - unique.len()))
                    .map_err(|_| AppError::Allocation {
                        bytes: unique
                            .len()
                            .saturating_add(1_024)
                            .saturating_mul(size_of::<u32>() + size_of::<usize>()),
                    })?;
            }
            if indices.len() == indices.capacity() {
                indices
                    .try_reserve(1_024.min(MAX_PREVIEW_VOXELS - indices.len()))
                    .map_err(|_| AppError::Allocation {
                        bytes: indices
                            .len()
                            .saturating_add(1_024)
                            .saturating_mul(size_of::<u32>()),
                    })?;
            }
            unique.insert(index);
            indices.push(index);
        }
        Ok(())
    })?;
    cancel.check()?;
    Ok(indices)
}

fn preview_voxel_limit_error() -> AppError {
    AppError::InvalidDimensions(format!(
        "brush preview exceeds the {MAX_PREVIEW_VOXELS}-voxel safety limit; reduce the brush size or bridge length"
    ))
}

fn bridge_footprint_indices(
    document: &VolumeDocument,
    start: [u32; 3],
    end: [u32; 3],
    spec: BrushSpec,
    space: StrokeSpace,
) -> Result<Vec<u32>> {
    if linear_index(document.geometry().dimensions, start).is_none()
        || linear_index(document.geometry().dimensions, end).is_none()
    {
        return Err(AppError::InvalidDimensions(
            "bridge endpoint lies outside the volume".into(),
        ));
    }
    let mut unique = HashSet::new();
    walk_supercover(start, end, |center| {
        for index in footprint_indices(document, center, spec, space)? {
            if unique.contains(&index) {
                continue;
            }
            if unique.len() == unique.capacity() {
                unique
                    .try_reserve(1_024)
                    .map_err(|_| AppError::Allocation {
                        bytes: unique
                            .len()
                            .saturating_add(1_024)
                            .saturating_mul(size_of::<u32>() + size_of::<usize>()),
                    })?;
            }
            unique.insert(index);
        }
        Ok(())
    })?;
    let mut indices = unique.into_iter().collect::<Vec<_>>();
    indices.sort_unstable();
    Ok(indices)
}

fn changed_voxels_cancellable(
    document: &VolumeDocument,
    indices: Vec<u32>,
    value: Option<u8>,
    cancel: &CancellationToken,
    maximum_voxels: Option<usize>,
) -> Result<Vec<[u32; 3]>> {
    cancel.check()?;
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let dimensions = document.geometry().dimensions;
    let occupancy = document.view().occupancy;
    let mut voxels = Vec::new();
    let initial_capacity = indices.len().min(maximum_voxels.unwrap_or(usize::MAX));
    voxels
        .try_reserve(initial_capacity)
        .map_err(|_| AppError::Allocation {
            bytes: initial_capacity.saturating_mul(size_of::<[u32; 3]>()),
        })?;
    for (offset, index) in indices.into_iter().enumerate() {
        if offset & 0x0fff == 0 {
            cancel.check()?;
        }
        if occupancy.get(index as usize).copied() == Some(value) {
            continue;
        }
        if maximum_voxels.is_some_and(|maximum| voxels.len() >= maximum) {
            return Err(preview_voxel_limit_error());
        }
        let voxel = voxel_from_linear(dimensions, index).ok_or_else(|| {
            AppError::InvalidDimensions("brush produced an out-of-volume voxel".into())
        })?;
        voxels.push(voxel);
    }
    cancel.check()?;
    Ok(voxels)
}

fn voxel_from_linear(dimensions: [u32; 3], index: u32) -> Option<[u32; 3]> {
    let plane = u64::from(dimensions[0]).checked_mul(u64::from(dimensions[1]))?;
    if plane == 0 {
        return None;
    }
    let index = u64::from(index);
    let z = index / plane;
    let remainder = index % plane;
    let y = remainder / u64::from(dimensions[0]);
    let x = remainder % u64::from(dimensions[0]);
    let voxel = [
        u32::try_from(x).ok()?,
        u32::try_from(y).ok()?,
        u32::try_from(z).ok()?,
    ];
    linear_index(dimensions, voxel).is_some().then_some(voxel)
}

fn invalidated_chunks_for_changes(
    dimensions: [u32; 3],
    changes: Option<ChangedVoxels<'_>>,
) -> Vec<ChunkCoord> {
    match changes {
        Some(ChangedVoxels::Indices(indices)) => {
            invalidated_chunks_for_indices(dimensions, indices.iter().copied())
        }
        Some(ChangedVoxels::Spans(spans)) => invalidated_chunks_for_index_spans(
            dimensions,
            spans.iter().map(|span| (span.start, span.len)),
        ),
        None => Vec::new(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::brush::{BrushShape, BrushSize};
    use crate::volume::{SpatialUnits, VolumeGeometry};

    fn one_voxel_brush() -> BrushSpec {
        BrushSpec {
            shape: BrushShape::Round,
            depth: BrushDepth::Slice,
            size: BrushSize::voxels(1).unwrap(),
        }
    }

    fn detached_engine_handle() -> EngineHandle {
        let (commands, _command_rx) = bounded::<EngineCommand>(1);
        let (_event_tx, events) = bounded::<EngineEvent>(1);
        let (latest_samples, sample_drain) = bounded::<PointerSample>(1);
        let (latest_previews, preview_drain) = bounded::<CancellableBrushPreview>(1);
        EngineHandle {
            commands,
            events,
            latest_samples,
            sample_drain,
            latest_previews,
            preview_drain,
            active_preview: Mutex::new(None),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            next_job: AtomicU64::new(1),
        }
    }

    fn state_with_empty_document(dimensions: [u32; 3]) -> EngineState {
        let geometry = VolumeGeometry::new(
            dimensions,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Millimeter,
        )
        .unwrap();
        state_with_empty_geometry(geometry)
    }

    fn state_with_empty_geometry(geometry: VolumeGeometry) -> EngineState {
        use std::{path::PathBuf, time::SystemTime};

        use crate::{
            codecs::{BinvoxMetadata, CodecSaveState, SourceVariant},
            volume::SourceIdentity,
        };

        let dimensions = geometry.dimensions;
        let (_command_tx, command_rx) = bounded(1);
        let (_sample_tx, sample_rx) = bounded(1);
        let (_preview_tx, preview_rx) = bounded(1);
        let (event_tx, _event_rx) = bounded(16);
        let (mesh_tx, mesh_rx) = bounded(2);
        let mut state = EngineState::new(
            command_rx,
            sample_rx,
            preview_rx,
            event_tx,
            mesh_tx,
            mesh_rx,
            Arc::new(Mutex::new(HashMap::new())),
            egui::Context::default(),
            NonZeroUsize::MIN,
        );
        let identity = SourceIdentity {
            canonical_path: PathBuf::from("/tmp/mesh-cancellation-test.binvox"),
            size: 0,
            modified: SystemTime::UNIX_EPOCH,
            sha256: [0; 32],
        };
        state.document = Some(
            VolumeDocument::new(
                geometry,
                vec![0; dimensions.into_iter().map(|axis| axis as usize).product()],
                identity,
                SourceVariant::Binvox { version: 1 },
                CodecSaveState::Binvox(BinvoxMetadata {
                    version: 1,
                    dimensions,
                    translation: [0.0; 3],
                    scale: 1.0,
                }),
                false,
            )
            .unwrap(),
        );
        let grid = chunk_grid(dimensions);
        state.mesh_dirty.configure(grid);
        state.mesh_scan_grid = grid;
        state.mesh_scan_total = grid.into_iter().map(u64::from).product();
        state.mesh_scan_next = state.mesh_scan_total;
        state
    }

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
    fn latest_preview_channel_discards_intermediate_hover_requests() {
        let engine = EngineHandle::spawn(egui::Context::default());
        for generation in 0..10_000 {
            engine
                .send_latest_preview(BrushPreviewRequest::Brush {
                    generation,
                    tool: Tool::Paint,
                    spec: BrushSpec::default(),
                    space: StrokeSpace::Slice(SliceAxis::Axial),
                    sample: PointerSample::Slice {
                        axis: SliceAxis::Axial,
                        voxel: [generation as u32, 0, 0],
                    },
                })
                .unwrap();
        }
    }

    #[test]
    fn superseding_a_preview_cancels_work_already_taken_by_the_receiver() {
        use std::{sync::Barrier, thread};

        let engine = detached_engine_handle();
        let request = BrushPreviewRequest::Brush {
            generation: 1,
            tool: Tool::Paint,
            spec: one_voxel_brush(),
            space: StrokeSpace::Slice(SliceAxis::Axial),
            sample: PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [0, 0, 0],
            },
        };
        engine.send_latest_preview(request).unwrap();
        let in_progress = engine.preview_drain.try_recv().unwrap();
        let started = Arc::new(Barrier::new(2));
        let worker_started = Arc::clone(&started);
        let worker = thread::spawn(move || {
            worker_started.wait();
            while !in_progress.cancel.is_cancelled() {
                std::hint::spin_loop();
            }
            in_progress.cancel.check()
        });
        started.wait();

        engine
            .send_latest_preview(BrushPreviewRequest::Brush {
                generation: 2,
                tool: Tool::Paint,
                spec: one_voxel_brush(),
                space: StrokeSpace::Slice(SliceAxis::Axial),
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [0, 0, 0],
                },
            })
            .unwrap();
        assert!(matches!(worker.join().unwrap(), Err(AppError::Cancelled)));
    }

    #[test]
    fn active_stroke_preview_matches_the_interpolated_supercover_edit() {
        let mut state = state_with_empty_document([6, 1, 1]);
        let spec = one_voxel_brush();
        let space = StrokeSpace::Slice(SliceAxis::Axial);
        state.begin_stroke(Tool::Paint, spec, space);
        state.apply_target([1, 0, 0]);

        let preview = calculate_brush_preview(
            state.document.as_ref().unwrap(),
            state.stroke.as_ref(),
            BrushPreviewRequest::Brush {
                generation: 7,
                tool: Tool::Paint,
                spec,
                space,
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [4, 0, 0],
                },
            },
            &CancellationToken::default(),
        )
        .unwrap();

        assert_eq!(preview.target, Some([4, 0, 0]));
        assert_eq!(preview.voxels, vec![[2, 0, 0], [3, 0, 0], [4, 0, 0]]);
    }

    #[test]
    fn oversized_preview_emits_failure_instead_of_an_empty_success() {
        let dimensions = [201; 3];
        let geometry = VolumeGeometry::new(
            dimensions,
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Millimeter,
        )
        .unwrap();
        let mut state = state_with_empty_geometry(geometry);
        let (event_tx, event_rx) = bounded(2);
        state.event_tx = event_tx;
        state.compute_brush_preview(CancellableBrushPreview {
            request: BrushPreviewRequest::Brush {
                generation: 99,
                tool: Tool::Paint,
                spec: BrushSpec {
                    shape: BrushShape::Square,
                    depth: BrushDepth::Volume,
                    size: BrushSize::millimeters(50.0).unwrap(),
                },
                space: StrokeSpace::Slice(SliceAxis::Axial),
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [100; 3],
                },
            },
            cancel: CancellationToken::default(),
        });

        let event = event_rx.try_recv().unwrap();
        assert!(matches!(
            event,
            EngineEvent::BrushPreviewFailed {
                generation: 99,
                message,
                ..
            } if message.contains("2000000")
        ));
        assert!(event_rx.try_recv().is_err());
    }

    #[test]
    fn cancelled_mesh_results_consumed_during_a_stroke_remain_queued() {
        let mut state = state_with_empty_document([64, 1, 1]);
        let chunks = [ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0)];
        state.mesh_in_flight = chunks.len();
        state.mesh_in_flight_chunks.extend(chunks);
        state.begin_stroke(
            Tool::Paint,
            BrushSpec::default(),
            StrokeSpace::Slice(SliceAxis::Axial),
        );
        assert!(state.mesh_cancel.is_cancelled());

        for chunk in chunks {
            let tag = MeshJobTag {
                document_epoch: state.epoch,
                revision: 1,
                chunk_generation: state.mesh_pass_generation,
                chunk,
            };
            state.handle_mesh_result((tag, Err(AppError::Cancelled)));
        }
        assert!(state.mesh_in_flight_chunks.is_empty());

        // The cancelled results arrived before stroke completion, so the
        // completion path can no longer recover them from the in-flight set.
        state.complete_stroke();
        assert_eq!(state.mesh_dirty.len(), chunks.len());
        assert!(
            chunks
                .into_iter()
                .all(|chunk| state.mesh_dirty.members.contains(&chunk))
        );
    }

    #[test]
    fn mesh_budget_failure_stays_latched_until_an_explicit_rebuild() {
        let mut state = state_with_empty_document([64, 1, 1]);
        state.preview_too_complex = true;
        state.mesh_dirty.clear();
        state.mesh_scan_next = state.mesh_scan_total;

        state.restart_mesh_after_change(vec![ChunkCoord::new(0, 0, 0)], 2);
        assert!(state.preview_too_complex);
        assert!(state.mesh_dirty.members.contains(&ChunkCoord::new(0, 0, 0)));
        state.dispatch_mesh_jobs();
        assert_eq!(state.mesh_in_flight, 0);

        state.reset_mesh_preview();
        assert!(!state.preview_too_complex);
        assert_eq!(state.mesh_scan_next, 0);
        assert!(state.mesh_scan_total > 0);
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

        let mut dirty = DirtyChunkQueue::default();
        dirty.configure([4, 1, 1]);
        for x in 0..4 {
            assert!(dirty.enqueue_back(ChunkCoord::new(x, 0, 0)));
        }
        assert!(dirty.enqueue_back(ChunkCoord::new(2, 0, 0)));
        assert!(!dirty.enqueue_back(ChunkCoord::new(4, 0, 0)));
        assert_eq!(dirty.len(), 4);

        dirty.promote(ChunkCoord::new(2, 0, 0));
        assert_eq!(dirty.pop_front(), Some(ChunkCoord::new(2, 0, 0)));
        assert_eq!(dirty.len(), 3);
    }
}
