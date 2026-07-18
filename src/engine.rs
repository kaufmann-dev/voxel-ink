use std::{
    collections::{HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, select_biased};

use crate::{
    AppError, Result,
    brush::{
        BrushDepth, BrushSpec, Ray, brush_indices_cancellable, ray_plane_voxel,
        surface_edit_target_with_occupancy, walk_supercover,
    },
    cancel::CancellationToken,
    io::{SaveMode, load_path, save_document},
    mesh::{
        ChunkCoord, ChunkInput, CompactQuad, MeshBudgetError, MeshJobTag, MeshMemoryBudget,
        MeshUpdatePriority, TaggedChunkMesh, bounded_mesh_queue, chunk_grid,
        invalidated_chunks_for_index_spans, invalidated_chunks_for_indices, mesh_chunk,
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
const LIVE_FEEDBACK_GRACE: Duration = Duration::from_millis(10);

pub type JobId = u64;
pub type EditId = u64;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StrokeDisposition {
    Complete { final_sample: Option<PointerSample> },
    Cancel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditOutcome {
    Completed,
    Cancelled,
    Rejected,
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
        edit_id: EditId,
        tool: Tool,
        spec: BrushSpec,
        space: StrokeSpace,
        initial_sample: PointerSample,
        cancel: CancellationToken,
    },
    CommitBridge {
        edit_id: EditId,
        anchor: [u32; 3],
        plane_normal: [f64; 3],
        sample: PointerSample,
        spec: BrushSpec,
        space: StrokeSpace,
    },
    EndStroke {
        edit_id: EditId,
        disposition: StrokeDisposition,
    },
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
    EditFinished {
        epoch: u64,
        edit_id: EditId,
        outcome: EditOutcome,
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
        priority: MeshUpdatePriority,
    },
    PreviewTooComplex {
        epoch: u64,
        revision: u64,
    },
}

pub struct EngineHandle {
    commands: Sender<EngineCommand>,
    events: Receiver<EngineEvent>,
    latest_samples: Sender<StrokeSample>,
    sample_drain: Receiver<StrokeSample>,
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct StrokeSample {
    edit_id: EditId,
    sample: PointerSample,
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

    pub fn send_latest_sample(&self, edit_id: EditId, sample: PointerSample) -> Result<()> {
        let sample = StrokeSample { edit_id, sample };
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
    edit_id: EditId,
    tool: Tool,
    spec: BrushSpec,
    space: StrokeSpace,
    cancel: CancellationToken,
    last_sample: Option<PointerSample>,
    last_applied: Option<[u32; 3]>,
    affected_chunks: HashSet<ChunkCoord>,
}

type MeshResult = (MeshJobTag, MeshUpdatePriority, Result<TaggedChunkMesh>);

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

    fn remove(&mut self, chunk: ChunkCoord) -> bool {
        if !self.members.remove(&chunk) {
            return false;
        }
        let position = self
            .order
            .iter()
            .position(|candidate| *candidate == chunk)
            .expect("dirty chunk membership and order diverged");
        self.order.remove(position);
        true
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
    sample_rx: Receiver<StrokeSample>,
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
    mesh_in_flight_priorities: HashMap<ChunkCoord, MeshUpdatePriority>,
    mesh_dirty: DirtyChunkQueue,
    mesh_live_chunks: HashSet<ChunkCoord>,
    mesh_recovery_chunks: HashSet<ChunkCoord>,
    mesh_recovery_edit: Option<EditId>,
    deferred_recovery_chunks: HashSet<ChunkCoord>,
    progressive_live_jobs: HashSet<MeshJobTag>,
    last_live_feedback_generation: u64,
    last_live_feedback_at: Option<Instant>,
    mesh_scan_grid: [u32; 3],
    mesh_scan_next: u64,
    mesh_scan_total: u64,
    mesh_pass_generation: u64,
    mesh_generations: HashMap<ChunkCoord, u64>,
    mesh_edit_generations: HashMap<ChunkCoord, u64>,
    mesh_budget: MeshMemoryBudget,
    mesh_cancel: CancellationToken,
    mesh_background_cancel: CancellationToken,
    preview_too_complex: bool,
    pending_live_mesh_chunks: VecDeque<(MeshJobTag, Vec<CompactQuad>)>,
    pending_background_mesh_chunks: VecDeque<(MeshJobTag, Vec<CompactQuad>)>,
}

impl EngineState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        command_rx: Receiver<EngineCommand>,
        sample_rx: Receiver<StrokeSample>,
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
            mesh_worker_limit: worker_count.get(),
            mesh_in_flight: 0,
            mesh_in_flight_chunks: HashSet::new(),
            mesh_in_flight_priorities: HashMap::new(),
            mesh_dirty: DirtyChunkQueue::default(),
            mesh_live_chunks: HashSet::new(),
            mesh_recovery_chunks: HashSet::new(),
            mesh_recovery_edit: None,
            deferred_recovery_chunks: HashSet::new(),
            progressive_live_jobs: HashSet::new(),
            last_live_feedback_generation: 0,
            last_live_feedback_at: None,
            mesh_scan_grid: [0; 3],
            mesh_scan_next: 0,
            mesh_scan_total: 0,
            mesh_pass_generation: 0,
            mesh_generations: HashMap::new(),
            mesh_edit_generations: HashMap::new(),
            mesh_budget: MeshMemoryBudget::default(),
            mesh_cancel: CancellationToken::default(),
            mesh_background_cancel: CancellationToken::default(),
            preview_too_complex: false,
            pending_live_mesh_chunks: VecDeque::new(),
            pending_background_mesh_chunks: VecDeque::new(),
        }
    }

    fn run(mut self) {
        loop {
            self.flush_pending_mesh_chunk();
            self.dispatch_mesh_jobs();
            if self.live_feedback_pending()
                && self.command_rx.is_empty()
                && let Ok(result) = self.mesh_rx.try_recv()
            {
                self.handle_mesh_result(result);
            }
            let idle_wait = if self.has_pending_mesh_chunks() {
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
                recv(self.mesh_rx) -> result => {
                    if let Ok(result) = result {
                        self.handle_mesh_result(result);
                    }
                }
                default(idle_wait) => {}
            }
        }
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
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
                edit_id,
                tool,
                spec,
                space,
                initial_sample,
                cancel,
            } => self.begin_stroke(edit_id, tool, spec, space, initial_sample, cancel),
            EngineCommand::CommitBridge {
                edit_id,
                anchor,
                plane_normal,
                sample,
                spec,
                space,
            } => self.commit_bridge(edit_id, anchor, plane_normal, sample, spec, space),
            EngineCommand::EndStroke {
                edit_id,
                disposition,
            } => match disposition {
                StrokeDisposition::Complete { final_sample } => {
                    if let Some(sample) = final_sample {
                        self.apply_pointer_sample(StrokeSample { edit_id, sample });
                    }
                    self.complete_stroke(edit_id);
                }
                StrokeDisposition::Cancel => self.cancel_stroke(edit_id),
            },
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
            EngineCommand::RebuildPreview => {
                if self.stroke.is_some() || self.mesh_recovery_edit.is_some() {
                    self.emit_error(
                        None,
                        AppError::InvalidDimensions(
                            "the 3D preview cannot rebuild during an active edit".into(),
                        ),
                    );
                } else {
                    self.reset_mesh_preview();
                }
            }
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
                self.mesh_background_cancel.cancel();
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
        self.cancel_active_stroke();
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
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
        self.cancel_active_stroke();
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
        self.document = None;
        self.epoch = self.epoch.saturating_add(1);
        self.mesh_dirty.configure([0; 3]);
        self.mesh_scan_grid = [0; 3];
        self.mesh_scan_next = 0;
        self.mesh_scan_total = 0;
        self.mesh_generations.clear();
        self.mesh_edit_generations.clear();
        self.mesh_live_chunks.clear();
        self.mesh_recovery_chunks.clear();
        self.mesh_recovery_edit = None;
        self.deferred_recovery_chunks.clear();
        self.progressive_live_jobs.clear();
        self.last_live_feedback_generation = 0;
        self.last_live_feedback_at = None;
        self.mesh_budget.clear();
        self.pending_live_mesh_chunks.clear();
        self.pending_background_mesh_chunks.clear();
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

    fn begin_stroke(
        &mut self,
        edit_id: EditId,
        tool: Tool,
        spec: BrushSpec,
        space: StrokeSpace,
        initial_sample: PointerSample,
        cancel: CancellationToken,
    ) {
        if self.stroke.is_some() || self.mesh_recovery_edit.is_some() {
            self.emit_error(
                None,
                AppError::InvalidDimensions(
                    "another edit stroke is active or still restoring its 3D surface".into(),
                ),
            );
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        }
        if tool == Tool::Bridge {
            self.emit_error(
                None,
                AppError::InvalidDimensions(
                    "bridge edits must be committed after both endpoints are chosen".into(),
                ),
            );
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        }
        let Some(document) = self.document.as_mut() else {
            self.emit_error(None, AppError::NoDocument);
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        };
        if let Some(value) = edit_value(tool)
            && let Err(error) = document.begin_stroke(value)
        {
            self.emit_error(None, error);
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        }
        self.stroke = Some(StrokeContext {
            edit_id,
            tool,
            spec,
            space,
            cancel,
            last_sample: None,
            last_applied: None,
            affected_chunks: HashSet::new(),
        });
        self.apply_pointer_sample(StrokeSample {
            edit_id,
            sample: initial_sample,
        });
    }

    fn apply_pointer_sample(&mut self, pending: StrokeSample) {
        let Some(stroke) = self.stroke.as_mut() else {
            return;
        };
        if stroke.edit_id != pending.edit_id {
            return;
        }
        let sample = pending.sample;
        if stroke.last_sample == Some(sample) {
            return;
        }
        stroke.last_sample = Some(sample);
        let target = self
            .document
            .as_ref()
            .and_then(|document| edit_target(document, stroke.tool, stroke.space, sample));
        let Some(target) = target else {
            return;
        };
        if stroke.last_applied == Some(target) {
            return;
        }
        self.apply_target(target);
    }

    fn apply_target(&mut self, target: [u32; 3]) {
        let Some((tool, spec, space, start, cancel)) = self.stroke.as_ref().map(|stroke| {
            (
                stroke.tool,
                stroke.spec,
                stroke.space,
                stroke.last_applied.unwrap_or(target),
                stroke.cancel.clone(),
            )
        }) else {
            return;
        };
        let Some(document) = self.document.as_mut() else {
            return;
        };
        if linear_index(document.geometry().dimensions, target).is_none() {
            return;
        }
        let previous_cursor = document.cursor();
        if tool == Tool::Inspect {
            document.set_cursor(target);
            self.stroke
                .as_mut()
                .expect("the inspect stroke remains active")
                .last_applied = Some(target);
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
        let stroke_change_start = document.active_stroke_indices().map_or(0, <[u32]>::len);
        let apply_result = walk_supercover(start, target, |center| {
            cancel.check()?;
            let footprint = footprint_indices(document, center, spec, space, &cancel)?;
            for batch in footprint.chunks(4_096) {
                cancel.check()?;
                document.apply_stroke_indices(batch.iter().copied())?;
                add_slice_patch_tiles(
                    document,
                    batch.iter().copied(),
                    &replaced_axes,
                    &mut patch_tiles,
                )?;
            }
            Ok(())
        });
        if let Err(error) = apply_result {
            let _ = document;
            self.rollback_active_stroke(Some(previous_cursor));
            if !matches!(error, AppError::Cancelled) {
                self.emit_error(None, error);
            }
            return;
        }
        document.set_cursor(target);
        self.stroke
            .as_mut()
            .expect("the editable stroke remains active")
            .last_applied = Some(target);
        self.preview_generation = self.preview_generation.saturating_add(1);
        let revision = document.revision();
        let dimensions = document.geometry().dimensions;
        let newly_affected = invalidated_chunks_for_indices(
            dimensions,
            document
                .active_stroke_indices()
                .expect("an editable stroke owns its exact changes")[stroke_change_start..]
                .iter()
                .copied(),
        );
        if newly_affected
            .iter()
            .copied()
            .any(|chunk| self.mesh_budget.reserve_current_if_absent(chunk).is_err())
        {
            let _ = document;
            self.rollback_active_stroke(Some(previous_cursor));
            self.emit_error(
                None,
                AppError::Allocation {
                    bytes: std::mem::size_of::<(ChunkCoord, usize)>(),
                },
            );
            return;
        }
        self.stroke
            .as_mut()
            .expect("the editable stroke remains active")
            .affected_chunks
            .extend(newly_affected.iter().copied());
        let patches = match slice_patches_from_tiles(document, patch_tiles) {
            Ok(patches) => patches,
            Err(error) => {
                let _ = document;
                self.rollback_active_stroke(Some(previous_cursor));
                self.emit_error(None, error);
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
        if !newly_affected.is_empty() {
            self.preempt_mesh_for_live_feedback();
        }
        self.promote_live_mesh_chunks(newly_affected);
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
                self.rollback_active_stroke(Some(previous_cursor));
                self.emit_error(None, error);
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
        match calculate_brush_preview(document, pending.request, &pending.cancel) {
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
        edit_id: EditId,
        anchor: [u32; 3],
        plane_normal: [f64; 3],
        sample: PointerSample,
        spec: BrushSpec,
        space: StrokeSpace,
    ) {
        if self.stroke.is_some() || self.mesh_recovery_edit.is_some() {
            self.emit_error(
                None,
                AppError::InvalidDimensions(
                    "another edit stroke is active or still restoring its 3D surface".into(),
                ),
            );
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        }
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
                    self.emit_edit_finished(edit_id, EditOutcome::Rejected);
                    return;
                }
            };
        let Some(document) = self.document.as_mut() else {
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
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
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        }
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
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
                self.emit_edit_finished(edit_id, EditOutcome::Cancelled);
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
            self.emit_edit_finished(edit_id, EditOutcome::Completed);
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
        self.emit_edit_finished(edit_id, EditOutcome::Completed);
    }

    fn complete_stroke(&mut self, edit_id: EditId) {
        if self
            .stroke
            .as_ref()
            .is_none_or(|stroke| stroke.edit_id != edit_id)
        {
            return;
        }
        let Some(stroke) = self.stroke.take() else {
            return;
        };
        if stroke.tool == Tool::Inspect {
            self.emit_edit_finished(edit_id, EditOutcome::Completed);
            return;
        }
        let Some(document) = self.document.as_mut() else {
            self.emit_edit_finished(edit_id, EditOutcome::Rejected);
            return;
        };
        if document.finish_stroke().is_none() {
            for chunk in stroke.affected_chunks {
                self.mesh_budget.release_reservation(chunk);
            }
            self.emit_edit_finished(edit_id, EditOutcome::Completed);
            return;
        }
        let invalidated = stroke.affected_chunks.into_iter().collect();
        for &chunk in &invalidated {
            self.mesh_budget.release_reservation(chunk);
        }
        let summary = document.summary();
        let revision = document.revision();
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        self.restart_mesh_after_change(invalidated, revision);
        self.emit_edit_finished(edit_id, EditOutcome::Completed);
    }

    fn cancel_stroke(&mut self, edit_id: EditId) {
        if self
            .stroke
            .as_ref()
            .is_none_or(|stroke| stroke.edit_id != edit_id)
        {
            return;
        }
        self.rollback_active_stroke(None);
    }

    fn cancel_active_stroke(&mut self) {
        self.rollback_active_stroke(None);
    }

    fn rollback_active_stroke(&mut self, restore_cursor: Option<[u32; 3]>) {
        let Some(stroke) = self.stroke.take() else {
            return;
        };
        let edit_id = stroke.edit_id;
        if stroke.tool == Tool::Inspect {
            self.emit_edit_finished(edit_id, EditOutcome::Cancelled);
            return;
        }
        let mut affected = stroke.affected_chunks;
        let restoration = self.document.as_mut().map(|document| {
            let dimensions = document.geometry().dimensions;
            if let Some(indices) = document.active_stroke_indices() {
                affected.extend(invalidated_chunks_for_indices(
                    dimensions,
                    indices.iter().copied(),
                ));
            }
            document.cancel_stroke();
            if let Some(cursor) = restore_cursor {
                document.set_cursor(cursor);
            }
            let summary = document.summary();
            let revision = document.revision();
            let slices = visible_slices(document);
            (summary, revision, slices)
        });
        let affected = affected.into_iter().collect::<Vec<_>>();
        let mut preserved_live = if affected.is_empty() {
            HashSet::new()
        } else {
            self.cancel_mesh_for_barrier()
        };
        let affected_set = affected.iter().copied().collect::<HashSet<_>>();
        preserved_live.retain(|chunk| !affected_set.contains(chunk));
        self.promote_live_mesh_chunks(preserved_live);
        self.promote_live_mesh_chunks(affected.iter().copied());
        let Some((summary, revision, slices)) = restoration else {
            self.emit_edit_finished(edit_id, EditOutcome::Cancelled);
            return;
        };
        self.mesh_recovery_edit = Some(edit_id);
        let mut absent_before_stroke = Vec::new();
        let mut never_promoted = Vec::new();
        for &chunk in &affected {
            match self.mesh_budget.reserved_bytes(chunk) {
                Some(0) => absent_before_stroke.push(chunk),
                Some(_) => {
                    self.mesh_recovery_chunks.insert(chunk);
                }
                None => never_promoted.push(chunk),
            }
        }
        for chunk in never_promoted {
            // Cancellation can interrupt a sample after it mutates occupancy but
            // before that sample reserves or promotes any mesh work. The resident
            // geometry is therefore already the restored geometry. Demote the
            // bookkeeping-only promotion and let ordinary background work revisit
            // a chunk that had not reached the renderer yet.
            self.mesh_dirty.remove(chunk);
            self.mesh_live_chunks.remove(&chunk);
            if !self.preview_too_complex {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
            }
        }
        for chunk in absent_before_stroke {
            self.mesh_dirty.remove(chunk);
            self.mesh_live_chunks.remove(&chunk);
            self.mesh_budget.remove(chunk);
            let generation = self
                .mesh_generations
                .get(&chunk)
                .copied()
                .unwrap_or(self.mesh_pass_generation);
            let tag = MeshJobTag {
                document_epoch: self.epoch,
                revision,
                chunk_generation: generation,
                edit_generation: self.preview_generation,
                chunk,
            };
            self.queue_mesh_chunk(tag, Vec::new(), MeshUpdatePriority::Live);
            self.mesh_budget.release_reservation(chunk);
            if !self.preview_too_complex {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
            }
        }
        self.emit(EngineEvent::Summary {
            epoch: self.epoch,
            summary,
        });
        match slices {
            Ok(slices) => self.emit(EngineEvent::SlicesReplaced {
                epoch: self.epoch,
                revision,
                slices,
            }),
            Err(error) => self.emit_error(None, error),
        }
        self.maybe_finish_mesh_recovery();
    }

    fn undo(&mut self) {
        self.cancel_active_stroke();
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
        self.cancel_active_stroke();
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
        self.mesh_background_cancel.cancel();
        self.mesh_dirty.clear();
        self.mesh_live_chunks.clear();
        self.mesh_recovery_chunks.clear();
        self.mesh_recovery_edit = None;
        self.deferred_recovery_chunks.clear();
        self.progressive_live_jobs.clear();
        self.last_live_feedback_generation = 0;
        self.last_live_feedback_at = None;
        self.mesh_budget.clear();
        self.pending_live_mesh_chunks.clear();
        self.pending_background_mesh_chunks.clear();
        self.preview_too_complex = false;
        self.mesh_cancel = CancellationToken::default();
        self.mesh_background_cancel = CancellationToken::default();
        self.mesh_pass_generation = self
            .mesh_generations
            .values()
            .copied()
            .fold(self.mesh_pass_generation, u64::max)
            .saturating_add(1);
        self.mesh_generations.clear();
        self.mesh_edit_generations.clear();
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
        self.mesh_background_cancel.cancel();
        self.mesh_cancel = CancellationToken::default();
        self.mesh_background_cancel = CancellationToken::default();
        self.progressive_live_jobs.clear();
        if self.document.is_none() {
            return;
        }
        let mut preserved_live = self.requeue_pending_mesh_chunks();
        preserved_live.extend(self.requeue_in_flight_mesh_chunks());
        let invalidated_set = invalidated.iter().copied().collect::<HashSet<_>>();
        preserved_live.retain(|chunk| !invalidated_set.contains(chunk));
        self.prioritize_mesh_near_cursor();
        self.promote_live_mesh_chunks(preserved_live);
        self.promote_live_mesh_chunks(invalidated.iter().copied());
        self.emit(EngineEvent::MeshInvalidated {
            epoch: self.epoch,
            revision,
            chunks: invalidated,
        });
    }

    fn promote_live_mesh_chunks(&mut self, chunks: impl IntoIterator<Item = ChunkCoord>) {
        for chunk in chunks {
            if !self.mesh_dirty.contains_coordinate(chunk) {
                continue;
            }
            let generation = self
                .mesh_generations
                .get(&chunk)
                .copied()
                .unwrap_or(self.mesh_pass_generation)
                .saturating_add(1);
            self.mesh_generations.insert(chunk, generation);
            self.mesh_edit_generations
                .insert(chunk, self.preview_generation);
            self.mesh_live_chunks.insert(chunk);
            self.mesh_dirty.promote(chunk);
        }
    }

    fn preempt_mesh_for_live_feedback(&mut self) {
        let background_chunks = self
            .mesh_in_flight_priorities
            .iter()
            .filter_map(|(&chunk, &priority)| {
                (priority == MeshUpdatePriority::Background).then_some(chunk)
            })
            .collect::<Vec<_>>();
        if background_chunks.is_empty() {
            return;
        }
        self.mesh_background_cancel.cancel();
        self.mesh_background_cancel = CancellationToken::default();
        for chunk in background_chunks {
            let queued = self.mesh_dirty.enqueue_back(chunk);
            debug_assert!(queued);
        }
    }

    fn cancel_mesh_for_barrier(&mut self) -> HashSet<ChunkCoord> {
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
        self.mesh_cancel = CancellationToken::default();
        self.mesh_background_cancel = CancellationToken::default();
        self.progressive_live_jobs.clear();
        let mut preserved_live = self.requeue_pending_mesh_chunks();
        preserved_live.extend(self.requeue_in_flight_mesh_chunks());
        preserved_live
    }

    fn resume_mesh_after_cancel(&mut self) {
        self.mesh_cancel = CancellationToken::default();
        self.mesh_background_cancel = CancellationToken::default();
        self.progressive_live_jobs.clear();
        let preserved_live = self.requeue_in_flight_mesh_chunks();
        self.prioritize_mesh_near_cursor();
        self.promote_live_mesh_chunks(preserved_live);
    }

    fn requeue_pending_mesh_chunks(&mut self) -> HashSet<ChunkCoord> {
        let mut preserved_live = HashSet::new();
        for (tag, _) in self.pending_live_mesh_chunks.drain(..) {
            preserved_live.insert(tag.chunk);
            let queued = self.mesh_dirty.enqueue_back(tag.chunk);
            debug_assert!(queued);
        }
        for (tag, _) in self.pending_background_mesh_chunks.drain(..) {
            let queued = self.mesh_dirty.enqueue_back(tag.chunk);
            debug_assert!(queued);
        }
        preserved_live
    }

    fn requeue_in_flight_mesh_chunks(&mut self) -> HashSet<ChunkCoord> {
        let mut preserved_live = HashSet::new();
        for &chunk in &self.mesh_in_flight_chunks {
            if self.mesh_in_flight_priorities.get(&chunk) == Some(&MeshUpdatePriority::Live) {
                preserved_live.insert(chunk);
            }
            let queued = self.mesh_dirty.enqueue_back(chunk);
            debug_assert!(queued);
        }
        preserved_live
    }

    fn live_feedback_pending(&self) -> bool {
        self.stroke.is_some()
            || !self.mesh_live_chunks.is_empty()
            || !self.mesh_recovery_chunks.is_empty()
            || !self.progressive_live_jobs.is_empty()
            || !self.pending_live_mesh_chunks.is_empty()
            || self
                .mesh_in_flight_priorities
                .values()
                .any(|priority| *priority == MeshUpdatePriority::Live)
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
        if (self.preview_too_complex && self.mesh_recovery_chunks.is_empty())
            || (self.has_pending_mesh_chunks()
                && self.mesh_live_chunks.is_empty()
                && self.mesh_recovery_chunks.is_empty())
        {
            return;
        }
        while self.mesh_in_flight < self.mesh_worker_limit {
            if self.stroke.is_some()
                && self
                    .mesh_in_flight_priorities
                    .values()
                    .any(|priority| *priority == MeshUpdatePriority::Live)
            {
                break;
            }
            if self.has_pending_mesh_chunks()
                && self.mesh_live_chunks.is_empty()
                && self.mesh_recovery_chunks.is_empty()
            {
                break;
            }
            if !self.command_rx.is_empty() {
                break;
            }
            if (!self.sample_rx.is_empty() || !self.preview_rx.is_empty())
                && self.mesh_live_chunks.is_empty()
            {
                break;
            }
            let next = if self.preview_too_complex {
                self.next_recovery_mesh_chunk()
            } else {
                self.next_mesh_chunk()
            };
            let Some((chunk, priority)) = next else {
                break;
            };
            let edit_generation = self.mesh_edit_generations.get(&chunk).copied().unwrap_or(0);
            if priority == MeshUpdatePriority::Live
                && self.stroke.is_some()
                && edit_generation <= self.last_live_feedback_generation
                && self
                    .last_live_feedback_at
                    .is_some_and(|at| at.elapsed() < LIVE_FEEDBACK_GRACE)
            {
                self.mesh_live_chunks.insert(chunk);
                self.mesh_dirty.promote(chunk);
                break;
            }
            if priority == MeshUpdatePriority::Background && self.stroke.is_some() {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
                break;
            }
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
                edit_generation,
                chunk,
            };
            let cancel = match priority {
                MeshUpdatePriority::Background => self.mesh_background_cancel.clone(),
                MeshUpdatePriority::Live => self.mesh_cancel.clone(),
            };
            let input = match ChunkInput::copy_from_volume(
                document.geometry().dimensions,
                document.view().occupancy,
                chunk,
                &cancel,
            ) {
                Ok(input) => input,
                Err(AppError::Cancelled) => {
                    if self.mesh_recovery_chunks.contains(&chunk) {
                        self.mesh_cancel = CancellationToken::default();
                        self.mesh_live_chunks.insert(chunk);
                        self.mesh_dirty.promote(chunk);
                    }
                    break;
                }
                Err(error) => {
                    if self.mesh_recovery_chunks.contains(&chunk) {
                        self.fail_mesh_recovery(format!(
                            "chunk {:?} could not be prepared for remeshing: {error}",
                            chunk.as_array()
                        ));
                        break;
                    }
                    self.emit_error(None, error);
                    continue;
                }
            };
            let sender = self.mesh_tx.clone();
            self.mesh_in_flight += 1;
            self.mesh_in_flight_chunks.insert(chunk);
            self.mesh_in_flight_priorities.insert(chunk, priority);
            if priority == MeshUpdatePriority::Live {
                self.progressive_live_jobs.insert(tag);
            }
            rayon::spawn(move || {
                let result = mesh_chunk(&input, tag, &cancel);
                let _ = sender.send((tag, priority, result));
            });
        }
    }

    fn next_mesh_chunk(&mut self) -> Option<(ChunkCoord, MeshUpdatePriority)> {
        let pending = self.mesh_dirty.len();
        for _ in 0..pending {
            let chunk = self.mesh_dirty.pop_front()?;
            if self.mesh_in_flight_chunks.contains(&chunk) {
                let queued = self.mesh_dirty.enqueue_back(chunk);
                debug_assert!(queued);
            } else {
                let priority = if self.mesh_live_chunks.remove(&chunk) {
                    MeshUpdatePriority::Live
                } else {
                    MeshUpdatePriority::Background
                };
                return Some((chunk, priority));
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
            return Some((chunk, MeshUpdatePriority::Background));
        }
        None
    }

    fn next_recovery_mesh_chunk(&mut self) -> Option<(ChunkCoord, MeshUpdatePriority)> {
        let chunk = self.mesh_recovery_chunks.iter().copied().find(|chunk| {
            !self.mesh_in_flight_chunks.contains(chunk) && self.mesh_dirty.members.contains(chunk)
        })?;
        if !self.mesh_dirty.remove(chunk) {
            return None;
        }
        self.mesh_live_chunks.remove(&chunk);
        Some((chunk, MeshUpdatePriority::Live))
    }

    fn handle_mesh_result(&mut self, (tag, priority, result): MeshResult) {
        self.mesh_in_flight = self.mesh_in_flight.saturating_sub(1);
        self.mesh_in_flight_chunks.remove(&tag.chunk);
        self.mesh_in_flight_priorities.remove(&tag.chunk);
        let progressive_live = self.progressive_live_jobs.remove(&tag);
        let recovery_result = self.mesh_recovery_chunks.contains(&tag.chunk);
        if self.mesh_recovery_edit.is_some() && !recovery_result {
            return;
        }
        if self.preview_too_complex && !recovery_result {
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
        let identity_is_stale =
            tag.document_epoch != self.epoch || tag.revision != document.revision();
        let progressive_generation = priority == MeshUpdatePriority::Live
            && progressive_live
            && tag.chunk_generation < generation;
        if identity_is_stale || (tag.chunk_generation != generation && !progressive_generation) {
            return;
        }
        let tagged = match result {
            Ok(tagged) => tagged,
            Err(AppError::Cancelled) => {
                // A cancelled shared pass can leave the document identity and
                // this chunk generation unchanged. Keep that still-current
                // chunk dirty so work already passed by the lazy scan cursor is
                // not lost when the replacement pass resumes.
                let queued = self.mesh_dirty.enqueue_back(tag.chunk);
                debug_assert!(queued);
                if recovery_result {
                    if self.mesh_cancel.is_cancelled() {
                        self.mesh_cancel = CancellationToken::default();
                    }
                    self.mesh_live_chunks.insert(tag.chunk);
                    self.mesh_dirty.promote(tag.chunk);
                }
                return;
            }
            Err(error) => {
                if recovery_result {
                    self.fail_mesh_recovery(format!(
                        "chunk {:?} could not be remeshed: {error}",
                        tag.chunk.as_array()
                    ));
                } else {
                    self.emit_error(None, error);
                }
                return;
            }
        };
        if recovery_result {
            match self.mesh_budget.try_replace_mesh(tag.chunk, &tagged.mesh) {
                Ok(()) => {
                    self.accept_recovery_mesh(tagged);
                    self.requeue_deferred_recovery_chunks();
                }
                Err(MeshBudgetError::Limit { .. }) => {
                    self.deferred_recovery_chunks.insert(tag.chunk);
                    self.fail_recovery_if_no_progress_is_possible();
                }
                Err(MeshBudgetError::Allocation) => {
                    self.fail_mesh_recovery(format!(
                        "mesh accounting for chunk {:?} ran out of memory",
                        tag.chunk.as_array()
                    ));
                }
            }
            self.maybe_finish_mesh_recovery();
            return;
        }
        if self
            .mesh_budget
            .try_replace_mesh_or_cancel(tag.chunk, &tagged.mesh, &self.mesh_cancel)
            .is_err()
        {
            self.mesh_background_cancel.cancel();
            self.preview_too_complex = true;
            self.mesh_dirty.clear();
            self.mesh_scan_next = self.mesh_scan_total;
            self.emit(EngineEvent::PreviewTooComplex {
                epoch: self.epoch,
                revision: document.revision(),
            });
            return;
        }
        self.queue_mesh_chunk(tagged.tag, tagged.mesh.quads, priority);
        if priority == MeshUpdatePriority::Live && self.stroke.is_some() {
            self.last_live_feedback_generation =
                self.last_live_feedback_generation.max(tag.edit_generation);
            self.last_live_feedback_at = Some(Instant::now());
        }
    }

    fn accept_recovery_mesh(&mut self, tagged: TaggedChunkMesh) {
        let tag = tagged.tag;
        self.mesh_budget.release_reservation(tag.chunk);
        self.mesh_recovery_chunks.remove(&tag.chunk);
        self.queue_mesh_chunk(tag, tagged.mesh.quads, MeshUpdatePriority::Live);
    }

    fn requeue_deferred_recovery_chunks(&mut self) {
        for chunk in self.deferred_recovery_chunks.drain() {
            self.mesh_live_chunks.insert(chunk);
            let queued = self.mesh_dirty.enqueue_back(chunk);
            debug_assert!(queued);
        }
    }

    fn fail_recovery_if_no_progress_is_possible(&mut self) {
        let work_can_still_finish = self.mesh_recovery_chunks.iter().any(|chunk| {
            self.mesh_in_flight_chunks.contains(chunk) || self.mesh_dirty.members.contains(chunk)
        });
        if !self.deferred_recovery_chunks.is_empty() && !work_can_still_finish {
            self.fail_mesh_recovery(
                "the restored chunk meshes exceed the available 3D preview memory".into(),
            );
        }
    }

    fn fail_mesh_recovery(&mut self, message: String) {
        let failed_chunks = self.mesh_recovery_chunks.clone();
        self.mesh_cancel.cancel();
        self.mesh_background_cancel.cancel();
        self.preview_too_complex = true;
        self.mesh_dirty.clear();
        self.mesh_live_chunks.clear();
        self.mesh_scan_next = self.mesh_scan_total;
        self.progressive_live_jobs.clear();
        self.pending_live_mesh_chunks
            .retain(|(tag, _)| !failed_chunks.contains(&tag.chunk));
        for chunk in self.mesh_recovery_chunks.drain() {
            self.mesh_budget.release_reservation(chunk);
        }
        self.deferred_recovery_chunks.clear();
        self.emit(EngineEvent::Error {
            job_id: None,
            message: format!("3D surface rollback failed: {message}"),
            source_changed: false,
        });
        if let Some(document) = &self.document {
            self.emit(EngineEvent::PreviewTooComplex {
                epoch: self.epoch,
                revision: document.revision(),
            });
        }
        self.maybe_finish_mesh_recovery();
    }

    fn maybe_finish_mesh_recovery(&mut self) {
        if self.mesh_recovery_chunks.is_empty()
            && self.deferred_recovery_chunks.is_empty()
            && self.pending_live_mesh_chunks.is_empty()
            && let Some(edit_id) = self.mesh_recovery_edit.take()
        {
            self.emit_edit_finished(edit_id, EditOutcome::Cancelled);
        }
    }

    fn queue_mesh_chunk(
        &mut self,
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
        priority: MeshUpdatePriority,
    ) {
        if self.has_pending_mesh_chunks() || self.event_tx.len() >= MESH_EVENT_BACKLOG_LIMIT {
            self.enqueue_pending_mesh_chunk(tag, quads, priority);
            return;
        }
        match self.event_tx.try_send(EngineEvent::MeshChunk {
            tag,
            quads,
            priority,
        }) {
            Ok(()) => self.repaint.request_repaint(),
            Err(TrySendError::Full(EngineEvent::MeshChunk {
                tag,
                quads,
                priority,
            })) => {
                self.enqueue_pending_mesh_chunk(tag, quads, priority);
            }
            Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => unreachable!("only mesh chunks are queued here"),
        }
    }

    fn enqueue_pending_mesh_chunk(
        &mut self,
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
        priority: MeshUpdatePriority,
    ) {
        match priority {
            MeshUpdatePriority::Live => {
                let newer_is_queued = self.pending_live_mesh_chunks.iter().any(|(queued, _)| {
                    queued.document_epoch == tag.document_epoch
                        && queued.revision == tag.revision
                        && queued.chunk == tag.chunk
                        && queued.chunk_generation >= tag.chunk_generation
                });
                if newer_is_queued {
                    return;
                }
                self.pending_live_mesh_chunks.retain(|(queued, _)| {
                    queued.document_epoch != tag.document_epoch
                        || queued.revision != tag.revision
                        || queued.chunk != tag.chunk
                        || queued.chunk_generation > tag.chunk_generation
                });
                self.pending_background_mesh_chunks.retain(|(queued, _)| {
                    queued.document_epoch != tag.document_epoch
                        || queued.revision != tag.revision
                        || queued.chunk != tag.chunk
                        || queued.chunk_generation > tag.chunk_generation
                });
                self.pending_live_mesh_chunks.push_back((tag, quads));
            }
            MeshUpdatePriority::Background => {
                if self.pending_live_mesh_chunks.iter().any(|(queued, _)| {
                    queued.document_epoch == tag.document_epoch
                        && queued.revision == tag.revision
                        && queued.chunk == tag.chunk
                        && queued.chunk_generation >= tag.chunk_generation
                }) {
                    return;
                }
                self.pending_background_mesh_chunks.push_back((tag, quads))
            }
        }
    }

    fn has_pending_mesh_chunks(&self) -> bool {
        !self.pending_live_mesh_chunks.is_empty() || !self.pending_background_mesh_chunks.is_empty()
    }

    fn flush_pending_mesh_chunk(&mut self) {
        if self.event_tx.len() >= MESH_EVENT_BACKLOG_LIMIT {
            return;
        }
        let (priority, pending) = if let Some(pending) = self.pending_live_mesh_chunks.pop_front() {
            (MeshUpdatePriority::Live, pending)
        } else if let Some(pending) = self.pending_background_mesh_chunks.pop_front() {
            (MeshUpdatePriority::Background, pending)
        } else {
            return;
        };
        let (tag, quads) = pending;
        match self.event_tx.try_send(EngineEvent::MeshChunk {
            tag,
            quads,
            priority,
        }) {
            Ok(()) => {
                self.repaint.request_repaint();
                if priority == MeshUpdatePriority::Live {
                    self.maybe_finish_mesh_recovery();
                }
            }
            Err(TrySendError::Full(EngineEvent::MeshChunk {
                tag,
                quads,
                priority,
            })) => match priority {
                MeshUpdatePriority::Live => {
                    self.pending_live_mesh_chunks.push_front((tag, quads));
                }
                MeshUpdatePriority::Background => {
                    self.pending_background_mesh_chunks.push_front((tag, quads));
                }
            },
            Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => unreachable!("only mesh chunks are queued here"),
        }
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

    fn emit_edit_finished(&self, edit_id: EditId, outcome: EditOutcome) {
        self.emit(EngineEvent::EditFinished {
            epoch: self.epoch,
            edit_id,
            outcome,
        });
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
            surface_edit_target_with_occupancy(document.geometry(), ray, tool, |index| {
                document.stroke_start_occupied(index)
            })
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
            if tool == Tool::Bridge {
                cancel.check()?;
                return Ok(BrushPreviewComputation {
                    generation,
                    tool,
                    target,
                    voxels: target.into_iter().collect(),
                });
            }
            let candidates = match target {
                None => Vec::new(),
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
    cancel: &CancellationToken,
) -> Result<Vec<u32>> {
    let (normal_axis, spec) = match space {
        StrokeSpace::Slice(axis) => (axis.normal_axis(), spec),
        StrokeSpace::Volume3D => (0, spec.with_depth(BrushDepth::Volume)),
    };
    brush_indices_cancellable(document.geometry(), center, normal_axis, spec, cancel, None)
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
        for index in
            footprint_indices(document, center, spec, space, &CancellationToken::default())?
        {
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
    use crate::mesh::ChunkMesh;
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
        let (latest_samples, sample_drain) = bounded::<StrokeSample>(1);
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

    fn current_chunk_mesh(state: &EngineState, chunk: ChunkCoord) -> TaggedChunkMesh {
        let document = state.document.as_ref().unwrap();
        let tag = MeshJobTag {
            document_epoch: state.epoch,
            revision: document.revision(),
            chunk_generation: state
                .mesh_generations
                .get(&chunk)
                .copied()
                .unwrap_or(state.mesh_pass_generation),
            edit_generation: state
                .mesh_edit_generations
                .get(&chunk)
                .copied()
                .unwrap_or(0),
            chunk,
        };
        let input = ChunkInput::copy_from_volume(
            document.geometry().dimensions,
            document.view().occupancy,
            chunk,
            &CancellationToken::default(),
        )
        .unwrap();
        mesh_chunk(&input, tag, &CancellationToken::default()).unwrap()
    }

    #[test]
    fn latest_sample_channel_discards_intermediate_motion() {
        let context = egui::Context::default();
        let engine = EngineHandle::spawn(context);
        engine
            .send_latest_sample(
                4,
                PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [1, 1, 1],
                },
            )
            .unwrap();
        engine
            .send_latest_sample(
                4,
                PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [9, 9, 9],
                },
            )
            .unwrap();
        assert_eq!(
            engine.sample_drain.try_recv().unwrap(),
            StrokeSample {
                edit_id: 4,
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [9, 9, 9],
                },
            }
        );
        assert!(engine.sample_drain.try_recv().is_err());
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
    fn bridge_anchor_preview_contains_only_the_selected_voxel() {
        let state = state_with_empty_document([6, 3, 1]);
        let preview = calculate_brush_preview(
            state.document.as_ref().unwrap(),
            BrushPreviewRequest::Brush {
                generation: 7,
                tool: Tool::Bridge,
                spec: BrushSpec {
                    shape: BrushShape::Square,
                    depth: BrushDepth::Volume,
                    size: BrushSize::voxels(5).unwrap(),
                },
                space: StrokeSpace::Slice(SliceAxis::Axial),
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [3, 1, 0],
                },
            },
            &CancellationToken::default(),
        )
        .unwrap();

        assert_eq!(preview.target, Some([3, 1, 0]));
        assert_eq!(preview.voxels, vec![[3, 1, 0]]);
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
    fn active_stroke_keeps_mesh_workers_live_and_promotes_exact_halo_chunks() {
        let mut state = state_with_empty_document([64, 1, 1]);
        let chunks = [ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0)];
        state.begin_stroke(
            1,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [31, 0, 0],
            },
            CancellationToken::default(),
        );

        assert!(!state.mesh_cancel.is_cancelled());
        assert_eq!(state.mesh_dirty.len(), chunks.len());
        assert!(
            chunks
                .into_iter()
                .all(|chunk| state.mesh_dirty.members.contains(&chunk))
        );
        assert_eq!(state.mesh_live_chunks, chunks.into_iter().collect());
        assert!(chunks.into_iter().all(|chunk| {
            state.mesh_generations.get(&chunk) == Some(&(state.mesh_pass_generation + 1))
        }));
    }

    #[test]
    fn nearby_ray_samples_do_not_advance_a_painted_surface_within_one_stroke() {
        let mut state = state_with_empty_document([3, 1, 1]);
        let document = state.document.as_mut().unwrap();
        document.begin_stroke(1).unwrap();
        document.apply_stroke_indices([0]).unwrap();
        document.finish_stroke().unwrap();
        let initial_sample = PointerSample::Ray {
            ray: Ray {
                origin: [4.0, 0.0, 0.0],
                direction: [-1.0, 0.0, 0.0],
            },
        };

        state.begin_stroke(
            77,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Volume3D,
            initial_sample,
            CancellationToken::default(),
        );
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);
        let generation = state.preview_generation;

        state.apply_pointer_sample(StrokeSample {
            edit_id: 77,
            sample: PointerSample::Ray {
                ray: Ray {
                    origin: [4.0, 0.1, 0.0],
                    direction: [-1.0, 0.0, 0.0],
                },
            },
        });

        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);
        assert_eq!(state.preview_generation, generation);
    }

    #[test]
    fn nearby_ray_samples_do_not_tunnel_through_an_erased_surface_within_one_stroke() {
        let mut state = state_with_empty_document([3, 1, 1]);
        let document = state.document.as_mut().unwrap();
        document.begin_stroke(1).unwrap();
        document.apply_stroke_indices([0, 1, 2]).unwrap();
        document.finish_stroke().unwrap();

        state.begin_stroke(
            78,
            Tool::Erase,
            one_voxel_brush(),
            StrokeSpace::Volume3D,
            PointerSample::Ray {
                ray: Ray {
                    origin: [4.0, 0.0, 0.0],
                    direction: [-1.0, 0.0, 0.0],
                },
            },
            CancellationToken::default(),
        );
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);
        let generation = state.preview_generation;

        state.apply_pointer_sample(StrokeSample {
            edit_id: 78,
            sample: PointerSample::Ray {
                ray: Ray {
                    origin: [4.0, 0.1, 0.0],
                    direction: [-1.0, 0.0, 0.0],
                },
            },
        });

        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);
        assert_eq!(state.preview_generation, generation);
    }

    #[test]
    fn one_layer_stroke_still_follows_distinct_surface_targets() {
        let mut state = state_with_empty_document([3, 2, 1]);
        let document = state.document.as_mut().unwrap();
        document.begin_stroke(1).unwrap();
        document.apply_stroke_indices([0, 3]).unwrap();
        document.finish_stroke().unwrap();

        state.begin_stroke(
            79,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Volume3D,
            PointerSample::Ray {
                ray: Ray {
                    origin: [4.0, 0.0, 0.0],
                    direction: [-1.0, 0.0, 0.0],
                },
            },
            CancellationToken::default(),
        );
        state.apply_pointer_sample(StrokeSample {
            edit_id: 79,
            sample: PointerSample::Ray {
                ray: Ray {
                    origin: [4.0, 1.0, 0.0],
                    direction: [-1.0, 0.0, 0.0],
                },
            },
        });

        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 4);
        assert_eq!(
            state.document.as_ref().unwrap().view().occupancy,
            &[1, 1, 0, 1, 1, 0]
        );
    }

    #[test]
    fn a_new_stroke_can_intentionally_add_the_next_surface_layer() {
        let mut state = state_with_empty_document([4, 1, 1]);
        let document = state.document.as_mut().unwrap();
        document.begin_stroke(1).unwrap();
        document.apply_stroke_indices([0]).unwrap();
        document.finish_stroke().unwrap();
        let sample = PointerSample::Ray {
            ray: Ray {
                origin: [5.0, 0.0, 0.0],
                direction: [-1.0, 0.0, 0.0],
            },
        };

        state.begin_stroke(
            80,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Volume3D,
            sample,
            CancellationToken::default(),
        );
        state.complete_stroke(80);
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);

        state.begin_stroke(
            81,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Volume3D,
            sample,
            CancellationToken::default(),
        );
        state.complete_stroke(81);
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 3);

        let document = state.document.as_mut().unwrap();
        document.undo().unwrap();
        assert_eq!(document.summary().occupied_count, 2);
        document.undo().unwrap();
        assert_eq!(document.summary().occupied_count, 1);
    }

    #[test]
    fn terminal_commands_do_not_consume_the_next_strokes_sample() {
        let mut state = state_with_empty_document([4, 1, 1]);
        state.begin_stroke(
            1,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [0, 0, 0],
            },
            CancellationToken::default(),
        );
        let (sample_tx, sample_rx) = bounded(1);
        state.sample_rx = sample_rx;
        sample_tx
            .send(StrokeSample {
                edit_id: 2,
                sample: PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [2, 0, 0],
                },
            })
            .unwrap();

        state.handle_command(EngineCommand::EndStroke {
            edit_id: 1,
            disposition: StrokeDisposition::Complete {
                final_sample: Some(PointerSample::Slice {
                    axis: SliceAxis::Axial,
                    voxel: [1, 0, 0],
                }),
            },
        });
        assert_eq!(state.sample_rx.len(), 1);
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 2);

        state.begin_stroke(
            2,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [3, 0, 0],
            },
            CancellationToken::default(),
        );
        state.apply_pointer_sample(state.sample_rx.try_recv().unwrap());
        state.handle_command(EngineCommand::EndStroke {
            edit_id: 1,
            disposition: StrokeDisposition::Cancel,
        });

        assert_eq!(state.stroke.as_ref().map(|stroke| stroke.edit_id), Some(2));
        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 4);
    }

    #[test]
    fn live_feedback_preempts_background_without_cancelling_live_jobs() {
        let mut state = state_with_empty_document([64, 1, 1]);
        let background = ChunkCoord::new(0, 0, 0);
        let live = ChunkCoord::new(1, 0, 0);
        state.mesh_in_flight = state.mesh_worker_limit;
        state.mesh_in_flight_chunks.extend([background, live]);
        state
            .mesh_in_flight_priorities
            .insert(background, MeshUpdatePriority::Background);
        state
            .mesh_in_flight_priorities
            .insert(live, MeshUpdatePriority::Live);
        let cancelled_background_batch = state.mesh_background_cancel.clone();
        let retained_live_batch = state.mesh_cancel.clone();

        state.preempt_mesh_for_live_feedback();

        assert!(cancelled_background_batch.is_cancelled());
        assert!(!retained_live_batch.is_cancelled());
        assert!(state.mesh_dirty.members.contains(&background));
        assert!(!state.mesh_dirty.members.contains(&live));
    }

    #[test]
    fn reset_generation_is_newer_than_every_in_flight_chunk_generation() {
        let mut state = state_with_empty_document([32, 1, 1]);
        let chunk = ChunkCoord::new(0, 0, 0);
        state.mesh_pass_generation = 5;
        state.mesh_generations.insert(chunk, 8);

        state.reset_mesh_preview();

        assert_eq!(state.mesh_pass_generation, 9);
        let old = MeshJobTag {
            document_epoch: state.epoch,
            revision: state.document.as_ref().unwrap().revision(),
            chunk_generation: 8,
            edit_generation: 0,
            chunk,
        };
        assert!(old.is_stale(state.epoch, old.revision, state.mesh_pass_generation));
    }

    #[test]
    fn committed_live_chunks_stay_ahead_of_cursor_background_work() {
        let mut state = state_with_empty_document([96, 1, 1]);
        let invalidated = ChunkCoord::new(2, 0, 0);

        state.restart_mesh_after_change(vec![invalidated], 2);

        assert_eq!(
            state.next_mesh_chunk(),
            Some((invalidated, MeshUpdatePriority::Live))
        );
    }

    #[test]
    fn revision_barrier_preserves_older_live_work_ahead_of_background() {
        let mut state = state_with_empty_document([128, 1, 1]);
        let background = ChunkCoord::new(0, 0, 0);
        let current = ChunkCoord::new(1, 0, 0);
        let pending_live = ChunkCoord::new(2, 0, 0);
        let in_flight_live = ChunkCoord::new(3, 0, 0);
        assert!(state.mesh_dirty.enqueue_back(background));
        let tag = MeshJobTag {
            document_epoch: state.epoch,
            revision: state.document.as_ref().unwrap().revision(),
            chunk_generation: state.mesh_pass_generation,
            edit_generation: 1,
            chunk: pending_live,
        };
        state.pending_live_mesh_chunks.push_back((tag, Vec::new()));
        state.mesh_in_flight_chunks.insert(in_flight_live);
        state
            .mesh_in_flight_priorities
            .insert(in_flight_live, MeshUpdatePriority::Live);

        state.restart_mesh_after_change(vec![current], 2);
        // Model the cancelled worker returning before replacement dispatch.
        state.mesh_in_flight_chunks.remove(&in_flight_live);
        state.mesh_in_flight_priorities.remove(&in_flight_live);

        assert_eq!(
            state.next_mesh_chunk(),
            Some((current, MeshUpdatePriority::Live))
        );
        let older_live = [state.next_mesh_chunk(), state.next_mesh_chunk()];
        assert!(older_live.iter().all(|entry| {
            entry.is_some_and(|(_, priority)| priority == MeshUpdatePriority::Live)
        }));
        assert_eq!(
            older_live
                .into_iter()
                .flatten()
                .map(|(chunk, _)| chunk)
                .collect::<HashSet<_>>(),
            HashSet::from([pending_live, in_flight_live])
        );
        assert_eq!(
            state.next_mesh_chunk(),
            Some((background, MeshUpdatePriority::Background))
        );
    }

    #[test]
    fn rebuild_is_rejected_without_resetting_an_active_stroke() {
        let mut state = state_with_empty_document([1, 1, 1]);
        let (event_tx, event_rx) = bounded(16);
        state.event_tx = event_tx;
        state.begin_stroke(
            92,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [0, 0, 0],
            },
            CancellationToken::default(),
        );
        while event_rx.try_recv().is_ok() {}

        state.handle_command(EngineCommand::RebuildPreview);

        assert_eq!(state.stroke.as_ref().map(|stroke| stroke.edit_id), Some(92));
        let events = event_rx.try_iter().collect::<Vec<_>>();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, EngineEvent::Error { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, EngineEvent::MeshReset { .. }))
        );
    }

    #[test]
    fn pending_live_uploads_keep_only_the_latest_chunk_generation() {
        let mut state = state_with_empty_document([64, 1, 1]);
        let chunk = ChunkCoord::new(0, 0, 0);
        for generation in 1..=100 {
            state.enqueue_pending_mesh_chunk(
                MeshJobTag {
                    document_epoch: state.epoch,
                    revision: state.document.as_ref().unwrap().revision(),
                    chunk_generation: generation,
                    edit_generation: generation,
                    chunk,
                },
                vec![
                    CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .unwrap(),
                ],
                MeshUpdatePriority::Live,
            );
        }

        assert_eq!(state.pending_live_mesh_chunks.len(), 1);
        assert_eq!(
            state
                .pending_live_mesh_chunks
                .front()
                .unwrap()
                .0
                .chunk_generation,
            100
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
    fn cancel_recovers_resident_chunks_after_partial_live_budget_failure() {
        let dimensions = [64, 3, 3];
        let chunks = [ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0)];
        let mut state = state_with_empty_document(dimensions);
        let document = state.document.as_mut().unwrap();
        document.begin_stroke(1).unwrap();
        document
            .apply_stroke_indices(0..dimensions.into_iter().product())
            .unwrap();
        document.finish_stroke().unwrap();
        let baseline = chunks.map(|chunk| current_chunk_mesh(&state, chunk));
        let baseline_bytes: [usize; 2] =
            std::array::from_fn(|index| baseline[index].mesh.gpu_bytes());
        let baseline_quads: [Vec<CompactQuad>; 2] =
            std::array::from_fn(|index| baseline[index].mesh.quads.clone());

        {
            let document = state.document.as_mut().unwrap();
            document.begin_stroke(0).unwrap();
            let line = (16..=48).map(|x| x + dimensions[0] * 4);
            document.apply_stroke_indices(line).unwrap();
        }
        let edited = chunks.map(|chunk| current_chunk_mesh(&state, chunk));
        state.document.as_mut().unwrap().cancel_stroke();
        let edited_bytes = edited.map(|tagged| tagged.mesh.gpu_bytes());
        assert!(
            edited_bytes
                .iter()
                .zip(baseline_bytes)
                .all(|(&edited, baseline)| edited > baseline)
        );

        let cap = edited_bytes[0].max(baseline_bytes[0]) + baseline_bytes[1];
        state.mesh_budget = MeshMemoryBudget::new(cap);
        for (chunk, bytes) in chunks.into_iter().zip(baseline_bytes) {
            state.mesh_budget.try_replace(chunk, bytes).unwrap();
        }
        let (event_tx, event_rx) = bounded(128);
        state.event_tx = event_tx;
        state.begin_stroke(
            90,
            Tool::Erase,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [16, 1, 1],
            },
            CancellationToken::default(),
        );
        state.apply_pointer_sample(StrokeSample {
            edit_id: 90,
            sample: PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [48, 1, 1],
            },
        });

        for chunk in chunks {
            let tagged = current_chunk_mesh(&state, chunk);
            state.handle_mesh_result((tagged.tag, MeshUpdatePriority::Live, Ok(tagged)));
        }
        assert!(state.preview_too_complex);
        assert_eq!(
            state.mesh_budget.reserved_bytes(chunks[0]),
            Some(baseline_bytes[0])
        );
        assert_eq!(
            state.mesh_budget.reserved_bytes(chunks[1]),
            Some(baseline_bytes[1])
        );

        state.cancel_stroke(90);
        assert_eq!(state.mesh_recovery_chunks, chunks.into_iter().collect());
        state.handle_command(EngineCommand::RebuildPreview);
        assert_eq!(state.mesh_recovery_chunks, chunks.into_iter().collect());
        assert_eq!(state.mesh_recovery_edit, Some(90));
        while event_rx.try_recv().is_ok() {}
        let recovery_generations = chunks.map(|chunk| state.mesh_generations[&chunk]);
        for chunk in chunks {
            let tagged = current_chunk_mesh(&state, chunk);
            state.handle_mesh_result((tagged.tag, MeshUpdatePriority::Live, Ok(tagged)));
        }
        while state.has_pending_mesh_chunks() {
            state.flush_pending_mesh_chunk();
        }

        assert!(state.preview_too_complex);
        assert!(state.mesh_recovery_chunks.is_empty());
        assert!(state.mesh_recovery_edit.is_none());
        assert_eq!(
            state.mesh_budget.used_bytes(),
            baseline_bytes.into_iter().sum()
        );
        for chunk in chunks {
            assert_eq!(state.mesh_budget.reserved_bytes(chunk), None);
        }
        let mut restored = HashMap::new();
        let mut acknowledged = false;
        while let Ok(event) = event_rx.try_recv() {
            match event {
                EngineEvent::MeshChunk {
                    tag,
                    quads,
                    priority: MeshUpdatePriority::Live,
                } => {
                    restored.insert(tag.chunk, (tag.chunk_generation, quads));
                }
                EngineEvent::EditFinished {
                    edit_id: 90,
                    outcome: EditOutcome::Cancelled,
                    ..
                } => acknowledged = true,
                EngineEvent::MeshReset { .. } => panic!("rollback must not reset the mesh"),
                _ => {}
            }
        }
        assert!(acknowledged);
        for (index, chunk) in chunks.into_iter().enumerate() {
            let (generation, quads) = restored.remove(&chunk).unwrap();
            assert_eq!(generation, recovery_generations[index]);
            assert_eq!(quads, baseline_quads[index]);
        }
    }

    #[test]
    fn cancel_removes_a_live_chunk_that_was_not_resident_before_the_stroke() {
        let mut state = state_with_empty_document([1, 1, 1]);
        let chunk = ChunkCoord::new(0, 0, 0);
        let (event_tx, event_rx) = bounded(32);
        state.event_tx = event_tx;
        state.begin_stroke(
            91,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [0, 0, 0],
            },
            CancellationToken::default(),
        );
        let live = current_chunk_mesh(&state, chunk);
        state.handle_mesh_result((live.tag, MeshUpdatePriority::Live, Ok(live)));
        assert!(state.mesh_budget.used_bytes() > 0);
        assert_eq!(state.mesh_budget.reserved_bytes(chunk), Some(0));
        while event_rx.try_recv().is_ok() {}

        state.cancel_stroke(91);

        assert_eq!(state.mesh_budget.used_bytes(), 0);
        assert_eq!(state.mesh_budget.reserved_bytes(chunk), None);
        assert!(state.mesh_dirty.members.contains(&chunk));
        assert!(!state.mesh_live_chunks.contains(&chunk));
        let events = event_rx.try_iter().collect::<Vec<_>>();
        assert!(events.iter().any(|event| {
            matches!(
                event,
                EngineEvent::MeshChunk {
                    tag,
                    quads,
                    priority: MeshUpdatePriority::Live,
                } if tag.chunk == chunk && quads.is_empty()
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                EngineEvent::EditFinished {
                    edit_id: 91,
                    outcome: EditOutcome::Cancelled,
                    ..
                }
            )
        }));
    }

    #[test]
    fn partial_cancel_before_mesh_promotion_needs_no_recovery_mesh() {
        let mut state = state_with_empty_document([1, 1, 1]);
        let chunk = ChunkCoord::new(0, 0, 0);
        let (event_tx, event_rx) = bounded(32);
        state.event_tx = event_tx;
        state.mesh_budget = MeshMemoryBudget::new(0);
        state.preview_too_complex = true;
        state.document.as_mut().unwrap().begin_stroke(1).unwrap();
        state
            .document
            .as_mut()
            .unwrap()
            .apply_stroke_indices([0])
            .unwrap();
        state.stroke = Some(StrokeContext {
            edit_id: 93,
            tool: Tool::Paint,
            spec: one_voxel_brush(),
            space: StrokeSpace::Slice(SliceAxis::Axial),
            cancel: CancellationToken::default(),
            last_sample: None,
            last_applied: None,
            affected_chunks: HashSet::new(),
        });

        state.rollback_active_stroke(None);

        assert_eq!(state.document.as_ref().unwrap().summary().occupied_count, 0);
        assert!(state.mesh_recovery_chunks.is_empty());
        assert!(state.deferred_recovery_chunks.is_empty());
        assert!(state.mesh_recovery_edit.is_none());
        assert!(!state.mesh_live_chunks.contains(&chunk));
        assert_eq!(state.mesh_budget.used_bytes(), 0);
        let events = event_rx.try_iter().collect::<Vec<_>>();
        assert!(!events.iter().any(|event| {
            matches!(event, EngineEvent::MeshChunk { tag, .. } if tag.chunk == chunk)
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                EngineEvent::EditFinished {
                    edit_id: 93,
                    outcome: EditOutcome::Cancelled,
                    ..
                }
            )
        }));
    }

    #[test]
    fn recovery_retries_growth_after_a_shrinking_chunk_frees_budget() {
        let mut state = state_with_empty_document([64, 1, 1]);
        let grow = ChunkCoord::new(0, 0, 0);
        let shrink = ChunkCoord::new(1, 0, 0);
        let quad =
            CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4]).unwrap();
        let bytes = size_of::<CompactQuad>();
        state.mesh_budget = MeshMemoryBudget::new(3 * bytes);
        state.mesh_budget.try_replace(grow, bytes).unwrap();
        state.mesh_budget.try_replace(shrink, 2 * bytes).unwrap();
        state.mesh_budget.reserve_current_if_absent(grow).unwrap();
        state.mesh_budget.reserve_current_if_absent(shrink).unwrap();
        state.mesh_recovery_chunks.extend([grow, shrink]);
        state.mesh_recovery_edit = Some(94);
        state.mesh_in_flight = 2;
        state.mesh_in_flight_chunks.extend([grow, shrink]);
        let revision = state.document.as_ref().unwrap().revision();
        let epoch = state.epoch;
        let generation = state.mesh_pass_generation;
        let tag = |chunk| MeshJobTag {
            document_epoch: epoch,
            revision,
            chunk_generation: generation,
            edit_generation: 1,
            chunk,
        };
        let cancelled_live_batch = state.mesh_cancel.clone();
        let (event_tx, event_rx) = bounded(32);
        state.event_tx = event_tx;

        state.handle_mesh_result((
            tag(grow),
            MeshUpdatePriority::Live,
            Ok(TaggedChunkMesh {
                tag: tag(grow),
                mesh: ChunkMesh {
                    quads: vec![quad; 2],
                    exposed_voxel_faces: 2,
                },
            }),
        ));
        assert!(state.deferred_recovery_chunks.contains(&grow));
        assert!(!cancelled_live_batch.is_cancelled());

        state.handle_mesh_result((
            tag(shrink),
            MeshUpdatePriority::Live,
            Ok(TaggedChunkMesh {
                tag: tag(shrink),
                mesh: ChunkMesh {
                    quads: vec![quad],
                    exposed_voxel_faces: 1,
                },
            }),
        ));
        assert!(!state.deferred_recovery_chunks.contains(&grow));
        assert!(state.mesh_dirty.members.contains(&grow));
        state.mesh_in_flight = 1;
        state.mesh_in_flight_chunks.insert(grow);
        state.handle_mesh_result((
            tag(grow),
            MeshUpdatePriority::Live,
            Ok(TaggedChunkMesh {
                tag: tag(grow),
                mesh: ChunkMesh {
                    quads: vec![quad; 2],
                    exposed_voxel_faces: 2,
                },
            }),
        ));

        assert!(state.mesh_recovery_chunks.is_empty());
        assert!(state.mesh_recovery_edit.is_none());
        assert_eq!(state.mesh_budget.used_bytes(), 3 * bytes);
        assert_eq!(state.mesh_budget.reserved_bytes(grow), None);
        assert_eq!(state.mesh_budget.reserved_bytes(shrink), None);
        let events = event_rx.try_iter().collect::<Vec<_>>();
        let acknowledgement = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    EngineEvent::EditFinished {
                        edit_id: 94,
                        outcome: EditOutcome::Cancelled,
                        ..
                    }
                )
            })
            .unwrap();
        let restored = events
            .iter()
            .enumerate()
            .filter(|(_, event)| matches!(event, EngineEvent::MeshChunk { .. }))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(restored.len(), 2);
        assert!(restored.into_iter().all(|index| index < acknowledgement));
    }

    #[test]
    fn recovery_ack_waits_until_a_pending_restoration_is_delivered() {
        let mut state = state_with_empty_document([1, 1, 1]);
        let (event_tx, event_rx) = bounded(64);
        state.event_tx = event_tx;
        state.begin_stroke(
            95,
            Tool::Paint,
            one_voxel_brush(),
            StrokeSpace::Slice(SliceAxis::Axial),
            PointerSample::Slice {
                axis: SliceAxis::Axial,
                voxel: [0, 0, 0],
            },
            CancellationToken::default(),
        );
        while event_rx.try_recv().is_ok() {}
        for index in 0..MESH_EVENT_BACKLOG_LIMIT {
            state
                .event_tx
                .try_send(EngineEvent::Error {
                    job_id: None,
                    message: format!("backlog {index}"),
                    source_changed: false,
                })
                .unwrap();
        }

        state.cancel_stroke(95);

        assert_eq!(state.pending_live_mesh_chunks.len(), 1);
        let before_flush = event_rx.try_iter().collect::<Vec<_>>();
        assert!(
            !before_flush
                .iter()
                .any(|event| matches!(event, EngineEvent::EditFinished { edit_id: 95, .. }))
        );
        state.flush_pending_mesh_chunk();
        let delivered = event_rx.try_iter().collect::<Vec<_>>();
        let mesh = delivered
            .iter()
            .position(|event| matches!(event, EngineEvent::MeshChunk { .. }))
            .unwrap();
        let acknowledgement = delivered
            .iter()
            .position(|event| matches!(event, EngineEvent::EditFinished { edit_id: 95, .. }))
            .unwrap();
        assert!(mesh < acknowledgement);
    }

    #[test]
    fn deterministic_recovery_error_terminates_with_a_visible_failure() {
        let mut state = state_with_empty_document([1, 1, 1]);
        let chunk = ChunkCoord::new(0, 0, 0);
        state.mesh_budget.try_replace(chunk, 1).unwrap();
        state.mesh_budget.reserve_current_if_absent(chunk).unwrap();
        state.mesh_recovery_chunks.insert(chunk);
        state.mesh_recovery_edit = Some(96);
        state.mesh_in_flight = 1;
        state.mesh_in_flight_chunks.insert(chunk);
        let tag = MeshJobTag {
            document_epoch: state.epoch,
            revision: state.document.as_ref().unwrap().revision(),
            chunk_generation: state.mesh_pass_generation,
            edit_generation: 1,
            chunk,
        };
        let (event_tx, event_rx) = bounded(16);
        state.event_tx = event_tx;

        state.handle_mesh_result((
            tag,
            MeshUpdatePriority::Live,
            Err(AppError::Allocation { bytes: 1 }),
        ));

        assert!(state.preview_too_complex);
        assert!(state.mesh_recovery_chunks.is_empty());
        assert!(state.mesh_recovery_edit.is_none());
        assert_eq!(state.mesh_budget.reserved_bytes(chunk), None);
        let events = event_rx.try_iter().collect::<Vec<_>>();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, EngineEvent::Error { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, EngineEvent::PreviewTooComplex { .. }))
        );
        assert!(events.iter().any(|event| {
            matches!(
                event,
                EngineEvent::EditFinished {
                    edit_id: 96,
                    outcome: EditOutcome::Cancelled,
                    ..
                }
            )
        }));
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
