use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crossbeam_channel::{Receiver, bounded};
use egui::{Color32, TextureHandle, TextureOptions, Vec2};

use crate::{
    EngineCommand, EngineEvent, EngineHandle,
    brush::{BrushDepth, BrushShape, BrushSize, BrushSpec},
    cancel::CancellationToken,
    codecs::{FormatFamily, SourceVariant},
    engine::{
        BrushPreviewRequest, BusyKind, EditId, JobId, PointerSample, SlicePatch, StrokeDisposition,
        StrokeSpace,
    },
    io::SaveMode,
    renderer::{SceneBridge, VolumePaintCallback, VoxelHighlightKind},
    volume::{DocumentSummary, SliceAxis, SliceImage, SpatialUnits, Tool},
};

const LOGO_BYTES: &[u8] = include_bytes!("../assets/voxel-ink.png");
const BACKGROUND: Color32 = Color32::BLACK;
const SURFACE: Color32 = Color32::from_gray(8);
const RAISED: Color32 = Color32::from_gray(24);
const LINE: Color32 = Color32::from_gray(64);
const TEXT: Color32 = Color32::from_gray(245);
const MUTED: Color32 = Color32::from_gray(166);
const TEAL: Color32 = Color32::from_rgb(36, 200, 219);
const YELLOW: Color32 = Color32::from_rgb(255, 193, 49);
const PAINT_GREEN: Color32 = Color32::from_rgb(0x42, 0xd6, 0x7c);
const ERASE_RED: Color32 = Color32::from_rgb(0xff, 0x5b, 0x45);
const BRIDGE_AMBER: Color32 = Color32::from_rgb(0xf4, 0xa6, 0x2a);
const CARD_RADIUS: u8 = 6;
const CONTROL_RADIUS: u8 = 4;
const MAX_ENGINE_EVENTS_PER_FRAME: usize = 256;
const SLICE_POINT_SCROLL_THRESHOLD: f32 = 24.0;
const REPAIR_PANEL_DEFAULT_WIDTH: f32 = 320.0;
const REPAIR_PANEL_MIN_WIDTH: f32 = 280.0;
const REPAIR_PANEL_MAX_WINDOW_FRACTION: f32 = 0.45;
const REPAIR_PANEL_ABSOLUTE_MAX_WIDTH: f32 = 440.0;
const WORKSPACE_GAP: f32 = 8.0;
const CONTROL_MIN_HEIGHT: f32 = 30.0;
const PANE_HEADER_CONTROL_INSET: f32 = 5.0;
const PANE_HEADER_HEIGHT: f32 = CONTROL_MIN_HEIGHT + 2.0 * PANE_HEADER_CONTROL_INSET;
const PANE_CONTENT_INSET: f32 = 6.0;
const WORKSPACE_MIN_PANE_WIDTH: f32 = 280.0;
const WORKSPACE_MIN_PANE_HEIGHT: f32 = 160.0;
const WORKSPACE_KEYBOARD_RESIZE_STEP: f32 = 16.0;
const SLICE_TEXTURE_OPTIONS: TextureOptions = TextureOptions {
    magnification: egui::TextureFilter::Nearest,
    minification: egui::TextureFilter::Linear,
    wrap_mode: egui::TextureWrapMode::ClampToEdge,
    mipmap_mode: None,
};

#[derive(Clone, Copy, Debug)]
enum GuardAction {
    OpenDialog,
    CloseDocument,
    Exit,
}

#[derive(Clone, Copy, Debug, Default)]
enum SavePromptKind {
    #[default]
    None,
    Unsaved(GuardAction),
    Normalized,
    SourceChanged,
}

#[derive(Clone, Debug)]
struct SaveRequest {
    destination: PathBuf,
    mode: SaveMode,
    after: Option<GuardAction>,
    restore_prompt: SavePromptKind,
}

#[derive(Clone, Debug)]
enum Prompt {
    Unsaved(GuardAction),
    Normalized(SaveRequest),
    SourceChanged(SaveRequest),
}

#[derive(Clone, Copy, Debug)]
enum DialogPurpose {
    Open,
    SaveAs,
    SaveCopy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptChoice {
    Save,
    Discard,
    Cancel,
    Overwrite,
    SaveCopy,
}

#[derive(Default)]
struct FrameStrokeActions {
    preview_hovered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PreviewAction {
    BridgeAnchor { plane_normal: [f64; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingPreview {
    generation: u64,
    request: BrushPreviewRequest,
    action: PreviewAction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DeferredBridgeRelease {
    sample: PointerSample,
    spec: BrushSpec,
    space: StrokeSpace,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BridgeAnchor {
    voxel: [u32; 3],
    space: StrokeSpace,
    plane_normal: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum UiStrokePhase {
    Open,
    Pending(StrokeDisposition),
    AwaitingAcknowledgement,
}

#[derive(Clone, Debug)]
struct UiStroke {
    edit_id: EditId,
    space: StrokeSpace,
    last_submitted: PointerSample,
    cancel: CancellationToken,
    phase: UiStrokePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BridgeCommitPhase {
    Pending,
    AwaitingAcknowledgement,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingBridgeCommit {
    edit_id: EditId,
    anchor: BridgeAnchor,
    release: DeferredBridgeRelease,
    phase: BridgeCommitPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkspacePane {
    Slice(SliceAxis),
    Volume3D,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceGrid {
    axial: egui::Rect,
    coronal: egui::Rect,
    sagittal: egui::Rect,
    volume: egui::Rect,
    column_splitter: egui::Rect,
    row_splitter: egui::Rect,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum BrushSizeUnit {
    #[default]
    Voxels,
    Millimeters,
}

#[derive(Clone, Copy, Debug)]
struct SliceViewState {
    zoom: f32,
    pan: Vec2,
    point_scroll_remainder: f32,
}

impl Default for SliceViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
            point_scroll_remainder: 0.0,
        }
    }
}

impl SliceViewState {
    fn wheel_steps(&mut self, events: &[egui::Event]) -> i32 {
        let mut steps = 0;
        for event in events {
            let egui::Event::MouseWheel {
                unit, delta, phase, ..
            } = event
            else {
                continue;
            };
            if matches!(phase, egui::TouchPhase::End | egui::TouchPhase::Cancel) {
                self.point_scroll_remainder = 0.0;
                continue;
            }
            let dominant = if delta.y.abs() >= delta.x.abs() {
                delta.y
            } else {
                delta.x
            };
            if dominant.abs() <= f32::EPSILON {
                continue;
            }
            match unit {
                egui::MouseWheelUnit::Line | egui::MouseWheelUnit::Page => {
                    self.point_scroll_remainder = 0.0;
                    steps += dominant.signum() as i32;
                }
                egui::MouseWheelUnit::Point => {
                    if self.point_scroll_remainder.signum() != dominant.signum()
                        && self.point_scroll_remainder.abs() > f32::EPSILON
                    {
                        self.point_scroll_remainder = 0.0;
                    }
                    self.point_scroll_remainder += dominant;
                    if self.point_scroll_remainder.abs() >= SLICE_POINT_SCROLL_THRESHOLD {
                        steps += self.point_scroll_remainder.signum() as i32;
                        self.point_scroll_remainder = 0.0;
                    }
                }
            }
        }
        steps
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn zoom_at(&mut self, zoom_delta: f32, pointer: egui::Pos2, fitted_center: egui::Pos2) {
        if !zoom_delta.is_finite() || zoom_delta <= 0.0 {
            return;
        }
        let previous_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_delta).clamp(0.25, 32.0);
        let scale = self.zoom / previous_zoom;
        let old_center = fitted_center + self.pan;
        let new_center = pointer - (pointer - old_center) * scale;
        self.pan = new_center - fitted_center;
    }
}

pub struct VoxelInkApp {
    egui_ctx: egui::Context,
    engine: EngineHandle,
    epoch: u64,
    summary: Option<DocumentSummary>,
    slices: HashMap<SliceAxis, SliceImage>,
    textures: HashMap<SliceAxis, TextureHandle>,
    slice_revision: u64,
    slice_preview_generation: u64,
    brush_preview_generation: u64,
    last_preview_request: Option<BrushPreviewRequest>,
    brush_preview_kind: Option<VoxelHighlightKind>,
    brush_preview_target: Option<[u32; 3]>,
    brush_preview_voxels: Vec<[u32; 3]>,
    pending_preview: Option<PendingPreview>,
    deferred_bridge_release: Option<DeferredBridgeRelease>,
    bridge_anchor: Option<BridgeAnchor>,
    logo: TextureHandle,
    scene: SceneBridge,
    target_format: wgpu::TextureFormat,
    adapter_name: String,
    tool: Tool,
    brush_shape: BrushShape,
    brush_depth: BrushDepth,
    brush_size_unit: BrushSizeUnit,
    brush_voxel_diameter: u8,
    brush_millimeters: f32,
    active_slice: SliceAxis,
    slice_views: HashMap<SliceAxis, SliceViewState>,
    busy: HashMap<JobId, BusyKind>,
    pending_saves: HashMap<JobId, SaveRequest>,
    prompt: Option<Prompt>,
    error_message: Option<String>,
    preview_notice: Option<String>,
    mesh_preview_failed: bool,
    dialog: Option<(DialogPurpose, Receiver<Option<PathBuf>>)>,
    dialog_copy_request: Option<SaveRequest>,
    allow_exit: bool,
    forced_colors: bool,
    reduced_motion: bool,
    stroke: Option<UiStroke>,
    bridge_commit: Option<PendingBridgeCommit>,
    next_edit_id: EditId,
    pane_keyboard_focus: bool,
    maximized_pane: Option<WorkspacePane>,
    workspace_column_ratio: f32,
    workspace_row_ratio: f32,
}

impl VoxelInkApp {
    #[must_use]
    pub fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        let forced_colors = env_flag("VOXEL_INK_FORCED_COLORS");
        let reduced_motion = env_flag("VOXEL_INK_REDUCED_MOTION");
        configure_style(&cc.egui_ctx, forced_colors, reduced_motion);
        let icon = eframe::icon_data::from_png_bytes(LOGO_BYTES)
            .expect("the embedded VoxelInk logo must be a valid PNG");
        let logo_image = egui::ColorImage::from_rgba_unmultiplied(
            [icon.width as usize, icon.height as usize],
            &icon.rgba,
        );
        let logo = cc
            .egui_ctx
            .load_texture("voxel-ink-logo", logo_image, TextureOptions::LINEAR);
        let (target_format, adapter_name) = cc.wgpu_render_state.as_ref().map_or_else(
            || {
                (
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                    "GPU unavailable in UI test harness".to_owned(),
                )
            },
            |state| {
                let info = state.adapter.get_info();
                tracing::info!(
                    adapter = %info.name,
                    backend = ?info.backend,
                    device_type = ?info.device_type,
                    driver = %info.driver,
                    "selected graphics adapter"
                );
                (
                    state.target_format,
                    format!("{} ({:?})", info.name, info.backend),
                )
            },
        );
        let engine = EngineHandle::spawn(cc.egui_ctx.clone());
        let scene = SceneBridge::default();
        scene.set_repaint_context(cc.egui_ctx.clone());
        let mut app = Self {
            egui_ctx: cc.egui_ctx.clone(),
            engine,
            epoch: 0,
            summary: None,
            slices: HashMap::new(),
            textures: HashMap::new(),
            slice_revision: 0,
            slice_preview_generation: 0,
            brush_preview_generation: 0,
            last_preview_request: None,
            brush_preview_kind: None,
            brush_preview_target: None,
            brush_preview_voxels: Vec::new(),
            pending_preview: None,
            deferred_bridge_release: None,
            bridge_anchor: None,
            logo,
            scene,
            target_format,
            adapter_name,
            tool: Tool::Inspect,
            brush_shape: BrushShape::Round,
            brush_depth: BrushDepth::Slice,
            brush_size_unit: BrushSizeUnit::Voxels,
            brush_voxel_diameter: 5,
            brush_millimeters: 3.0,
            active_slice: SliceAxis::Axial,
            slice_views: [SliceAxis::Axial, SliceAxis::Coronal, SliceAxis::Sagittal]
                .into_iter()
                .map(|axis| (axis, SliceViewState::default()))
                .collect(),
            busy: HashMap::new(),
            pending_saves: HashMap::new(),
            prompt: None,
            error_message: None,
            preview_notice: None,
            mesh_preview_failed: false,
            dialog: None,
            dialog_copy_request: None,
            allow_exit: false,
            forced_colors,
            reduced_motion,
            stroke: None,
            bridge_commit: None,
            next_edit_id: 1,
            pane_keyboard_focus: false,
            maximized_pane: None,
            workspace_column_ratio: 0.5,
            workspace_row_ratio: 0.5,
        };
        if let Some(path) = initial_path {
            match app.engine.start_load(path) {
                Ok(job_id) => {
                    app.busy.insert(job_id, BusyKind::Loading);
                }
                Err(error) => app.error_message = Some(error.to_string()),
            }
        }
        app
    }

    fn process_engine_events(&mut self, ctx: &egui::Context) {
        let mut exhausted_budget = true;
        for _ in 0..MAX_ENGINE_EVENTS_PER_FRAME {
            let Some(event) = self.engine.try_event() else {
                exhausted_budget = false;
                break;
            };
            match event {
                EngineEvent::Busy {
                    job_id,
                    kind,
                    active,
                } => {
                    if active {
                        self.busy.insert(job_id, kind);
                    } else {
                        self.busy.remove(&job_id);
                        if let Some(request) = self.pending_saves.remove(&job_id) {
                            self.restore_save_prompt(request);
                        }
                    }
                }
                EngineEvent::Error {
                    job_id,
                    message,
                    source_changed,
                } => {
                    let request = job_id.and_then(|id| self.pending_saves.remove(&id));
                    if source_changed {
                        if let Some(request) = request {
                            self.prompt = Some(Prompt::SourceChanged(SaveRequest {
                                restore_prompt: SavePromptKind::SourceChanged,
                                ..request
                            }));
                        }
                    } else if let Some(request) = request {
                        self.restore_save_prompt(request);
                    }
                    self.error_message = Some(message);
                }
                EngineEvent::EditFinished { epoch, edit_id, .. } if epoch == self.epoch => {
                    if self
                        .stroke
                        .as_ref()
                        .is_some_and(|stroke| stroke.edit_id == edit_id)
                    {
                        self.stroke = None;
                    }
                    if self
                        .bridge_commit
                        .is_some_and(|commit| commit.edit_id == edit_id)
                    {
                        self.bridge_commit = None;
                    }
                }
                EngineEvent::DocumentReplaced {
                    epoch,
                    summary,
                    slices,
                } => {
                    self.brush_size_unit = if summary.units == SpatialUnits::Unknown {
                        BrushSizeUnit::Voxels
                    } else {
                        BrushSizeUnit::Millimeters
                    };
                    self.epoch = epoch;
                    self.slice_revision = summary.revision;
                    self.slice_preview_generation = 0;
                    self.brush_preview_generation = 0;
                    self.last_preview_request = None;
                    self.brush_preview_kind = None;
                    self.brush_preview_target = None;
                    self.brush_preview_voxels.clear();
                    self.pending_preview = None;
                    self.deferred_bridge_release = None;
                    self.bridge_anchor = None;
                    self.summary = Some(summary);
                    self.stroke = None;
                    self.bridge_commit = None;
                    self.maximized_pane = None;
                    for view in self.slice_views.values_mut() {
                        view.reset();
                    }
                    self.replace_slices(ctx, slices);
                    self.preview_notice = None;
                    self.mesh_preview_failed = false;
                }
                EngineEvent::DocumentClosed { epoch } => {
                    self.epoch = epoch;
                    self.summary = None;
                    self.slices.clear();
                    self.textures.clear();
                    self.slice_revision = 0;
                    self.slice_preview_generation = 0;
                    self.brush_preview_generation = 0;
                    self.last_preview_request = None;
                    self.brush_preview_kind = None;
                    self.brush_preview_target = None;
                    self.brush_preview_voxels.clear();
                    self.pending_preview = None;
                    self.deferred_bridge_release = None;
                    self.bridge_anchor = None;
                    self.stroke = None;
                    self.bridge_commit = None;
                    self.maximized_pane = None;
                    for view in self.slice_views.values_mut() {
                        view.reset();
                    }
                    self.scene.clear();
                    self.preview_notice = None;
                    self.mesh_preview_failed = false;
                }
                EngineEvent::Summary { epoch, summary } if epoch == self.epoch => {
                    let revision_changed = self
                        .summary
                        .as_ref()
                        .is_some_and(|current| current.revision != summary.revision);
                    self.summary = Some(summary);
                    if revision_changed {
                        self.preview_notice = None;
                        self.brush_preview_generation =
                            self.brush_preview_generation.saturating_add(1);
                        self.engine.cancel_preview();
                        self.last_preview_request = None;
                        self.pending_preview = None;
                        self.deferred_bridge_release = None;
                        self.bridge_anchor = None;
                        self.brush_preview_target = None;
                        self.brush_preview_voxels.clear();
                        self.brush_preview_kind = None;
                        self.refresh_scene_brush_preview(epoch, self.summary_revision());
                    }
                }
                EngineEvent::SlicesReplaced {
                    epoch,
                    revision,
                    slices,
                } if epoch == self.epoch
                    && self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.revision == revision) =>
                {
                    self.slice_revision = revision;
                    self.slice_preview_generation = 0;
                    self.last_preview_request = None;
                    self.brush_preview_target = None;
                    self.brush_preview_voxels.clear();
                    self.brush_preview_kind = None;
                    self.refresh_scene_brush_preview(epoch, revision);
                    self.replace_slices(ctx, slices);
                }
                EngineEvent::SlicePatch {
                    epoch,
                    revision,
                    preview_generation,
                    patch,
                } if epoch == self.epoch
                    && self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.revision == revision)
                    && (revision != self.slice_revision
                        || preview_generation >= self.slice_preview_generation) =>
                {
                    if revision != self.slice_revision {
                        self.slice_revision = revision;
                        self.slice_preview_generation = 0;
                    }
                    self.slice_preview_generation = preview_generation;
                    self.apply_slice_patch(ctx, patch);
                }
                EngineEvent::BrushPreview {
                    epoch,
                    revision,
                    generation,
                    tool,
                    target,
                    voxels,
                } if epoch == self.epoch
                    && generation == self.brush_preview_generation
                    && self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.revision == revision) =>
                {
                    self.accept_brush_preview(epoch, revision, generation, tool, target, voxels);
                }
                EngineEvent::BrushPreviewFailed {
                    epoch,
                    revision,
                    generation,
                    message,
                } if epoch == self.epoch
                    && generation == self.brush_preview_generation
                    && self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.revision == revision) =>
                {
                    self.handle_brush_preview_failure(generation, message);
                }
                EngineEvent::Saved {
                    job_id,
                    mode: _,
                    summary,
                } => {
                    self.summary = Some(summary);
                    self.prompt = None;
                    if let Some(request) = self.pending_saves.remove(&job_id)
                        && let Some(action) = request.after
                    {
                        if request.mode == SaveMode::Copy {
                            self.prompt = Some(Prompt::Unsaved(action));
                        } else {
                            self.execute_guard(action);
                        }
                    }
                }
                EngineEvent::CavitiesFilled {
                    epoch,
                    revision: _,
                    added_voxels,
                } if epoch == self.epoch => {
                    self.preview_notice = Some(if added_voxels == 0 {
                        "No enclosed cavities found — the interior is already filled, or an opening still reaches the outside".into()
                    } else {
                        cavities_filled_notice(self.summary.as_ref(), added_voxels)
                    });
                }
                EngineEvent::MeshReset {
                    epoch,
                    revision,
                    dimensions,
                    index_to_world,
                } => {
                    self.mesh_preview_failed = false;
                    self.scene
                        .reset(epoch, revision, dimensions, index_to_world);
                }
                EngineEvent::MeshInvalidated {
                    epoch,
                    revision,
                    chunks,
                } => {
                    self.scene.invalidate(epoch, revision, chunks);
                }
                EngineEvent::MeshChunk {
                    tag,
                    quads,
                    priority,
                } => {
                    self.scene.insert_chunk(tag, quads, priority);
                }
                EngineEvent::PreviewTooComplex { epoch, .. } if epoch == self.epoch => {
                    self.mesh_preview_failed = true;
                }
                _ => {}
            }
        }
        if exhausted_budget {
            ctx.request_repaint();
        }
        if !self.has_edit_interaction()
            && self.scene.take_remesh_request()
            && self.summary.is_some()
        {
            let _ = self.send_command(EngineCommand::RebuildPreview);
        }
        if let Some(message) = self.scene.take_highlight_failure() {
            self.error_message = Some(message.to_string());
        }
    }

    fn process_dialog(&mut self) {
        let result = self
            .dialog
            .as_ref()
            .and_then(|(_, receiver)| receiver.try_recv().ok());
        let Some(path) = result else {
            return;
        };
        let purpose = self.dialog.take().map(|(purpose, _)| purpose);
        let Some(path) = path else {
            if matches!(purpose, Some(DialogPurpose::SaveCopy)) {
                self.dialog_copy_request = None;
            }
            return;
        };
        match purpose {
            Some(DialogPurpose::Open) => {
                self.cancel_edit_interaction();
                match self.engine.start_load(path) {
                    Ok(job_id) => {
                        self.busy.insert(job_id, BusyKind::Loading);
                    }
                    Err(error) => self.error_message = Some(error.to_string()),
                }
            }
            Some(DialogPurpose::SaveAs) => {
                let request = SaveRequest {
                    destination: path,
                    mode: SaveMode::SaveAs,
                    after: None,
                    restore_prompt: SavePromptKind::None,
                };
                if self
                    .summary
                    .as_ref()
                    .is_some_and(|summary| summary.normalized_non_binary)
                {
                    self.prompt = Some(Prompt::Normalized(SaveRequest {
                        restore_prompt: SavePromptKind::Normalized,
                        ..request
                    }));
                } else {
                    self.start_save(request);
                }
            }
            Some(DialogPurpose::SaveCopy) => {
                self.prompt = None;
                let original = self.dialog_copy_request.take();
                self.start_save(SaveRequest {
                    destination: path,
                    mode: SaveMode::Copy,
                    after: original.as_ref().and_then(|request| request.after),
                    restore_prompt: original
                        .map_or(SavePromptKind::None, |request| request.restore_prompt),
                });
            }
            None => {}
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.prompt.is_some()
            || self.error_message.is_some()
            || !self.busy.is_empty()
            || self.dialog.is_some()
        {
            return;
        }
        let has_edit_interaction = self.has_edit_interaction();
        if (has_edit_interaction || self.maximized_pane.is_some())
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            if has_edit_interaction {
                self.cancel_edit_interaction();
            } else {
                self.maximized_pane = None;
            }
            ctx.request_repaint();
        }
        let open = consume_shortcut(ctx, egui::Modifiers::CTRL, egui::Key::O);
        let save = consume_shortcut(ctx, egui::Modifiers::CTRL, egui::Key::S);
        let save_as = consume_shortcut(
            ctx,
            egui::Modifiers {
                ctrl: true,
                shift: true,
                ..Default::default()
            },
            egui::Key::S,
        );
        let close = consume_shortcut(ctx, egui::Modifiers::CTRL, egui::Key::W);
        let undo = consume_shortcut(ctx, egui::Modifiers::CTRL, egui::Key::Z);
        let redo_shift = consume_shortcut(
            ctx,
            egui::Modifiers {
                ctrl: true,
                shift: true,
                ..Default::default()
            },
            egui::Key::Z,
        );
        let redo_y = consume_shortcut(ctx, egui::Modifiers::CTRL, egui::Key::Y);
        let edit_commands_ready = !self.has_edit_interaction();
        if open && edit_commands_ready {
            self.guard_or_execute(GuardAction::OpenDialog);
        }
        if save_as && edit_commands_ready {
            self.open_save_dialog(DialogPurpose::SaveAs);
        } else if save && edit_commands_ready {
            self.request_overwrite(None);
        }
        if close && edit_commands_ready {
            self.guard_or_execute(GuardAction::CloseDocument);
        }
        if (redo_shift || redo_y) && edit_commands_ready {
            self.clear_brush_preview();
            let _ = self.send_command(EngineCommand::Redo);
        } else if undo && edit_commands_ready {
            self.clear_brush_preview();
            let _ = self.send_command(EngineCommand::Undo);
        }
        let previous_tool = self.tool;
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::I)) {
            self.tool = Tool::Inspect;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::B)) {
            self.tool = Tool::Paint;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::E)) {
            self.tool = Tool::Erase;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::L)) {
            self.tool = Tool::Bridge;
        }
        if self.tool != previous_tool {
            self.cancel_edit_interaction();
        }
        if !self.has_edit_interaction()
            && (self.pane_keyboard_focus || !ctx.egui_wants_keyboard_input())
        {
            self.handle_cursor_keys(ctx);
        }
    }

    fn handle_cursor_keys(&mut self, ctx: &egui::Context) {
        let Some(summary) = self.summary.as_ref() else {
            return;
        };
        let mut cursor = summary.cursor;
        let varying = match self.active_slice {
            SliceAxis::Axial => [0, 1],
            SliceAxis::Coronal => [0, 2],
            SliceAxis::Sagittal => [1, 2],
        };
        let normal = self.active_slice.normal_axis();
        let mut moved = false;
        for (key, axis, delta) in [
            (egui::Key::ArrowLeft, varying[0], -1_i32),
            (egui::Key::ArrowRight, varying[0], 1),
            (egui::Key::ArrowUp, varying[1], 1),
            (egui::Key::ArrowDown, varying[1], -1),
            (egui::Key::PageUp, normal, 1),
            (egui::Key::PageDown, normal, -1),
        ] {
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, key)) {
                cursor[axis] = if delta > 0 {
                    (cursor[axis] + 1).min(summary.dimensions[axis] - 1)
                } else {
                    cursor[axis].saturating_sub(1)
                };
                moved = true;
            }
        }
        if moved {
            let _ = self.send_command(EngineCommand::SetCursor(cursor));
        }
        let edit = ctx.input_mut(|input| {
            input.consume_key(egui::Modifiers::NONE, egui::Key::Space)
                || input.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
        });
        if edit && matches!(self.tool, Tool::Paint | Tool::Erase) {
            let space = StrokeSpace::Slice(self.active_slice);
            let sample = PointerSample::Slice {
                axis: self.active_slice,
                voxel: cursor,
            };
            if self.begin_stroke(self.tool, self.brush_spec(), space, sample) {
                let _ = self.complete_active_stroke(None);
            }
        }
    }

    fn toolbar(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        egui::Panel::top("voxel-ink-toolbar")
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .stroke(egui::Stroke::new(1.0, self.line_color()))
                    .inner_margin(egui::Margin::symmetric(10, 7)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Image::new(&self.logo)
                            .fit_to_exact_size(Vec2::splat(28.0))
                            .alt_text("VoxelInk teal and yellow logo"),
                    );
                    ui.label(egui::RichText::new("VoxelInk").size(18.0).strong());
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(2.0);
                    let edit_commands_ready = !self.has_edit_interaction();
                    if ui
                        .add_enabled(
                            self.busy.is_empty() && edit_commands_ready,
                            egui::Button::new("Open…"),
                        )
                        .on_hover_text("Open volume (Ctrl+O)")
                        .clicked()
                    {
                        self.guard_or_execute(GuardAction::OpenDialog);
                    }
                    let has_document = self.summary.is_some();
                    let ready = has_document && self.busy.is_empty() && edit_commands_ready;
                    if ui
                        .add_enabled(ready, egui::Button::new("Save"))
                        .on_hover_text("Save (Ctrl+S)")
                        .clicked()
                    {
                        self.request_overwrite(None);
                    }
                    if ui
                        .add_enabled(ready, egui::Button::new("Save As…"))
                        .on_hover_text("Save As (Ctrl+Shift+S)")
                        .clicked()
                    {
                        self.open_save_dialog(DialogPurpose::SaveAs);
                    }
                    if ui
                        .add_enabled(ready, egui::Button::new("Close"))
                        .on_hover_text("Close volume (Ctrl+W)")
                        .clicked()
                    {
                        self.guard_or_execute(GuardAction::CloseDocument);
                    }
                    ui.add_space(2.0);
                    ui.separator();
                    ui.add_space(2.0);
                    let can_undo = self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.can_undo);
                    let can_redo = self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.can_redo);
                    if ui
                        .add_enabled(ready && can_undo, egui::Button::new("Undo"))
                        .on_hover_text("Undo (Ctrl+Z)")
                        .clicked()
                    {
                        self.clear_brush_preview();
                        let _ = self.send_command(EngineCommand::Undo);
                    }
                    if ui
                        .add_enabled(ready && can_redo, egui::Button::new("Redo"))
                        .on_hover_text("Redo (Ctrl+Shift+Z or Ctrl+Y)")
                        .clicked()
                    {
                        self.clear_brush_preview();
                        let _ = self.send_command(EngineCommand::Redo);
                    }
                    if let Some(summary) = &self.summary {
                        ui.add_space(8.0);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let text = if summary.is_dirty() {
                                egui::RichText::new(format!(
                                    "{}  • modified",
                                    display_name(&summary.source_path)
                                ))
                                .color(YELLOW)
                            } else {
                                egui::RichText::new(display_name(&summary.source_path)).color(MUTED)
                            };
                            egui::Frame::new()
                                .fill(self.background_color())
                                .stroke(egui::Stroke::new(1.0, self.line_color()))
                                .corner_radius(CONTROL_RADIUS)
                                .inner_margin(egui::Margin::symmetric(9, 5))
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(text).truncate());
                                });
                        });
                    }
                });
            });
    }

    fn repair_panel(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        let maximum_width = (root.available_width() * REPAIR_PANEL_MAX_WINDOW_FRACTION)
            .clamp(REPAIR_PANEL_MIN_WIDTH, REPAIR_PANEL_ABSOLUTE_MAX_WIDTH);
        let panel = egui::Panel::left("voxel-ink-repair-panel")
            .default_size(REPAIR_PANEL_DEFAULT_WIDTH)
            .min_size(REPAIR_PANEL_MIN_WIDTH)
            .max_size(maximum_width)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(root, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("voxel-ink-repair-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let has_document = self.summary.is_some();
                        let ready = has_document
                            && self.busy.is_empty()
                            && !self.has_edit_interaction();
                        ui.heading("Repair volume");
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(
                                    "Choose the repair workflow that fits the volume.",
                                )
                                .color(MUTED),
                            )
                            .wrap(),
                        );
                        ui.add_space(12.0);

                        let card_width = ui.available_width();
                        let card = egui::Frame::new()
                            .fill(self.background_color())
                            .stroke(egui::Stroke::new(1.0, self.line_color()))
                            .corner_radius(CARD_RADIUS)
                            .inner_margin(egui::Margin::same(12));
                        card.show(ui, |ui| {
                            ui.set_width(
                                (card_width - card.total_margin().sum().x).max(1.0),
                            );
                            ui.strong("Seal openings");
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new("Paint missing voxels or bridge gaps.")
                                        .color(MUTED),
                                )
                                .wrap(),
                            );
                            ui.add_space(10.0);

                            let previous_tool = self.tool;
                            ui.add_enabled_ui(self.busy.is_empty(), |ui| {
                                ui.columns(2, |columns| {
                                    if selection_button(
                                        &mut columns[0],
                                        "Inspect (I)",
                                        self.tool == Tool::Inspect,
                                        TEAL,
                                        self.forced_colors,
                                    )
                                    .on_hover_text(
                                        "Move through slices and rotate or pan the 3D view",
                                    )
                                    .clicked()
                                    {
                                        self.tool = Tool::Inspect;
                                    }
                                    if selection_button(
                                        &mut columns[1],
                                        "Paint (B)",
                                        self.tool == Tool::Paint,
                                        PAINT_GREEN,
                                        self.forced_colors,
                                    )
                                    .on_hover_text("Fill the highlighted voxels")
                                    .clicked()
                                    {
                                        self.tool = Tool::Paint;
                                    }
                                });
                                ui.add_space(6.0);
                                ui.columns(2, |columns| {
                                    if selection_button(
                                        &mut columns[0],
                                        "Erase (E)",
                                        self.tool == Tool::Erase,
                                        ERASE_RED,
                                        self.forced_colors,
                                    )
                                    .on_hover_text("Remove the highlighted voxels")
                                    .clicked()
                                    {
                                        self.tool = Tool::Erase;
                                    }
                                    if selection_button(
                                        &mut columns[1],
                                        "Bridge (L)",
                                        self.tool == Tool::Bridge,
                                        BRIDGE_AMBER,
                                        self.forced_colors,
                                    )
                                    .on_hover_text(
                                        "Drag between two points to add one solid connection",
                                    )
                                    .clicked()
                                    {
                                        self.tool = Tool::Bridge;
                                    }
                                });
                            });
                            if self.tool != previous_tool {
                                self.cancel_edit_interaction();
                            }

                            if self.tool != Tool::Inspect {
                                ui.add_space(12.0);
                                ui.separator();
                                ui.add_space(8.0);
                                ui.add_enabled_ui(ready, |ui| {
                                    ui.label(egui::RichText::new("Shape").small().strong());
                                    ui.columns(2, |columns| {
                                        if selection_button(
                                            &mut columns[0],
                                            "Round",
                                            self.brush_shape == BrushShape::Round,
                                            TEAL,
                                            self.forced_colors,
                                        )
                                        .clicked()
                                        {
                                            self.brush_shape = BrushShape::Round;
                                        }
                                        if selection_button(
                                            &mut columns[1],
                                            "Square",
                                            self.brush_shape == BrushShape::Square,
                                            TEAL,
                                            self.forced_colors,
                                        )
                                        .clicked()
                                        {
                                            self.brush_shape = BrushShape::Square;
                                        }
                                    });
                                    ui.add_space(8.0);
                                    ui.label(egui::RichText::new("Depth").small().strong());
                                    ui.columns(2, |columns| {
                                        if selection_button(
                                            &mut columns[0],
                                            "One slice",
                                            self.brush_depth == BrushDepth::Slice,
                                            TEAL,
                                            self.forced_colors,
                                        )
                                        .clicked()
                                        {
                                            self.brush_depth = BrushDepth::Slice;
                                        }
                                        if selection_button(
                                            &mut columns[1],
                                            "3D volume",
                                            self.brush_depth == BrushDepth::Volume,
                                            TEAL,
                                            self.forced_colors,
                                        )
                                        .clicked()
                                        {
                                            self.brush_depth = BrushDepth::Volume;
                                        }
                                    });
                                    ui.label(
                                        egui::RichText::new(
                                            "Depth applies to slice views; 3D always uses a volume brush.",
                                        )
                                        .small()
                                        .color(MUTED),
                                    );
                                    ui.add_space(8.0);

                                    let has_physical_units = self
                                        .summary
                                        .as_ref()
                                        .is_some_and(|summary| summary.units != SpatialUnits::Unknown);
                                    if !has_physical_units {
                                        self.brush_size_unit = BrushSizeUnit::Voxels;
                                    }
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("Size").small().strong());
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.add_enabled_ui(has_physical_units, |ui| {
                                                    if compact_selection_button(
                                                        ui,
                                                        "mm",
                                                        self.brush_size_unit
                                                            == BrushSizeUnit::Millimeters,
                                                        self.forced_colors,
                                                    )
                                                    .clicked()
                                                    {
                                                        self.brush_size_unit =
                                                            BrushSizeUnit::Millimeters;
                                                    }
                                                });
                                                if compact_selection_button(
                                                    ui,
                                                    "voxels",
                                                    self.brush_size_unit == BrushSizeUnit::Voxels,
                                                    self.forced_colors,
                                                )
                                                .clicked()
                                                {
                                                    self.brush_size_unit = BrushSizeUnit::Voxels;
                                                }
                                            },
                                        );
                                    });
                                    ui.spacing_mut().slider_width =
                                        (ui.available_width() - 62.0).max(80.0);
                                    match self.brush_size_unit {
                                        BrushSizeUnit::Voxels => {
                                            let label = ui.add(
                                                egui::Label::new("Brush diameter in voxels").wrap(),
                                            );
                                            let changed = ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.brush_voxel_diameter,
                                                        BrushSize::MIN_VOXELS
                                                            ..=BrushSize::MAX_VOXELS,
                                                    )
                                                    .step_by(2.0)
                                                    .suffix(" vox"),
                                                )
                                                .labelled_by(label.id)
                                                .changed();
                                            if changed
                                                && self.brush_voxel_diameter.is_multiple_of(2)
                                            {
                                                self.brush_voxel_diameter = self
                                                    .brush_voxel_diameter
                                                    .saturating_add(1)
                                                    .min(BrushSize::MAX_VOXELS);
                                            }
                                        }
                                        BrushSizeUnit::Millimeters => {
                                            let label = ui.add(
                                                egui::Label::new(
                                                    "Brush diameter in millimetres",
                                                )
                                                .wrap(),
                                            );
                                            ui.add(
                                                egui::Slider::new(
                                                    &mut self.brush_millimeters,
                                                    BrushSize::MIN_MILLIMETERS
                                                        ..=BrushSize::MAX_MILLIMETERS,
                                                )
                                                .step_by(0.5)
                                                .suffix(" mm"),
                                            )
                                            .labelled_by(label.id);
                                        }
                                    }
                                });
                            }
                        });

                        ui.add_space(10.0);
                        let card_width = ui.available_width();
                        card.show(ui, |ui| {
                            ui.set_width(
                                (card_width - card.total_margin().sum().x).max(1.0),
                            );
                            ui.strong("Fill interior");
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(
                                        "Fill empty regions that cannot reach a volume boundary.",
                                    )
                                    .color(MUTED),
                                )
                                .wrap(),
                            );
                            ui.add_space(10.0);
                            let action = egui::Button::new(
                                egui::RichText::new("Fill enclosed cavities").strong(),
                            )
                            .min_size(egui::vec2(ui.available_width(), 38.0))
                            .corner_radius(CONTROL_RADIUS);
                            let action = if ready && !self.forced_colors {
                                action
                                    .fill(TEAL)
                                    .stroke(egui::Stroke::new(1.0, TEAL))
                            } else {
                                action
                            };
                            if ui
                                .add_enabled(ready, action)
                                .on_hover_text(
                                    "Fill every empty 3D region that cannot reach a volume boundary. The result is one Undo step.",
                                )
                                .clicked()
                            {
                                self.preview_notice = None;
                                self.clear_brush_preview();
                                if let Err(error) = self.engine.start_fill_enclosed_cavities() {
                                    self.error_message = Some(error.to_string());
                                }
                            }
                        });
                    });
            });
        panel.response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Panel, true, "Repair volume controls")
        });
    }

    fn status_bar(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        egui::Panel::bottom("voxel-ink-status")
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .stroke(egui::Stroke::new(1.0, self.line_color()))
                    .inner_margin(egui::Margin::symmetric(8, 6)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    if let Some(summary) = &self.summary {
                        status_chip(
                            ui,
                            "Size",
                            format!(
                                "{} × {} × {}",
                                summary.dimensions[0], summary.dimensions[1], summary.dimensions[2]
                            ),
                            self.forced_colors,
                        );
                        status_chip(
                            ui,
                            "Solid",
                            format!("{} occupied", format_count(summary.occupied_count)),
                            self.forced_colors,
                        );
                        status_chip(
                            ui,
                            "Cursor",
                            format!(
                                "cursor {}, {}, {}",
                                summary.cursor[0], summary.cursor[1], summary.cursor[2]
                            ),
                            self.forced_colors,
                        );
                        if summary.normalized_non_binary {
                            ui.colored_label(YELLOW, "non-binary source normalized; save required");
                        }
                    } else {
                        ui.label("Open a NIfTI, BINVOX, or VOX volume to begin");
                    }
                    if let Some(notice) = &self.preview_notice {
                        ui.colored_label(YELLOW, notice);
                    }
                    if self.mesh_preview_failed {
                        ui.colored_label(YELLOW, "3D preview paused at its memory limit");
                        if ui
                            .add_enabled(
                                !self.has_edit_interaction(),
                                egui::Button::new("Retry 3D preview"),
                            )
                            .clicked()
                            && self.send_command(EngineCommand::RebuildPreview)
                        {
                            self.mesh_preview_failed = false;
                        }
                    }
                    if let Some((&job_id, kind)) = self.busy.iter().next() {
                        if self.reduced_motion {
                            ui.label("Working");
                        } else {
                            ui.spinner();
                        }
                        ui.label(match kind {
                            BusyKind::Loading => "Loading…",
                            BusyKind::Saving => "Saving…",
                            BusyKind::FillingCavities => "Finding enclosed cavities…",
                        });
                        if ui.button("Cancel").clicked() {
                            self.engine.cancel_job(job_id);
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&self.adapter_name).color(MUTED).small(),
                            )
                            .truncate(),
                        );
                    });
                });
            });
    }

    fn workspace(&mut self, root: &mut egui::Ui) {
        self.pane_keyboard_focus = false;
        let background = self.background_color();
        let dialog_open = self.dialog.is_some();
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(background)
                    .inner_margin(egui::Margin::same(6)),
            )
            .show(root, |ui| {
                if dialog_open {
                    ui.disable();
                }
                let rect = ui.max_rect();
                let mut actions = FrameStrokeActions::default();
                match self.maximized_pane {
                    Some(WorkspacePane::Slice(axis)) => {
                        ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                            self.slice_pane(ui, axis, &mut actions);
                        });
                    }
                    Some(WorkspacePane::Volume3D) => {
                        ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                            self.volume_pane(ui, &mut actions);
                        });
                    }
                    None => {
                        let grid = self.resizable_workspace_grid(ui, rect);
                        for (axis, pane_rect) in [
                            (SliceAxis::Axial, grid.axial),
                            (SliceAxis::Coronal, grid.coronal),
                            (SliceAxis::Sagittal, grid.sagittal),
                        ] {
                            ui.scope_builder(egui::UiBuilder::new().max_rect(pane_rect), |ui| {
                                self.slice_pane(ui, axis, &mut actions);
                            });
                        }
                        ui.scope_builder(egui::UiBuilder::new().max_rect(grid.volume), |ui| {
                            self.volume_pane(ui, &mut actions);
                        });
                    }
                }
                if !actions.preview_hovered
                    && self.stroke.is_none()
                    && self.bridge_anchor.is_none()
                    && self.pending_preview.is_none()
                {
                    self.clear_brush_preview();
                }
            });
    }

    fn resizable_workspace_grid(
        &mut self,
        ui: &mut egui::Ui,
        workspace: egui::Rect,
    ) -> WorkspaceGrid {
        let mut grid = workspace_grid(
            workspace,
            self.workspace_column_ratio,
            self.workspace_row_ratio,
        );

        let row_response = ui
            .interact(
                grid.row_splitter,
                ui.id().with("workspace-row-splitter"),
                egui::Sense::drag(),
            )
            .on_hover_and_drag_cursor(egui::CursorIcon::ResizeVertical);
        row_response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::ResizeHandle,
                true,
                "Resize workspace rows",
            )
        });
        if row_response.dragged()
            && let Some(pointer) = row_response.interact_pointer_pos()
        {
            self.workspace_row_ratio = split_ratio_at_pointer(
                pointer.y,
                workspace.top(),
                workspace.height(),
                WORKSPACE_MIN_PANE_HEIGHT,
            );
        }
        if row_response.has_focus() {
            let (pixel_delta, reset) = ui.input_mut(|input| {
                let up = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp);
                let down = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown);
                let reset = input.consume_key(egui::Modifiers::NONE, egui::Key::Home);
                (
                    if up {
                        -WORKSPACE_KEYBOARD_RESIZE_STEP
                    } else if down {
                        WORKSPACE_KEYBOARD_RESIZE_STEP
                    } else {
                        0.0
                    },
                    reset,
                )
            });
            if reset {
                self.workspace_row_ratio = 0.5;
            } else if pixel_delta != 0.0 {
                self.workspace_row_ratio = split_ratio_after_delta(
                    workspace.height(),
                    self.workspace_row_ratio,
                    WORKSPACE_MIN_PANE_HEIGHT,
                    pixel_delta,
                );
            }
        }
        grid = workspace_grid(
            workspace,
            self.workspace_column_ratio,
            self.workspace_row_ratio,
        );

        let column_response = ui
            .interact(
                grid.column_splitter,
                ui.id().with("workspace-column-splitter"),
                egui::Sense::drag(),
            )
            .on_hover_and_drag_cursor(egui::CursorIcon::ResizeHorizontal);
        column_response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::ResizeHandle,
                true,
                "Resize workspace columns",
            )
        });
        if column_response.dragged()
            && let Some(pointer) = column_response.interact_pointer_pos()
        {
            self.workspace_column_ratio = split_ratio_at_pointer(
                pointer.x,
                workspace.left(),
                workspace.width(),
                WORKSPACE_MIN_PANE_WIDTH,
            );
        }
        if column_response.has_focus() {
            let (pixel_delta, reset) = ui.input_mut(|input| {
                let left = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft);
                let right = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight);
                let reset = input.consume_key(egui::Modifiers::NONE, egui::Key::Home);
                (
                    if left {
                        -WORKSPACE_KEYBOARD_RESIZE_STEP
                    } else if right {
                        WORKSPACE_KEYBOARD_RESIZE_STEP
                    } else {
                        0.0
                    },
                    reset,
                )
            });
            if reset {
                self.workspace_column_ratio = 0.5;
            } else if pixel_delta != 0.0 {
                self.workspace_column_ratio = split_ratio_after_delta(
                    workspace.width(),
                    self.workspace_column_ratio,
                    WORKSPACE_MIN_PANE_WIDTH,
                    pixel_delta,
                );
            }
        }
        grid = workspace_grid(
            workspace,
            self.workspace_column_ratio,
            self.workspace_row_ratio,
        );

        let splitter_color = |response: &egui::Response| {
            if response.hovered() || response.dragged() || response.has_focus() {
                if self.forced_colors {
                    Color32::WHITE
                } else {
                    TEAL
                }
            } else {
                self.line_color()
            }
        };
        ui.painter().rect_filled(
            egui::Rect::from_center_size(
                grid.row_splitter.center(),
                egui::vec2(grid.row_splitter.width(), 2.0),
            ),
            0.0,
            splitter_color(&row_response),
        );
        ui.painter().rect_filled(
            egui::Rect::from_center_size(
                grid.column_splitter.center(),
                egui::vec2(2.0, grid.column_splitter.height()),
            ),
            0.0,
            splitter_color(&column_response),
        );
        if row_response.dragged()
            || column_response.dragged()
            || row_response.has_focus()
            || column_response.has_focus()
        {
            ui.ctx().request_repaint();
        }
        grid
    }

    fn slice_pane(&mut self, ui: &mut egui::Ui, axis: SliceAxis, actions: &mut FrameStrokeActions) {
        let pane_rect = ui.max_rect();
        ui.painter()
            .rect_filled(pane_rect, CARD_RADIUS, self.background_color());
        let header_rect = egui::Rect::from_min_max(
            pane_rect.min,
            egui::pos2(pane_rect.right(), pane_rect.top() + PANE_HEADER_HEIGHT),
        );
        let header_controls = header_rect.shrink(PANE_HEADER_CONTROL_INSET);
        ui.painter()
            .rect_filled(header_rect, CARD_RADIUS, self.background_color());
        let title = match axis {
            SliceAxis::Axial => "Axial",
            SliceAxis::Coronal => "Coronal",
            SliceAxis::Sagittal => "Sagittal",
        };
        let coordinate = self.slices.get(&axis).map_or_else(
            || {
                self.summary
                    .as_ref()
                    .map_or(0, |summary| summary.cursor[axis.normal_axis()])
            },
            |slice| slice.coordinate,
        );
        let last_coordinate = self.summary.as_ref().map_or(0, |summary| {
            summary.dimensions[axis.normal_axis()].saturating_sub(1)
        });
        ui.painter().text(
            egui::pos2(pane_rect.left() + 9.0, header_rect.center().y),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(14.0),
            TEXT,
        );
        ui.painter().text(
            egui::pos2(pane_rect.left() + 72.0, header_rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{coordinate} / {last_coordinate}"),
            egui::FontId::proportional(12.0),
            MUTED,
        );
        let pane = WorkspacePane::Slice(axis);
        let maximized = self.maximized_pane == Some(pane);
        let maximize_label = if maximized {
            "Restore".to_owned()
        } else {
            "Expand".to_owned()
        };
        let button_width = 68.0;
        let button_rect = egui::Rect::from_min_size(
            egui::pos2(
                header_controls.right() - button_width,
                header_controls.top(),
            ),
            egui::vec2(button_width, header_controls.height()),
        );
        let maximize_response = ui
            .put(button_rect, egui::Button::new(maximize_label))
            .on_hover_text(if maximized {
                "Return to the four-view layout (Esc)"
            } else {
                "Use the full workspace for this view"
            });
        maximize_response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Button,
                true,
                if maximized {
                    "Restore four views".to_owned()
                } else {
                    format!("Full screen {title}")
                },
            )
        });
        if maximize_response.clicked() {
            self.active_slice = axis;
            self.maximized_pane = if maximized { None } else { Some(pane) };
            ui.ctx().request_repaint();
        }
        let fit_button_rect = egui::Rect::from_min_size(
            egui::pos2(button_rect.left() - 47.0, header_controls.top()),
            egui::vec2(42.0, header_controls.height()),
        );
        if ui
            .put(fit_button_rect, egui::Button::new("Fit"))
            .on_hover_text("Fit the complete slice in this pane")
            .clicked()
            && let Some(view) = self.slice_views.get_mut(&axis)
        {
            view.reset();
        }
        let coordinate_rect = egui::Rect::from_min_size(
            egui::pos2(fit_button_rect.left() - 75.0, header_controls.top()),
            egui::vec2(70.0, header_controls.height()),
        );
        let mut requested_coordinate = coordinate;
        if ui
            .put(
                coordinate_rect,
                egui::DragValue::new(&mut requested_coordinate)
                    .range(0..=last_coordinate)
                    .speed(1.0)
                    .prefix("slice "),
            )
            .on_hover_text("Enter or drag to choose an exact slice")
            .changed()
            && requested_coordinate != coordinate
            && let Some(summary) = &self.summary
        {
            let mut cursor = summary.cursor;
            cursor[axis.normal_axis()] = requested_coordinate;
            self.active_slice = axis;
            let _ = self.send_command(EngineCommand::SetCursor(cursor));
        }
        let content = egui::Rect::from_min_max(
            egui::pos2(
                pane_rect.left() + PANE_CONTENT_INSET,
                header_rect.bottom() + PANE_CONTENT_INSET,
            ),
            pane_rect.max - egui::vec2(PANE_CONTENT_INSET, PANE_CONTENT_INSET),
        );
        let Some(slice) = self.slices.get(&axis) else {
            ui.painter().text(
                content.center(),
                egui::Align2::CENTER_CENTER,
                "No volume",
                egui::FontId::proportional(14.0),
                MUTED,
            );
            return;
        };
        let Some(texture_id) = self.textures.get(&axis).map(TextureHandle::id) else {
            return;
        };
        let width = slice.width;
        let height = slice.height;
        let physical_aspect = self.summary.as_ref().map_or_else(
            || width as f32 / height as f32,
            |summary| slice_physical_aspect(axis, width, height, summary.spacing),
        );
        let fitted_rect = fit_rect(content, physical_aspect);
        let mut view = self.slice_views.get(&axis).copied().unwrap_or_default();
        let image_rect = egui::Rect::from_center_size(
            fitted_rect.center() + view.pan,
            fitted_rect.size() * view.zoom,
        );
        let content_painter = ui.painter().with_clip_rect(content);
        content_painter.image(
            texture_id,
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        if let Some(kind) = self.brush_preview_kind {
            paint_brush_preview(
                &content_painter,
                image_rect,
                axis,
                coordinate,
                width,
                height,
                &self.brush_preview_voxels,
                kind,
                self.forced_colors,
            );
        }
        if let Some(anchor) = self.bridge_anchor {
            paint_brush_preview(
                &content_painter,
                image_rect,
                axis,
                coordinate,
                width,
                height,
                std::slice::from_ref(&anchor.voxel),
                VoxelHighlightKind::BridgeAnchor,
                self.forced_colors,
            );
        }
        paint_interaction_hint(
            ui.painter(),
            content,
            if content.width() < 400.0 {
                "Wheel slice · Ctrl+wheel zoom · Drag move"
            } else {
                "Wheel: slice · Ctrl+wheel: zoom · Middle/right drag: move"
            },
            self.forced_colors,
        );
        let interaction_rect = image_rect.intersect(content);
        let response = ui.interact(
            interaction_rect,
            ui.id().with(("slice", axis)),
            egui::Sense::click_and_drag(),
        );
        response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Image,
                true,
                format!("{title} slice editor"),
            )
        });
        if response.has_focus() {
            self.pane_keyboard_focus = true;
            ui.painter().rect_stroke(
                image_rect,
                CONTROL_RADIUS,
                egui::Stroke::new(
                    2.0,
                    if self.forced_colors {
                        Color32::WHITE
                    } else {
                        YELLOW
                    },
                ),
                egui::StrokeKind::Inside,
            );
        }
        if response.clicked() {
            response.request_focus();
            self.active_slice = axis;
        }
        if response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
        {
            view.pan += response.drag_delta();
        }
        if response.hovered()
            && self.busy.is_empty()
            && self.stroke.is_none()
            && self.bridge_commit.is_none()
        {
            let control = ui.input(|input| input.modifiers.ctrl);
            if control {
                let zoom_delta = ui.input(|input| input.zoom_delta());
                if (zoom_delta - 1.0).abs() > f32::EPSILON
                    && let Some(pointer) = response.hover_pos()
                {
                    view.zoom_at(zoom_delta, pointer, fitted_rect.center());
                }
                view.point_scroll_remainder = 0.0;
            } else {
                let steps = ui.input(|input| view.wheel_steps(&input.events));
                if steps != 0
                    && let Some(summary) = &self.summary
                {
                    let mut cursor = summary.cursor;
                    let normal = axis.normal_axis();
                    cursor[normal] = step_slice_coordinate(
                        cursor[normal],
                        summary.dimensions[normal] - 1,
                        steps,
                    );
                    self.active_slice = axis;
                    let _ = self.send_command(EngineCommand::SetCursor(cursor));
                }
            }
        } else if !response.hovered() {
            view.point_scroll_remainder = 0.0;
        }
        self.slice_views.insert(axis, view);
        if let Some(summary) = &self.summary {
            paint_crosshair(
                &content_painter,
                image_rect,
                axis,
                summary.cursor,
                width,
                height,
            );
        }
        let displayed_cursor = self.summary.as_ref().map(|summary| {
            let mut cursor = summary.cursor;
            cursor[axis.normal_axis()] = coordinate;
            cursor
        });
        let pointer_to_voxel = |position| {
            rect_contains_half_open(interaction_rect, position)
                .then_some(position)
                .and_then(|position| {
                    displayed_cursor.and_then(|cursor| {
                        slice_pointer_to_voxel(axis, position, image_rect, width, height, cursor)
                    })
                })
        };
        let hovered_voxel = persistent_hover_position(&response).and_then(pointer_to_voxel);
        let interaction_voxel = response.interact_pointer_pos().and_then(pointer_to_voxel);
        let (primary_press_position, primary_release_position) =
            ui.input(|input| primary_pointer_event_positions(&input.events));
        let pressed_voxel = primary_press_position.and_then(pointer_to_voxel);
        let released_voxel = primary_release_position.and_then(pointer_to_voxel);
        let primary_pressed = pressed_voxel.is_some();
        let primary_released = primary_release_position.is_some();
        if self.edit_input_blocked() {
            return;
        }
        let space = StrokeSpace::Slice(axis);
        if self.tool != Tool::Inspect
            && response.hovered()
            && self.stroke.is_none()
            && self.bridge_commit.is_none()
            && let Some(voxel) = hovered_voxel
        {
            actions.preview_hovered = true;
            let sample = PointerSample::Slice { axis, voxel };
            let request = match self.bridge_anchor {
                Some(anchor) if self.tool == Tool::Bridge && anchor.space == space => {
                    BrushPreviewRequest::Bridge {
                        generation: 0,
                        spec: self.brush_spec(),
                        space,
                        anchor: anchor.voxel,
                        plane_normal: anchor.plane_normal,
                        sample,
                    }
                }
                _ => BrushPreviewRequest::Brush {
                    generation: 0,
                    tool: self.tool,
                    spec: self.brush_spec(),
                    space,
                    sample,
                },
            };
            self.request_brush_preview(request);
        }
        if matches!(self.tool, Tool::Paint | Tool::Erase) {
            if primary_pressed && let Some(voxel) = pressed_voxel {
                self.active_slice = axis;
                let _ = self.begin_stroke(
                    self.tool,
                    self.brush_spec(),
                    space,
                    PointerSample::Slice { axis, voxel },
                );
            } else if response.dragged_by(egui::PointerButton::Primary)
                && let Some(edit_id) = self.open_stroke_id(space)
                && let Some(voxel) = interaction_voxel
                && let Err(error) =
                    self.send_stroke_sample(edit_id, PointerSample::Slice { axis, voxel })
            {
                self.error_message = Some(error.to_string());
                self.cancel_edit_interaction();
            }
            if primary_released && self.open_stroke_id(space).is_some() {
                self.finish_stroke_pointer_release(
                    released_voxel.map(|voxel| PointerSample::Slice { axis, voxel }),
                );
            }
        } else if self.tool == Tool::Bridge && self.stroke.is_none() && self.bridge_commit.is_none()
        {
            if primary_pressed && let Some(voxel) = pressed_voxel {
                self.active_slice = axis;
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: Tool::Bridge,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Slice { axis, voxel },
                    },
                    PreviewAction::BridgeAnchor {
                        plane_normal: [0.0; 3],
                    },
                );
            }
            let bridge_interaction_matches = self
                .bridge_anchor
                .is_some_and(|anchor| anchor.space == space)
                || self.pending_preview.is_some_and(|pending| {
                    matches!(
                        pending.request,
                        BrushPreviewRequest::Brush {
                            space: pending_space,
                            ..
                        } if pending_space == space
                    )
                });
            if primary_released && bridge_interaction_matches {
                if let Some(voxel) = released_voxel {
                    let release = DeferredBridgeRelease {
                        sample: PointerSample::Slice { axis, voxel },
                        spec: self.brush_spec(),
                        space,
                    };
                    if self.bridge_anchor.is_some() {
                        self.request_bridge_commit(release);
                    } else if self.pending_preview.is_some_and(|pending| {
                        matches!(pending.action, PreviewAction::BridgeAnchor { .. })
                    }) {
                        self.deferred_bridge_release = Some(release);
                    }
                } else {
                    self.cancel_edit_interaction();
                }
            }
        } else if response.clicked()
            && let Some(voxel) = interaction_voxel
        {
            let _ = self.send_command(EngineCommand::SetCursor(voxel));
        }
    }

    fn volume_pane(&mut self, ui: &mut egui::Ui, actions: &mut FrameStrokeActions) {
        let pane_rect = ui.max_rect();
        ui.painter()
            .rect_filled(pane_rect, CARD_RADIUS, self.background_color());
        let header_rect = egui::Rect::from_min_max(
            pane_rect.min,
            egui::pos2(pane_rect.right(), pane_rect.top() + PANE_HEADER_HEIGHT),
        );
        let header_controls = header_rect.shrink(PANE_HEADER_CONTROL_INSET);
        ui.painter()
            .rect_filled(header_rect, CARD_RADIUS, self.background_color());
        ui.painter().text(
            egui::pos2(pane_rect.left() + 9.0, header_rect.center().y),
            egui::Align2::LEFT_CENTER,
            "3D View",
            egui::FontId::proportional(14.0),
            TEXT,
        );
        let pane = WorkspacePane::Volume3D;
        let maximized = self.maximized_pane == Some(pane);
        let maximize_label = if maximized { "Restore" } else { "Expand" };
        let maximize_width = 68.0;
        let maximize_rect = egui::Rect::from_min_size(
            egui::pos2(
                header_controls.right() - maximize_width,
                header_controls.top(),
            ),
            egui::vec2(maximize_width, header_controls.height()),
        );
        let maximize_response = ui
            .put(maximize_rect, egui::Button::new(maximize_label))
            .on_hover_text(if maximized {
                "Return to the four-view layout (Esc)"
            } else {
                "Use the full workspace for the 3D view"
            });
        maximize_response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Button,
                true,
                if maximized {
                    "Restore four views"
                } else {
                    "Full screen 3D"
                },
            )
        });
        if maximize_response.clicked() {
            self.maximized_pane = if maximized { None } else { Some(pane) };
            ui.ctx().request_repaint();
        }
        let reset_width = 58.0;
        let reset_rect = egui::Rect::from_min_size(
            egui::pos2(
                maximize_rect.left() - reset_width - 5.0,
                header_controls.top(),
            ),
            egui::vec2(reset_width, header_controls.height()),
        );
        let reset_response = ui
            .put(reset_rect, egui::Button::new("Reset"))
            .on_hover_text("Center the volume and restore the default rotation and zoom");
        reset_response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Reset view")
        });
        if reset_response.clicked() {
            self.scene.reset_view();
        }
        let content = egui::Rect::from_min_max(
            egui::pos2(
                pane_rect.left() + PANE_CONTENT_INSET,
                header_rect.bottom() + PANE_CONTENT_INSET,
            ),
            pane_rect.max - egui::vec2(PANE_CONTENT_INSET, PANE_CONTENT_INSET),
        );
        ui.painter()
            .rect_filled(content, CONTROL_RADIUS, self.background_color());
        self.scene
            .set_aspect(content.width() / content.height().max(1.0));
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            content,
            VolumePaintCallback::new(self.scene.clone(), self.target_format),
        ));
        let response = ui.interact(
            content,
            ui.id().with("3d-volume-editor"),
            egui::Sense::click_and_drag(),
        );
        response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Other,
                true,
                "Interactive 3D volume editor",
            )
        });
        let controls = match self.tool {
            Tool::Inspect => "Left / middle drag: rotate   Right drag: move   Wheel: zoom",
            Tool::Bridge => {
                "Left drag: bridge   Middle drag: rotate   Right drag: move   Wheel: zoom"
            }
            Tool::Paint | Tool::Erase => {
                "Left drag: edit   Middle drag: rotate   Right drag: move   Wheel: zoom"
            }
        };
        let compact_controls = match self.tool {
            Tool::Inspect => "Rotate · Move · Zoom",
            Tool::Bridge => "Bridge · Rotate · Move · Zoom",
            Tool::Paint | Tool::Erase => "Edit · Rotate · Move · Zoom",
        };
        paint_interaction_hint(
            ui.painter(),
            content,
            if content.width() < 400.0 {
                compact_controls
            } else {
                controls
            },
            self.forced_colors,
        );
        if response.has_focus() {
            self.pane_keyboard_focus = true;
            ui.painter().rect_stroke(
                content,
                CONTROL_RADIUS,
                egui::Stroke::new(
                    2.0,
                    if self.forced_colors {
                        Color32::WHITE
                    } else {
                        YELLOW
                    },
                ),
                egui::StrokeKind::Inside,
            );
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                self.scene.zoom((scroll * 0.002).exp());
            }
        }
        if response.dragged_by(egui::PointerButton::Secondary) {
            self.scene.pan(response.drag_delta(), content.size());
        } else if response.dragged_by(egui::PointerButton::Middle)
            || (self.tool == Tool::Inspect && response.dragged_by(egui::PointerButton::Primary))
        {
            self.scene.orbit(response.drag_delta());
        }
        if self.tool == Tool::Inspect || self.summary.is_none() {
            return;
        }
        if self.edit_input_blocked() {
            return;
        }
        let pointer_to_ray = |position| {
            rect_contains_half_open(content, position).then(|| {
                let normalized = [
                    ((position.x - content.left()) / content.width()) * 2.0 - 1.0,
                    1.0 - ((position.y - content.top()) / content.height()) * 2.0,
                ];
                self.scene.ray_for_pointer(normalized)
            })
        };
        let hovered_ray = persistent_hover_position(&response).and_then(pointer_to_ray);
        let interaction_ray = response.interact_pointer_pos().and_then(pointer_to_ray);
        let (primary_press_position, primary_release_position) =
            ui.input(|input| primary_pointer_event_positions(&input.events));
        let pressed_ray = primary_press_position.and_then(pointer_to_ray);
        let released_ray = primary_release_position.and_then(pointer_to_ray);
        let primary_pressed = pressed_ray.is_some();
        let primary_released = primary_release_position.is_some();
        let space = StrokeSpace::Volume3D;
        if (response.hovered() || response.dragged_by(egui::PointerButton::Primary))
            && self.stroke.is_none()
            && self.bridge_commit.is_none()
            && let Some(ray) = hovered_ray
        {
            actions.preview_hovered = true;
            let sample = PointerSample::Ray { ray };
            let request = match self.bridge_anchor {
                Some(anchor) if self.tool == Tool::Bridge && anchor.space == space => {
                    BrushPreviewRequest::Bridge {
                        generation: 0,
                        spec: self.brush_spec(),
                        space,
                        anchor: anchor.voxel,
                        plane_normal: anchor.plane_normal,
                        sample,
                    }
                }
                _ => BrushPreviewRequest::Brush {
                    generation: 0,
                    tool: self.tool,
                    spec: self.brush_spec(),
                    space,
                    sample,
                },
            };
            self.request_brush_preview(request);
        }
        if matches!(self.tool, Tool::Paint | Tool::Erase) {
            if primary_pressed && let Some(ray) = pressed_ray {
                let _ = self.begin_stroke(
                    self.tool,
                    self.brush_spec(),
                    space,
                    PointerSample::Ray { ray },
                );
            } else if response.dragged_by(egui::PointerButton::Primary)
                && let Some(edit_id) = self.open_stroke_id(space)
                && let Some(ray) = interaction_ray
                && let Err(error) = self.send_stroke_sample(edit_id, PointerSample::Ray { ray })
            {
                self.error_message = Some(error.to_string());
                self.cancel_edit_interaction();
            }
            if primary_released && self.open_stroke_id(space).is_some() {
                self.finish_stroke_pointer_release(
                    released_ray.map(|ray| PointerSample::Ray { ray }),
                );
            }
        } else if self.tool == Tool::Bridge && self.stroke.is_none() && self.bridge_commit.is_none()
        {
            if primary_pressed && let Some(ray) = pressed_ray {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: Tool::Bridge,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Ray { ray },
                    },
                    PreviewAction::BridgeAnchor {
                        plane_normal: ray.direction,
                    },
                );
            }
            let bridge_interaction_matches = self
                .bridge_anchor
                .is_some_and(|anchor| anchor.space == space)
                || self.pending_preview.is_some_and(|pending| {
                    matches!(
                        pending.request,
                        BrushPreviewRequest::Brush {
                            space: pending_space,
                            ..
                        } if pending_space == space
                    )
                });
            if primary_released && bridge_interaction_matches {
                if let Some(ray) = released_ray {
                    let release = DeferredBridgeRelease {
                        sample: PointerSample::Ray { ray },
                        spec: self.brush_spec(),
                        space,
                    };
                    if self.bridge_anchor.is_some() {
                        self.request_bridge_commit(release);
                    } else if self.pending_preview.is_some_and(|pending| {
                        matches!(pending.action, PreviewAction::BridgeAnchor { .. })
                    }) {
                        self.deferred_bridge_release = Some(release);
                    }
                } else {
                    self.cancel_edit_interaction();
                }
            }
        }
    }

    fn show_prompts(&mut self, ctx: &egui::Context) {
        if self.dialog.is_some() {
            return;
        }
        if let Some(message) = self.error_message.clone() {
            let response = egui::Modal::new(egui::Id::new("voxel-ink-error")).show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.heading("VoxelInk could not complete the operation");
                ui.add_space(8.0);
                ui.label(message);
                ui.add_space(12.0);
                ui.horizontal(|ui| ui.button("OK").clicked()).inner
            });
            if response.inner || response.should_close() {
                self.error_message = None;
            }
            return;
        }
        let Some(prompt) = self.prompt.clone() else {
            return;
        };
        let response = egui::Modal::new(egui::Id::new("voxel-ink-prompt")).show(ctx, |ui| {
            ui.set_min_width(390.0);
            match &prompt {
                Prompt::Unsaved(_) => {
                    ui.heading("Save changes?");
                    ui.label("This volume has unsaved changes.");
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            Some(PromptChoice::Save)
                        } else if ui.button("Discard").clicked() {
                            Some(PromptChoice::Discard)
                        } else if ui.button("Cancel").clicked() {
                            Some(PromptChoice::Cancel)
                        } else {
                            None
                        }
                    })
                    .inner
                }
                Prompt::Normalized(request) => {
                    ui.heading("Normalized non-binary source");
                    ui.label(if request.mode == SaveMode::SaveAs {
                        "The saved file will contain canonical binary occupancy."
                    } else {
                        "Saving will replace the source values with canonical binary occupancy."
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Save a Copy…").clicked() {
                            Some(PromptChoice::SaveCopy)
                        } else if ui
                            .button(if request.mode == SaveMode::SaveAs {
                                "Save"
                            } else {
                                "Overwrite"
                            })
                            .clicked()
                        {
                            Some(PromptChoice::Overwrite)
                        } else if ui.button("Cancel").clicked() {
                            Some(PromptChoice::Cancel)
                        } else {
                            None
                        }
                    })
                    .inner
                }
                Prompt::SourceChanged(_) => {
                    ui.heading("Source changed outside VoxelInk");
                    ui.label("VoxelInk will not overwrite a changed, replaced, or deleted source.");
                    ui.horizontal(|ui| {
                        if ui.button("Save a Copy…").clicked() {
                            Some(PromptChoice::SaveCopy)
                        } else if ui.button("Cancel").clicked() {
                            Some(PromptChoice::Cancel)
                        } else {
                            None
                        }
                    })
                    .inner
                }
            }
        });
        let choice = response
            .inner
            .or_else(|| response.should_close().then_some(PromptChoice::Cancel));
        if let Some(choice) = choice {
            self.apply_prompt_choice(prompt, choice);
        }
    }

    fn apply_prompt_choice(&mut self, prompt: Prompt, choice: PromptChoice) {
        match (prompt, choice) {
            (Prompt::Unsaved(action), PromptChoice::Save) => {
                self.request_overwrite(Some(action));
            }
            (Prompt::Unsaved(action), PromptChoice::Discard) => {
                self.prompt = None;
                self.execute_guard(action);
            }
            (Prompt::Normalized(request), PromptChoice::Overwrite) => {
                self.prompt = None;
                self.start_save(request);
            }
            (Prompt::Normalized(request), PromptChoice::SaveCopy)
            | (Prompt::SourceChanged(request), PromptChoice::SaveCopy) => {
                self.dialog_copy_request = Some(request);
                self.open_save_dialog(DialogPurpose::SaveCopy);
            }
            (Prompt::SourceChanged(request), PromptChoice::Cancel)
            | (Prompt::Normalized(request), PromptChoice::Cancel) => {
                self.prompt = request.after.map(Prompt::Unsaved);
            }
            (_, PromptChoice::Cancel) => self.prompt = None,
            _ => {}
        }
    }

    fn guard_or_execute(&mut self, action: GuardAction) {
        if self.has_edit_interaction() {
            return;
        }
        if self.summary.as_ref().is_some_and(DocumentSummary::is_dirty) {
            self.prompt = Some(Prompt::Unsaved(action));
        } else {
            self.execute_guard(action);
        }
    }

    fn execute_guard(&mut self, action: GuardAction) {
        self.cancel_edit_interaction();
        match action {
            GuardAction::OpenDialog => self.open_file_dialog(),
            GuardAction::CloseDocument => {
                let _ = self.send_command(EngineCommand::Close);
            }
            GuardAction::Exit => {
                self.allow_exit = true;
                self.egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn request_overwrite(&mut self, after: Option<GuardAction>) {
        if self.has_edit_interaction() {
            return;
        }
        let Some(summary) = &self.summary else {
            return;
        };
        let request = SaveRequest {
            destination: summary.source_path.clone(),
            mode: SaveMode::Overwrite,
            after,
            restore_prompt: after.map_or(SavePromptKind::None, SavePromptKind::Unsaved),
        };
        if summary.normalized_non_binary {
            self.prompt = Some(Prompt::Normalized(SaveRequest {
                restore_prompt: SavePromptKind::Normalized,
                ..request
            }));
        } else {
            self.prompt = None;
            self.start_save(request);
        }
    }

    fn send_command(&mut self, command: EngineCommand) -> bool {
        match self.engine.send(command) {
            Ok(()) => true,
            Err(error) => {
                self.error_message = Some(error.to_string());
                false
            }
        }
    }

    fn brush_spec(&self) -> BrushSpec {
        let size = match self.brush_size_unit {
            BrushSizeUnit::Voxels => {
                BrushSize::voxels(self.brush_voxel_diameter).unwrap_or_default()
            }
            BrushSizeUnit::Millimeters => {
                BrushSize::millimeters(self.brush_millimeters).unwrap_or_default()
            }
        };
        BrushSpec {
            shape: self.brush_shape,
            depth: self.brush_depth,
            size,
        }
    }

    fn request_brush_preview(&mut self, request: BrushPreviewRequest) {
        if self.pending_preview.is_some() || self.edit_input_blocked() {
            return;
        }
        let template = preview_with_generation(request, 0);
        if self.last_preview_request == Some(template) {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        let request = preview_with_generation(request, self.brush_preview_generation);
        match self.engine.send_latest_preview(request) {
            Ok(()) => self.last_preview_request = Some(template),
            Err(error) => {
                self.last_preview_request = None;
                self.discard_current_brush_preview();
                self.error_message = Some(error.to_string());
            }
        }
    }

    fn request_action_preview(&mut self, request: BrushPreviewRequest, action: PreviewAction) {
        if self.pending_preview.is_some() || self.edit_input_blocked() {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        if self.brush_preview_kind.is_some()
            || self.brush_preview_target.is_some()
            || !self.brush_preview_voxels.is_empty()
        {
            self.discard_current_brush_preview();
        }
        let request = preview_with_generation(request, self.brush_preview_generation);
        match self.engine.send_latest_preview(request) {
            Ok(()) => {
                self.last_preview_request = Some(preview_with_generation(request, 0));
                self.pending_preview = Some(PendingPreview {
                    generation: self.brush_preview_generation,
                    request,
                    action,
                });
            }
            Err(error) => {
                self.last_preview_request = None;
                self.pending_preview = None;
                self.error_message = Some(error.to_string());
            }
        }
    }

    fn accept_brush_preview(
        &mut self,
        epoch: u64,
        revision: u64,
        generation: u64,
        tool: Tool,
        target: Option<[u32; 3]>,
        voxels: Vec<[u32; 3]>,
    ) {
        let Some(pending) = self
            .pending_preview
            .take()
            .filter(|pending| pending.generation == generation)
        else {
            self.brush_preview_kind = Some(match tool {
                Tool::Paint => VoxelHighlightKind::Paint,
                Tool::Erase => VoxelHighlightKind::Erase,
                Tool::Bridge if self.bridge_anchor.is_some() => VoxelHighlightKind::BridgePath,
                Tool::Bridge => VoxelHighlightKind::BridgeAnchor,
                Tool::Inspect => return,
            });
            self.brush_preview_target = target;
            self.brush_preview_voxels = voxels;
            self.refresh_scene_brush_preview(epoch, revision);
            return;
        };
        self.last_preview_request = None;
        self.brush_preview_kind = None;
        self.brush_preview_target = None;
        self.brush_preview_voxels.clear();
        if let (
            PreviewAction::BridgeAnchor { plane_normal },
            BrushPreviewRequest::Brush { space, .. },
        ) = (pending.action, pending.request)
        {
            if let Some(voxel) = target {
                self.bridge_anchor = Some(BridgeAnchor {
                    voxel,
                    space,
                    plane_normal,
                });
                if let Some(release) = self.deferred_bridge_release.take()
                    && release.space == space
                {
                    self.request_bridge_commit(release);
                }
            } else {
                self.deferred_bridge_release = None;
            }
        }
        self.refresh_scene_brush_preview(epoch, revision);
    }

    fn handle_brush_preview_failure(&mut self, generation: u64, message: String) {
        let pending_matches = self
            .pending_preview
            .is_some_and(|pending| pending.generation == generation);
        if !pending_matches && self.pending_preview.is_some() {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.last_preview_request = None;
        self.pending_preview = None;
        self.deferred_bridge_release = None;
        self.bridge_anchor = None;
        self.engine.cancel_preview();
        self.discard_current_brush_preview();
        if self.stroke.is_some() {
            let _ = self.complete_active_stroke(None);
        }
        self.error_message = Some(message);
    }

    fn discard_current_brush_preview(&mut self) {
        self.brush_preview_kind = None;
        self.brush_preview_target = None;
        self.brush_preview_voxels.clear();
        let revision = self.summary.as_ref().map_or(0, |summary| summary.revision);
        self.refresh_scene_brush_preview(self.epoch, revision);
        self.egui_ctx.request_repaint();
    }

    fn refresh_scene_brush_preview(&self, epoch: u64, revision: u64) {
        let mut highlights = Vec::new();
        if let Some(kind) = self.brush_preview_kind {
            highlights.extend(
                self.brush_preview_voxels
                    .iter()
                    .copied()
                    .map(|voxel| (voxel, kind)),
            );
        }
        if let Some(anchor) = self.bridge_anchor {
            highlights.push((anchor.voxel, VoxelHighlightKind::BridgeAnchor));
        }
        if highlights.is_empty() {
            self.scene.clear_voxel_highlight();
            return;
        }
        let _ = self.scene.set_voxel_highlight(epoch, revision, highlights);
    }

    fn request_bridge_commit(&mut self, release: DeferredBridgeRelease) {
        if self.edit_input_blocked() {
            self.cancel_edit_interaction();
            return;
        }
        let Some(anchor) = self.bridge_anchor else {
            self.deferred_bridge_release = Some(release);
            return;
        };
        if self.bridge_commit.is_some() || self.stroke.is_some() {
            return;
        }
        let edit_id = self.allocate_edit_id();
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.engine.cancel_preview();
        self.last_preview_request = None;
        self.pending_preview = None;
        self.deferred_bridge_release = None;
        self.bridge_anchor = None;
        self.discard_current_brush_preview();
        self.bridge_commit = Some(PendingBridgeCommit {
            edit_id,
            anchor,
            release,
            phase: BridgeCommitPhase::Pending,
        });
        self.retry_pending_bridge_commit();
    }

    fn clear_brush_preview(&mut self) {
        if self.last_preview_request.is_none()
            && self.brush_preview_kind.is_none()
            && self.brush_preview_voxels.is_empty()
            && self.pending_preview.is_none()
            && self.deferred_bridge_release.is_none()
        {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.engine.cancel_preview();
        self.last_preview_request = None;
        self.pending_preview = None;
        self.deferred_bridge_release = None;
        self.discard_current_brush_preview();
    }

    fn cancel_edit_interaction(&mut self) {
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.engine.cancel_preview();
        self.last_preview_request = None;
        self.pending_preview = None;
        self.deferred_bridge_release = None;
        self.bridge_anchor = None;
        if let Some(commit) = self.bridge_commit
            && commit.phase == BridgeCommitPhase::Pending
        {
            self.bridge_commit = None;
        }
        if let Some(stroke) = self.stroke.as_mut()
            && stroke.phase != UiStrokePhase::AwaitingAcknowledgement
        {
            stroke.cancel.cancel();
            stroke.phase = UiStrokePhase::Pending(StrokeDisposition::Cancel);
            self.retry_pending_stroke_terminal();
        }
        self.discard_current_brush_preview();
    }

    fn finish_stroke_pointer_release(&mut self, final_sample: Option<PointerSample>) {
        let _ = self.complete_active_stroke(final_sample);
    }

    fn has_edit_interaction(&self) -> bool {
        self.pending_preview.is_some()
            || self.deferred_bridge_release.is_some()
            || self.bridge_anchor.is_some()
            || self.stroke.is_some()
            || self.bridge_commit.is_some()
    }

    fn edit_input_blocked(&self) -> bool {
        !self.busy.is_empty()
            || self.dialog.is_some()
            || self.prompt.is_some()
            || self.error_message.is_some()
            || self.allow_exit
            || !self.egui_ctx.input(|input| input.focused)
    }

    fn open_stroke_id(&self, space: StrokeSpace) -> Option<EditId> {
        self.stroke.as_ref().and_then(|stroke| {
            (stroke.space == space && stroke.phase == UiStrokePhase::Open).then_some(stroke.edit_id)
        })
    }

    fn summary_revision(&self) -> u64 {
        self.summary.as_ref().map_or(0, |summary| summary.revision)
    }

    fn begin_stroke(
        &mut self,
        tool: Tool,
        spec: BrushSpec,
        space: StrokeSpace,
        initial_sample: PointerSample,
    ) -> bool {
        if self.stroke.is_some() || self.bridge_commit.is_some() || self.edit_input_blocked() {
            return false;
        }
        self.clear_brush_preview();
        let edit_id = self.allocate_edit_id();
        let cancel = CancellationToken::default();
        if self.send_command(EngineCommand::BeginStroke {
            edit_id,
            tool,
            spec,
            space,
            initial_sample,
            cancel: cancel.clone(),
        }) {
            self.stroke = Some(UiStroke {
                edit_id,
                space,
                last_submitted: initial_sample,
                cancel,
                phase: UiStrokePhase::Open,
            });
            true
        } else {
            false
        }
    }

    fn complete_active_stroke(&mut self, final_sample: Option<PointerSample>) -> bool {
        let Some(stroke) = self.stroke.as_mut() else {
            return true;
        };
        match stroke.phase {
            UiStrokePhase::Open => {
                stroke.phase = UiStrokePhase::Pending(StrokeDisposition::Complete {
                    final_sample: final_sample.or(Some(stroke.last_submitted)),
                });
            }
            UiStrokePhase::Pending(StrokeDisposition::Complete { .. }) => {}
            UiStrokePhase::Pending(StrokeDisposition::Cancel) => return false,
            UiStrokePhase::AwaitingAcknowledgement => return true,
        }
        self.retry_pending_stroke_terminal()
    }

    fn retry_pending_stroke_terminal(&mut self) -> bool {
        let Some((edit_id, disposition)) = self.stroke.as_ref().and_then(|stroke| {
            if let UiStrokePhase::Pending(disposition) = stroke.phase {
                Some((stroke.edit_id, disposition))
            } else {
                None
            }
        }) else {
            return true;
        };
        match self.engine.send(EngineCommand::EndStroke {
            edit_id,
            disposition,
        }) {
            Ok(()) => {
                if let Some(stroke) = self
                    .stroke
                    .as_mut()
                    .filter(|stroke| stroke.edit_id == edit_id)
                {
                    stroke.phase = UiStrokePhase::AwaitingAcknowledgement;
                }
                true
            }
            Err(error) => {
                self.error_message = Some(error.to_string());
                self.egui_ctx.request_repaint();
                false
            }
        }
    }

    fn send_stroke_sample(&mut self, edit_id: EditId, sample: PointerSample) -> crate::Result<()> {
        self.engine.send_latest_sample(edit_id, sample)?;
        if let Some(stroke) = self
            .stroke
            .as_mut()
            .filter(|stroke| stroke.edit_id == edit_id)
        {
            stroke.last_submitted = sample;
        }
        Ok(())
    }

    fn retry_pending_bridge_commit(&mut self) {
        let Some(commit) = self
            .bridge_commit
            .filter(|commit| commit.phase == BridgeCommitPhase::Pending)
        else {
            return;
        };
        match self.engine.send(EngineCommand::CommitBridge {
            edit_id: commit.edit_id,
            anchor: commit.anchor.voxel,
            plane_normal: commit.anchor.plane_normal,
            sample: commit.release.sample,
            spec: commit.release.spec,
            space: commit.release.space,
        }) {
            Ok(()) => {
                if let Some(queued) = self
                    .bridge_commit
                    .as_mut()
                    .filter(|queued| queued.edit_id == commit.edit_id)
                {
                    queued.phase = BridgeCommitPhase::AwaitingAcknowledgement;
                }
            }
            Err(error) => {
                self.error_message = Some(error.to_string());
                self.egui_ctx.request_repaint();
            }
        }
    }

    fn retry_pending_edit_commands(&mut self) {
        let _ = self.retry_pending_stroke_terminal();
        self.retry_pending_bridge_commit();
    }

    fn allocate_edit_id(&mut self) -> EditId {
        let edit_id = self.next_edit_id;
        self.next_edit_id = self.next_edit_id.wrapping_add(1).max(1);
        edit_id
    }

    const fn background_color(&self) -> Color32 {
        if self.forced_colors {
            Color32::BLACK
        } else {
            BACKGROUND
        }
    }

    const fn surface_color(&self) -> Color32 {
        if self.forced_colors {
            Color32::BLACK
        } else {
            SURFACE
        }
    }

    const fn line_color(&self) -> Color32 {
        if self.forced_colors {
            Color32::WHITE
        } else {
            LINE
        }
    }

    fn start_save(&mut self, request: SaveRequest) {
        match self
            .engine
            .start_save(request.destination.clone(), request.mode)
        {
            Ok(job_id) => {
                self.pending_saves.insert(job_id, request);
            }
            Err(error) => {
                self.restore_save_prompt(request);
                self.error_message = Some(error.to_string());
            }
        }
    }

    fn restore_save_prompt(&mut self, request: SaveRequest) {
        match request.restore_prompt {
            SavePromptKind::None => {}
            SavePromptKind::Unsaved(action) => self.prompt = Some(Prompt::Unsaved(action)),
            SavePromptKind::Normalized => self.prompt = Some(Prompt::Normalized(request)),
            SavePromptKind::SourceChanged => self.prompt = Some(Prompt::SourceChanged(request)),
        }
    }

    fn open_file_dialog(&mut self) {
        if self.dialog.is_some() || self.has_edit_interaction() || !self.busy.is_empty() {
            return;
        }
        let (sender, receiver) = bounded(1);
        let repaint = self.egui_ctx.clone();
        std::thread::spawn(move || {
            let result = rfd::FileDialog::new()
                .set_title("Open volume")
                .add_filter("Supported volumes", &["nii", "gz", "binvox", "vox"])
                .add_filter("NIfTI", &["nii", "gz"])
                .add_filter("BINVOX", &["binvox"])
                .add_filter("MagicaVoxel", &["vox"])
                .pick_file();
            let _ = sender.send(result);
            repaint.request_repaint();
        });
        self.dialog = Some((DialogPurpose::Open, receiver));
    }

    fn open_save_dialog(&mut self, purpose: DialogPurpose) {
        if self.has_edit_interaction() {
            return;
        }
        if self.dialog.is_some() || self.summary.is_none() {
            return;
        }
        let summary = self.summary.as_ref().expect("checked above").clone();
        let (sender, receiver) = bounded(1);
        let repaint = self.egui_ctx.clone();
        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new()
                .set_title(match purpose {
                    DialogPurpose::SaveAs => "Save volume as",
                    DialogPurpose::SaveCopy => "Save a copy",
                    DialogPurpose::Open => "Open volume",
                })
                .set_file_name(suggested_save_name(&summary, purpose));
            dialog = match summary.source_variant.family() {
                FormatFamily::Nifti => dialog.add_filter("NIfTI", &["nii", "gz"]),
                FormatFamily::Binvox => dialog.add_filter("BINVOX", &["binvox"]),
                FormatFamily::Vox => dialog.add_filter("MagicaVoxel", &["vox"]),
            };
            let _ = sender.send(dialog.save_file());
            repaint.request_repaint();
        });
        self.dialog = Some((purpose, receiver));
    }

    fn replace_slices(&mut self, ctx: &egui::Context, slices: Vec<SliceImage>) {
        for slice in slices {
            let axis = slice.axis;
            self.slices.insert(axis, slice);
            self.refresh_texture(ctx, axis);
        }
    }

    fn apply_slice_patch(&mut self, ctx: &egui::Context, patch: SlicePatch) {
        let Some(pixel_count) = (patch.width as usize).checked_mul(patch.height as usize) else {
            return;
        };
        {
            let Some(slice) = self.slices.get_mut(&patch.axis) else {
                return;
            };
            if slice.coordinate != patch.coordinate
                || patch
                    .x
                    .checked_add(patch.width)
                    .is_none_or(|right| right > slice.width)
                || patch
                    .y
                    .checked_add(patch.height)
                    .is_none_or(|bottom| bottom > slice.height)
                || patch.pixels.len() != pixel_count
            {
                return;
            }
            for row in 0..patch.height {
                let source = row as usize * patch.width as usize;
                let destination =
                    (patch.y + row) as usize * slice.width as usize + patch.x as usize;
                slice.pixels[destination..destination + patch.width as usize]
                    .copy_from_slice(&patch.pixels[source..source + patch.width as usize]);
            }
        }
        let filled = if self.forced_colors {
            Color32::WHITE
        } else {
            TEAL
        };
        let empty = if self.forced_colors {
            Color32::BLACK
        } else {
            BACKGROUND
        };
        let mut pixels = Vec::new();
        if pixels.try_reserve_exact(pixel_count).is_err() {
            self.error_message = Some(format!(
                "not enough memory to allocate {pixel_count} pixels for a slice texture patch"
            ));
            return;
        }
        pixels.extend(
            patch
                .pixels
                .iter()
                .map(|value| if *value == 0 { empty } else { filled }),
        );
        if let Some(texture) = self.textures.get_mut(&patch.axis) {
            texture.set_partial(
                [patch.x as usize, patch.y as usize],
                egui::ColorImage::new([patch.width as usize, patch.height as usize], pixels),
                SLICE_TEXTURE_OPTIONS,
            );
        } else {
            self.refresh_texture(ctx, patch.axis);
        }
    }

    fn refresh_texture(&mut self, ctx: &egui::Context, axis: SliceAxis) {
        let Some(slice) = self.slices.get(&axis) else {
            return;
        };
        let filled = if self.forced_colors {
            Color32::WHITE
        } else {
            TEAL
        };
        let empty = if self.forced_colors {
            Color32::BLACK
        } else {
            BACKGROUND
        };
        let pixel_count = match (slice.width as usize).checked_mul(slice.height as usize) {
            Some(pixel_count) => pixel_count,
            None => {
                self.error_message = Some("slice texture dimensions overflow memory limits".into());
                return;
            }
        };
        let mut pixels = Vec::new();
        if pixels.try_reserve_exact(pixel_count).is_err() {
            self.error_message = Some(format!(
                "not enough memory to allocate {pixel_count} pixels for a slice texture"
            ));
            return;
        }
        pixels.extend(
            slice
                .pixels
                .iter()
                .map(|value| if *value == 0 { empty } else { filled }),
        );
        let image = egui::ColorImage::new([slice.width as usize, slice.height as usize], pixels);
        if let Some(texture) = self.textures.get_mut(&axis) {
            texture.set(image, SLICE_TEXTURE_OPTIONS);
        } else {
            let texture = ctx.load_texture(
                format!("voxel-slice-{axis:?}"),
                image,
                SLICE_TEXTURE_OPTIONS,
            );
            self.textures.insert(axis, texture);
        }
    }
}

impl eframe::App for VoxelInkApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.process_engine_events(&ctx);
        let close_requested = ctx.input(|input| input.viewport().close_requested());
        let had_edit_interaction = self.has_edit_interaction();
        let escape_requested = had_edit_interaction
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        let focus_lost = had_edit_interaction && ctx.input(|input| !input.focused);
        if escape_requested || focus_lost || (close_requested && had_edit_interaction) {
            self.cancel_edit_interaction();
        }
        self.process_dialog();
        self.retry_pending_edit_commands();
        if close_requested {
            if self.allow_exit {
                return;
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if had_edit_interaction {
                self.preview_notice = Some("Current edit cancelled; close again to exit".into());
            } else if self.prompt.is_none() {
                self.guard_or_execute(GuardAction::Exit);
            }
        }
        if self.allow_exit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if self.dialog.is_some() {
            ui.disable();
        }
        self.toolbar(ui);
        self.status_bar(ui);
        self.repair_panel(ui);
        self.workspace(ui);
        self.handle_shortcuts(&ctx);
        self.show_prompts(&ctx);
    }
}

fn selection_button(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    accent: Color32,
    forced_colors: bool,
) -> egui::Response {
    let (fill, stroke, text) = if forced_colors {
        if selected {
            (Color32::WHITE, Color32::WHITE, Color32::BLACK)
        } else {
            (Color32::BLACK, Color32::WHITE, Color32::WHITE)
        }
    } else if selected {
        (
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 52),
            accent,
            TEXT,
        )
    } else {
        (RAISED, LINE, MUTED)
    };
    let width = ui.available_width();
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(text))
            .selected(selected)
            .fill(fill)
            .stroke(egui::Stroke::new(if selected { 1.5 } else { 1.0 }, stroke))
            .corner_radius(if forced_colors { 0 } else { CONTROL_RADIUS })
            .min_size(egui::vec2(width, 34.0)),
    )
}

fn compact_selection_button(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    forced_colors: bool,
) -> egui::Response {
    let (fill, stroke, text) = if forced_colors {
        if selected {
            (Color32::WHITE, Color32::WHITE, Color32::BLACK)
        } else {
            (Color32::BLACK, Color32::WHITE, Color32::WHITE)
        }
    } else if selected {
        (
            Color32::from_rgba_unmultiplied(TEAL.r(), TEAL.g(), TEAL.b(), 52),
            TEAL,
            TEXT,
        )
    } else {
        (RAISED, LINE, MUTED)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(label).small().color(text))
            .selected(selected)
            .fill(fill)
            .stroke(egui::Stroke::new(1.0, stroke))
            .corner_radius(if forced_colors { 0 } else { CONTROL_RADIUS })
            .min_size(egui::vec2(if label == "mm" { 42.0 } else { 58.0 }, 26.0)),
    )
}

fn status_chip(ui: &mut egui::Ui, label: &str, value: String, forced_colors: bool) {
    egui::Frame::new()
        .fill(if forced_colors {
            Color32::BLACK
        } else {
            BACKGROUND
        })
        .stroke(egui::Stroke::new(
            1.0,
            if forced_colors { Color32::WHITE } else { LINE },
        ))
        .corner_radius(if forced_colors { 0 } else { CONTROL_RADIUS })
        .inner_margin(egui::Margin::symmetric(7, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(label).small().color(if forced_colors {
                    Color32::WHITE
                } else {
                    MUTED
                }));
                ui.label(egui::RichText::new(value).small().strong());
            });
        });
}

fn configure_style(ctx: &egui::Context, forced_colors: bool, reduced_motion: bool) {
    ctx.set_theme(egui::Theme::Dark);
    let mut visuals = egui::Visuals::dark();
    let control_radius = if forced_colors {
        egui::CornerRadius::ZERO
    } else {
        egui::CornerRadius::same(CONTROL_RADIUS)
    };
    visuals.window_corner_radius = if forced_colors {
        egui::CornerRadius::ZERO
    } else {
        egui::CornerRadius::same(8)
    };
    visuals.menu_corner_radius = if forced_colors {
        egui::CornerRadius::ZERO
    } else {
        egui::CornerRadius::same(CARD_RADIUS)
    };
    visuals.widgets.noninteractive.corner_radius = control_radius;
    visuals.widgets.inactive.corner_radius = control_radius;
    visuals.widgets.hovered.corner_radius = control_radius;
    visuals.widgets.active.corner_radius = control_radius;
    visuals.widgets.open.corner_radius = control_radius;
    visuals.panel_fill = if forced_colors {
        Color32::BLACK
    } else {
        BACKGROUND
    };
    visuals.window_fill = if forced_colors {
        Color32::BLACK
    } else {
        SURFACE
    };
    visuals.extreme_bg_color = if forced_colors {
        Color32::BLACK
    } else {
        RAISED
    };
    visuals.override_text_color = Some(if forced_colors { Color32::WHITE } else { TEXT });
    visuals.selection.bg_fill = if forced_colors { Color32::WHITE } else { TEAL };
    visuals.selection.stroke = egui::Stroke::new(
        2.0,
        if forced_colors {
            Color32::BLACK
        } else {
            BACKGROUND
        },
    );
    visuals.widgets.inactive.bg_fill = if forced_colors {
        Color32::BLACK
    } else {
        RAISED
    };
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, LINE);
    visuals.widgets.hovered.bg_fill = if forced_colors {
        Color32::WHITE
    } else {
        Color32::from_gray(36)
    };
    visuals.widgets.hovered.fg_stroke =
        egui::Stroke::new(1.5, if forced_colors { Color32::BLACK } else { TEXT });
    visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(1.0, if forced_colors { Color32::WHITE } else { TEAL });
    visuals.widgets.active.bg_fill = if forced_colors { Color32::WHITE } else { TEAL };
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, MUTED);
    ctx.set_visuals(visuals);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.spacing.item_spacing = egui::vec2(8.0, 7.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);
        style.spacing.interact_size.y = CONTROL_MIN_HEIGHT;
        if reduced_motion {
            style.animation_time = 0.0;
        }
    });
}

fn cavities_filled_notice(summary: Option<&DocumentSummary>, added_voxels: u64) -> String {
    let count = format_count(added_voxels);
    let voxel_label = if added_voxels == 1 { "voxel" } else { "voxels" };
    let physical_volume = summary.and_then(|summary| {
        let voxel_volume = summary.spacing.into_iter().product::<f64>();
        let cubic_centimeters = match summary.units {
            SpatialUnits::Meter => voxel_volume * 1_000_000.0,
            SpatialUnits::Millimeter => voxel_volume / 1_000.0,
            SpatialUnits::Micron => voxel_volume / 1_000_000_000_000.0,
            SpatialUnits::Unknown => return None,
        } * added_voxels as f64;
        cubic_centimeters.is_finite().then_some(cubic_centimeters)
    });
    physical_volume.map_or_else(
        || format!("Filled {count} enclosed {voxel_label} · Undo restores them"),
        |volume| {
            let volume = if volume >= 0.1 {
                format!("{volume:.1}")
            } else {
                format!("{volume:.3}")
            };
            format!("Filled {count} enclosed {voxel_label} ({volume} cm³) · Undo restores them")
        },
    )
}

fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index != 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }
    formatted
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
}

fn consume_shortcut(ctx: &egui::Context, modifiers: egui::Modifiers, key: egui::Key) -> bool {
    ctx.input_mut(|input| input.consume_shortcut(&egui::KeyboardShortcut::new(modifiers, key)))
}

fn primary_pointer_event_positions(
    events: &[egui::Event],
) -> (Option<egui::Pos2>, Option<egui::Pos2>) {
    let mut pressed = None;
    let mut released = None;
    for event in events {
        let egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: is_pressed,
            ..
        } = event
        else {
            continue;
        };
        if *is_pressed {
            pressed.get_or_insert(*pos);
        } else {
            released = Some(*pos);
        }
    }
    (pressed, released)
}

fn preview_with_generation(request: BrushPreviewRequest, generation: u64) -> BrushPreviewRequest {
    match request {
        BrushPreviewRequest::Brush {
            tool,
            spec,
            space,
            sample,
            ..
        } => BrushPreviewRequest::Brush {
            generation,
            tool,
            spec,
            space,
            sample,
        },
        BrushPreviewRequest::Bridge {
            spec,
            space,
            anchor,
            plane_normal,
            sample,
            ..
        } => BrushPreviewRequest::Bridge {
            generation,
            spec,
            space,
            anchor,
            plane_normal,
            sample,
        },
    }
}

fn slice_physical_aspect(axis: SliceAxis, width: u32, height: u32, spacing: [f64; 3]) -> f32 {
    let [horizontal_axis, vertical_axis] = match axis {
        SliceAxis::Axial => [0, 1],
        SliceAxis::Coronal => [0, 2],
        SliceAxis::Sagittal => [1, 2],
    };
    let physical_width = f64::from(width) * spacing[horizontal_axis];
    let physical_height = f64::from(height) * spacing[vertical_axis];
    if physical_width.is_finite() && physical_height.is_finite() && physical_height > 0.0 {
        (physical_width / physical_height) as f32
    } else {
        width as f32 / height.max(1) as f32
    }
}

fn step_slice_coordinate(current: u32, last: u32, steps: i32) -> u32 {
    if steps >= 0 {
        current.saturating_add(steps as u32).min(last)
    } else {
        current.saturating_sub(steps.unsigned_abs())
    }
}

fn workspace_grid(workspace: egui::Rect, column_ratio: f32, row_ratio: f32) -> WorkspaceGrid {
    let column_gap = WORKSPACE_GAP.min(workspace.width().max(0.0));
    let row_gap = WORKSPACE_GAP.min(workspace.height().max(0.0));
    let left_width = split_extent(workspace.width(), column_ratio, WORKSPACE_MIN_PANE_WIDTH);
    let top_height = split_extent(workspace.height(), row_ratio, WORKSPACE_MIN_PANE_HEIGHT);
    let left_edge = workspace.left() + left_width;
    let right_edge = left_edge + column_gap;
    let top_edge = workspace.top() + top_height;
    let bottom_edge = top_edge + row_gap;

    WorkspaceGrid {
        axial: egui::Rect::from_min_max(workspace.min, egui::pos2(left_edge, top_edge)),
        coronal: egui::Rect::from_min_max(
            egui::pos2(right_edge, workspace.top()),
            egui::pos2(workspace.right(), top_edge),
        ),
        sagittal: egui::Rect::from_min_max(
            egui::pos2(workspace.left(), bottom_edge),
            egui::pos2(left_edge, workspace.bottom()),
        ),
        volume: egui::Rect::from_min_max(egui::pos2(right_edge, bottom_edge), workspace.max),
        column_splitter: egui::Rect::from_min_max(
            egui::pos2(left_edge, workspace.top()),
            egui::pos2(right_edge, workspace.bottom()),
        ),
        row_splitter: egui::Rect::from_min_max(
            egui::pos2(workspace.left(), top_edge),
            egui::pos2(workspace.right(), bottom_edge),
        ),
    }
}

fn split_extent(span: f32, ratio: f32, minimum: f32) -> f32 {
    let gap = WORKSPACE_GAP.min(span.max(0.0));
    let usable = (span - gap).max(0.0);
    if usable == 0.0 {
        return 0.0;
    }
    let minimum = minimum.min(usable * 0.5);
    let ratio = if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.5
    };
    (usable * ratio).clamp(minimum, usable - minimum)
}

fn split_ratio_at_pointer(pointer: f32, start: f32, span: f32, minimum: f32) -> f32 {
    let gap = WORKSPACE_GAP.min(span.max(0.0));
    let usable = (span - gap).max(0.0);
    if usable == 0.0 || !pointer.is_finite() || !start.is_finite() {
        return 0.5;
    }
    let requested_extent = pointer - start - gap * 0.5;
    let minimum = minimum.min(usable * 0.5);
    requested_extent.clamp(minimum, usable - minimum) / usable
}

fn split_ratio_after_delta(span: f32, ratio: f32, minimum: f32, delta: f32) -> f32 {
    let gap = WORKSPACE_GAP.min(span.max(0.0));
    let extent = split_extent(span, ratio, minimum);
    split_ratio_at_pointer(extent + gap * 0.5 + delta, 0.0, span, minimum)
}

fn fit_rect(available: egui::Rect, aspect: f32) -> egui::Rect {
    let size = if available.width() / available.height().max(1.0) > aspect {
        Vec2::new(available.height() * aspect, available.height())
    } else {
        Vec2::new(available.width(), available.width() / aspect.max(0.001))
    };
    egui::Rect::from_center_size(available.center(), size)
}

fn slice_pointer_to_voxel(
    axis: SliceAxis,
    position: egui::Pos2,
    rect: egui::Rect,
    width: u32,
    height: u32,
    cursor: [u32; 3],
) -> Option<[u32; 3]> {
    if width == 0 || height == 0 || !rect_contains_half_open(rect, position) {
        return None;
    }
    let u = ((position.x - rect.left()) / rect.width() * width as f32) as u32;
    let v = ((position.y - rect.top()) / rect.height() * height as f32) as u32;
    Some(match axis {
        SliceAxis::Axial => [u, v, cursor[2]],
        SliceAxis::Coronal => [u, cursor[1], v],
        SliceAxis::Sagittal => [cursor[0], u, v],
    })
}

fn rect_contains_half_open(rect: egui::Rect, position: egui::Pos2) -> bool {
    position.x >= rect.left()
        && position.x < rect.right()
        && position.y >= rect.top()
        && position.y < rect.bottom()
}

fn persistent_hover_position(response: &egui::Response) -> Option<egui::Pos2> {
    response.hover_pos()
}

fn paint_crosshair(
    painter: &egui::Painter,
    rect: egui::Rect,
    axis: SliceAxis,
    cursor: [u32; 3],
    width: u32,
    height: u32,
) {
    let (u, v) = match axis {
        SliceAxis::Axial => (cursor[0], cursor[1]),
        SliceAxis::Coronal => (cursor[0], cursor[2]),
        SliceAxis::Sagittal => (cursor[1], cursor[2]),
    };
    let x = rect.left() + (u as f32 + 0.5) / width as f32 * rect.width();
    let y = rect.top() + (v as f32 + 0.5) / height as f32 * rect.height();
    let stroke = egui::Stroke::new(1.0, YELLOW);
    painter.line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        stroke,
    );
    painter.line_segment(
        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
        stroke,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_brush_preview(
    painter: &egui::Painter,
    rect: egui::Rect,
    axis: SliceAxis,
    coordinate: u32,
    width: u32,
    height: u32,
    voxels: &[[u32; 3]],
    kind: VoxelHighlightKind,
    forced_colors: bool,
) {
    let color = if forced_colors {
        [u8::MAX; 3]
    } else {
        kind.color()
    };
    let fill = Color32::from_rgba_unmultiplied(color[0], color[1], color[2], kind.fill_alpha());
    let outline =
        Color32::from_rgba_unmultiplied(color[0], color[1], color[2], kind.outline_alpha());
    let normal_axis = axis.normal_axis();
    for &voxel in voxels {
        if voxel[normal_axis] != coordinate {
            continue;
        }
        let [u, v] = match axis {
            SliceAxis::Axial => [voxel[0], voxel[1]],
            SliceAxis::Coronal => [voxel[0], voxel[2]],
            SliceAxis::Sagittal => [voxel[1], voxel[2]],
        };
        if u >= width || v >= height {
            continue;
        }
        let voxel_rect = egui::Rect::from_min_max(
            egui::pos2(
                rect.left() + u as f32 / width as f32 * rect.width(),
                rect.top() + v as f32 / height as f32 * rect.height(),
            ),
            egui::pos2(
                rect.left() + (u + 1) as f32 / width as f32 * rect.width(),
                rect.top() + (v + 1) as f32 / height as f32 * rect.height(),
            ),
        );
        painter.rect_filled(voxel_rect, 0.0, fill);
        painter.rect_stroke(
            voxel_rect,
            0.0,
            egui::Stroke::new(1.0, outline),
            egui::StrokeKind::Inside,
        );
    }
}

fn paint_interaction_hint(
    painter: &egui::Painter,
    content: egui::Rect,
    text: &str,
    forced_colors: bool,
) {
    let text_color = if forced_colors { Color32::WHITE } else { MUTED };
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(12.0),
        text_color,
    );
    let size = galley.size() + egui::vec2(12.0, 6.0);
    let rect = egui::Rect::from_min_size(
        egui::pos2(content.left() + 6.0, content.bottom() - size.y - 6.0),
        size,
    );
    painter.rect_filled(
        rect,
        if forced_colors { 0 } else { CONTROL_RADIUS },
        if forced_colors {
            Color32::BLACK
        } else {
            Color32::from_rgba_unmultiplied(SURFACE.r(), SURFACE.g(), SURFACE.b(), 232)
        },
    );
    painter.rect_stroke(
        rect,
        if forced_colors { 0 } else { CONTROL_RADIUS },
        egui::Stroke::new(1.0, if forced_colors { Color32::WHITE } else { LINE }),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.min + egui::vec2(6.0, 3.0), galley, text_color);
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| path.display().to_string(), str::to_owned)
}

fn suggested_save_name(summary: &DocumentSummary, purpose: DialogPurpose) -> String {
    let current = display_name(&summary.source_path);
    if matches!(purpose, DialogPurpose::SaveAs) {
        return current;
    }
    let suffix = match summary.source_variant {
        SourceVariant::Nifti { .. } => {
            if current.to_ascii_lowercase().ends_with(".nii.gz") {
                ".nii.gz"
            } else {
                ".nii"
            }
        }
        SourceVariant::Binvox { .. } => ".binvox",
        SourceVariant::Vox => ".vox",
    };
    let stem = current.strip_suffix(suffix).unwrap_or(&current);
    format!("{stem}-copy{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_pointer_remains_available_for_hover_picking() {
        let context = egui::Context::default();
        let pointer = egui::pos2(40.0, 30.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 80.0));
        let mut hovered = None;
        for events in [vec![egui::Event::PointerMoved(pointer)], Vec::new()] {
            let _ = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(screen),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let response = ui.interact(
                        screen,
                        ui.id().with("stationary-hover-target"),
                        egui::Sense::click_and_drag(),
                    );
                    hovered = persistent_hover_position(&response);
                },
            );
        }
        assert_eq!(hovered, Some(pointer));
    }

    #[test]
    fn slice_pointer_mapping_keeps_synchronized_normal_coordinate() {
        let rect = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(100.0, 50.0));
        assert_eq!(
            slice_pointer_to_voxel(
                SliceAxis::Coronal,
                egui::pos2(75.0, 25.0),
                rect,
                20,
                10,
                [3, 7, 2],
            ),
            Some([15, 7, 5])
        );
    }

    #[test]
    fn slice_pointer_outside_image_does_not_clamp_to_border_voxel() {
        let rect = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(100.0, 50.0));
        assert_eq!(
            slice_pointer_to_voxel(
                SliceAxis::Axial,
                egui::pos2(100.0, 25.0),
                rect,
                20,
                10,
                [3, 7, 2],
            ),
            None
        );
        assert_eq!(
            slice_pointer_to_voxel(
                SliceAxis::Axial,
                egui::pos2(-0.1, 25.0),
                rect,
                20,
                10,
                [3, 7, 2],
            ),
            None
        );
    }

    #[test]
    fn primary_press_and_release_keep_their_own_same_frame_positions() {
        let events = [
            egui::Event::PointerButton {
                pos: egui::pos2(12.0, 18.0),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerMoved(egui::pos2(42.0, 36.0)),
            egui::Event::PointerButton {
                pos: egui::pos2(42.0, 36.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];

        assert_eq!(
            primary_pointer_event_positions(&events),
            (Some(egui::pos2(12.0, 18.0)), Some(egui::pos2(42.0, 36.0)))
        );
    }

    #[test]
    fn raw_wheel_notch_moves_once_without_smoothed_frame_replay() {
        let mut view = SliceViewState::default();
        let event = egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 1.0),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        };

        assert_eq!(view.wheel_steps(&[event]), 1);
        assert_eq!(view.wheel_steps(&[]), 0);
    }

    #[test]
    fn control_wheel_is_reported_as_zoom_instead_of_smooth_scroll() {
        let context = egui::Context::default();
        let modifiers = egui::Modifiers {
            ctrl: true,
            ..Default::default()
        };
        let mut observed = (egui::Vec2::ZERO, 1.0);
        let _ = context.run_ui(
            egui::RawInput {
                modifiers,
                events: vec![egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    modifiers,
                }],
                ..Default::default()
            },
            |ui| {
                observed = ui.input(|input| (input.smooth_scroll_delta, input.zoom_delta()));
            },
        );

        assert_eq!(observed.0, egui::Vec2::ZERO);
        assert!(observed.1 > 1.0);
    }

    #[test]
    fn slice_zoom_keeps_the_voxel_under_the_pointer_stationary() {
        let fitted_center = egui::pos2(50.0, 40.0);
        let pointer = egui::pos2(75.0, 55.0);
        let mut view = SliceViewState::default();
        let before = (pointer - (fitted_center + view.pan)) / view.zoom;

        view.zoom_at(2.0, pointer, fitted_center);

        let after = (pointer - (fitted_center + view.pan)) / view.zoom;
        assert!((view.zoom - 2.0).abs() < f32::EPSILON);
        assert!((after - before).length() < f32::EPSILON);
    }

    #[test]
    fn point_scroll_accumulates_to_one_controlled_step() {
        let mut view = SliceViewState::default();
        let point = |amount| egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, amount),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        };

        assert_eq!(view.wheel_steps(&[point(8.0)]), 0);
        assert_eq!(view.wheel_steps(&[point(8.0)]), 0);
        assert_eq!(view.wheel_steps(&[point(8.0)]), 1);
        assert_eq!(view.wheel_steps(&[]), 0);
        assert_eq!(view.wheel_steps(&[point(-8.0)]), 0);
        assert_eq!(view.wheel_steps(&[point(-16.0)]), -1);
    }

    #[test]
    fn workspace_grid_accounts_for_the_splitter_gap_and_minimum_sizes() {
        let workspace =
            egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(1_000.0, 700.0));
        let centered = workspace_grid(workspace, 0.5, 0.5);
        assert!((centered.axial.width() - 496.0).abs() < f32::EPSILON);
        assert!((centered.coronal.width() - 496.0).abs() < f32::EPSILON);
        assert!((centered.axial.height() - 346.0).abs() < f32::EPSILON);
        assert!((centered.sagittal.height() - 346.0).abs() < f32::EPSILON);
        assert!((centered.column_splitter.width() - WORKSPACE_GAP).abs() < f32::EPSILON);
        assert!((centered.row_splitter.height() - WORKSPACE_GAP).abs() < f32::EPSILON);

        let minimum = workspace_grid(workspace, 0.0, 1.0);
        assert!((minimum.axial.width() - WORKSPACE_MIN_PANE_WIDTH).abs() < f32::EPSILON);
        assert!((minimum.sagittal.height() - WORKSPACE_MIN_PANE_HEIGHT).abs() < f32::EPSILON);
        assert!(
            (minimum.coronal.width() - (1_000.0 - WORKSPACE_GAP - WORKSPACE_MIN_PANE_WIDTH)).abs()
                < f32::EPSILON
        );
        assert!(
            (minimum.axial.height() - (700.0 - WORKSPACE_GAP - WORKSPACE_MIN_PANE_HEIGHT)).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn workspace_grid_stays_valid_when_the_window_is_smaller_than_both_minimums() {
        let workspace = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 200.0));
        let grid = workspace_grid(workspace, f32::NAN, f32::INFINITY);
        for pane in [grid.axial, grid.coronal, grid.sagittal, grid.volume] {
            assert!(pane.width() >= 0.0);
            assert!(pane.height() >= 0.0);
        }
        assert!((grid.axial.width() - grid.coronal.width()).abs() < f32::EPSILON);
        assert!((grid.axial.height() - grid.sagittal.height()).abs() < f32::EPSILON);
    }

    #[test]
    fn slice_aspect_uses_physical_spacing() {
        let aspect = slice_physical_aspect(SliceAxis::Coronal, 617, 231, [0.315, 0.315, 0.8]);
        assert!((aspect - 1.051_7).abs() < 0.001);
        assert_eq!(
            SLICE_TEXTURE_OPTIONS.magnification,
            egui::TextureFilter::Nearest
        );
        assert_eq!(
            SLICE_TEXTURE_OPTIONS.minification,
            egui::TextureFilter::Linear
        );
    }

    #[test]
    fn copy_name_preserves_double_nifti_extension() {
        let summary = DocumentSummary {
            dimensions: [1; 3],
            spacing: [1.0; 3],
            units: crate::volume::SpatialUnits::Millimeter,
            source_path: PathBuf::from("face.nii.gz"),
            source_variant: SourceVariant::Nifti {
                version: crate::codecs::NiftiVersion::Two,
                compression: crate::codecs::Compression::Gzip,
            },
            cursor: [0; 3],
            occupied_count: 0,
            normalized_non_binary: false,
            revision: 1,
            saved_revision: 1,
            can_undo: false,
            can_redo: false,
        };
        assert_eq!(
            suggested_save_name(&summary, DialogPurpose::SaveCopy),
            "face-copy.nii.gz"
        );
    }
}
