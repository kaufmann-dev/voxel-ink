use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use crossbeam_channel::{Receiver, bounded};
use egui::{Color32, TextureHandle, TextureOptions, Vec2};

use crate::{
    EngineCommand, EngineEvent, EngineHandle,
    brush::{BrushDepth, BrushShape, BrushSize, BrushSpec},
    codecs::{FormatFamily, SourceVariant},
    engine::{BrushPreviewRequest, BusyKind, JobId, PointerSample, SlicePatch, StrokeSpace},
    io::SaveMode,
    renderer::{SceneBridge, VolumePaintCallback, VoxelPreviewKind},
    volume::{DocumentSummary, SliceAxis, SliceImage, SpatialUnits, Tool},
};

const LOGO_BYTES: &[u8] = include_bytes!("../assets/voxel-ink.png");
const BACKGROUND: Color32 = Color32::from_rgb(11, 17, 20);
const SURFACE: Color32 = Color32::from_rgb(16, 25, 29);
const RAISED: Color32 = Color32::from_rgb(22, 35, 41);
const LINE: Color32 = Color32::from_rgb(49, 69, 78);
const TEXT: Color32 = Color32::from_rgb(236, 243, 245);
const MUTED: Color32 = Color32::from_rgb(170, 185, 191);
const TEAL: Color32 = Color32::from_rgb(36, 200, 219);
const YELLOW: Color32 = Color32::from_rgb(255, 193, 49);
const MAX_ENGINE_EVENTS_PER_FRAME: usize = 256;
const SLICE_POINT_SCROLL_THRESHOLD: f32 = 24.0;
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
    Stroke { complete: bool },
    BridgeAnchor { plane_normal: [f64; 3] },
    BridgeCommit,
}

impl PreviewAction {
    const fn finishing_stroke(self) -> bool {
        matches!(self, Self::Stroke { complete: true })
    }

    fn mark_stroke_complete(&mut self) {
        if let Self::Stroke { complete } = self {
            *complete = true;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ActionPreview {
    request: BrushPreviewRequest,
    action: PreviewAction,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkspacePane {
    Slice(SliceAxis),
    Volume3D,
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
    brush_preview_tool: Option<Tool>,
    brush_preview_target: Option<[u32; 3]>,
    brush_preview_voxels: Vec<[u32; 3]>,
    stroke_overlay_tool: Option<Tool>,
    stroke_overlay_voxels: Vec<[u32; 3]>,
    stroke_overlay_members: HashSet<[u32; 3]>,
    stroke_overlay_revision: Option<u64>,
    stroke_overlay_waiting_for_mesh: bool,
    pending_preview: Option<PendingPreview>,
    queued_action_preview: Option<ActionPreview>,
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
    stroke_active: bool,
    stroke_completion_pending: bool,
    pane_keyboard_focus: bool,
    maximized_pane: Option<WorkspacePane>,
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
            brush_preview_tool: None,
            brush_preview_target: None,
            brush_preview_voxels: Vec::new(),
            stroke_overlay_tool: None,
            stroke_overlay_voxels: Vec::new(),
            stroke_overlay_members: HashSet::new(),
            stroke_overlay_revision: None,
            stroke_overlay_waiting_for_mesh: false,
            pending_preview: None,
            queued_action_preview: None,
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
            stroke_active: false,
            stroke_completion_pending: false,
            pane_keyboard_focus: false,
            maximized_pane: None,
        };
        if let Some(path) = initial_path
            && let Err(error) = app.engine.start_load(path)
        {
            app.error_message = Some(error.to_string());
        }
        app
    }

    fn process_engine_events(&mut self, ctx: &egui::Context) {
        let mut exhausted_budget = true;
        let mut overlay_refresh_revision = None;
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
                    self.brush_preview_tool = None;
                    self.brush_preview_target = None;
                    self.brush_preview_voxels.clear();
                    self.stroke_overlay_tool = None;
                    self.stroke_overlay_voxels.clear();
                    self.stroke_overlay_members.clear();
                    self.stroke_overlay_revision = None;
                    self.stroke_overlay_waiting_for_mesh = false;
                    self.pending_preview = None;
                    self.queued_action_preview = None;
                    self.deferred_bridge_release = None;
                    self.bridge_anchor = None;
                    self.summary = Some(summary);
                    self.stroke_active = false;
                    self.stroke_completion_pending = false;
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
                    self.brush_preview_tool = None;
                    self.brush_preview_target = None;
                    self.brush_preview_voxels.clear();
                    self.stroke_overlay_tool = None;
                    self.stroke_overlay_voxels.clear();
                    self.stroke_overlay_members.clear();
                    self.stroke_overlay_revision = None;
                    self.stroke_overlay_waiting_for_mesh = false;
                    self.pending_preview = None;
                    self.queued_action_preview = None;
                    self.deferred_bridge_release = None;
                    self.bridge_anchor = None;
                    self.stroke_active = false;
                    self.stroke_completion_pending = false;
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
                        self.queued_action_preview = None;
                        self.deferred_bridge_release = None;
                        self.bridge_anchor = None;
                        self.brush_preview_target = None;
                        self.brush_preview_voxels.clear();
                        self.brush_preview_tool = None;
                        if !self.stroke_overlay_waiting_for_mesh {
                            self.clear_stroke_overlay_data();
                        }
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
                    self.brush_preview_tool = None;
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
                    if epoch == self.epoch && self.stroke_overlay_waiting_for_mesh {
                        self.stroke_overlay_revision = Some(revision);
                        self.stroke_overlay_waiting_for_mesh = false;
                        self.refresh_scene_brush_preview(epoch, revision);
                    }
                }
                EngineEvent::MeshChunk { tag, quads } => {
                    self.scene.insert_chunk(tag, quads);
                    if tag.document_epoch == self.epoch
                        && self.stroke_overlay_revision == Some(tag.revision)
                        && self.retire_stroke_overlay_chunk(tag.chunk)
                    {
                        overlay_refresh_revision = Some(tag.revision);
                    }
                }
                EngineEvent::PreviewTooComplex { epoch, .. } if epoch == self.epoch => {
                    self.mesh_preview_failed = true;
                }
                _ => {}
            }
        }
        if let Some(revision) = overlay_refresh_revision {
            self.refresh_scene_brush_preview(self.epoch, revision);
        }
        if exhausted_budget {
            ctx.request_repaint();
        }
        if self.scene.take_remesh_request() && self.summary.is_some() {
            let _ = self.send_command(EngineCommand::RebuildPreview);
        }
        if let Some(message) = self.scene.take_preview_failure() {
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
                if let Err(error) = self.engine.start_load(path) {
                    self.error_message = Some(error.to_string());
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
        if self.prompt.is_some() || self.error_message.is_some() || !self.busy.is_empty() {
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
            self.clear_stroke_overlay();
            let _ = self.send_command(EngineCommand::Redo);
        } else if undo && edit_commands_ready {
            self.clear_brush_preview();
            self.clear_stroke_overlay();
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
            self.request_action_preview(
                BrushPreviewRequest::Brush {
                    generation: 0,
                    tool: self.tool,
                    spec: self.brush_spec(),
                    space,
                    sample: PointerSample::Slice {
                        axis: self.active_slice,
                        voxel: cursor,
                    },
                },
                PreviewAction::Stroke { complete: true },
            );
        }
    }

    fn toolbar(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        egui::Panel::top("voxel-ink-toolbar")
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .inner_margin(egui::Margin::same(8)),
            )
            .show(root, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::Image::new(&self.logo)
                            .fit_to_exact_size(Vec2::splat(34.0))
                            .alt_text("VoxelInk teal and yellow logo"),
                    );
                    ui.heading("VoxelInk");
                    ui.separator();
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
                    ui.separator();
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
                        self.clear_stroke_overlay();
                        let _ = self.send_command(EngineCommand::Undo);
                    }
                    if ui
                        .add_enabled(ready && can_redo, egui::Button::new("Redo"))
                        .on_hover_text("Redo (Ctrl+Shift+Z or Ctrl+Y)")
                        .clicked()
                    {
                        self.clear_brush_preview();
                        self.clear_stroke_overlay();
                        let _ = self.send_command(EngineCommand::Redo);
                    }
                    if let Some(summary) = &self.summary {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let dirty = if summary.is_dirty() {
                                " • modified"
                            } else {
                                ""
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}{}",
                                    display_name(&summary.source_path),
                                    dirty
                                ))
                                .color(MUTED),
                            );
                        });
                    }
                });
            });
    }

    fn repair_panel(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        egui::Panel::left("voxel-ink-repair-panel")
            .default_size(232.0)
            .min_size(232.0)
            .max_size(232.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(root, |ui| {
                let has_document = self.summary.is_some();
                let ready = has_document
                    && self.busy.is_empty()
                    && !self.has_edit_interaction();
                ui.heading("Repair volume");
                ui.add_space(4.0);
                ui.strong("1 · Seal every opening");
                ui.label(
                    egui::RichText::new(
                        "Paint missing voxels or bridge two surfaces until the skull interior is closed.",
                    )
                    .color(MUTED),
                );
                ui.add_space(8.0);
                let previous_tool = self.tool;
                ui.add_enabled_ui(self.busy.is_empty(), |ui| {
                    ui.selectable_value(&mut self.tool, Tool::Inspect, "Inspect / navigate (I)")
                        .on_hover_text("Move through slices and rotate or pan the 3D view");
                    ui.selectable_value(&mut self.tool, Tool::Paint, "Paint solid (B)")
                        .on_hover_text("Fill the highlighted voxels with the primary mouse button");
                    ui.selectable_value(&mut self.tool, Tool::Erase, "Erase solid (E)")
                        .on_hover_text("Remove the highlighted voxels with the primary mouse button");
                    ui.selectable_value(&mut self.tool, Tool::Bridge, "Bridge surfaces (L)")
                        .on_hover_text("Drag between two points to add a solid connection in one Undo step");
                });
                if self.tool != previous_tool {
                    self.cancel_edit_interaction();
                }

                ui.add_space(10.0);
                ui.add_enabled_ui(ready && self.tool != Tool::Inspect, |ui| {
                    ui.strong("Brush footprint");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.brush_shape, BrushShape::Round, "Round");
                        ui.selectable_value(&mut self.brush_shape, BrushShape::Square, "Square");
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.brush_depth, BrushDepth::Slice, "One slice");
                        ui.selectable_value(&mut self.brush_depth, BrushDepth::Volume, "3D volume");
                    });
                    ui.label(
                        egui::RichText::new("Depth applies in slice views; 3D editing uses a 3D brush.")
                            .small()
                            .color(MUTED),
                    );
                    ui.add_space(5.0);

                    let has_physical_units = self
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.units != SpatialUnits::Unknown);
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.brush_size_unit,
                            BrushSizeUnit::Voxels,
                            "Voxels",
                        );
                        ui.add_enabled_ui(has_physical_units, |ui| {
                            ui.selectable_value(
                                &mut self.brush_size_unit,
                                BrushSizeUnit::Millimeters,
                                "Millimetres",
                            );
                        });
                    });
                    if !has_physical_units {
                        self.brush_size_unit = BrushSizeUnit::Voxels;
                    }
                    match self.brush_size_unit {
                        BrushSizeUnit::Voxels => {
                            let changed = ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.brush_voxel_diameter,
                                        BrushSize::MIN_VOXELS..=BrushSize::MAX_VOXELS,
                                    )
                                    .step_by(2.0)
                                    .text("Brush diameter in voxels"),
                                )
                                .changed();
                            if changed && self.brush_voxel_diameter.is_multiple_of(2) {
                                self.brush_voxel_diameter =
                                    self.brush_voxel_diameter.saturating_add(1).min(BrushSize::MAX_VOXELS);
                            }
                        }
                        BrushSizeUnit::Millimeters => {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.brush_millimeters,
                                    BrushSize::MIN_MILLIMETERS..=BrushSize::MAX_MILLIMETERS,
                                )
                                .step_by(0.5)
                                .suffix(" mm")
                                .text("Brush diameter in millimetres"),
                            );
                        }
                    }
                });

                ui.add_space(14.0);
                ui.separator();
                ui.add_space(10.0);
                ui.strong("2 · Fill the closed interior");
                ui.label(
                    egui::RichText::new(
                        "After every opening is sealed, fill all empty regions that can no longer reach a volume boundary.",
                    )
                    .color(MUTED),
                );
                ui.add_space(8.0);
                if ui
                    .add_enabled(
                        ready,
                        egui::Button::new(
                            egui::RichText::new("Fill enclosed cavities").strong(),
                        )
                        .min_size(egui::vec2(ui.available_width(), 34.0)),
                    )
                    .on_hover_text(
                        "Fill every empty 3D region that cannot reach any outside volume face. The result is one Undo step.",
                    )
                    .clicked()
                {
                    self.preview_notice = None;
                    self.clear_brush_preview();
                    self.clear_stroke_overlay();
                    if let Err(error) = self.engine.start_fill_enclosed_cavities() {
                        self.error_message = Some(error.to_string());
                    }
                }
                ui.label(
                    egui::RichText::new("No point selection is needed. Cancel is available while scanning.")
                        .small()
                        .color(MUTED),
                );
            });
    }

    fn status_bar(&mut self, root: &mut egui::Ui) {
        let surface = self.surface_color();
        egui::Panel::bottom("voxel-ink-status")
            .frame(
                egui::Frame::new()
                    .fill(surface)
                    .inner_margin(egui::Margin::symmetric(8, 5)),
            )
            .show(root, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if let Some(summary) = &self.summary {
                        ui.label(format!(
                            "{} × {} × {}   {} occupied   cursor {}, {}, {}",
                            summary.dimensions[0],
                            summary.dimensions[1],
                            summary.dimensions[2],
                            summary.occupied_count,
                            summary.cursor[0],
                            summary.cursor[1],
                            summary.cursor[2]
                        ));
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
                        ui.colored_label(
                            YELLOW,
                            "3D preview paused at its memory limit; slice editing is unaffected",
                        );
                        if ui.button("Retry 3D preview").clicked()
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
                        ui.label(egui::RichText::new(&self.adapter_name).color(MUTED).small());
                    });
                });
            });
    }

    fn workspace(&mut self, root: &mut egui::Ui) {
        self.pane_keyboard_focus = false;
        let background = self.background_color();
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(background)
                    .inner_margin(egui::Margin::same(6)),
            )
            .show(root, |ui| {
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
                        let gap = 6.0;
                        let half_width = (rect.width() - gap) * 0.5;
                        let half_height = (rect.height() - gap) * 0.5;
                        let panes = [
                            (
                                SliceAxis::Axial,
                                egui::Rect::from_min_size(
                                    rect.min,
                                    Vec2::new(half_width, half_height),
                                ),
                            ),
                            (
                                SliceAxis::Coronal,
                                egui::Rect::from_min_size(
                                    egui::pos2(rect.min.x + half_width + gap, rect.min.y),
                                    Vec2::new(half_width, half_height),
                                ),
                            ),
                            (
                                SliceAxis::Sagittal,
                                egui::Rect::from_min_size(
                                    egui::pos2(rect.min.x, rect.min.y + half_height + gap),
                                    Vec2::new(half_width, half_height),
                                ),
                            ),
                        ];
                        for (axis, pane_rect) in panes {
                            ui.scope_builder(egui::UiBuilder::new().max_rect(pane_rect), |ui| {
                                self.slice_pane(ui, axis, &mut actions);
                            });
                        }
                        let volume_rect = egui::Rect::from_min_size(
                            egui::pos2(
                                rect.min.x + half_width + gap,
                                rect.min.y + half_height + gap,
                            ),
                            Vec2::new(half_width, half_height),
                        );
                        ui.scope_builder(egui::UiBuilder::new().max_rect(volume_rect), |ui| {
                            self.volume_pane(ui, &mut actions);
                        });
                    }
                }
                if !actions.preview_hovered
                    && !self.stroke_active
                    && self.bridge_anchor.is_none()
                    && self.pending_preview.is_none()
                {
                    self.clear_brush_preview();
                }
            });
    }

    fn slice_pane(&mut self, ui: &mut egui::Ui, axis: SliceAxis, actions: &mut FrameStrokeActions) {
        let pane_rect = ui.max_rect();
        ui.painter()
            .rect_filled(pane_rect, 0.0, self.surface_color());
        ui.painter().rect_stroke(
            pane_rect,
            0.0,
            egui::Stroke::new(1.0, self.line_color()),
            egui::StrokeKind::Inside,
        );
        let title = match axis {
            SliceAxis::Axial => "Axial",
            SliceAxis::Coronal => "Coronal",
            SliceAxis::Sagittal => "Sagittal",
        };
        let coordinate = self
            .summary
            .as_ref()
            .map_or(0, |summary| summary.cursor[axis.normal_axis()]);
        let last_coordinate = self.summary.as_ref().map_or(0, |summary| {
            summary.dimensions[axis.normal_axis()].saturating_sub(1)
        });
        ui.painter().text(
            pane_rect.min + egui::vec2(9.0, 8.0),
            egui::Align2::LEFT_TOP,
            format!("{title}  {coordinate} / {last_coordinate}"),
            egui::FontId::proportional(14.0),
            TEXT,
        );
        let pane = WorkspacePane::Slice(axis);
        let maximized = self.maximized_pane == Some(pane);
        let maximize_label = if maximized {
            "Restore four views".to_owned()
        } else {
            format!("Full screen {title}")
        };
        let button_width = if maximized { 132.0 } else { 148.0 };
        let button_rect = egui::Rect::from_min_size(
            egui::pos2(
                pane_rect.right() - button_width - 5.0,
                pane_rect.top() + 4.0,
            ),
            egui::vec2(button_width, 22.0),
        );
        if ui
            .put(button_rect, egui::Button::new(maximize_label))
            .on_hover_text(if maximized {
                "Return to the four-view layout (Esc)"
            } else {
                "Use the full workspace for this view"
            })
            .clicked()
        {
            self.active_slice = axis;
            self.maximized_pane = if maximized { None } else { Some(pane) };
            ui.ctx().request_repaint();
        }
        let fit_button_rect = egui::Rect::from_min_size(
            egui::pos2(button_rect.left() - 47.0, pane_rect.top() + 4.0),
            egui::vec2(42.0, 22.0),
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
            egui::pos2(fit_button_rect.left() - 75.0, pane_rect.top() + 4.0),
            egui::vec2(70.0, 22.0),
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
            pane_rect.min + egui::vec2(6.0, 30.0),
            pane_rect.max - egui::vec2(6.0, 6.0),
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
        ui.painter().image(
            texture_id,
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        if let Some(preview_tool) = self.stroke_overlay_tool {
            paint_brush_preview(
                &ui.painter().with_clip_rect(content),
                image_rect,
                axis,
                coordinate,
                width,
                height,
                &self.stroke_overlay_voxels,
                preview_tool,
                self.forced_colors,
            );
        }
        if let Some(preview_tool) = self.brush_preview_tool {
            paint_brush_preview(
                &ui.painter().with_clip_rect(content),
                image_rect,
                axis,
                coordinate,
                width,
                height,
                &self.brush_preview_voxels,
                preview_tool,
                self.forced_colors,
            );
        }
        ui.painter().text(
            content.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "Wheel: 1 slice   Ctrl + wheel: zoom   Middle / right drag: move",
            egui::FontId::proportional(12.0),
            MUTED,
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
                0.0,
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
            && !self.stroke_active
            && !self.stroke_completion_pending
        {
            let control = ui.input(|input| input.modifiers.ctrl);
            if control {
                let zoom_delta = ui.input(|input| input.smooth_scroll_delta.y);
                if zoom_delta.abs() > f32::EPSILON {
                    let previous_zoom = view.zoom;
                    view.zoom = (view.zoom * (zoom_delta * 0.002).exp()).clamp(0.25, 32.0);
                    if let Some(pointer) = response.hover_pos() {
                        let old_center = fitted_rect.center() + view.pan;
                        let scale = view.zoom / previous_zoom;
                        let new_center = pointer - (pointer - old_center) * scale;
                        view.pan = new_center - fitted_rect.center();
                    }
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
                ui.painter(),
                image_rect,
                axis,
                summary.cursor,
                width,
                height,
            );
        }
        let pointer_voxel = response
            .interact_pointer_pos()
            .and_then(|position| {
                rect_contains_half_open(interaction_rect, position).then_some(position)
            })
            .and_then(|position| {
                self.summary.as_ref().and_then(|summary| {
                    slice_pointer_to_voxel(
                        axis,
                        position,
                        image_rect,
                        width,
                        height,
                        summary.cursor,
                    )
                })
            });
        if !self.busy.is_empty() {
            return;
        }
        let space = StrokeSpace::Slice(axis);
        if self.tool != Tool::Inspect
            && response.hovered()
            && !self.stroke_active
            && !self.stroke_completion_pending
            && let Some(voxel) = pointer_voxel
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
            if response.drag_started_by(egui::PointerButton::Primary)
                && let Some(voxel) = pointer_voxel
            {
                self.active_slice = axis;
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Slice { axis, voxel },
                    },
                    PreviewAction::Stroke { complete: false },
                );
            } else if response.dragged_by(egui::PointerButton::Primary)
                && let Some(voxel) = pointer_voxel
            {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Slice { axis, voxel },
                    },
                    PreviewAction::Stroke { complete: false },
                );
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(voxel) = pointer_voxel {
                    self.request_action_preview(
                        BrushPreviewRequest::Brush {
                            generation: 0,
                            tool: self.tool,
                            spec: self.brush_spec(),
                            space,
                            sample: PointerSample::Slice { axis, voxel },
                        },
                        PreviewAction::Stroke { complete: true },
                    );
                } else {
                    self.finish_stroke_pointer_release();
                }
            } else if response.clicked_by(egui::PointerButton::Primary)
                && let Some(voxel) = pointer_voxel
            {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Slice { axis, voxel },
                    },
                    PreviewAction::Stroke { complete: true },
                );
            }
        } else if self.tool == Tool::Bridge {
            if response.drag_started_by(egui::PointerButton::Primary)
                && let Some(voxel) = pointer_voxel
            {
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
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(voxel) = pointer_voxel {
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
            && let Some(voxel) = pointer_voxel
        {
            let _ = self.send_command(EngineCommand::SetCursor(voxel));
        }
    }

    fn volume_pane(&mut self, ui: &mut egui::Ui, actions: &mut FrameStrokeActions) {
        let pane_rect = ui.max_rect();
        ui.painter()
            .rect_filled(pane_rect, 0.0, self.surface_color());
        ui.painter().rect_stroke(
            pane_rect,
            0.0,
            egui::Stroke::new(1.0, self.line_color()),
            egui::StrokeKind::Inside,
        );
        ui.painter().text(
            pane_rect.min + egui::vec2(9.0, 8.0),
            egui::Align2::LEFT_TOP,
            "3D View",
            egui::FontId::proportional(14.0),
            TEXT,
        );
        let pane = WorkspacePane::Volume3D;
        let maximized = self.maximized_pane == Some(pane);
        let maximize_label = if maximized {
            "Restore four views"
        } else {
            "Full screen 3D"
        };
        let maximize_width = if maximized { 132.0 } else { 126.0 };
        let maximize_rect = egui::Rect::from_min_size(
            egui::pos2(
                pane_rect.right() - maximize_width - 5.0,
                pane_rect.top() + 4.0,
            ),
            egui::vec2(maximize_width, 22.0),
        );
        if ui
            .put(maximize_rect, egui::Button::new(maximize_label))
            .on_hover_text(if maximized {
                "Return to the four-view layout (Esc)"
            } else {
                "Use the full workspace for the 3D view"
            })
            .clicked()
        {
            self.maximized_pane = if maximized { None } else { Some(pane) };
            ui.ctx().request_repaint();
        }
        let reset_width = 82.0;
        let reset_rect = egui::Rect::from_min_size(
            egui::pos2(
                maximize_rect.left() - reset_width - 5.0,
                pane_rect.top() + 4.0,
            ),
            egui::vec2(reset_width, 22.0),
        );
        if ui
            .put(reset_rect, egui::Button::new("Reset view"))
            .on_hover_text("Center the volume and restore the default rotation and zoom")
            .clicked()
        {
            self.scene.reset_view();
        }
        let content = egui::Rect::from_min_max(
            pane_rect.min + egui::vec2(6.0, 30.0),
            pane_rect.max - egui::vec2(6.0, 6.0),
        );
        ui.painter()
            .rect_filled(content, 0.0, self.background_color());
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
        ui.painter().text(
            content.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            controls,
            egui::FontId::proportional(12.0),
            MUTED,
        );
        if response.has_focus() {
            self.pane_keyboard_focus = true;
            ui.painter().rect_stroke(
                content,
                0.0,
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
        if !self.busy.is_empty() {
            return;
        }
        let pointer_ray = response
            .interact_pointer_pos()
            .filter(|position| rect_contains_half_open(content, *position))
            .map(|position| {
                let normalized = [
                    ((position.x - content.left()) / content.width()) * 2.0 - 1.0,
                    1.0 - ((position.y - content.top()) / content.height()) * 2.0,
                ];
                self.scene.ray_for_pointer(normalized)
            });
        let space = StrokeSpace::Volume3D;
        if (response.hovered() || response.dragged_by(egui::PointerButton::Primary))
            && !self.stroke_active
            && !self.stroke_completion_pending
            && let Some(ray) = pointer_ray
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
            if response.drag_started_by(egui::PointerButton::Primary)
                && let Some(ray) = pointer_ray
            {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Ray { ray },
                    },
                    PreviewAction::Stroke { complete: false },
                );
            } else if response.dragged_by(egui::PointerButton::Primary)
                && let Some(ray) = pointer_ray
            {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Ray { ray },
                    },
                    PreviewAction::Stroke { complete: false },
                );
            }
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(ray) = pointer_ray {
                    self.request_action_preview(
                        BrushPreviewRequest::Brush {
                            generation: 0,
                            tool: self.tool,
                            spec: self.brush_spec(),
                            space,
                            sample: PointerSample::Ray { ray },
                        },
                        PreviewAction::Stroke { complete: true },
                    );
                } else {
                    self.finish_stroke_pointer_release();
                }
            } else if response.clicked_by(egui::PointerButton::Primary)
                && let Some(ray) = pointer_ray
            {
                self.request_action_preview(
                    BrushPreviewRequest::Brush {
                        generation: 0,
                        tool: self.tool,
                        spec: self.brush_spec(),
                        space,
                        sample: PointerSample::Ray { ray },
                    },
                    PreviewAction::Stroke { complete: true },
                );
            }
        } else if self.tool == Tool::Bridge {
            if response.drag_started_by(egui::PointerButton::Primary)
                && let Some(ray) = pointer_ray
            {
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
            if response.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(ray) = pointer_ray {
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
        if self.pending_preview.is_some() || self.queued_action_preview.is_some() {
            return;
        }
        let template = preview_with_generation(request, 0);
        if self.last_preview_request == Some(template) {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.discard_current_brush_preview();
        let request = preview_with_generation(request, self.brush_preview_generation);
        match self.engine.send_latest_preview(request) {
            Ok(()) => self.last_preview_request = Some(template),
            Err(error) => {
                self.last_preview_request = None;
                self.error_message = Some(error.to_string());
            }
        }
    }

    fn request_action_preview(&mut self, request: BrushPreviewRequest, action: PreviewAction) {
        let mut queued = ActionPreview { request, action };
        if self.pending_preview.is_some() {
            if self
                .pending_preview
                .is_some_and(|current| current.action.finishing_stroke())
                || self
                    .queued_action_preview
                    .is_some_and(|current| current.action.finishing_stroke())
            {
                queued.action.mark_stroke_complete();
            }
            self.queued_action_preview = Some(queued);
            return;
        }
        self.dispatch_action_preview(queued);
    }

    fn dispatch_action_preview(&mut self, action_preview: ActionPreview) {
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        if self.brush_preview_tool.is_some()
            || self.brush_preview_target.is_some()
            || !self.brush_preview_voxels.is_empty()
        {
            self.discard_current_brush_preview();
        }
        let request =
            preview_with_generation(action_preview.request, self.brush_preview_generation);
        match self.engine.send_latest_preview(request) {
            Ok(()) => {
                self.last_preview_request = Some(preview_with_generation(request, 0));
                self.pending_preview = Some(PendingPreview {
                    generation: self.brush_preview_generation,
                    request,
                    action: action_preview.action,
                });
            }
            Err(error) => {
                self.last_preview_request = None;
                self.pending_preview = None;
                self.queued_action_preview = None;
                if self.stroke_active {
                    let _ = self.complete_active_stroke();
                }
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
            self.brush_preview_tool = Some(tool);
            self.brush_preview_target = target;
            self.brush_preview_voxels = voxels;
            self.refresh_scene_brush_preview(epoch, revision);
            return;
        };
        self.last_preview_request = None;
        self.brush_preview_tool = None;
        self.brush_preview_target = None;
        self.brush_preview_voxels.clear();
        let finishing_stroke = pending.action.finishing_stroke();
        match (pending.action, pending.request) {
            (
                PreviewAction::Stroke { complete },
                BrushPreviewRequest::Brush {
                    tool,
                    spec,
                    space,
                    sample,
                    ..
                },
            ) => {
                if target.is_some() {
                    if !self.stroke_active && !self.begin_stroke(tool, spec, space) {
                        self.queued_action_preview = None;
                    } else if self.stroke_active {
                        self.accumulate_stroke_overlay(tool, &voxels);
                        if let Err(error) = self.engine.send_latest_sample(sample) {
                            self.queued_action_preview = None;
                            self.error_message = Some(error.to_string());
                            let _ = self.complete_active_stroke();
                        }
                    }
                }
                if complete {
                    let _ = self.complete_active_stroke();
                }
            }
            (
                PreviewAction::BridgeAnchor { plane_normal },
                BrushPreviewRequest::Brush { space, .. },
            ) => {
                if let Some(voxel) = target {
                    self.clear_stroke_overlay_data();
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
            (
                PreviewAction::BridgeCommit,
                BrushPreviewRequest::Bridge {
                    spec,
                    space,
                    anchor,
                    plane_normal,
                    sample,
                    ..
                },
            ) => {
                if target.is_some() {
                    self.accumulate_stroke_overlay(Tool::Bridge, &voxels);
                    self.stroke_overlay_waiting_for_mesh = !self.stroke_overlay_voxels.is_empty();
                    self.stroke_overlay_revision = None;
                    if !self.send_command(EngineCommand::CommitBridge {
                        anchor,
                        plane_normal,
                        sample,
                        spec,
                        space,
                    }) {
                        self.clear_stroke_overlay();
                    }
                }
                self.bridge_anchor = None;
                self.deferred_bridge_release = None;
            }
            _ => {
                self.queued_action_preview = None;
            }
        }
        self.refresh_scene_brush_preview(epoch, revision);
        if !finishing_stroke
            && !self.stroke_completion_pending
            && self.pending_preview.is_none()
            && let Some(queued) = self.queued_action_preview.take()
        {
            self.dispatch_action_preview(queued);
        } else if finishing_stroke {
            self.queued_action_preview = None;
        }
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
        self.queued_action_preview = None;
        self.deferred_bridge_release = None;
        self.bridge_anchor = None;
        self.engine.cancel_preview();
        self.discard_current_brush_preview();
        if self.stroke_active || self.stroke_completion_pending {
            let _ = self.complete_active_stroke();
        }
        self.error_message = Some(message);
    }

    fn accumulate_stroke_overlay(&mut self, tool: Tool, voxels: &[[u32; 3]]) {
        if voxels.is_empty() {
            return;
        }
        if self.stroke_overlay_tool != Some(tool) {
            self.stroke_overlay_tool = Some(tool);
            self.stroke_overlay_voxels.clear();
            self.stroke_overlay_members.clear();
        }
        self.stroke_overlay_revision = None;
        self.stroke_overlay_waiting_for_mesh = false;
        for voxel in voxels.iter().copied() {
            if self.stroke_overlay_members.insert(voxel) {
                self.stroke_overlay_voxels.push(voxel);
            }
        }
    }

    fn retire_stroke_overlay_chunk(&mut self, chunk: crate::mesh::ChunkCoord) -> bool {
        let previous_len = self.stroke_overlay_voxels.len();
        self.stroke_overlay_voxels
            .retain(|voxel| crate::mesh::ChunkCoord::from_voxel(*voxel) != chunk);
        if self.stroke_overlay_voxels.len() == previous_len {
            return false;
        }
        self.stroke_overlay_members = self.stroke_overlay_voxels.iter().copied().collect();
        if self.stroke_overlay_voxels.is_empty() {
            self.stroke_overlay_tool = None;
            self.stroke_overlay_revision = None;
        }
        true
    }

    fn clear_stroke_overlay_data(&mut self) {
        self.stroke_overlay_tool = None;
        self.stroke_overlay_voxels.clear();
        self.stroke_overlay_members.clear();
        self.stroke_overlay_revision = None;
        self.stroke_overlay_waiting_for_mesh = false;
    }

    fn clear_stroke_overlay(&mut self) {
        self.clear_stroke_overlay_data();
        self.refresh_scene_brush_preview(self.epoch, self.summary_revision());
    }

    fn discard_current_brush_preview(&mut self) {
        self.brush_preview_tool = None;
        self.brush_preview_target = None;
        self.brush_preview_voxels.clear();
        let revision = self.summary.as_ref().map_or(0, |summary| summary.revision);
        self.refresh_scene_brush_preview(self.epoch, revision);
        self.egui_ctx.request_repaint();
    }

    fn refresh_scene_brush_preview(&self, epoch: u64, revision: u64) {
        let (tool, mut voxels) = if let Some(tool) = self.brush_preview_tool
            && self.stroke_overlay_tool != Some(tool)
        {
            (tool, self.brush_preview_voxels.clone())
        } else if let Some(tool) = self.stroke_overlay_tool {
            (tool, self.stroke_overlay_voxels.clone())
        } else if let Some(tool) = self.brush_preview_tool {
            (tool, self.brush_preview_voxels.clone())
        } else {
            self.scene.clear_voxel_preview();
            return;
        };
        if self.stroke_overlay_tool == self.brush_preview_tool {
            voxels.extend(
                self.brush_preview_voxels
                    .iter()
                    .copied()
                    .filter(|voxel| !self.stroke_overlay_members.contains(voxel)),
            );
        }
        if voxels.is_empty() {
            self.scene.clear_voxel_preview();
            return;
        }
        let kind = match tool {
            Tool::Erase => VoxelPreviewKind::Erase,
            Tool::Bridge => VoxelPreviewKind::Bridge,
            Tool::Inspect | Tool::Paint => VoxelPreviewKind::Paint,
        };
        let _ = self.scene.set_voxel_preview(epoch, revision, voxels, kind);
    }

    fn request_bridge_commit(&mut self, release: DeferredBridgeRelease) {
        let Some(anchor) = self.bridge_anchor else {
            self.deferred_bridge_release = Some(release);
            return;
        };
        let request = BrushPreviewRequest::Bridge {
            generation: 0,
            spec: release.spec,
            space: release.space,
            anchor: anchor.voxel,
            plane_normal: anchor.plane_normal,
            sample: release.sample,
        };
        self.request_action_preview(request, PreviewAction::BridgeCommit);
    }

    fn clear_brush_preview(&mut self) {
        if self.last_preview_request.is_none()
            && self.brush_preview_tool.is_none()
            && self.brush_preview_voxels.is_empty()
            && self.pending_preview.is_none()
            && self.queued_action_preview.is_none()
            && self.deferred_bridge_release.is_none()
        {
            return;
        }
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.engine.cancel_preview();
        self.last_preview_request = None;
        self.pending_preview = None;
        self.queued_action_preview = None;
        self.deferred_bridge_release = None;
        self.discard_current_brush_preview();
    }

    fn cancel_edit_interaction(&mut self) {
        self.brush_preview_generation = self.brush_preview_generation.saturating_add(1);
        self.engine.cancel_preview();
        self.last_preview_request = None;
        self.pending_preview = None;
        self.queued_action_preview = None;
        self.deferred_bridge_release = None;
        self.bridge_anchor = None;
        let had_active_stroke = self.stroke_active || self.stroke_completion_pending;
        if had_active_stroke && self.send_command(EngineCommand::CancelStroke) {
            self.stroke_active = false;
            self.stroke_completion_pending = false;
        }
        self.clear_stroke_overlay_data();
        self.discard_current_brush_preview();
    }

    fn finish_stroke_pointer_release(&mut self) {
        if let Some(queued) = self.queued_action_preview.as_mut() {
            queued.action.mark_stroke_complete();
        } else if let Some(pending) = self.pending_preview.as_mut() {
            pending.action.mark_stroke_complete();
        } else {
            let _ = self.complete_active_stroke();
        }
    }

    fn has_edit_interaction(&self) -> bool {
        self.pending_preview.is_some()
            || self.queued_action_preview.is_some()
            || self.deferred_bridge_release.is_some()
            || self.bridge_anchor.is_some()
            || self.stroke_active
            || self.stroke_completion_pending
    }

    fn summary_revision(&self) -> u64 {
        self.summary.as_ref().map_or(0, |summary| summary.revision)
    }

    fn begin_stroke(&mut self, tool: Tool, spec: BrushSpec, space: StrokeSpace) -> bool {
        if self.stroke_active || self.stroke_completion_pending || !self.busy.is_empty() {
            return false;
        }
        self.clear_stroke_overlay_data();
        if self.send_command(EngineCommand::BeginStroke { tool, spec, space }) {
            self.stroke_active = tool != Tool::Inspect;
            true
        } else {
            false
        }
    }

    fn complete_active_stroke(&mut self) -> bool {
        if !self.stroke_active && !self.stroke_completion_pending {
            return true;
        }
        if self.stroke_active && !self.stroke_overlay_voxels.is_empty() {
            self.stroke_overlay_revision = None;
            self.stroke_overlay_waiting_for_mesh = true;
        }
        match self.engine.send(EngineCommand::CompleteStroke) {
            Ok(()) => {
                self.stroke_active = false;
                self.stroke_completion_pending = false;
                true
            }
            Err(error) => {
                self.stroke_completion_pending = true;
                self.error_message = Some(error.to_string());
                self.egui_ctx.request_repaint();
                false
            }
        }
    }

    fn retry_pending_stroke_completion(&mut self) {
        if self.stroke_completion_pending {
            let _ = self.complete_active_stroke();
        }
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
        if self.dialog.is_some() {
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
        self.retry_pending_stroke_completion();
        self.process_dialog();
        if ctx.input(|input| input.viewport().close_requested()) {
            if self.allow_exit {
                return;
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if self.has_edit_interaction() {
                self.cancel_edit_interaction();
                self.preview_notice = Some("Current edit cancelled; close again to exit".into());
            } else if self.prompt.is_none() {
                self.guard_or_execute(GuardAction::Exit);
            }
        }
        if self.allow_exit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        self.toolbar(ui);
        self.status_bar(ui);
        self.repair_panel(ui);
        self.workspace(ui);
        self.handle_shortcuts(&ctx);
        self.show_prompts(&ctx);
    }
}

fn configure_style(ctx: &egui::Context, forced_colors: bool, reduced_motion: bool) {
    ctx.set_theme(egui::Theme::Dark);
    let mut visuals = egui::Visuals::dark();
    visuals.window_corner_radius = egui::CornerRadius::ZERO;
    visuals.menu_corner_radius = egui::CornerRadius::ZERO;
    visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::ZERO;
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::ZERO;
    visuals.widgets.hovered.corner_radius = egui::CornerRadius::ZERO;
    visuals.widgets.active.corner_radius = egui::CornerRadius::ZERO;
    visuals.widgets.open.corner_radius = egui::CornerRadius::ZERO;
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
    visuals.widgets.hovered.bg_fill = if forced_colors {
        Color32::WHITE
    } else {
        Color32::from_rgb(28, 45, 52)
    };
    visuals.widgets.hovered.fg_stroke =
        egui::Stroke::new(1.5, if forced_colors { Color32::BLACK } else { TEXT });
    visuals.widgets.active.bg_fill = if forced_colors { Color32::WHITE } else { TEAL };
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, MUTED);
    ctx.set_visuals(visuals);
    if reduced_motion {
        ctx.style_mut_of(egui::Theme::Dark, |style| style.animation_time = 0.0);
    }
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
    tool: Tool,
    forced_colors: bool,
) {
    let (fill, outline) = if forced_colors {
        (Color32::from_white_alpha(120), Color32::WHITE)
    } else if tool == Tool::Erase {
        (
            Color32::from_rgba_unmultiplied(255, 91, 69, 130),
            Color32::from_rgb(255, 154, 89),
        )
    } else {
        (
            Color32::from_rgba_unmultiplied(36, 200, 219, 125),
            Color32::from_rgb(121, 238, 248),
        )
    };
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
