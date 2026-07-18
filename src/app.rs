use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crossbeam_channel::{Receiver, bounded};
use egui::{Color32, TextureHandle, TextureOptions, Vec2};

use crate::{
    EngineCommand, EngineEvent, EngineHandle,
    brush::BrushDiameter,
    codecs::{FormatFamily, SourceVariant},
    engine::{BusyKind, JobId, PointerSample, SlicePatch, StrokeSpace},
    io::SaveMode,
    renderer::{SceneBridge, VolumePaintCallback},
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
    sample: Option<PointerSample>,
    complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkspacePane {
    Slice(SliceAxis),
    Volume3D,
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
    logo: TextureHandle,
    scene: SceneBridge,
    target_format: wgpu::TextureFormat,
    adapter_name: String,
    tool: Tool,
    brush_diameter: u8,
    active_slice: SliceAxis,
    busy: HashMap<JobId, BusyKind>,
    pending_saves: HashMap<JobId, SaveRequest>,
    prompt: Option<Prompt>,
    error_message: Option<String>,
    preview_notice: Option<String>,
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
            logo,
            scene,
            target_format,
            adapter_name,
            tool: Tool::Inspect,
            brush_diameter: 1,
            active_slice: SliceAxis::Axial,
            busy: HashMap::new(),
            pending_saves: HashMap::new(),
            prompt: None,
            error_message: None,
            preview_notice: None,
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
                    self.epoch = epoch;
                    self.slice_revision = summary.revision;
                    self.slice_preview_generation = 0;
                    self.summary = Some(summary);
                    self.stroke_active = false;
                    self.stroke_completion_pending = false;
                    self.maximized_pane = None;
                    self.replace_slices(ctx, slices);
                    self.preview_notice = None;
                }
                EngineEvent::DocumentClosed { epoch } => {
                    self.epoch = epoch;
                    self.summary = None;
                    self.slices.clear();
                    self.textures.clear();
                    self.slice_revision = 0;
                    self.slice_preview_generation = 0;
                    self.stroke_active = false;
                    self.stroke_completion_pending = false;
                    self.maximized_pane = None;
                    self.scene.clear();
                    self.preview_notice = None;
                }
                EngineEvent::Summary { epoch, summary } if epoch == self.epoch => {
                    let revision_changed = self
                        .summary
                        .as_ref()
                        .is_some_and(|current| current.revision != summary.revision);
                    self.summary = Some(summary);
                    if revision_changed {
                        self.preview_notice = None;
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
                EngineEvent::MadeSolid {
                    epoch,
                    revision: _,
                    added_voxels,
                } if epoch == self.epoch => {
                    self.preview_notice = Some(if added_voxels == 0 {
                        "Already solid left ↔ right".into()
                    } else {
                        solidify_notice(self.summary.as_ref(), added_voxels)
                    });
                }
                EngineEvent::MeshReset {
                    epoch,
                    revision,
                    dimensions,
                    index_to_world,
                } => self
                    .scene
                    .reset(epoch, revision, dimensions, index_to_world),
                EngineEvent::MeshInvalidated {
                    epoch,
                    revision,
                    chunks,
                } => self.scene.invalidate(epoch, revision, chunks),
                EngineEvent::MeshChunk { tag, quads } => self.scene.insert_chunk(tag, quads),
                EngineEvent::PreviewTooComplex { epoch, .. } if epoch == self.epoch => {
                    self.preview_notice = Some(
                        "3D preview too complex; slice editing remains fully available".into(),
                    );
                }
                _ => {}
            }
        }
        if exhausted_budget {
            ctx.request_repaint();
        }
        if self.scene.take_remesh_request() && self.summary.is_some() {
            let _ = self.send_command(EngineCommand::RebuildPreview);
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
        if self.maximized_pane.is_some()
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.maximized_pane = None;
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
        if open {
            self.guard_or_execute(GuardAction::OpenDialog);
        }
        if save_as {
            self.open_save_dialog(DialogPurpose::SaveAs);
        } else if save {
            self.request_overwrite(None);
        }
        if close {
            self.guard_or_execute(GuardAction::CloseDocument);
        }
        if redo_shift || redo_y {
            let _ = self.send_command(EngineCommand::Redo);
        } else if undo {
            let _ = self.send_command(EngineCommand::Undo);
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::I)) {
            self.tool = Tool::Inspect;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::B)) {
            self.tool = Tool::Brush;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::E)) {
            self.tool = Tool::Eraser;
        }
        if self.pane_keyboard_focus || !ctx.egui_wants_keyboard_input() {
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
        if edit && self.tool != Tool::Inspect {
            let diameter = BrushDiameter::new(self.brush_diameter).unwrap_or_default();
            if self.begin_stroke(self.tool, diameter, StrokeSpace::Slice(self.active_slice)) {
                let _ = self.send_command(EngineCommand::SliceSample {
                    axis: self.active_slice,
                    voxel: cursor,
                });
                let _ = self.complete_active_stroke();
            }
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
                    if ui
                        .add_enabled(self.busy.is_empty(), egui::Button::new("Open…"))
                        .on_hover_text("Open volume (Ctrl+O)")
                        .clicked()
                    {
                        self.guard_or_execute(GuardAction::OpenDialog);
                    }
                    let has_document = self.summary.is_some();
                    let ready = has_document && self.busy.is_empty();
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
                        let _ = self.send_command(EngineCommand::Undo);
                    }
                    if ui
                        .add_enabled(ready && can_redo, egui::Button::new("Redo"))
                        .on_hover_text("Redo (Ctrl+Shift+Z or Ctrl+Y)")
                        .clicked()
                    {
                        let _ = self.send_command(EngineCommand::Redo);
                    }
                    ui.separator();
                    ui.strong("VOLUME");
                    if ui
                        .add_enabled(
                            ready,
                            egui::Button::new(
                                egui::RichText::new("Make solid left ↔ right").strong(),
                            ),
                        )
                        .on_hover_text(
                            "Fill every gap between the model's left and right surfaces in one undoable step; this also fills internal openings such as eye sockets and sinuses",
                        )
                        .clicked()
                    {
                        self.preview_notice = None;
                        if let Err(error) = self.engine.start_make_solid() {
                            self.error_message = Some(error.to_string());
                        }
                    }
                    ui.separator();
                    ui.strong("TOOLS");
                    ui.add_enabled_ui(self.busy.is_empty(), |ui| {
                        ui.selectable_value(&mut self.tool, Tool::Inspect, "Inspect (I)")
                            .on_hover_text("Inspect slices; drag the 3D view to rotate it");
                        ui.selectable_value(&mut self.tool, Tool::Brush, "Brush (B)")
                            .on_hover_text("Add filled voxels with the primary mouse button");
                        ui.selectable_value(&mut self.tool, Tool::Eraser, "Eraser (E)")
                            .on_hover_text("Remove voxels with the primary mouse button");
                    });
                    ui.add_enabled_ui(self.tool != Tool::Inspect && self.busy.is_empty(), |ui| {
                        ui.label("Brush size (voxels)");
                        let changed = ui
                            .add(
                                egui::Slider::new(&mut self.brush_diameter, 1..=31)
                                    .step_by(2.0)
                                    .show_value(true)
                                    .text("brush size in voxels"),
                            )
                            .changed();
                        if changed && self.brush_diameter.is_multiple_of(2) {
                            self.brush_diameter = (self.brush_diameter + 1).min(31);
                        }
                    });
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
                    if let Some((&job_id, kind)) = self.busy.iter().next() {
                        if self.reduced_motion {
                            ui.label("Working");
                        } else {
                            ui.spinner();
                        }
                        ui.label(match kind {
                            BusyKind::Loading => "Loading…",
                            BusyKind::Saving => "Saving…",
                            BusyKind::MakingSolid => "Making the volume solid…",
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
                if let Some(sample) = actions.sample
                    && let Err(error) = self.engine.send_latest_sample(sample)
                {
                    self.error_message = Some(error.to_string());
                }
                if actions.complete {
                    let _ = self.complete_active_stroke();
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
        let image_rect = fit_rect(content, width as f32 / height as f32);
        ui.painter().image(
            texture_id,
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().text(
            image_rect.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "Wheel: change slice   Shift + wheel: jump 10",
            egui::FontId::proportional(12.0),
            MUTED,
        );
        let response = ui.interact(
            image_rect,
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
        if response.hovered()
            && self.busy.is_empty()
            && !self.stroke_active
            && !self.stroke_completion_pending
        {
            let (scroll, fast) = ui.input(|input| {
                let delta = input.smooth_scroll_delta;
                let dominant = if delta.y.abs() >= delta.x.abs() {
                    delta.y
                } else {
                    delta.x
                };
                (dominant, input.modifiers.shift)
            });
            if scroll.abs() > f32::EPSILON
                && let Some(summary) = &self.summary
            {
                let mut cursor = summary.cursor;
                let normal = axis.normal_axis();
                let step = if fast { 10 } else { 1 };
                cursor[normal] = if scroll > 0.0 {
                    cursor[normal]
                        .saturating_add(step)
                        .min(summary.dimensions[normal] - 1)
                } else {
                    cursor[normal].saturating_sub(step)
                };
                self.active_slice = axis;
                let _ = self.send_command(EngineCommand::SetCursor(cursor));
            }
        }
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
        let pointer_voxel = response.interact_pointer_pos().and_then(|position| {
            self.summary.as_ref().map(|summary| {
                slice_pointer_to_voxel(axis, position, image_rect, width, height, summary.cursor)
            })
        });
        if !self.busy.is_empty() {
            return;
        }
        if self.tool != Tool::Inspect {
            if response.drag_started()
                && let Some(voxel) = pointer_voxel
            {
                self.active_slice = axis;
                if self.begin_stroke(
                    self.tool,
                    BrushDiameter::new(self.brush_diameter).unwrap_or_default(),
                    StrokeSpace::Slice(axis),
                ) && !self.send_command(EngineCommand::SliceSample { axis, voxel })
                {
                    let _ = self.complete_active_stroke();
                }
            } else if response.dragged()
                && let Some(voxel) = pointer_voxel
            {
                actions.sample = Some(PointerSample::Slice { axis, voxel });
            }
            if response.drag_stopped() {
                if let Some(voxel) = pointer_voxel {
                    actions.sample = Some(PointerSample::Slice { axis, voxel });
                }
                actions.complete = true;
            } else if response.clicked()
                && let Some(voxel) = pointer_voxel
                && self.begin_stroke(
                    self.tool,
                    BrushDiameter::new(self.brush_diameter).unwrap_or_default(),
                    StrokeSpace::Slice(axis),
                )
            {
                let _ = self.send_command(EngineCommand::SliceSample { axis, voxel });
                let _ = self.complete_active_stroke();
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
        let controls = if self.tool == Tool::Inspect {
            "Left / middle drag: rotate   Right drag: move   Wheel: zoom"
        } else {
            "Left drag: edit   Middle drag: rotate   Right drag: move   Wheel: zoom"
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
        let pointer_ray = response.interact_pointer_pos().map(|position| {
            let normalized = [
                ((position.x - content.left()) / content.width()) * 2.0 - 1.0,
                1.0 - ((position.y - content.top()) / content.height()) * 2.0,
            ];
            self.scene.ray_for_pointer(normalized)
        });
        let Some(summary) = &self.summary else {
            return;
        };
        let slice_coordinate = summary.cursor[self.active_slice.normal_axis()];
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(ray) = pointer_ray
        {
            if self.begin_stroke(
                self.tool,
                BrushDiameter::new(self.brush_diameter).unwrap_or_default(),
                StrokeSpace::Volume3D,
            ) && !self.send_command(EngineCommand::RaySample {
                ray,
                active_slice: self.active_slice,
                slice_coordinate,
            }) {
                let _ = self.complete_active_stroke();
            }
        } else if response.dragged_by(egui::PointerButton::Primary)
            && let Some(ray) = pointer_ray
        {
            actions.sample = Some(PointerSample::Ray {
                ray,
                active_slice: self.active_slice,
                slice_coordinate,
            });
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(ray) = pointer_ray {
                actions.sample = Some(PointerSample::Ray {
                    ray,
                    active_slice: self.active_slice,
                    slice_coordinate,
                });
            }
            actions.complete = true;
        } else if response.clicked_by(egui::PointerButton::Primary)
            && let Some(ray) = pointer_ray
            && self.begin_stroke(
                self.tool,
                BrushDiameter::new(self.brush_diameter).unwrap_or_default(),
                StrokeSpace::Volume3D,
            )
        {
            let _ = self.send_command(EngineCommand::RaySample {
                ray,
                active_slice: self.active_slice,
                slice_coordinate,
            });
            let _ = self.complete_active_stroke();
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
        let had_active_stroke = self.stroke_active || self.stroke_completion_pending;
        if had_active_stroke && !self.complete_active_stroke() {
            return;
        }
        if had_active_stroke || self.summary.as_ref().is_some_and(DocumentSummary::is_dirty) {
            self.prompt = Some(Prompt::Unsaved(action));
        } else {
            self.execute_guard(action);
        }
    }

    fn execute_guard(&mut self, action: GuardAction) {
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
        if !self.complete_active_stroke() {
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

    fn begin_stroke(&mut self, tool: Tool, diameter: BrushDiameter, space: StrokeSpace) -> bool {
        if self.stroke_active || self.stroke_completion_pending || !self.busy.is_empty() {
            return false;
        }
        if self.send_command(EngineCommand::BeginStroke {
            tool,
            diameter,
            space,
        }) {
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
        if !self.complete_active_stroke() {
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
                TextureOptions::NEAREST,
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
            texture.set(image, TextureOptions::NEAREST);
        } else {
            let texture = ctx.load_texture(
                format!("voxel-slice-{axis:?}"),
                image,
                TextureOptions::NEAREST,
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
            if self.prompt.is_none() {
                self.guard_or_execute(GuardAction::Exit);
            }
        }
        if self.allow_exit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        self.toolbar(ui);
        self.status_bar(ui);
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

fn solidify_notice(summary: Option<&DocumentSummary>, added_voxels: u64) -> String {
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
        || format!("Solid left ↔ right: +{count} {voxel_label} · Undo restores original"),
        |volume| {
            let volume = if volume >= 0.1 {
                format!("{volume:.1}")
            } else {
                format!("{volume:.3}")
            };
            format!(
                "Solid left ↔ right: +{count} {voxel_label} ({volume} cm³) · Undo restores original"
            )
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
) -> [u32; 3] {
    let u =
        (((position.x - rect.left()) / rect.width()).clamp(0.0, 0.999_999) * width as f32) as u32;
    let v =
        (((position.y - rect.top()) / rect.height()).clamp(0.0, 0.999_999) * height as f32) as u32;
    match axis {
        SliceAxis::Axial => [u, v, cursor[2]],
        SliceAxis::Coronal => [u, cursor[1], v],
        SliceAxis::Sagittal => [cursor[0], u, v],
    }
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
            [15, 7, 5]
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
