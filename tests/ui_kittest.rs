mod common;

use std::{
    fs,
    path::PathBuf,
    process::Command,
    thread,
    time::{Duration, Instant},
};

use egui::accesskit::{Role, Toggled};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};
use tempfile::tempdir;
use voxel_ink::VoxelInkApp;

use common::write_binvox;

#[test]
fn toolbar_exposes_names_disabled_state_focus_order_and_tool_state() {
    let mut harness = app_harness(None);

    assert_eq!(
        harness
            .get_by_role_and_label(Role::Image, "VoxelInk teal and yellow logo")
            .accesskit_node()
            .label(),
        Some("VoxelInk teal and yellow logo".to_owned())
    );
    assert!(!harness.get_by_label("Open…").accesskit_node().is_disabled());
    for label in ["Save", "Save As…", "Close", "Undo", "Redo"] {
        assert!(
            harness.get_by_label(label).accesskit_node().is_disabled(),
            "{label} should be disabled without a document"
        );
    }
    assert!(
        harness
            .get_by_label("Fill enclosed cavities")
            .accesskit_node()
            .is_disabled()
    );

    assert_eq!(tool_toggled(&harness, "Inspect (I)"), Some(Toggled::True));
    assert_eq!(tool_toggled(&harness, "Paint (B)"), Some(Toggled::False));
    assert_eq!(tool_toggled(&harness, "Erase (E)"), Some(Toggled::False));
    assert_eq!(tool_toggled(&harness, "Bridge (L)"), Some(Toggled::False));

    harness.get_by_label("Open…").focus();
    harness.step();
    assert_eq!(focused_label(&harness).as_deref(), Some("Open…"));
    harness.key_press(egui::Key::Tab);
    harness.step();
    assert_eq!(
        focused_label(&harness).as_deref(),
        Some("Inspect (I)"),
        "disabled toolbar actions must be skipped in focus order"
    );
    harness.key_press(egui::Key::Tab);
    harness.step();
    assert_eq!(focused_label(&harness).as_deref(), Some("Paint (B)"));

    harness.key_press(egui::Key::B);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Paint (B)"), Some(Toggled::True));
    assert!(
        harness
            .get_by_role_and_label(Role::Slider, "Brush diameter in voxels")
            .accesskit_node()
            .is_disabled(),
        "brush settings should stay disabled until a volume is loaded"
    );

    harness.key_press(egui::Key::E);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Erase (E)"), Some(Toggled::True));
    harness.key_press(egui::Key::L);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Bridge (L)"), Some(Toggled::True));
    harness.key_press(egui::Key::I);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Inspect (I)"), Some(Toggled::True));
    assert!(
        harness
            .query_by_role_and_label(Role::Slider, "Brush diameter in voxels")
            .is_none(),
        "Inspect should hide repair-only brush controls"
    );
}

#[test]
fn repair_panel_guidance_and_controls_stay_inside_the_panel() {
    let harness = app_harness_with_size(None, [900.0, 620.0]);
    let panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    let content_right = panel.right() - 12.0;

    for label in [
        "Choose the repair workflow that fits the volume.",
        "Paint missing voxels or bridge gaps.",
        "Fill empty regions that cannot reach a volume boundary.",
        "Fill enclosed cavities",
    ] {
        let rect = harness.get_by_label(label).rect();
        assert!(
            rect.right() <= content_right,
            "{label:?} extends beyond the repair panel {panel:?}: {rect:?}"
        );
    }
    for removed_label in ["01", "02", "No point selection is needed."] {
        assert!(
            harness.query_by_label_contains(removed_label).is_none(),
            "obsolete workflow sequencing text {removed_label:?} should not be shown"
        );
    }
}

#[test]
fn loaded_document_has_synchronized_named_panes_and_keyboard_editing() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("keyboard.binvox");
    write_binvox(&path, [4, 3, 2], &[0; 24]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "4 × 3 × 2");

    assert!(!harness.get_by_label("Save").accesskit_node().is_disabled());
    for (role, label) in [
        (Role::Image, "Axial slice editor"),
        (Role::Image, "Coronal slice editor"),
        (Role::Image, "Sagittal slice editor"),
        (Role::Unknown, "Interactive 3D volume editor"),
    ] {
        assert_eq!(
            harness
                .get_by_role_and_label(role, label)
                .accesskit_node()
                .label(),
            Some(label.to_owned())
        );
    }

    harness.key_press(egui::Key::ArrowRight);
    harness.step();
    wait_for_label(&mut harness, "cursor 3, 1, 1");
    harness.key_press(egui::Key::PageDown);
    harness.step();
    wait_for_label(&mut harness, "cursor 3, 1, 0");

    harness.key_press(egui::Key::B);
    harness.step();
    harness.key_press(egui::Key::Space);
    harness.step();
    wait_for_label(&mut harness, "7 occupied");
    wait_for_label(&mut harness, "modified");
    assert_eq!(tool_toggled(&harness, "Paint (B)"), Some(Toggled::True));
}

#[test]
fn a_same_frame_slice_click_applies_and_completes_one_edit() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("quick-click.binvox");
    write_binvox(&path, [5, 5, 1], &[0; 25]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "5 × 5 × 1");
    harness.key_press(egui::Key::B);
    harness.step();

    harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .click();
    harness.step();

    wait_for_label(&mut harness, "13 occupied");
    wait_for_label(&mut harness, "modified");
}

#[test]
fn a_same_frame_slice_drag_keeps_distinct_press_and_release_samples() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("quick-drag.binvox");
    write_binvox(&path, [20, 7, 1], &[0; 140]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "20 × 7 × 1");
    harness.key_press(egui::Key::B);
    harness.step();

    let rect = harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .rect();
    let press = egui::pos2(rect.left() + rect.width() * 0.2, rect.center().y);
    let release = egui::pos2(rect.left() + rect.width() * 0.8, rect.center().y);
    for (pos, pressed) in [(press, true), (release, false)] {
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    harness.step();

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if current_occupied_count(&harness).is_some_and(|count| count > 13) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "same-frame drag collapsed to one brush footprint: {:?}",
            current_occupied_count(&harness)
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
    wait_for_label(&mut harness, "modified");
}

#[test]
fn every_view_can_use_the_full_workspace_and_escape_restores_the_grid() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("maximize.binvox");
    write_binvox(&path, [4, 3, 2], &[0; 24]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "4 × 3 × 2");

    for (button, visible_editor) in [
        ("Full screen Axial", "Axial slice editor"),
        ("Full screen Coronal", "Coronal slice editor"),
        ("Full screen Sagittal", "Sagittal slice editor"),
        ("Full screen 3D", "Interactive 3D volume editor"),
    ] {
        harness.get_by_label(button).click_accesskit();
        harness.step();
        harness.step();
        assert!(harness.query_by_label("Restore four views").is_some());
        for editor in [
            "Axial slice editor",
            "Coronal slice editor",
            "Sagittal slice editor",
            "Interactive 3D volume editor",
        ] {
            assert_eq!(
                harness.query_by_label(editor).is_some(),
                editor == visible_editor,
                "{button} showed the wrong workspace content"
            );
        }

        harness.key_press(egui::Key::Escape);
        harness.step();
        harness.step();
        for editor in [
            "Axial slice editor",
            "Coronal slice editor",
            "Sagittal slice editor",
            "Interactive 3D volume editor",
        ] {
            assert!(harness.query_by_label(editor).is_some());
        }
    }
    assert!(harness.query_by_label("Reset view").is_some());
}

#[test]
fn four_view_splitters_resize_the_grid_and_survive_full_screen_restore() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("resize-panes.binvox");
    write_binvox(&path, [4, 3, 2], &[0; 24]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "4 × 3 × 2");

    let column = harness
        .get_by_role_and_label(Role::Splitter, "Resize workspace columns")
        .rect();
    let row = harness
        .get_by_role_and_label(Role::Splitter, "Resize workspace rows")
        .rect();
    let column_grab = egui::pos2(column.center().x, column.top() + column.height() * 0.25);
    harness.hover_at(column_grab);
    harness.step();
    assert_eq!(
        harness.output().platform_output.cursor_icon,
        egui::CursorIcon::ResizeHorizontal
    );

    let target = column_grab + egui::vec2(120.0, 0.0);
    harness.drag_at(column_grab);
    harness.step();
    harness.hover_at(target);
    harness.step();
    harness.drop_at(target);
    harness.step();
    harness.step();

    let resized_column = harness
        .get_by_role_and_label(Role::Splitter, "Resize workspace columns")
        .rect();
    let resized_row = harness
        .get_by_role_and_label(Role::Splitter, "Resize workspace rows")
        .rect();
    assert!(
        resized_column.center().x > column.center().x + 100.0,
        "column splitter did not follow the drag: {column:?} -> {resized_column:?}"
    );
    assert!(
        (resized_row.center().y - row.center().y).abs() < 1.0,
        "column resizing unexpectedly moved the row splitter"
    );

    let saved_column_x = resized_column.center().x;
    let saved_panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    harness.get_by_label("Full screen Axial").click_accesskit();
    harness.step();
    harness.step();
    assert!(
        harness
            .query_by_role_and_label(Role::Splitter, "Resize workspace columns")
            .is_none()
    );
    assert!(
        harness
            .query_by_role_and_label(Role::Splitter, "Resize workspace rows")
            .is_none()
    );

    harness.key_press(egui::Key::Escape);
    harness.step();
    harness.step();
    let restored_column_x = harness
        .get_by_role_and_label(Role::Splitter, "Resize workspace columns")
        .rect()
        .center()
        .x;
    let restored_panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    assert!(
        (restored_column_x - saved_column_x).abs() < 1.0,
        "column splitter moved across full-screen restore: {saved_column_x} -> {restored_column_x}; panel {saved_panel:?} -> {restored_panel:?}"
    );
}

#[test]
fn fill_enclosed_cavities_is_a_single_clear_action_with_visible_feedback() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("make-solid.binvox");
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
        occupancy[(voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2])) as usize] = 1;
    }
    write_binvox(&path, dimensions, &occupancy);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "6 occupied");

    let action = harness.get_by_label("Fill enclosed cavities");
    assert!(!action.accesskit_node().is_disabled());
    action.click();
    harness.step();

    wait_for_label(&mut harness, "7 occupied");
    wait_for_label(&mut harness, "Filled 1 enclosed voxel");
    wait_for_label(&mut harness, "modified");
    assert!(!harness.get_by_label("Undo").accesskit_node().is_disabled());
}

#[test]
fn dirty_close_prompt_is_non_destructive_until_discard_is_chosen() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("dirty.binvox");
    write_binvox(&path, [2, 2, 1], &[0; 4]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "2 × 2 × 1");

    harness.key_press(egui::Key::B);
    harness.step();
    harness.key_press(egui::Key::Enter);
    harness.step();
    wait_for_label(&mut harness, "4 occupied");
    wait_for_label(&mut harness, "modified");

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::W);
    harness.step();
    assert!(harness.query_by_label("Save changes?").is_some());
    for label in ["Save", "Discard", "Cancel"] {
        assert!(
            !harness
                .query_all_by_label(label)
                .collect::<Vec<_>>()
                .is_empty()
        );
    }

    harness.get_by_label("Cancel").click();
    harness.step();
    wait_for_label_absence(&mut harness, "Save changes?");
    assert!(harness.query_by_label_contains("4 occupied").is_some());

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::W);
    harness.step();
    harness.get_by_label("Discard").click();
    harness.step();
    wait_for_label(&mut harness, "Open a NIfTI, BINVOX, or VOX volume to begin");
    assert!(harness.query_by_label("Axial slice editor").is_none());
}

#[test]
fn raw_pane_clicks_cannot_edit_behind_a_modal_prompt() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("modal-blocks-edit.binvox");
    write_binvox(&path, [20, 7, 1], &[0; 140]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "20 × 7 × 1");

    harness.key_press(egui::Key::B);
    harness.step();
    harness.key_press(egui::Key::Space);
    harness.step();
    wait_for_label(&mut harness, "13 occupied");

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::W);
    harness.step();
    assert!(harness.query_by_label("Save changes?").is_some());
    let rect = harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .rect();
    let position = egui::pos2(rect.left() + rect.width() * 0.8, rect.center().y);
    for pressed in [true, false] {
        harness.input_mut().events.push(egui::Event::PointerButton {
            pos: position,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
    }
    harness.step();
    for _ in 0..8 {
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }

    assert_eq!(current_occupied_count(&harness), Some(13));
}

#[test]
fn a_same_frame_focus_loss_prevents_a_new_raw_stroke() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("focus-loss.binvox");
    write_binvox(&path, [20, 7, 1], &[0; 140]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "20 × 7 × 1");
    harness.key_press(egui::Key::B);
    harness.step();

    let rect = harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .rect();
    let position = rect.center();
    harness.input_mut().events.push(egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    harness
        .input_mut()
        .events
        .push(egui::Event::WindowFocused(false));
    harness.input_mut().focused = false;
    harness.step();
    harness.input_mut().events.push(egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    harness
        .input_mut()
        .events
        .push(egui::Event::WindowFocused(true));
    harness.input_mut().focused = true;
    harness.step();
    for _ in 0..8 {
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }

    assert_eq!(current_occupied_count(&harness), Some(0));
}

#[test]
fn normalized_overwrite_requires_an_explicit_native_prompt_choice() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("normalized.binvox");
    let mut bytes = b"#binvox 2\ndim 1 1 1\ntranslate 0 0 0\nscale 1\ndata\n".to_vec();
    bytes.extend_from_slice(&[2, 1]);
    fs::write(&path, bytes).unwrap();
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "non-binary source normalized; save required");

    harness.get_by_label("Save").click();
    harness.step();
    assert!(
        harness
            .query_by_label("Normalized non-binary source")
            .is_some()
    );
    assert!(harness.query_by_label("Save a Copy…").is_some());
    assert!(harness.query_by_label("Overwrite").is_some());
    assert!(harness.query_by_label("Cancel").is_some());

    harness.get_by_label("Cancel").click();
    harness.step();
    wait_for_label_absence(&mut harness, "Normalized non-binary source");
    assert!(
        harness
            .query_by_label("non-binary source normalized; save required")
            .is_some()
    );
}

#[test]
fn changed_source_surfaces_error_then_refuses_overwrite() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("changed.binvox");
    write_binvox(&path, [1, 1, 1], &[0]);
    let mut harness = app_harness(Some(path.clone()));
    wait_for_label(&mut harness, "1 × 1 × 1");

    harness.key_press(egui::Key::B);
    harness.step();
    harness.key_press(egui::Key::Space);
    harness.step();
    wait_for_label(&mut harness, "1 occupied");
    wait_for_label(&mut harness, "modified");
    fs::write(&path, b"externally replaced").unwrap();

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::S);
    harness.step();
    wait_for_label(&mut harness, "VoxelInk could not complete the operation");
    assert!(
        harness
            .query_by_label_contains("source file changed outside VoxelInk")
            .is_some()
    );
    assert_eq!(fs::read(&path).unwrap(), b"externally replaced");

    harness.get_by_label("OK").click_accesskit();
    harness.step();
    wait_for_label(&mut harness, "Source changed outside VoxelInk");
    assert!(harness.query_by_label("Save a Copy…").is_some());
    assert!(harness.query_by_label("Cancel").is_some());
    assert!(harness.query_by_label("Overwrite").is_none());
    harness.get_by_label("Cancel").click_accesskit();
    harness.step();
    assert!(harness.query_by_label_contains("1 occupied").is_some());
}

#[test]
fn forced_colors_and_reduced_motion_are_applied() {
    const CHILD_MARKER: &str = "VOXEL_INK_KITTEST_STYLE_CHILD";
    if std::env::var_os(CHILD_MARKER).is_some() {
        struct ConfiguredStyleApp {
            app: VoxelInkApp,
            visuals: egui::Visuals,
            animation_time: f32,
        }

        impl eframe::App for ConfiguredStyleApp {
            fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
                self.app.ui(ui, frame);
            }
        }

        let harness = Harness::builder()
            .with_size([1_600.0, 900.0])
            .with_max_steps(64)
            .build_eframe(|creation_context| {
                let app = VoxelInkApp::new(creation_context, None);
                let configured = creation_context.egui_ctx.style_of(egui::Theme::Dark);
                ConfiguredStyleApp {
                    app,
                    visuals: configured.visuals.clone(),
                    animation_time: configured.animation_time,
                }
            });
        let visuals = &harness.state().visuals;
        assert_eq!(visuals.panel_fill, egui::Color32::BLACK);
        assert_eq!(visuals.window_fill, egui::Color32::BLACK);
        assert_eq!(visuals.override_text_color, Some(egui::Color32::WHITE));
        assert_eq!(visuals.selection.bg_fill, egui::Color32::WHITE);
        assert_eq!(visuals.widgets.inactive.bg_fill, egui::Color32::BLACK);
        assert_eq!(visuals.widgets.hovered.bg_fill, egui::Color32::WHITE);
        assert_eq!(visuals.window_corner_radius, egui::CornerRadius::ZERO);
        assert_eq!(visuals.menu_corner_radius, egui::CornerRadius::ZERO);
        assert_eq!(
            visuals.widgets.noninteractive.corner_radius,
            egui::CornerRadius::ZERO
        );
        assert_eq!(
            visuals.widgets.inactive.corner_radius,
            egui::CornerRadius::ZERO
        );
        assert_eq!(
            visuals.widgets.hovered.corner_radius,
            egui::CornerRadius::ZERO
        );
        assert_eq!(
            visuals.widgets.active.corner_radius,
            egui::CornerRadius::ZERO
        );
        assert_eq!(visuals.widgets.open.corner_radius, egui::CornerRadius::ZERO);
        assert_eq!(harness.state().animation_time, 0.0);
        assert!(contrast_ratio(egui::Color32::WHITE, egui::Color32::BLACK) >= 7.0);
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("forced_colors_and_reduced_motion_are_applied")
        .arg("--nocapture")
        .env(CHILD_MARKER, "1")
        .env("VOXEL_INK_FORCED_COLORS", "1")
        .env("VOXEL_INK_REDUCED_MOTION", "1")
        .status()
        .unwrap();
    assert!(status.success());
}

fn app_harness(initial_path: Option<PathBuf>) -> Harness<'static, VoxelInkApp> {
    app_harness_with_size(initial_path, [1_600.0, 900.0])
}

fn app_harness_with_size(
    initial_path: Option<PathBuf>,
    size: [f32; 2],
) -> Harness<'static, VoxelInkApp> {
    Harness::builder()
        .with_size(size)
        .with_max_steps(64)
        .build_eframe(move |creation_context| VoxelInkApp::new(creation_context, initial_path))
}

fn wait_for_label(harness: &mut Harness<'_, VoxelInkApp>, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if harness.query_by_label_contains(text).is_some() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for accessible label containing {text:?}"
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
}

fn wait_for_label_absence(harness: &mut Harness<'_, VoxelInkApp>, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if harness.query_by_label_contains(text).is_none() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for accessible label containing {text:?} to disappear"
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
}

fn tool_toggled(harness: &Harness<'_, VoxelInkApp>, label: &str) -> Option<Toggled> {
    harness.get_by_label(label).accesskit_node().toggled()
}

fn focused_label(harness: &Harness<'_, VoxelInkApp>) -> Option<String> {
    harness
        .root()
        .children_recursive()
        .find(|node| node.is_focused())
        .and_then(|node| node.accesskit_node().label())
}

fn current_occupied_count(harness: &Harness<'_, VoxelInkApp>) -> Option<u64> {
    harness.root().children_recursive().find_map(|node| {
        let accessible = node.accesskit_node();
        let text = accessible.value().or_else(|| accessible.label())?;
        let words = text.split_whitespace().collect::<Vec<_>>();
        words
            .windows(2)
            .find(|pair| pair[1] == "occupied")
            .and_then(|pair| pair[0].replace(',', "").parse().ok())
    })
}

fn contrast_ratio(foreground: egui::Color32, background: egui::Color32) -> f32 {
    let foreground = relative_luminance(foreground);
    let background = relative_luminance(background);
    (foreground.max(background) + 0.05) / (foreground.min(background) + 0.05)
}

fn relative_luminance(color: egui::Color32) -> f32 {
    let [red, green, blue, _] = color.to_array().map(|channel| f32::from(channel) / 255.0);
    [red, green, blue]
        .into_iter()
        .zip([0.2126, 0.7152, 0.0722])
        .map(|(channel, weight)| {
            let linear = if channel <= 0.04045 {
                channel / 12.92
            } else {
                ((channel + 0.055) / 1.055).powf(2.4)
            };
            linear * weight
        })
        .sum()
}
