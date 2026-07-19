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
    assert!(
        harness
            .query_by_role_and_label(Role::Slider, "Brush diameter in voxels")
            .is_some(),
        "Erase should expose brush settings"
    );
    harness.key_press(egui::Key::L);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Bridge (L)"), Some(Toggled::True));
    assert!(
        harness
            .query_by_role_and_label(Role::Slider, "Brush diameter in voxels")
            .is_some(),
        "Bridge should expose brush settings"
    );
    harness.key_press(egui::Key::I);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Inspect (I)"), Some(Toggled::True));
    assert!(
        harness
            .query_by_role_and_label(Role::Slider, "Brush diameter in voxels")
            .is_none(),
        "Inspect should hide brush-only settings"
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
        "Adjust flight movement and voxel surface detail.",
        "Show voxel outlines",
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
fn loaded_repair_cards_keep_their_gutter_when_the_scrollbar_appears() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("scrolling-repair-panel.binvox");
    write_binvox(&path, [5, 5, 5], &[0; 125]);
    let mut harness = app_harness_with_size(Some(path), [900.0, 620.0]);
    wait_for_label(&mut harness, "5 × 5 × 5");

    let panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    let inspect_card = harness.get_by_label("Seal openings workflow").rect();
    let inspect_left_margin = inspect_card.left() - panel.left();
    let inspect_right_margin = panel.right() - inspect_card.right();
    let scroll_bar_width = egui::style::ScrollStyle::solid().allocated_width();
    assert!(
        inspect_left_margin >= 11.0
            && inspect_right_margin >= inspect_left_margin + scroll_bar_width - 1.0,
        "Inspect lost its content gutter or scrollbar rail: {inspect_left_margin} vs {inspect_right_margin}"
    );

    harness.key_press(egui::Key::B);
    for _ in 0..4 {
        harness.step();
    }

    let paint_card = harness.get_by_label("Seal openings workflow").rect();
    assert!(
        paint_card.height() > inspect_card.height() + 100.0,
        "Paint should reveal its brush settings below the compact Inspect card: {inspect_card:?} -> {paint_card:?}"
    );
    for label in [
        "Seal openings workflow",
        "3D view settings",
        "Fill interior workflow",
    ] {
        let card = harness.get_by_label(label).rect();
        let left_margin = card.left() - panel.left();
        let right_margin = panel.right() - card.right();
        assert!(
            left_margin >= 11.0 && right_margin >= 11.0,
            "{label:?} lost its panel margin after scrolling: panel {panel:?}, card {card:?}"
        );
        assert!(
            right_margin >= left_margin + scroll_bar_width - 1.0,
            "{label:?} does not reserve its scrollbar outside the content gutter: {left_margin} vs {right_margin}"
        );
    }
}

#[test]
fn expanded_workspace_keeps_repair_cards_inside_the_parent_gutter() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("expanded-repair-panel.nii");
    write_nifti_u8(&path, [5, 5, 5], &[0; 125]);
    let mut harness = app_harness_with_size(Some(path), [1_293.0, 816.0]);
    wait_for_label(&mut harness, "5 × 5 × 5");
    let initial_panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    assert!(
        initial_panel.width() <= 321.0,
        "millimetre controls expanded the repair panel without user input: {initial_panel:?}"
    );
    let resize_start = egui::pos2(initial_panel.right(), initial_panel.center().y);
    let resize_target = resize_start + egui::vec2(160.0, 0.0);
    harness.hover_at(resize_start);
    harness.step();
    harness.drag_at(resize_start);
    harness.step();
    harness.hover_at(resize_target);
    harness.step();
    harness.drop_at(resize_target);
    for _ in 0..3 {
        harness.step();
    }
    let resized_panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    assert!(
        resized_panel.width() > initial_panel.width() + 50.0,
        "the regression must exercise a wide repair panel: {initial_panel:?} -> {resized_panel:?}"
    );
    harness.key_press(egui::Key::B);
    harness.step();
    harness.get_by_label("Full screen 3D").click_accesskit();
    for _ in 0..4 {
        harness.step();
    }

    let panel = harness
        .get_by_role_and_label(Role::Pane, "Repair volume controls")
        .rect();
    let scroll_bar_width = egui::style::ScrollStyle::solid().allocated_width();
    for label in [
        "Seal openings workflow",
        "3D view settings",
        "Fill interior workflow",
    ] {
        let card = harness.get_by_label(label).rect();
        let left_margin = card.left() - panel.left();
        let right_margin = panel.right() - card.right();
        assert!(
            left_margin >= 11.0 && right_margin >= left_margin + scroll_bar_width - 1.0,
            "{label:?} escaped the repair panel gutter in the expanded workspace: panel {panel:?}, card {card:?}"
        );
    }
}

#[test]
fn zoomed_focused_slice_outline_is_clipped_to_its_editor() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("focused-slice-outline.binvox");
    write_binvox(&path, [4, 3, 2], &[0; 24]);
    let mut harness = app_harness_with_size(Some(path), [900.0, 700.0]);
    wait_for_label(&mut harness, "4 × 3 × 2");

    let editor = harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .rect();
    harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .click();
    harness.step();
    harness.hover_at(editor.center());
    harness.input_mut().modifiers = egui::Modifiers::CTRL;
    harness.input_mut().events.push(egui::Event::Zoom(2.5));
    harness.step();
    harness.input_mut().modifiers = egui::Modifiers::NONE;
    harness.step();

    fn collect_focus_outlines(
        shape: &egui::epaint::Shape,
        clip_rect: egui::Rect,
        outlines: &mut Vec<(egui::Rect, egui::Rect)>,
    ) {
        match shape {
            egui::epaint::Shape::Rect(rect)
                if rect.stroke.width == 2.0
                    && rect.stroke.color == egui::Color32::from_rgb(255, 193, 49) =>
            {
                outlines.push((rect.rect, clip_rect));
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_focus_outlines(shape, clip_rect, outlines);
                }
            }
            _ => {}
        }
    }

    let mut outlines = Vec::new();
    for clipped in &harness.output().shapes {
        collect_focus_outlines(&clipped.shape, clipped.clip_rect, &mut outlines);
    }
    let (outline, clip_rect) = outlines
        .into_iter()
        .find(|(outline, _)| outline.width() > editor.width() || outline.height() > editor.height())
        .expect("the focused zoomed slice should paint an oversized focus outline");
    let editor = harness
        .get_by_role_and_label(Role::Image, "Axial slice editor")
        .rect();
    assert!(
        !clip_rect.contains_rect(outline),
        "the regression needs an outline that actually crosses its clip: {outline:?} inside {clip_rect:?}"
    );
    assert!(
        (clip_rect.min - editor.min).length() < 1.0 && (clip_rect.max - editor.max).length() < 1.0,
        "the focused slice outline clip {clip_rect:?} does not match its editor {editor:?}"
    );

    fn collect_painted_shapes<'a>(
        shape: &'a egui::epaint::Shape,
        shapes: &mut Vec<&'a egui::epaint::Shape>,
    ) {
        if let egui::epaint::Shape::Vec(children) = shape {
            for child in children {
                collect_painted_shapes(child, shapes);
            }
        } else {
            shapes.push(shape);
        }
    }

    let mut painted_shapes = Vec::new();
    for clipped in &harness.output().shapes {
        collect_painted_shapes(&clipped.shape, &mut painted_shapes);
    }
    let (hint_text_index, hint_text) = painted_shapes
        .iter()
        .enumerate()
        .find_map(|(index, shape)| match shape {
            egui::epaint::Shape::Text(text) if text.galley.text().starts_with("Wheel") => {
                Some((index, text))
            }
            _ => None,
        })
        .expect("the slice interaction hint should be painted");
    let hint_text_rect = hint_text.visual_bounding_rect();
    let (hint_background_index, hint_rect) = painted_shapes[..hint_text_index]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, shape)| match shape {
            egui::epaint::Shape::Rect(rect)
                if rect.fill.a() > 200 && rect.rect.contains_rect(hint_text_rect) =>
            {
                Some((index, rect.rect))
            }
            _ => None,
        })
        .expect("the slice interaction hint should have an opaque background");
    let yellow = egui::Color32::from_rgb(255, 193, 49);
    let overlapping_slice_marks = painted_shapes
        .iter()
        .enumerate()
        .filter_map(|(index, shape)| match shape {
            egui::epaint::Shape::LineSegment { points, stroke }
                if stroke.color == yellow
                    && egui::Rect::from_two_pos(points[0], points[1])
                        .expand(stroke.width.max(0.5))
                        .intersects(hint_rect) =>
            {
                Some(index)
            }
            egui::epaint::Shape::Rect(rect)
                if rect.stroke.color == yellow && rect.rect.intersects(hint_rect) =>
            {
                Some(index)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        overlapping_slice_marks.len() >= 2,
        "the regression needs both focused-slice and crosshair marks under the hint: {overlapping_slice_marks:?}"
    );
    assert!(
        overlapping_slice_marks
            .iter()
            .all(|index| *index < hint_background_index),
        "slice marks were painted above the interaction hint: marks {overlapping_slice_marks:?}, hint background {hint_background_index}"
    );
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
fn fly_selector_is_accessible_without_resizing_and_expand_stays_independent() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fly-selector.binvox");
    write_binvox(&path, [4, 3, 2], &[0; 24]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "4 × 3 × 2");

    let orbit = harness.get_by_label("Orbit view");
    let fly = harness.get_by_label("Fly view");
    assert_eq!(orbit.accesskit_node().toggled(), Some(Toggled::True));
    assert_eq!(fly.accesskit_node().toggled(), Some(Toggled::False));
    assert_selected_view_button_text(&harness, "Orbit view", "Orbit");

    let fly_speed = harness.get_by_role_and_label(Role::Slider, "Flight speed");
    assert!(
        fly_speed.accesskit_node().is_disabled(),
        "flight speed must wait until the first Fly viewpoint is initialized"
    );
    assert_eq!(
        harness
            .get_by_label("Show voxel outlines")
            .accesskit_node()
            .toggled(),
        Some(Toggled::True)
    );

    let initial_editor = harness.get_by_label("Interactive 3D volume editor").rect();
    fly.click_accesskit();
    harness.step();
    harness.step();

    assert_eq!(
        harness
            .get_by_label("Orbit view")
            .accesskit_node()
            .toggled(),
        Some(Toggled::False)
    );
    assert_eq!(
        harness.get_by_label("Fly view").accesskit_node().toggled(),
        Some(Toggled::True)
    );
    assert_selected_view_button_text(&harness, "Fly view", "Fly");
    let fly_speed = harness.get_by_role_and_label(Role::Slider, "Flight speed");
    assert!(!fly_speed.accesskit_node().is_disabled());
    let speed = fly_speed.accesskit_node().numeric_value().unwrap();
    let maximum = fly_speed.accesskit_node().max_numeric_value().unwrap();
    assert!(
        speed <= maximum * 0.051,
        "the initial Fly speed should be much slower than the available maximum: {speed} / {maximum}"
    );

    harness
        .get_by_label("Show voxel outlines")
        .click_accesskit();
    harness.step();
    assert_eq!(
        harness
            .get_by_label("Show voxel outlines")
            .accesskit_node()
            .toggled(),
        Some(Toggled::False)
    );
    assert_rect_near(
        harness.get_by_label("Interactive 3D volume editor").rect(),
        initial_editor,
    );
    assert!(
        harness
            .query_by_label("Click the 3D view to capture the mouse")
            .is_some()
    );

    harness.get_by_label("Full screen 3D").click_accesskit();
    harness.step();
    harness.step();

    let expanded_editor = harness.get_by_label("Interactive 3D volume editor").rect();
    assert!(
        expanded_editor.width() > initial_editor.width()
            && expanded_editor.height() > initial_editor.height(),
        "Expand did not remain independent of Fly mode: {initial_editor:?} -> {expanded_editor:?}"
    );
    assert!(harness.query_by_label("Restore four views").is_some());
    assert_eq!(
        harness.get_by_label("Fly view").accesskit_node().toggled(),
        Some(Toggled::True)
    );
}

#[test]
fn fly_speed_slider_keeps_its_minimum_when_the_viewport_captures() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fly-speed-capture.binvox");
    write_binvox(&path, [9, 9, 9], &[1; 729]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "729 occupied");

    harness.get_by_label("Fly view").click_accesskit();
    harness.step();
    harness.step();

    let selected_speed = drag_slider_to_minimum(&mut harness, "Flight speed");

    let viewport_center = harness
        .get_by_label("Interactive 3D volume editor")
        .rect()
        .center();
    queue_primary_button(&mut harness, viewport_center, true);
    harness.step();
    queue_primary_button(&mut harness, viewport_center, false);
    harness.input_mut().events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, f32::MAX),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    });
    harness.step();
    harness.step();

    let captured_speed = harness
        .get_by_role_and_label(Role::Slider, "Flight speed")
        .accesskit_node()
        .numeric_value()
        .unwrap();
    assert_eq!(
        captured_speed, selected_speed,
        "a pointer-lock wheel spike during the capture release changed the selected speed"
    );

    harness.input_mut().events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, 120.0),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    });
    harness.step();
    harness.step();
    let adjusted_speed = harness
        .get_by_role_and_label(Role::Slider, "Flight speed")
        .accesskit_node()
        .numeric_value()
        .unwrap();
    assert!(
        adjusted_speed > selected_speed,
        "the wheel should adjust speed after the capture click is fully released"
    );
}

#[test]
fn brush_diameter_sliders_keep_their_minimum_when_fly_captures() {
    let directory = tempdir().unwrap();
    let binvox_path = directory.path().join("brush-capture.binvox");
    write_binvox(&binvox_path, [5, 5, 5], &[1; 125]);
    assert_brush_minimum_survives_fly_capture(
        binvox_path,
        "125 occupied",
        "Brush diameter in voxels",
    );

    let nifti_path = directory.path().join("brush-capture.nii");
    write_nifti_u8(&nifti_path, [5, 5, 5], &[1; 125]);
    assert_brush_minimum_survives_fly_capture(
        nifti_path,
        "125 occupied",
        "Brush diameter in millimetres",
    );
}

#[test]
fn fly_capture_is_first_click_safe_edits_with_erase_and_suppresses_shortcuts() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fly-edit.binvox");
    let dimensions = [9, 9, 9];
    let initial_occupied = u64::from(dimensions.into_iter().product::<u32>());
    write_binvox(
        &path,
        dimensions,
        &vec![1; usize::try_from(initial_occupied).unwrap()],
    );
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "729 occupied");

    harness.key_press(egui::Key::E);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Erase (E)"), Some(Toggled::True));
    harness.get_by_label("Fly view").click_accesskit();
    harness.step();
    harness.get_by_label("Full screen 3D").click_accesskit();
    harness.step();
    harness.step();

    let capture_position = harness
        .get_by_label("Interactive 3D volume editor")
        .rect()
        .center();
    queue_same_frame_primary_click(&mut harness, capture_position);
    harness.step();

    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::Locked),
            egui::ViewportCommand::CursorVisible(false),
        ],
    );
    assert!(
        harness
            .query_by_label("Mouse captured; press Escape to release")
            .is_some()
    );
    assert!(
        harness
            .query_by_label("Click the 3D view to capture the mouse")
            .is_none()
    );
    for _ in 0..8 {
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
    assert_eq!(
        current_occupied_count(&harness),
        Some(initial_occupied),
        "the click that captured the cursor must never edit"
    );

    queue_same_frame_primary_click(&mut harness, capture_position);
    harness.step();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if current_occupied_count(&harness).is_some_and(|count| count < initial_occupied) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "a captured Erase click did not edit the center-ray surface: {:?}",
            current_occupied_count(&harness)
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
    wait_for_label(&mut harness, "modified");
    let erased_occupied = current_occupied_count(&harness).unwrap();

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::Z);
    harness.step();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if current_occupied_count(&harness) == Some(initial_occupied) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Ctrl+Z did not undo while Fly capture remained active: {:?}",
            current_occupied_count(&harness)
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }
    assert!(
        harness
            .query_by_label("Mouse captured; press Escape to release")
            .is_some(),
        "undo must not release Fly capture"
    );

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::Y);
    harness.step();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if current_occupied_count(&harness) == Some(erased_occupied) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Ctrl+Y did not redo while Fly capture remained active: {:?}",
            current_occupied_count(&harness)
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }

    harness.key_press_modifiers(egui::Modifiers::CTRL, egui::Key::W);
    harness.step();
    assert!(
        harness.query_by_label("Save changes?").is_none(),
        "Ctrl+W must be suppressed while Ctrl is the Fly boost modifier"
    );
    assert!(
        harness
            .query_by_label("Mouse captured; press Escape to release")
            .is_some()
    );

    harness.key_press(egui::Key::B);
    harness.step();
    assert_eq!(tool_toggled(&harness, "Paint (B)"), Some(Toggled::True));

    harness.key_down(egui::Key::Escape);
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::None),
            egui::ViewportCommand::CursorVisible(true),
        ],
    );
    assert!(
        harness
            .query_by_label("Click the 3D view to capture the mouse")
            .is_some()
    );
    assert!(
        harness.query_by_label("Restore four views").is_some(),
        "the Escape that releases Fly capture must not also restore the grid"
    );
    harness.key_up(egui::Key::Escape);
    harness.step();

    harness.key_down(egui::Key::Escape);
    harness.step();
    harness.key_up(egui::Key::Escape);
    harness.step();
    harness.step();
    for editor in [
        "Axial slice editor",
        "Coronal slice editor",
        "Sagittal slice editor",
        "Interactive 3D volume editor",
    ] {
        assert!(
            harness.query_by_label(editor).is_some(),
            "the second, uncaptured Escape did not restore {editor}"
        );
    }
}

#[test]
fn fly_held_erase_stroke_updates_center_rays_and_completes() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fly-held-erase.binvox");
    let dimensions = [15, 15, 15];
    let initial_occupied = u64::from(dimensions.into_iter().product::<u32>());
    write_binvox(
        &path,
        dimensions,
        &vec![1; usize::try_from(initial_occupied).unwrap()],
    );
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "3,375 occupied");

    harness.key_press(egui::Key::E);
    harness.step();
    harness.get_by_label("Fly view").click_accesskit();
    harness.step();

    let viewport_center = harness
        .get_by_label("Interactive 3D volume editor")
        .rect()
        .center();
    queue_same_frame_primary_click(&mut harness, viewport_center);
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::Locked),
            egui::ViewportCommand::CursorVisible(false),
        ],
    );

    queue_primary_button(&mut harness, viewport_center, true);
    harness.step();
    let deadline = Instant::now() + Duration::from_secs(3);
    let count_after_press = loop {
        if let Some(count) = current_occupied_count(&harness)
            && count < initial_occupied
        {
            break count;
        }
        assert!(
            Instant::now() < deadline,
            "the held Erase stroke did not apply its initial center ray: {:?}",
            current_occupied_count(&harness)
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    };
    assert!(
        harness.get_by_label("Undo").accesskit_node().is_disabled(),
        "Undo must remain unavailable until the held stroke is released"
    );

    harness.key_down(egui::Key::W);
    harness.step();
    for _ in 0..4 {
        harness
            .input_mut()
            .events
            .push(egui::Event::MouseMoved(egui::vec2(20.0, 0.0)));
        harness.step();
        thread::sleep(Duration::from_millis(2));
    }
    harness.key_up(egui::Key::W);
    harness.step();
    queue_primary_button(&mut harness, viewport_center, false);
    harness.step();

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let occupied = current_occupied_count(&harness);
        let undo_ready = !harness.get_by_label("Undo").accesskit_node().is_disabled();
        let modified = harness.query_by_label_contains("modified").is_some();
        if occupied.is_some_and(|count| count < count_after_press) && undo_ready && modified {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the moved held stroke did not add another footprint and complete: initial {initial_occupied}, after press {count_after_press}, current {occupied:?}, undo ready {undo_ready}, modified {modified}"
        );
        thread::sleep(Duration::from_millis(2));
        harness.step();
    }

    harness.key_down(egui::Key::Escape);
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::None),
            egui::ViewportCommand::CursorVisible(true),
        ],
    );
    harness.key_up(egui::Key::Escape);
    harness.step();
}

#[test]
fn fly_capture_releases_on_focus_loss_and_mode_switch() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fly-release.binvox");
    write_binvox(&path, [5, 5, 5], &[1; 125]);
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, "125 occupied");

    harness.get_by_label("Fly view").click_accesskit();
    harness.step();
    let capture_position = harness
        .get_by_label("Interactive 3D volume editor")
        .rect()
        .center();
    queue_same_frame_primary_click(&mut harness, capture_position);
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::Locked),
            egui::ViewportCommand::CursorVisible(false),
        ],
    );

    harness
        .input_mut()
        .events
        .push(egui::Event::WindowFocused(false));
    harness.input_mut().focused = false;
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::None),
            egui::ViewportCommand::CursorVisible(true),
        ],
    );
    assert!(
        harness
            .query_by_label("Click the 3D view to capture the mouse")
            .is_some()
    );

    harness
        .input_mut()
        .events
        .push(egui::Event::WindowFocused(true));
    harness.input_mut().focused = true;
    harness.step();
    queue_same_frame_primary_click(&mut harness, capture_position);
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::Locked),
            egui::ViewportCommand::CursorVisible(false),
        ],
    );

    harness.get_by_label("Orbit view").click_accesskit();
    harness.step();
    assert_root_viewport_commands(
        &harness,
        &[
            egui::ViewportCommand::CursorGrab(egui::viewport::CursorGrab::None),
            egui::ViewportCommand::CursorVisible(true),
        ],
    );
    harness.step();
    assert_eq!(
        harness
            .get_by_label("Orbit view")
            .accesskit_node()
            .toggled(),
        Some(Toggled::True)
    );
    assert!(
        harness
            .query_by_label("Mouse captured; press Escape to release")
            .is_none()
    );
    assert!(
        harness
            .query_by_label("Click the 3D view to capture the mouse")
            .is_none()
    );
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

    assert!(
        !harness
            .get_by_label("Fill enclosed cavities")
            .accesskit_node()
            .is_disabled()
    );
    harness
        .get_by_label("Fill enclosed cavities")
        .scroll_to_me();
    for _ in 0..4 {
        harness.step();
    }
    harness
        .get_by_label("Fill enclosed cavities")
        .click_accesskit();
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

fn root_viewport_commands<'h>(
    harness: &'h Harness<'_, VoxelInkApp>,
) -> &'h [egui::ViewportCommand] {
    &harness
        .output()
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .expect("the app should always render the root viewport")
        .commands
}

fn assert_root_viewport_commands(
    harness: &Harness<'_, VoxelInkApp>,
    expected: &[egui::ViewportCommand],
) {
    let actual = root_viewport_commands(harness);
    for command in expected {
        assert!(
            actual.contains(command),
            "root viewport commands {actual:?} did not include {command:?}"
        );
    }
}

fn queue_same_frame_primary_click(harness: &mut Harness<'_, VoxelInkApp>, position: egui::Pos2) {
    harness
        .input_mut()
        .events
        .push(egui::Event::PointerMoved(position));
    for pressed in [true, false] {
        queue_primary_button(harness, position, pressed);
    }
}

fn queue_primary_button(
    harness: &mut Harness<'_, VoxelInkApp>,
    position: egui::Pos2,
    pressed: bool,
) {
    harness.input_mut().events.push(egui::Event::PointerButton {
        pos: position,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    });
}

fn drag_slider_to_minimum(harness: &mut Harness<'_, VoxelInkApp>, accessible_label: &str) -> f64 {
    harness
        .get_by_role_and_label(Role::Slider, accessible_label)
        .scroll_to_me();
    for _ in 0..4 {
        harness.step();
    }
    let slider = harness
        .get_by_role_and_label(Role::Slider, accessible_label)
        .rect();
    let minimum_position = egui::pos2(slider.left() + 1.0, slider.center().y);
    harness.hover_at(slider.center());
    harness.step();
    harness.drag_at(slider.center());
    harness.step();
    harness.hover_at(minimum_position);
    harness.step();
    harness.drop_at(minimum_position);
    harness.step();

    let node = harness
        .get_by_role_and_label(Role::Slider, accessible_label)
        .accesskit_node();
    let minimum = node.min_numeric_value().unwrap();
    let selected = node.numeric_value().unwrap();
    assert!(
        (selected - minimum).abs() <= f64::EPSILON,
        "the regression must first select the minimum {accessible_label}: {selected} != {minimum}"
    );
    selected
}

fn assert_brush_minimum_survives_fly_capture(
    path: PathBuf,
    occupied_label: &str,
    slider_label: &str,
) {
    let mut harness = app_harness(Some(path));
    wait_for_label(&mut harness, occupied_label);
    harness.key_press(egui::Key::B);
    harness.step();
    harness.get_by_label("Fly view").click_accesskit();
    harness.step();
    harness.step();

    let selected_diameter = drag_slider_to_minimum(&mut harness, slider_label);
    let viewport_center = harness
        .get_by_label("Interactive 3D volume editor")
        .rect()
        .center();
    queue_primary_button(&mut harness, viewport_center, true);
    harness.step();
    queue_primary_button(&mut harness, viewport_center, false);
    harness.input_mut().events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, f32::MAX),
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::NONE,
    });
    harness.step();
    harness.step();

    let captured_diameter = harness
        .get_by_role_and_label(Role::Slider, slider_label)
        .accesskit_node()
        .numeric_value()
        .unwrap();
    assert_eq!(
        captured_diameter, selected_diameter,
        "capturing Fly changed {slider_label}"
    );
}

fn assert_rect_near(actual: egui::Rect, expected: egui::Rect) {
    let delta = (actual.min - expected.min).length() + (actual.max - expected.max).length();
    assert!(
        delta <= 0.1,
        "rect changed unexpectedly: {expected:?} -> {actual:?}"
    );
}

fn assert_selected_view_button_text(
    harness: &Harness<'_, VoxelInkApp>,
    accessible_label: &str,
    visible_text: &str,
) {
    fn collect_text_shapes<'a>(
        shape: &'a egui::epaint::Shape,
        text_shapes: &mut Vec<&'a egui::epaint::TextShape>,
    ) {
        match shape {
            egui::epaint::Shape::Text(text) => text_shapes.push(text),
            egui::epaint::Shape::Vec(children) => {
                for child in children {
                    collect_text_shapes(child, text_shapes);
                }
            }
            _ => {}
        }
    }

    let button_rect = harness.get_by_label(accessible_label).rect();
    let mut text_shapes = Vec::new();
    for clipped in &harness.output().shapes {
        collect_text_shapes(&clipped.shape, &mut text_shapes);
    }
    let text = text_shapes
        .into_iter()
        .find(|text| {
            text.galley.text() == visible_text
                && text.visual_bounding_rect().intersects(button_rect)
        })
        .unwrap_or_else(|| panic!("{accessible_label} did not paint {visible_text:?}"));
    assert_eq!(
        text.galley.rows.len(),
        1,
        "{visible_text} wrapped inside {button_rect:?}"
    );
    assert!(
        !text.galley.elided,
        "{visible_text} was truncated inside {button_rect:?}"
    );
    assert!(
        text.galley
            .job
            .sections
            .iter()
            .all(|section| section.format.color == egui::Color32::BLACK),
        "selected {visible_text} text was not explicitly black: {:?}",
        text.galley.job.sections
    );
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

fn write_nifti_u8(path: &std::path::Path, dimensions: [u32; 3], occupancy: &[u8]) {
    let expected_len = dimensions
        .into_iter()
        .map(|axis| axis as usize)
        .product::<usize>();
    assert_eq!(occupancy.len(), expected_len);

    let mut bytes = vec![0_u8; 352 + occupancy.len()];
    bytes[0..4].copy_from_slice(&348_i32.to_le_bytes());
    let dimensions = dimensions.map(|axis| i16::try_from(axis).unwrap());
    for (index, value) in [3_i16, dimensions[0], dimensions[1], dimensions[2]]
        .into_iter()
        .enumerate()
    {
        let offset = 40 + index * 2;
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    for index in 4..8 {
        let offset = 40 + index * 2;
        bytes[offset..offset + 2].copy_from_slice(&1_i16.to_le_bytes());
    }
    bytes[70..72].copy_from_slice(&2_i16.to_le_bytes());
    bytes[72..74].copy_from_slice(&8_i16.to_le_bytes());
    for index in 1..=3 {
        let offset = 76 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&1.0_f32.to_le_bytes());
    }
    bytes[108..112].copy_from_slice(&352.0_f32.to_le_bytes());
    bytes[123] = 2;
    bytes[344..348].copy_from_slice(b"n+1\0");
    bytes[352..].copy_from_slice(occupancy);
    fs::write(path, bytes).expect("NIfTI fixture should be written");
}
