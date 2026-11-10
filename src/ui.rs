use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::history::UndoHistory;
use crate::renderer::{Renderer3D, ViewportShading};
use crate::scene::Scene;
use crate::voxel::{SculptSettings, VoxelBrushOp};
use glam::{Vec2, Vec3};
use std::collections::HashSet;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SmartSelectMode {
    /// 1-click selects the entire connected 3D sub-part / component (e.g. a single tree, gear, column, crystal, or airship).
    SubPart,
    /// 1-click flood-fills a contiguous surface patch bounded by sharp edges (`crease_angle_deg`).
    SmartSurface,
    /// 1-click selects all faces of the clicked object layer.
    WholeLayer,
    /// Click & drag spherical brush selection.
    Brush,
}

#[derive(Clone)]
pub struct UiState {
    pub busy_operation: Option<(String, std::time::Instant)>,
    pub actions: Vec<crate::ipc::IpcRequest>,
    pub last_error: Option<String>,
    pub last_agent_activity: Option<std::time::Instant>,
    pub activity: Vec<String>,
    pub project_dialog: Option<String>,
    pub project_file: String,
    pub outliner_search: String,
    pub comment_search: String,
    pub comment_filter: usize,
    pub review_tab: usize,
    pub scene_tab: usize,
    pub reply_to: Option<u64>,
    pub reply_draft: String,
    pub selection_name: String,
    pub view_name: String,
    pub clip_enabled: bool,
    pub clip_axis: usize,
    pub clip_flip: bool,
    pub clip_offset: f32,
    pub measure_active: bool,
    pub measure_points: Vec<Vec3>,
    pub viewport_rect: egui::Rect,
    pub focus_comment: Option<u64>,
    pub constraint_draft: String,
    pub screenshot_cache: std::collections::HashMap<String, Result<egui::TextureHandle, String>>,
    pub extrude_distance: f32,
    pub inset_factor: f32,
    pub inset_depth: f32,
    pub sculpt: SculptSettings,
    pub voxel_op: VoxelBrushOp,
    pub voxel_brush_radius: f32,
    pub voxel_paint_color: [f32; 3],
    pub voxel_symmetry_x: bool,
    pub io_file_path: String,
    pub status_message: String,
    pub show_io_modal: bool,
    pub select_mode: SmartSelectMode,
    pub multi_select: bool,
    pub crease_angle_deg: f32,
    pub selection_radius: f32,
    pub hover_preview_obj_id: Option<u64>,
    pub hover_preview_seed_face: Option<usize>,
    pub hover_preview_faces: HashSet<usize>,
    pub comment_draft: String,
    pub show_resolved_comments: bool,
    pub scene_dirty_for_disk: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            busy_operation: None,
            actions: Vec::new(),
            last_error: None,
            last_agent_activity: None,
            activity: Vec::new(),
            project_dialog: None,
            project_file: "project.spatial.json".into(),
            outliner_search: String::new(),
            comment_search: String::new(),
            comment_filter: 0,
            review_tab: 0,
            scene_tab: 0,
            reply_to: None,
            reply_draft: String::new(),
            selection_name: String::new(),
            view_name: String::new(),
            clip_enabled: false,
            clip_axis: 0,
            clip_flip: false,
            clip_offset: 0.0,
            measure_active: false,
            measure_points: Vec::new(),
            viewport_rect: egui::Rect::NOTHING,
            focus_comment: None,
            constraint_draft: String::new(),
            screenshot_cache: std::collections::HashMap::new(),
            extrude_distance: 0.45,
            inset_factor: 0.28,
            inset_depth: 0.0,
            sculpt: SculptSettings::default(),
            voxel_op: VoxelBrushOp::AddSphere,
            voxel_brush_radius: 0.45,
            voxel_paint_color: [0.88, 0.55, 0.35],
            voxel_symmetry_x: true,
            io_file_path: "model_export.obj".to_string(),
            status_message: "Ready · select a region to leave feedback".into(),
            show_io_modal: false,
            select_mode: SmartSelectMode::SubPart,
            multi_select: true,
            crease_angle_deg: 32.0,
            selection_radius: 0.45,
            hover_preview_obj_id: None,
            hover_preview_seed_face: None,
            hover_preview_faces: HashSet::new(),
            comment_draft: String::new(),
            show_resolved_comments: false,
            scene_dirty_for_disk: false,
        }
    }
}

pub fn update_pending_anchor_from_selection(scene: &mut Scene) {
    if let Some(obj) = scene.active_object() {
        let obj_id = obj.id;
        let model = obj.transform.matrix();
        if let Some((lc, ln, _min_b, _max_b)) = obj.mesh.selected_faces_bounds_and_normal() {
            let wc = model.transform_point3(lc);
            let wn = model
                .inverse()
                .transpose()
                .transform_vector3(ln)
                .normalize_or_zero();
            scene.pending_anchor = Some((obj_id, wc, lc, wn));
        } else {
            let (min_b, max_b) = obj.mesh.bounds();
            let lc = (min_b + max_b) * 0.5;
            let wc = model.transform_point3(lc);
            scene.pending_anchor = Some((obj_id, wc, lc, Vec3::Y));
        }
    }
}

pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(103, 226, 195);
const MUTED: egui::Color32 = egui::Color32::from_rgb(169, 183, 202);
const TEXT: egui::Color32 = egui::Color32::from_rgb(233, 239, 247);
const CARD: egui::Color32 = egui::Color32::from_rgb(28, 38, 52);

fn medium_font(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("Studio Medium".into()))
}

fn semibold(text: impl Into<String>, size: f32) -> egui::RichText {
    egui::RichText::new(text).font(egui::FontId::new(
        size,
        egui::FontFamily::Name("Studio Semibold".into()),
    ))
}

fn reading_text(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text)
        .size(15.0)
        .line_height(Some(22.0))
        .color(TEXT)
}

pub fn apply_theme(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let fallbacks = fonts.families[&egui::FontFamily::Proportional].clone();
    for (name, bytes) in [
        (
            "Studio Regular",
            include_bytes!("../assets/fonts/FiraSans-Regular.ttf").as_slice(),
        ),
        (
            "Studio Medium",
            include_bytes!("../assets/fonts/FiraSans-Medium.ttf").as_slice(),
        ),
        (
            "Studio Semibold",
            include_bytes!("../assets/fonts/FiraSans-SemiBold.ttf").as_slice(),
        ),
    ] {
        fonts
            .font_data
            .insert(name.into(), egui::FontData::from_static(bytes).into());
        let mut family = vec![name.into()];
        family.extend(fallbacks.clone());
        fonts
            .families
            .insert(egui::FontFamily::Name(name.into()), family);
    }
    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .insert(0, "Studio Regular".into());
    ctx.set_fonts(fonts);
    let mut style = egui::Style::default();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = egui::Color32::from_rgb(18, 26, 38);
    style.visuals.window_fill = egui::Color32::from_rgb(23, 33, 47);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(12, 19, 29);
    style.visuals.faint_bg_color = CARD;
    style.visuals.override_text_color = Some(TEXT);
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(37, 83, 79);
    style.visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    style.visuals.widgets.inactive.bg_fill = CARD;
    style.visuals.widgets.inactive.weak_bg_fill = CARD;
    style.visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(44, 58, 76));
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(42, 64, 77);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(35, 83, 78);
    style.visuals.window_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(47, 65, 82));
    style.visuals.window_corner_radius = egui::CornerRadius::same(12);
    style.spacing.item_spacing = egui::vec2(8.0, 9.0);
    style.spacing.button_padding = egui::vec2(10.0, 7.0);
    style.spacing.interact_size.y = 30.0;
    style.spacing.window_margin = egui::Margin::same(18);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(22.0, egui::FontFamily::Name("Studio Semibold".into())),
    );
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(15.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(12.0));
    ctx.set_style(style);
}

fn eyebrow(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .font(medium_font(12.0))
            .extra_letter_spacing(0.3)
            .line_height(Some(18.0))
            .color(MUTED),
    );
}
fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .corner_radius(8)
        .inner_margin(14)
}
fn action(state: &mut UiState, tool: &str, args: serde_json::Value) {
    state.actions.push(crate::ipc::IpcRequest {
        tool: tool.into(),
        args,
    });
}
fn primary(ui: &mut egui::Ui, text: &str, enabled: bool) -> bool {
    ui.add_enabled(
        enabled,
        egui::Button::new(
            egui::RichText::new(text)
                .font(medium_font(14.0))
                .color(egui::Color32::from_rgb(10, 29, 29)),
        )
        .fill(ACCENT),
    )
    .clicked()
}

pub fn frame_all(scene: &Scene, camera: &mut OrbitCamera) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for object in scene.objects.iter().filter(|o| o.visible) {
        let model = object.transform.matrix();
        for &p in &object.mesh.positions {
            let p = model.transform_point3(p);
            min = min.min(p);
            max = max.max(p);
        }
    }
    if min.x <= max.x {
        camera.target = (min + max) * 0.5;
        camera.distance =
            ((max - min).length() * 0.45 / (camera.fov_y_radians * 0.5).sin()).clamp(0.4, 200.0);
    }
}

pub fn frame_selection(scene: &Scene, camera: &mut OrbitCamera) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for (i, object) in scene.objects.iter().enumerate() {
        if !object.visible {
            continue;
        }
        if scene.has_any_selection()
            && scene.selected_object != Some(i)
            && object.mesh.selected_faces.is_empty()
        {
            continue;
        }
        let model = object.transform.matrix();
        if object.mesh.selected_faces.is_empty() {
            for &p in &object.mesh.positions {
                let p = model.transform_point3(p);
                min = min.min(p);
                max = max.max(p);
            }
        } else {
            for &fi in &object.mesh.selected_faces {
                for &vi in &object.mesh.faces[fi] {
                    let p = model.transform_point3(object.mesh.positions[vi]);
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
    }
    if min.x <= max.x {
        camera.target = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        let half = (camera.fov_y_radians * 0.5)
            .min(((camera.fov_y_radians * 0.5).tan() * camera.aspect).atan());
        camera.distance = (radius / half.sin().max(0.1) * 1.2).clamp(0.4, 200.0);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_ui(
    ctx: &egui::Context,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    _gizmo: &mut GizmoState,
    renderer: &mut Renderer3D,
    history: &mut UndoHistory,
    state: &mut UiState,
    _hover_world_pt: Option<Vec3>,
) {
    use crate::review::RevisionStatus;
    use crate::scene::{CommentStatus, SavedSelection, SavedView};
    use serde_json::json;
    let comparing = scene.preview_revision.is_some();
    let busy = state.busy_operation.is_some();
    let in_progress = busy
        || scene.revisions.iter().any(|r| {
            matches!(
                r.status,
                RevisionStatus::InProgress | RevisionStatus::ReadyForReview
            )
        });
    let mut view_changed = false;

    // Project controls have their own row so selection tools never compete with them.
    egui::TopBottomPanel::top("studio_header")
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(14, 22, 33))
                .inner_margin(egui::Margin::symmetric(18, 12)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (icon, _) =
                    ui.allocate_exact_size(egui::vec2(30.0, 32.0), egui::Sense::hover());
                let center = icon.center();
                ui.painter().rect_stroke(
                    egui::Rect::from_center_size(center, egui::vec2(20.0, 20.0)),
                    3,
                    egui::Stroke::new(1.5_f32, ACCENT),
                    egui::StrokeKind::Inside,
                );
                ui.painter().rect_filled(
                    egui::Rect::from_center_size(center, egui::vec2(7.0, 7.0)),
                    2,
                    ACCENT,
                );
                ui.vertical(|ui| {
                    ui.label(semibold("Spatial Studio", 20.0));
                    ui.label(
                        egui::RichText::new("AI modeling · human direction")
                            .small()
                            .color(MUTED),
                    );
                });
                ui.add_space(16.0);
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_max_width(300.0);
                    ui.add(egui::Label::new(semibold(&scene.project_name, 15.0)).truncate());
                    ui.label(
                        egui::RichText::new(if scene.project_path.is_some() {
                            "Project recovery enabled"
                        } else {
                            "Save a project to keep your work"
                        })
                        .small()
                        .color(MUTED),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if primary(ui, "Save project", !comparing) {
                        if scene.project_path.is_some() {
                            action(state, "save_project", json!({}));
                        } else {
                            state.project_dialog = Some("save".into());
                        }
                    }
                    ui.menu_button("Project", |ui| {
                        if ui.button("Open project…").clicked() {
                            state.project_dialog = Some("open".into());
                            ui.close_menu();
                        }
                        if ui.button("Save project as…").clicked() {
                            state.project_dialog = Some("save".into());
                            ui.close_menu();
                        }
                        if ui.button("Recover autosave…").clicked() {
                            state.project_dialog = Some("recover".into());
                            ui.close_menu();
                        }
                        if ui.button("Export model…").clicked() {
                            state.show_io_modal = true;
                            ui.close_menu();
                        }
                        ui.separator();
                        ui.label("Project title");
                        if ui.text_edit_singleline(&mut scene.project_name).changed() {
                            state.scene_dirty_for_disk = true;
                        }
                        ui.label("Display units (world coordinates use meters)");
                        for unit in ["m", "cm", "mm", "in"] {
                            if ui
                                .selectable_value(&mut scene.units, unit.into(), unit)
                                .changed()
                            {
                                state.scene_dirty_for_disk = true;
                            }
                        }
                    });
                    if ui
                        .add_enabled(
                            history.can_redo() && !comparing && !in_progress,
                            egui::Button::new("Redo"),
                        )
                        .clicked()
                    {
                        action(state, "redo", json!({}));
                    }
                    if ui
                        .add_enabled(
                            history.can_undo() && !comparing && !in_progress,
                            egui::Button::new("Undo"),
                        )
                        .clicked()
                    {
                        action(state, "undo", json!({}));
                    }
                });
            });
        });
    egui::TopBottomPanel::top("selection_toolbar")
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(21, 31, 44))
                .inner_margin(egui::Margin::symmetric(16, 8)),
        )
        .show(ctx, |ui| {
            ui.add_enabled_ui(!comparing, |ui| {
                ui.horizontal_wrapped(|ui| {
                    eyebrow(ui, "Select");
                    for (mode, label, tip) in [
                        (
                            SmartSelectMode::SubPart,
                            "1  Part",
                            "Select one connected piece",
                        ),
                        (
                            SmartSelectMode::SmartSurface,
                            "2  Surface",
                            "Select a surface bounded by sharp edges",
                        ),
                        (
                            SmartSelectMode::WholeLayer,
                            "3  Layer",
                            "Select the entire object",
                        ),
                        (
                            SmartSelectMode::Brush,
                            "4  Brush",
                            "Paint a selection; Ctrl erases",
                        ),
                    ] {
                        if ui
                            .selectable_value(&mut state.select_mode, mode, label)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            state.hover_preview_seed_face = None;
                            state.measure_active = false;
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut state.multi_select, "Multi-select");
                    if state.select_mode == SmartSelectMode::SmartSurface {
                        ui.label("Crease");
                        if ui
                            .add(
                                egui::Slider::new(&mut state.crease_angle_deg, 5.0..=85.0)
                                    .suffix("°"),
                            )
                            .changed()
                        {
                            state.hover_preview_seed_face = None;
                        }
                    } else if state.select_mode == SmartSelectMode::Brush {
                        ui.label("Radius");
                        ui.add(egui::Slider::new(&mut state.selection_radius, 0.05..=2.5));
                        state.sculpt.radius = state.selection_radius;
                    }
                    if ui.small_button("Grow ]").clicked() {
                        for o in &mut scene.objects {
                            if !o.mesh.selected_faces.is_empty() {
                                o.mesh.grow_selection();
                            }
                        }
                        update_pending_anchor_from_selection(scene);
                        state.scene_dirty_for_disk = true;
                    }
                    if ui.small_button("Shrink [").clicked() {
                        for o in &mut scene.objects {
                            if !o.mesh.selected_faces.is_empty() {
                                o.mesh.shrink_selection();
                            }
                        }
                        update_pending_anchor_from_selection(scene);
                        state.scene_dirty_for_disk = true;
                    }
                    if ui.small_button("Clear").clicked() {
                        scene.clear_all_selections();
                        scene.selected_object = None;
                        state.scene_dirty_for_disk = true;
                    }
                });
            });
        });
    egui::TopBottomPanel::bottom("studio_status")
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(14, 22, 33))
                .inner_margin(egui::Margin::symmetric(16, 7)),
        )
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new("•").color(if state.last_agent_activity.is_some() {
                        ACCENT
                    } else {
                        MUTED
                    }),
                );
                ui.label(
                    egui::RichText::new(
                        state
                            .last_agent_activity
                            .map(|t| format!("Agent activity {}s ago", t.elapsed().as_secs()))
                            .unwrap_or("Waiting for an agent".into()),
                    )
                    .small()
                    .color(MUTED),
                );
                ui.separator();
                ui.label(egui::RichText::new(&state.status_message).small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let faces: usize = scene.objects.iter().map(|o| o.mesh.faces.len()).sum();
                    ui.label(
                        egui::RichText::new(format!(
                            "{} parts   ·   {} faces   ·   {}",
                            scene.objects.len(),
                            faces,
                            scene.units
                        ))
                        .small()
                        .color(MUTED),
                    );
                });
            });
        });

    egui::SidePanel::left("scene_panel").default_width(275.0).width_range(220.0..=390.0).frame(egui::Frame::new().fill(egui::Color32::from_rgb(18,26,38)).inner_margin(14)).show(ctx, |ui| {
        ui.heading("Scene");
        ui.label(egui::RichText::new("Find a part. Give it direction.").small().color(MUTED));
        ui.horizontal(|ui| {
            ui.selectable_value(&mut state.scene_tab,0,"Parts");
            if ui.selectable_value(&mut state.scene_tab,1,"Inspect").changed() { state.outliner_search.clear(); }
            ui.selectable_value(&mut state.scene_tab,2,"Saved");
        });
        if state.scene_tab == 0 { ui.add(egui::TextEdit::singleline(&mut state.outliner_search).hint_text("Search parts…").desired_width(f32::INFINITY)); }
        ui.horizontal(|ui| {
            if ui.small_button("Show all").clicked() { for o in &mut scene.objects { o.visible = true; } state.scene_dirty_for_disk = true; }
            if ui.small_button("Isolate").on_hover_text("Keep every selected object visible").clicked() {
                let selected = scene.selected_object;
                if scene.has_any_selection() { for (i,o) in scene.objects.iter_mut().enumerate() { o.visible = selected == Some(i) || !o.mesh.selected_faces.is_empty(); } state.scene_dirty_for_disk = true; }
            }
            if ui.small_button("Frame").clicked() { frame_selection(scene,camera); }
        });
        ui.add_space(6.0);
        egui::ScrollArea::vertical().id_salt("scene_scroll").show(ui,|ui| {
            if state.scene_tab == 0 {
            let search = state.outliner_search.to_lowercase();
            let mut clicked = None; let mut frame = false;
            for (i,obj) in scene.objects.iter_mut().enumerate() {
                if !obj.name.to_lowercase().contains(&search) { continue; }
                let selected = scene.selected_object == Some(i) || !obj.mesh.selected_faces.is_empty();
                ui.push_id(obj.id,|ui| {
                    egui::Frame::new().fill(if selected { egui::Color32::from_rgb(30,61,62) } else { egui::Color32::TRANSPARENT }).corner_radius(6).inner_margin(6).show(ui,|ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button(if obj.visible { "on" } else { "off" }).on_hover_text("Toggle visibility").clicked() { obj.visible = !obj.visible; state.scene_dirty_for_disk = true; }
                            ui.vertical(|ui| {
                                let response = ui.selectable_label(selected,egui::RichText::new(&obj.name).size(14.0).line_height(Some(20.0)));
                                if response.clicked() || response.double_clicked() { clicked = Some(i); frame |= response.double_clicked(); }
                                let count = scene.comments.iter().filter(|c| !c.resolved && (c.object_id == obj.id || c.targets.iter().any(|t| t.object_id == obj.id))).count();
                                ui.label(egui::RichText::new(format!("{} faces{}",obj.mesh.faces.len(),if count > 0 { format!("  ·  {count} open") } else { String::new() })).small().color(MUTED));
                            });
                        });
                    });
                });
            }
            if let Some(i) = clicked {
                if !comparing {
                    scene.selected_object = Some(i);
                    let o = &scene.objects[i]; let (min,max) = o.mesh.bounds(); let local = (min+max)*0.5;
                    scene.pending_anchor = Some((o.id,o.transform.matrix().transform_point3(local),local,Vec3::Y));
                    state.scene_dirty_for_disk = true;
                }
                if frame { frame_selection(scene,camera); }
            }
            }
            if state.scene_tab == 2 {
            egui::CollapsingHeader::new("Saved selections").default_open(true).show(ui,|ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut state.selection_name).hint_text("Selection name").desired_width(120.0));
                    if ui.add_enabled(scene.has_any_selection() && !comparing,egui::Button::new("Save")).clicked() {
                        let targets = scene.collect_current_target_sections(state.selection_radius);
                        scene.saved_selections.push(SavedSelection { name: if state.selection_name.trim().is_empty() { format!("Selection {}",scene.saved_selections.len()+1) } else { state.selection_name.trim().into() },targets });
                        state.selection_name.clear(); state.scene_dirty_for_disk = true;
                    }
                });
                let mut restore = None; let mut remove = None;
                for (i,s) in scene.saved_selections.iter().enumerate() { ui.horizontal(|ui| {
                    if ui.add_enabled(!comparing,egui::Button::new(&s.name)).clicked() { restore = Some(s.targets.clone()); }
                    if ui.small_button("×").clicked() { remove = Some(i); }
                }); }
                if let Some(targets) = restore { match scene.restore_targets(&targets) { Ok(()) => { update_pending_anchor_from_selection(scene); state.scene_dirty_for_disk = true; }, Err(e) => state.last_error = Some(e) } }
                if let Some(i) = remove { scene.saved_selections.remove(i); state.scene_dirty_for_disk = true; }
            });
            egui::CollapsingHeader::new("Saved views").default_open(true).show(ui,|ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut state.view_name).hint_text("View name").desired_width(120.0));
                    if ui.button("Save").clicked() { scene.saved_views.push(SavedView { name: if state.view_name.trim().is_empty() { format!("View {}",scene.saved_views.len()+1) } else { state.view_name.trim().into() },camera: camera.clone() }); state.view_name.clear(); state.scene_dirty_for_disk = true; }
                });
                let mut remove = None;
                for (i,v) in scene.saved_views.iter().enumerate() { ui.horizontal(|ui| { if ui.button(&v.name).clicked() { let aspect = camera.aspect; *camera = v.camera.clone(); camera.aspect = aspect; } if ui.small_button("×").clicked() { remove = Some(i); } }); }
                if let Some(i) = remove { scene.saved_views.remove(i); state.scene_dirty_for_disk = true; }
            });
            }
            if state.scene_tab == 1 {
            egui::CollapsingHeader::new("Inspect & measure").default_open(true).show(ui,|ui| {
                ui.checkbox(&mut renderer.ghost_unselected,"Ghost unselected parts");
                ui.checkbox(&mut state.clip_enabled,"Section plane");
                if state.clip_enabled { ui.horizontal(|ui| { for (axis,label) in [(0,"X"),(1,"Y"),(2,"Z")] { ui.selectable_value(&mut state.clip_axis,axis,label); } ui.checkbox(&mut state.clip_flip,"Flip"); }); ui.add(egui::DragValue::new(&mut state.clip_offset).speed(0.05).prefix("Position  ").suffix(" m")); }
                let sign = if state.clip_flip { -1.0 } else { 1.0 };
                scene.clip_plane = if state.clip_enabled { let mut p = [0.0,0.0,0.0,-state.clip_offset*sign]; p[state.clip_axis] = sign; Some(p) } else { None };
                ui.separator();
                ui.checkbox(&mut state.measure_active,"Pick measurement points");
                ui.label(egui::RichText::new("Pick 2 points for distance, 3 for an angle at the middle point.").small().color(MUTED));
                if let Some(text) = measurement_text(&state.measure_points,scene.unit_factor(),&scene.units) {
                    ui.label(egui::RichText::new(&text).strong().color(ACCENT));
                    if ui.button("Attach to comment").clicked() { if !state.comment_draft.is_empty() { state.comment_draft.push('\n'); } state.comment_draft.push_str(&format!("Measurement: {text}. World points (m): {:?}. ",state.measure_points.iter().map(|p| p.to_array()).collect::<Vec<_>>())); }
                }
                if ui.small_button("Clear points").clicked() { state.measure_points.clear(); }
            });
            egui::CollapsingHeader::new("Activity").show(ui,|ui| {
                if state.activity.is_empty() { ui.label(egui::RichText::new("Agent operations appear here.").small().color(MUTED)); }
                for line in state.activity.iter().rev().take(8) { ui.label(egui::RichText::new(line).small().color(MUTED)); }
            });
            }
        });
    });

    egui::SidePanel::right("review_panel").default_width(390.0).width_range(310.0..=480.0).frame(egui::Frame::new().fill(egui::Color32::from_rgb(18,26,38)).inner_margin(16)).show(ctx, |ui| {
        let open = scene.comments.iter().filter(|c| !c.resolved).count();
        ui.horizontal(|ui| { ui.heading("Review"); ui.label(egui::RichText::new(format!("{open} open")).small().color(ACCENT)); });
        ui.label(egui::RichText::new("Your feedback, anchored in 3D.").small().color(MUTED));
        ui.horizontal(|ui| { ui.selectable_value(&mut state.review_tab,0,"Comments"); ui.selectable_value(&mut state.review_tab,1,format!("Revisions ({})",scene.revisions.len())); });
        ui.separator();
        if let Some(error) = state.last_error.clone() {
            egui::Frame::new().fill(egui::Color32::from_rgb(64,35,41)).corner_radius(6).inner_margin(10).show(ui,|ui| { ui.label(egui::RichText::new(error).color(egui::Color32::from_rgb(255,180,170))); if ui.small_button("Dismiss").clicked() { state.last_error = None; } });
        }
        if let Some((name,started)) = state.busy_operation.clone() {
            card().show(ui,|ui| {
                ui.horizontal(|ui| { ui.spinner(); ui.label(egui::RichText::new(name).strong()); });
                ui.label(egui::RichText::new(format!("Running · {:.1}s",started.elapsed().as_secs_f32())).small().color(MUTED));
                if ui.button("Cancel operation").on_hover_text("Discard this operation's result; your live scene stays unchanged").clicked() { action(state,"cancel_operation",json!({})); }
            });
        }
        if state.review_tab == 0 {
            card().show(ui,|ui| {
                eyebrow(ui,"New comment");
                let sections = scene.collect_current_target_sections(state.selection_radius);
                if sections.is_empty() { ui.label(egui::RichText::new("Select a region, or leave scene feedback.").small().color(MUTED)); }
                else {
                    let total: usize = sections.iter().map(|t| t.selected_faces.len()).sum();
                    ui.label(egui::RichText::new(if total == 0 { format!("{} whole part(s)",sections.len()) } else { format!("{} regions · {} faces",sections.len(),total) }).small().color(ACCENT));
                    let mut remove = None;
                    for (i,t) in sections.iter().enumerate().take(4) { ui.horizontal(|ui| { ui.label(egui::RichText::new(if t.selected_faces.is_empty() { format!("{} · whole part",t.object_name) } else { format!("{} · {} faces",t.object_name,t.selected_faces.len()) }).small()); if ui.small_button("×").clicked() { remove = Some(i); } }); }
                    if let Some(i) = remove { let t = &sections[i]; if let Some(o) = scene.objects.iter_mut().find(|o| o.id == t.object_id) { for fi in &t.selected_faces { o.mesh.selected_faces.remove(fi); } o.mesh.bump_version(); } if scene.objects.iter().all(|o| o.mesh.selected_faces.is_empty()) { scene.selected_object = None; } update_pending_anchor_from_selection(scene); state.scene_dirty_for_disk = true; }
                }
                ui.add(egui::TextEdit::multiline(&mut state.comment_draft).margin(8.0).hint_text(egui::RichText::new("What should change? Describe the result you want…").color(MUTED)).desired_rows(3).desired_width(f32::INFINITY));
                egui::CollapsingHeader::new("Dimensions & constraints").show(ui,|ui| {
                    ui.add(egui::TextEdit::singleline(&mut state.constraint_draft).hint_text("e.g. opening +20 mm; keep pillars equal").desired_width(f32::INFINITY));
                });
                ui.horizontal(|ui| {
                    let enabled = !state.comment_draft.trim().is_empty() && !comparing;
                    if primary(ui,if sections.is_empty() { "Add scene feedback" } else { "Comment on selection" },enabled) {
                        let mut text = state.comment_draft.trim().to_string();
                        if !state.constraint_draft.trim().is_empty() { text.push_str(&format!("\nConstraints: {}",state.constraint_draft.trim())); }
                        if sections.is_empty() { scene.add_general_comment(text); } else { scene.add_comment_for_selection(text,state.selection_radius); }
                        state.comment_draft.clear(); state.constraint_draft.clear(); state.scene_dirty_for_disk = true;
                    }
                    if !sections.is_empty() && ui.add_enabled(enabled,egui::Button::new("Scene-wide")).clicked() { let mut text = state.comment_draft.trim().to_string();
                        if !state.constraint_draft.trim().is_empty() { text.push_str(&format!("\nConstraints: {}",state.constraint_draft.trim())); }
                        scene.add_general_comment(text); state.comment_draft.clear(); state.constraint_draft.clear(); state.scene_dirty_for_disk = true; }
                });
            });
            ui.add_space(6.0);
            ui.add(egui::TextEdit::singleline(&mut state.comment_search).hint_text("Search feedback…").desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("comment_filter").selected_text(match state.comment_filter { 1=>"Open",2=>"In progress",3=>"Ready for review",4=>"Accepted",_=>"All feedback" }).show_ui(ui,|ui| { for (i,label) in [(0,"All feedback"),(1,"Open"),(2,"In progress"),(3,"Ready for review"),(4,"Accepted")] { ui.selectable_value(&mut state.comment_filter,i,label); } });
                ui.checkbox(&mut state.show_resolved_comments,"Accepted");
            });
            egui::ScrollArea::vertical().id_salt("feedback_scroll").show(ui,|ui| {
                let search = state.comment_search.to_lowercase();
                let mut focus = None;
                for c in &scene.comments {
                    let status = if c.resolved { CommentStatus::Accepted } else { c.status };
                    if !state.show_resolved_comments && c.resolved { continue; }
                    if state.comment_filter > 0 && state.comment_filter != match status { CommentStatus::Open=>1,CommentStatus::InProgress=>2,CommentStatus::ReadyForReview=>3,CommentStatus::Accepted=>4 } { continue; }
                    if !format!("{} {} {}",c.id,c.comment,c.object_name).to_lowercase().contains(&search) { continue; }
                    ui.push_id(c.id,|ui| { card().stroke(egui::Stroke::new(1.0_f32,if scene.active_comment_id == Some(c.id) { ACCENT } else { egui::Color32::from_rgb(41,56,74) })).show(ui,|ui| {
                        ui.horizontal_wrapped(|ui| { eyebrow(ui,&format!("#{:02}  {}",c.id,if c.is_general { "Scene" } else { "Region" })); ui.label(egui::RichText::new(status.label()).small().color(if status == CommentStatus::ReadyForReview { ACCENT } else { MUTED })); });
                        ui.label(reading_text(&c.comment));
                        if !c.is_general { ui.label(egui::RichText::new(&c.object_name).small().color(MUTED)); }
                        if scene.comment_needs_reattachment(c) {
                            ui.label(egui::RichText::new("Needs reattachment · geometry changed").small().color(egui::Color32::from_rgb(245,187,113)));
                            if ui.add_enabled(scene.has_any_selection() && !comparing,egui::Button::new("Attach current selection")).clicked() { action(state,"reattach_comment",json!({"comment_id":c.id})); }
                        }
                        if let Some(reply) = &c.agent_reply { ui.separator(); eyebrow(ui,"Agent update"); ui.label(reading_text(reply)); }
                        for reply in &c.replies { ui.label(egui::RichText::new(format!("{}: {}",reply.author,reply.text)).size(13.0).line_height(Some(19.0)).color(MUTED)); }
                        ui.horizontal_wrapped(|ui| {
                            if !c.is_general && ui.add_enabled(!comparing && !scene.comment_needs_reattachment(c),egui::Button::new("View region")).clicked() { focus = Some(c.id); }
                            if ui.add_enabled(!comparing && !scene.revisions.iter().any(|r| r.comment_id == Some(c.id) && r.status == RevisionStatus::InProgress),egui::Button::new(if c.resolved { "Reopen" } else { "Accept" })).clicked() {
                                if let Some(r) = scene.revisions.iter().find(|r| r.comment_id == Some(c.id) && r.status == RevisionStatus::ReadyForReview) { action(state,"accept_revision",json!({"revision_id":r.id})); }
                                else { action(state,"update_comment",json!({"comment_id":c.id,"status":if c.resolved { "open" } else { "accepted" }})); }
                            }
                            ui.menu_button("…",|ui| { if ui.add_enabled(!scene.revisions.iter().any(|r| r.comment_id == Some(c.id) && matches!(r.status,RevisionStatus::InProgress | RevisionStatus::ReadyForReview)),egui::Button::new("Delete comment")).clicked() { action(state,"delete_comment",json!({"comment_id":c.id})); ui.close_menu(); } });
                            if ui.small_button("Reply").clicked() { state.reply_to = Some(c.id); state.reply_draft.clear(); }
                        });
                        if state.reply_to == Some(c.id) { ui.add(egui::TextEdit::multiline(&mut state.reply_draft).hint_text("Add a reply…").desired_rows(2).desired_width(f32::INFINITY)); if ui.add_enabled(!state.reply_draft.trim().is_empty(),egui::Button::new("Post reply")).clicked() { action(state,"update_comment",json!({"comment_id":c.id,"reply":state.reply_draft.trim(),"author":"You"})); state.reply_draft.clear(); state.reply_to = None; } }
                    }); });
                    ui.add_space(6.0);
                }
                if scene.comments.is_empty() { ui.add_space(16.0); eyebrow(ui,"Start a conversation"); ui.label(egui::RichText::new("Select a part in the viewport and describe how you'd like it to change.").color(MUTED)); }
                if let Some(id) = focus { if let Some(anchor) = scene.focus_comment(id) { frame_selection(scene,camera); camera.target = anchor; state.scene_dirty_for_disk = true; } }
            });
        } else {
            ui.label(egui::RichText::new("Compare agent changes before you accept them.").small().color(MUTED));
            if comparing && primary(ui,"Return to live scene",true) { scene.preview_revision = None; view_changed = true; }
            if ui.checkbox(&mut scene.highlight_changes,"Highlight changed parts").changed() { view_changed = true; }
            egui::ScrollArea::vertical().id_salt("revision_scroll").show(ui,|ui| {
                if scene.revisions.is_empty() { card().show(ui,|ui| { eyebrow(ui,"No revisions yet"); ui.label("Ask your agent to begin a revision before editing, then submit it for review."); }); }
                for r in scene.revisions.iter().rev() {
                    card().show(ui,|ui| {
                        eyebrow(ui,&format!("Revision {:02}  ·  {}",r.id,match r.status { RevisionStatus::InProgress=>"In progress",RevisionStatus::ReadyForReview=>"Ready for review",RevisionStatus::Accepted=>"Accepted",RevisionStatus::Reverted=>"Reverted" }));
                        ui.label(semibold(&r.label,17.0).line_height(Some(23.0)));
                        if let Some(id) = r.comment_id { ui.label(egui::RichText::new(format!("Linked to comment #{id}")).small().color(MUTED)); }
                        if !r.summary.is_empty() { ui.label(reading_text(&r.summary)); }
                        let changed = if r.status == RevisionStatus::InProgress { Vec::new() } else { scene.changed_objects(r.id) };
                        ui.label(egui::RichText::new(format!("{} changed {}",changed.len(),if changed.len()==1 { "part" } else { "parts" })).small().color(ACCENT));
                        ui.horizontal(|ui| {
                            if ui.selectable_label(scene.preview_revision == Some((r.id,true)),"Before").clicked() { action(state,"preview_revision",json!({"revision_id":r.id,"before":true})); view_changed = true; }
                            if ui.selectable_label(scene.preview_revision == Some((r.id,false)),"After").clicked() { action(state,"preview_revision",json!({"revision_id":r.id,"before":false})); view_changed = true; }
                        });
                        if r.status == RevisionStatus::ReadyForReview {
                            ui.horizontal(|ui| { if primary(ui,"Accept revision",true) { action(state,"accept_revision",json!({"revision_id":r.id})); view_changed = true; } if ui.button("Revert").clicked() { action(state,"revert_revision",json!({"revision_id":r.id})); view_changed = true; } });
                        } else if r.status == RevisionStatus::InProgress && ui.button("Cancel revision").on_hover_text("Restore its starting geometry between agent operations").clicked() { action(state,"cancel_revision",json!({"revision_id":r.id})); view_changed = true; }
                        for path in &r.screenshots {
                            let texture = state.screenshot_cache.entry(path.clone()).or_insert_with(|| {
                                let image = image::open(path).map_err(|e| e.to_string())?.thumbnail(640,480).into_rgba8();
                                let pixels = egui::ColorImage::from_rgba_unmultiplied([image.width() as usize,image.height() as usize],image.as_raw());
                                Ok(ctx.load_texture(path,pixels,egui::TextureOptions::LINEAR))
                            });
                            match texture {
                                Ok(texture) => { ui.add(egui::Image::new((texture.id(),texture.size_vec2())).max_width(ui.available_width())); },
                                Err(_) => { ui.label(egui::RichText::new("Screenshot unavailable").small().color(MUTED)).on_hover_text(path); }
                            }
                        }
                    }); ui.add_space(8.0);
                }
            });
        }
    });

    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx,|ui| {
        state.viewport_rect = ui.max_rect();
        let rect = ui.max_rect();
        let caption = if let Some((id,before)) = scene.preview_revision { format!("REVISION {id:02}   /   {}",if before { "BEFORE" } else { "AFTER" }) } else { if camera.orthographic { "Orthographic".into() } else { "Perspective".into() } };
        egui::Area::new("viewport_controls".into()).fixed_pos(rect.left_top()+egui::vec2(18.0,16.0)).show(ctx,|ui| {
            egui::Frame::new().fill(egui::Color32::from_rgb(15,24,35)).corner_radius(8).inner_margin(10).show(ui,|ui| {
                eyebrow(ui,&caption);
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut renderer.shading,ViewportShading::StudioLit,"Lit");
                    ui.selectable_value(&mut renderer.shading,ViewportShading::SculptClay,"Clay");
                    ui.selectable_value(&mut renderer.shading,ViewportShading::WireframeOnly,"Wire");
                    ui.checkbox(&mut renderer.show_wireframe_overlay,"Edges"); ui.checkbox(&mut renderer.show_grid,"Grid");
                    ui.menu_button("Camera",|ui| { if ui.button("Front").clicked() { camera.set_view_front(); } if ui.button("Right").clicked() { camera.set_view_right(); } if ui.button("Top").clicked() { camera.set_view_top(); } ui.checkbox(&mut camera.orthographic,"Orthographic"); if ui.button("Frame selection  F").clicked() { frame_selection(scene,camera); } });
                });
            });
        });
        let viewport = Vec2::new(rect.width(),rect.height());
        if !comparing {
            for c in &scene.comments {
                if c.resolved || c.is_general || scene.comment_needs_reattachment(c) { continue; }
                if let Some(p) = camera.world_to_screen(scene.current_comment_anchor(c),viewport) {
                    let pos = rect.left_top()+egui::vec2(p.x,p.y);
                    if rect.shrink(20.0).contains(pos) {
                        egui::Area::new(egui::Id::new(("comment_pin",c.id))).fixed_pos(pos).show(ctx,|ui| {
                            if ui.add(egui::Button::new(egui::RichText::new(format!("#{:02}",c.id)).strong().color(ACCENT)).fill(egui::Color32::from_rgb(19,42,46))).on_hover_text(&c.comment).clicked() { state.focus_comment = Some(c.id); state.review_tab = 0; }
                        });
                    }
                }
            }
        }
        let painter = ui.painter().with_clip_rect(rect);
        let mut screen_points = Vec::new();
        for (i,&p) in state.measure_points.iter().enumerate() {
            if let Some(p) = camera.world_to_screen(p,viewport) {
                let p = rect.left_top()+egui::vec2(p.x,p.y); screen_points.push(p);
                painter.circle_filled(p,5.0,ACCENT); painter.text(p+egui::vec2(9.0,-9.0),egui::Align2::LEFT_BOTTOM,format!("{}",i+1),egui::FontId::proportional(12.0),ACCENT);
            }
        }
        for pair in screen_points.windows(2) { painter.line_segment([pair[0],pair[1]],egui::Stroke::new(2.0_f32,ACCENT)); }
        egui::Area::new("viewport_hint".into()).fixed_pos(egui::pos2(rect.left()+18.0,rect.bottom()-40.0)).show(ctx,|ui| {
            egui::Frame::new().fill(egui::Color32::from_rgb(15,24,35)).corner_radius(6).inner_margin(8).show(ui,|ui| {
            ui.label(egui::RichText::new(if comparing { "Comparison is read-only · return to live scene to select" } else if state.measure_active { "MEASURE  ·  Click surfaces to place points" } else { "Click to select   ·   Alt + drag to orbit   ·   Scroll to zoom   ·   F to frame" }).small().color(MUTED));
            });
        });
    });
    if let Some(id) = state.focus_comment.take() {
        if let Some(p) = scene.focus_comment(id) {
            frame_selection(scene, camera);
            camera.target = p;
            state.scene_dirty_for_disk = true;
        }
    }
    if view_changed {
        renderer.invalidate_cache();
    }
    if let Some(mode) = state.project_dialog.clone() {
        let mut open = true;
        egui::Window::new(match mode.as_str() { "open"=>"Open project","recover"=>"Recover project",_=>"Save project" }).open(&mut open).collapsible(false).resizable(false).default_width(470.0).anchor(egui::Align2::CENTER_CENTER,egui::Vec2::ZERO).show(ctx,|ui| {
            ui.label("Project file path (.spatial.json)");
            ui.add(egui::TextEdit::singleline(&mut state.project_file).desired_width(f32::INFINITY));
            ui.label(egui::RichText::new("Includes geometry, feedback, saved views and revision history. Saves keep a .bak copy; recovery loads .autosave.").small().color(MUTED));
            if primary(ui,match mode.as_str() { "open"=>"Open project","recover"=>"Recover autosave",_=>"Save project" },!state.project_file.trim().is_empty()) {
                action(state,if mode == "save" { "save_project" } else { "open_project" },json!({"path":state.project_file.trim(),"recover":mode == "recover"})); state.project_dialog = None;
            }
        });
        if !open {
            state.project_dialog = None;
        }
    }
    if state.show_io_modal {
        let mut open = true;
        egui::Window::new("Export model")
            .open(&mut open)
            .default_width(440.0)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label("Destination path (.obj, .stl, .glb)");
                ui.add(
                    egui::TextEdit::singleline(&mut state.io_file_path)
                        .desired_width(f32::INFINITY),
                );
                if primary(ui, "Export scene", !state.io_file_path.trim().is_empty()) {
                    action(state, "export_model", json!({"path":state.io_file_path}));
                    state.show_io_modal = false;
                }
            });
        if !open {
            state.show_io_modal = false;
        }
    }
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) && !comparing {
        if scene.project_path.is_some() {
            action(state, "save_project", json!({}));
        } else {
            state.project_dialog = Some("save".into());
        }
    }
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z))
        && !ctx.wants_keyboard_input()
        && !comparing
        && !in_progress
    {
        action(
            state,
            if ctx.input(|i| i.modifiers.shift) {
                "redo"
            } else {
                "undo"
            },
            json!({}),
        );
    }
}

pub fn measurement_text(points: &[Vec3], factor: f32, unit: &str) -> Option<String> {
    if points.len() < 2 {
        return None;
    }
    let distance = points[0].distance(points[1]) * factor;
    let mut result = format!("Distance {distance:.3} {unit}");
    if points.len() >= 3 {
        let a = points[0] - points[1];
        let b = points[2] - points[1];
        if a.length_squared() > 1e-12 && b.length_squared() > 1e-12 {
            result.push_str(&format!(
                " · Angle {:.1}°",
                a.normalize()
                    .dot(b.normalize())
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            ));
        }
    }
    Some(result)
}
