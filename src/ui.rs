use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::history::UndoHistory;
use crate::renderer::{Renderer3D, ViewportShading};
use crate::review::RevisionStatus;
use crate::scene::{CommentStatus, Scene, SpatialComment};
use crate::voxel::{SculptSettings, VoxelBrushOp};
use egui::{Color32, FontId, RichText};
use glam::{Vec2, Vec3};
use std::collections::{HashMap, HashSet};

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

/// Feedback list filters shown as chips above the comment list.
pub const FILTER_ACTIVE: usize = 0;
pub const FILTER_NEEDS_REVIEW: usize = 1;
pub const FILTER_ACCEPTED: usize = 2;
pub const FILTER_ALL: usize = 3;

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
    /// Comment card to scroll into view on the next frame (set by viewport pins and links).
    pub scroll_to_comment: Option<u64>,
    /// Comment cards whose long text is expanded.
    pub expanded_comments: HashSet<u64>,
    pub show_shortcuts: bool,
    /// Revision screenshot currently shown at full size.
    pub lightbox: Option<String>,
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
            comment_filter: FILTER_ACTIVE,
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
            scroll_to_comment: None,
            expanded_comments: HashSet::new(),
            show_shortcuts: false,
            lightbox: None,
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
            scene_dirty_for_disk: false,
        }
    }
}

impl UiState {
    /// True while a modal surface (dialog or lightbox) owns Escape and pointer input.
    pub fn modal_open(&self) -> bool {
        self.project_dialog.is_some() || self.show_io_modal || self.lightbox.is_some()
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

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

pub const ACCENT: Color32 = Color32::from_rgb(103, 226, 195);
const MUTED: Color32 = Color32::from_rgb(169, 183, 202);
const SUBTLE: Color32 = Color32::from_rgb(122, 138, 158);
const TEXT: Color32 = Color32::from_rgb(233, 239, 247);
const CARD: Color32 = Color32::from_rgb(28, 38, 52);
const BORDER: Color32 = Color32::from_rgb(41, 56, 74);
const PANEL: Color32 = Color32::from_rgb(18, 26, 38);
const HEADER: Color32 = Color32::from_rgb(14, 22, 33);
const FIELD: Color32 = Color32::from_rgb(12, 19, 29);
const OVERLAY: Color32 = Color32::from_rgba_premultiplied(14, 22, 32, 238);
const ON_ACCENT: Color32 = Color32::from_rgb(10, 29, 29);
const SELECTED_ROW: Color32 = Color32::from_rgb(30, 61, 62);
const WARN: Color32 = Color32::from_rgb(245, 187, 113);
const INFO: Color32 = Color32::from_rgb(130, 176, 255);
const DANGER: Color32 = Color32::from_rgb(255, 160, 150);
const DANGER_BG: Color32 = Color32::from_rgb(64, 35, 41);

fn comment_status_color(status: CommentStatus) -> Color32 {
    match status {
        CommentStatus::Open => WARN,
        CommentStatus::InProgress => INFO,
        CommentStatus::ReadyForReview => ACCENT,
        CommentStatus::Accepted => SUBTLE,
    }
}

fn revision_status(status: RevisionStatus) -> (&'static str, Color32) {
    match status {
        RevisionStatus::InProgress => ("In progress", INFO),
        RevisionStatus::ReadyForReview => ("Ready for review", ACCENT),
        RevisionStatus::Accepted => ("Accepted", SUBTLE),
        RevisionStatus::Reverted => ("Reverted", DANGER),
    }
}

fn effective_status(c: &SpatialComment) -> CommentStatus {
    if c.resolved {
        CommentStatus::Accepted
    } else {
        c.status
    }
}

// ---------------------------------------------------------------------------
// Typography & theme
// ---------------------------------------------------------------------------

fn medium_font(size: f32) -> FontId {
    FontId::new(size, egui::FontFamily::Name("Studio Medium".into()))
}

fn semibold(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(FontId::new(
        size,
        egui::FontFamily::Name("Studio Semibold".into()),
    ))
}

fn hint(text: &str) -> RichText {
    RichText::new(text).color(SUBTLE)
}

fn small_muted(text: impl Into<String>) -> RichText {
    RichText::new(text).small().color(MUTED)
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
    style.visuals.panel_fill = PANEL;
    style.visuals.window_fill = Color32::from_rgb(23, 33, 47);
    style.visuals.extreme_bg_color = FIELD;
    style.visuals.faint_bg_color = CARD;
    style.visuals.override_text_color = Some(TEXT);
    style.visuals.hyperlink_color = ACCENT;
    style.visuals.selection.bg_fill = Color32::from_rgb(37, 83, 79);
    style.visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    let radius = egui::CornerRadius::same(6);
    let widgets = &mut style.visuals.widgets;
    widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    widgets.noninteractive.corner_radius = radius;
    widgets.inactive.bg_fill = CARD;
    widgets.inactive.weak_bg_fill = CARD;
    widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(44, 58, 76));
    widgets.inactive.corner_radius = radius;
    widgets.hovered.bg_fill = Color32::from_rgb(42, 64, 77);
    widgets.hovered.weak_bg_fill = Color32::from_rgb(36, 52, 66);
    widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT);
    widgets.hovered.corner_radius = radius;
    widgets.active.bg_fill = Color32::from_rgb(35, 83, 78);
    widgets.active.weak_bg_fill = Color32::from_rgb(35, 83, 78);
    widgets.active.corner_radius = radius;
    widgets.open.corner_radius = radius;
    style.visuals.window_stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(47, 65, 82));
    style.visuals.window_corner_radius = egui::CornerRadius::same(12);
    style.visuals.menu_corner_radius = egui::CornerRadius::same(8);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.interact_size.y = 30.0;
    style.spacing.window_margin = egui::Margin::same(18);
    style.spacing.menu_margin = egui::Margin::same(8);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(21.0, egui::FontFamily::Name("Studio Semibold".into())),
    );
    style
        .text_styles
        .insert(egui::TextStyle::Body, FontId::proportional(15.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, FontId::proportional(12.0));
    ctx.set_style(style);
}

// ---------------------------------------------------------------------------
// Small widgets
// ---------------------------------------------------------------------------

fn eyebrow(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text.to_uppercase())
            .font(medium_font(11.5))
            .extra_letter_spacing(0.6)
            .line_height(Some(18.0))
            .color(SUBTLE),
    );
}

fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .corner_radius(8)
        .inner_margin(14)
}

fn overlay_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(OVERLAY)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(10, 8))
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
            RichText::new(text)
                .font(medium_font(14.0))
                .color(if enabled {
                    ON_ACCENT
                } else {
                    Color32::from_rgb(170, 205, 198)
                }),
        )
        .fill(if enabled {
            ACCENT
        } else {
            Color32::from_rgb(40, 76, 74)
        }),
    )
    .clicked()
}

/// Secondary button with a tooltip and an explanation when it is disabled.
fn tool_button(
    ui: &mut egui::Ui,
    text: &str,
    enabled: bool,
    tip: &str,
    disabled_tip: &str,
) -> bool {
    ui.add_enabled(enabled, egui::Button::new(text))
        .on_hover_text(tip)
        .on_disabled_hover_text(disabled_tip)
        .clicked()
}

fn quiet_button(ui: &mut egui::Ui, text: &str, tip: &str) -> bool {
    ui.add(egui::Button::new(RichText::new(text).color(MUTED)).frame(false))
        .on_hover_text(tip)
        .clicked()
}

fn pill(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.14))
        .stroke(egui::Stroke::new(1.0_f32, color.gamma_multiply(0.45)))
        .corner_radius(10)
        .inner_margin(egui::Margin::symmetric(8, 1))
        .show(ui, |ui| {
            ui.label(RichText::new(text).font(medium_font(11.5)).color(color))
        })
        .response
}

/// Segmented control: one frame containing mutually exclusive options.
fn segmented<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    value: &mut T,
    options: &[(T, String, &str)],
) -> bool {
    let mut changed = false;
    egui::Frame::new()
        .fill(FIELD)
        .corner_radius(8)
        .inner_margin(3)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.horizontal(|ui| {
                for (v, label, tip) in options {
                    let selected = *value == *v;
                    let text = if selected {
                        RichText::new(label).color(TEXT)
                    } else {
                        RichText::new(label).color(MUTED)
                    };
                    let mut r = ui.selectable_label(selected, text);
                    if !tip.is_empty() {
                        r = r.on_hover_text(*tip);
                    }
                    if r.clicked() && !selected {
                        *value = *v;
                        changed = true;
                    }
                }
            });
        });
    changed
}

fn text_field(
    ui: &mut egui::Ui,
    value: &mut String,
    placeholder: &str,
    width: f32,
) -> egui::Response {
    ui.add(
        egui::TextEdit::singleline(value)
            .hint_text(hint(placeholder))
            .margin(egui::vec2(8.0, 6.0))
            .desired_width(width),
    )
}

fn search_field(ui: &mut egui::Ui, value: &mut String, placeholder: &str) {
    ui.horizontal(|ui| {
        let clear = !value.is_empty();
        let reserve = if clear {
            30.0 + ui.spacing().item_spacing.x
        } else {
            0.0
        };
        let width = (ui.available_width() - reserve).max(60.0);
        let response = text_field(ui, value, placeholder, width);
        if response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            value.clear();
        }
        if clear && quiet_button(ui, "×", "Clear search") {
            value.clear();
        }
    });
}

/// Wrapped text clamped to `max_rows`; returns true when text was elided.
fn clamped_text(
    ui: &mut egui::Ui,
    text: &str,
    size: f32,
    line_height: f32,
    color: Color32,
    max_rows: Option<usize>,
) -> bool {
    let mut job = egui::text::LayoutJob::single_section(
        text.to_owned(),
        egui::TextFormat {
            font_id: FontId::proportional(size),
            color,
            line_height: Some(line_height),
            ..Default::default()
        },
    );
    job.wrap.max_width = ui.available_width();
    if let Some(rows) = max_rows {
        job.wrap.max_rows = rows;
        job.wrap.overflow_character = Some('…');
    }
    let galley = ui.fonts(|f| f.layout_job(job));
    let elided = galley.elided;
    ui.add(egui::Label::new(galley));
    elided
}

/// Single-line truncated label with the full text as tooltip.
fn truncated(ui: &mut egui::Ui, text: RichText, full: &str) {
    ui.add(egui::Label::new(text).truncate())
        .on_hover_text(full);
}

fn compact_count(n: usize) -> String {
    if n >= 10_000 {
        format!("{}k", n / 1000)
    } else if n >= 1000 {
        format!("{:.1}k", n as f32 / 1000.0)
    } else {
        n.to_string()
    }
}

fn ago(elapsed: std::time::Duration) -> String {
    let s = elapsed.as_secs();
    if s < 5 {
        "just now".into()
    } else if s < 60 {
        format!("{s}s ago")
    } else if s < 3600 {
        format!("{}m ago", s / 60)
    } else {
        format!("{}h ago", s / 3600)
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// Short and full descriptions of the parts a comment targets.
fn comment_targets(c: &SpatialComment) -> (String, String) {
    if c.targets.is_empty() {
        return (c.object_name.clone(), c.object_name.clone());
    }
    let mut names: Vec<&str> = Vec::new();
    for t in &c.targets {
        if !names.contains(&t.object_name.as_str()) {
            names.push(&t.object_name);
        }
    }
    let full = c
        .targets
        .iter()
        .map(|t| {
            if t.selected_faces.is_empty() {
                format!("{} · whole part", t.object_name)
            } else {
                format!(
                    "{} · {}",
                    t.object_name,
                    plural(t.selected_faces.len(), "face", "faces")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let short = if c.targets.len() > 1 {
        format!(
            "{} on {}",
            plural(c.targets.len(), "region", "regions"),
            names.join(", ")
        )
    } else {
        names.join(", ")
    };
    (short, full)
}

fn selection_summary(scene: &Scene) -> Option<String> {
    let selected: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| !o.mesh.selected_faces.is_empty())
        .collect();
    if selected.is_empty() {
        return scene
            .active_object()
            .map(|o| format!("{} · whole part", o.name));
    }
    let faces: usize = selected.iter().map(|o| o.mesh.selected_faces.len()).sum();
    if selected.len() == 1 {
        Some(format!(
            "{} · {}",
            selected[0].name,
            plural(faces, "face", "faces")
        ))
    } else {
        Some(format!(
            "{} on {}",
            plural(faces, "face", "faces"),
            plural(selected.len(), "part", "parts")
        ))
    }
}

struct RowResponse {
    clicked: bool,
    double_clicked: bool,
    toggle_visibility: bool,
}

/// Compact outliner row: visibility eye, truncated name, face count and open-feedback badge.
fn part_row(
    ui: &mut egui::Ui,
    name: &str,
    faces: usize,
    open_comments: usize,
    visible: bool,
    selected: bool,
) -> RowResponse {
    let height = 32.0;
    let (rect, row) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    let eye_rect = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 17.0, rect.center().y),
        egui::vec2(28.0, 26.0),
    );
    let eye = ui
        .interact(eye_rect, row.id.with("visibility"), egui::Sense::click())
        .on_hover_text(if visible { "Hide part" } else { "Show part" });
    let row = row.on_hover_ui(|ui| {
        ui.label(semibold(name, 14.0));
        ui.label(small_muted(format!(
            "{} faces{}",
            faces,
            if open_comments > 0 {
                format!(" · {} open", plural(open_comments, "comment", "comments"))
            } else {
                String::new()
            }
        )));
        ui.label(small_muted("Click to select · double-click to frame"));
    });

    let painter = ui.painter();
    let bg = if selected {
        SELECTED_ROW
    } else if row.hovered() {
        Color32::from_rgb(25, 36, 51)
    } else {
        Color32::TRANSPARENT
    };
    painter.rect_filled(rect, 6, bg);
    if selected {
        painter.rect_filled(
            egui::Rect::from_min_size(
                rect.left_top() + egui::vec2(0.0, 7.0),
                egui::vec2(3.0, height - 14.0),
            ),
            2,
            ACCENT,
        );
    }

    let c = eye_rect.center();
    let eye_color = if eye.hovered() {
        TEXT
    } else if visible {
        MUTED
    } else {
        Color32::from_rgb(84, 98, 116)
    };
    painter.add(egui::Shape::ellipse_stroke(
        c,
        egui::vec2(7.0, 4.2),
        egui::Stroke::new(1.3_f32, eye_color),
    ));
    if visible {
        painter.circle_filled(c, 2.1, eye_color);
    } else {
        painter.line_segment(
            [c + egui::vec2(-7.0, 5.0), c + egui::vec2(7.0, -5.0)],
            egui::Stroke::new(1.3_f32, eye_color),
        );
    }

    let mut right = rect.right() - 8.0;
    if open_comments > 0 {
        let g = painter.layout_no_wrap(open_comments.to_string(), medium_font(11.5), ON_ACCENT);
        let w = g.size().x + 12.0;
        let badge = egui::Rect::from_min_max(
            egui::pos2(right - w, rect.center().y - 9.0),
            egui::pos2(right, rect.center().y + 9.0),
        );
        painter.rect_filled(badge, 9, WARN);
        painter.galley(badge.center() - g.size() * 0.5, g, ON_ACCENT);
        right = badge.left() - 8.0;
    }
    let count = painter.layout_no_wrap(compact_count(faces), FontId::proportional(12.0), SUBTLE);
    let count_w = count.size().x;
    painter.galley(
        egui::pos2(right - count_w, rect.center().y - count.size().y * 0.5),
        count,
        SUBTLE,
    );
    right -= count_w + 10.0;

    let name_left = rect.left() + 36.0;
    let mut job = egui::text::LayoutJob::single_section(
        name.to_owned(),
        egui::TextFormat {
            font_id: FontId::proportional(14.0),
            color: if !visible {
                SUBTLE
            } else if selected {
                TEXT
            } else {
                Color32::from_rgb(214, 222, 233)
            },
            ..Default::default()
        },
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width((right - name_left).max(12.0));
    let galley = ui.fonts(|f| f.layout_job(job));
    ui.painter().galley(
        egui::pos2(name_left, rect.center().y - galley.size().y * 0.5),
        galley,
        TEXT,
    );

    RowResponse {
        clicked: row.clicked(),
        double_clicked: row.double_clicked(),
        toggle_visibility: eye.clicked(),
    }
}

// ---------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Main UI
// ---------------------------------------------------------------------------

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

    draw_header(ctx, scene, history, state, comparing, in_progress);
    draw_selection_toolbar(ctx, scene, state, comparing);
    draw_status_bar(ctx, scene, state);
    draw_scene_panel(ctx, scene, camera, renderer, state, comparing);
    draw_review_panel(ctx, scene, state, comparing, &mut view_changed);
    draw_viewport(
        ctx,
        scene,
        camera,
        renderer,
        state,
        comparing,
        &mut view_changed,
    );

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

    draw_dialogs(ctx, state);

    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) && !comparing {
        if scene.project_path.is_some() {
            action(state, "save_project", serde_json::json!({}));
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
            serde_json::json!({}),
        );
    }
}

fn draw_header(
    ctx: &egui::Context,
    scene: &mut Scene,
    history: &UndoHistory,
    state: &mut UiState,
    comparing: bool,
    in_progress: bool,
) {
    use serde_json::json;
    egui::TopBottomPanel::top("studio_header")
        .frame(
            egui::Frame::new()
                .fill(HEADER)
                .inner_margin(egui::Margin::symmetric(18, 10)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let (icon, _) =
                    ui.allocate_exact_size(egui::vec2(28.0, 32.0), egui::Sense::hover());
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
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.label(semibold("Spatial Studio", 19.0));
                    ui.label(small_muted("AI modeling · human direction"));
                });
                ui.add_space(14.0);
                ui.separator();
                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.set_max_width(320.0);
                    ui.add(egui::Label::new(semibold(&scene.project_name, 15.0)).truncate())
                        .on_hover_text(
                            scene
                                .project_path
                                .clone()
                                .unwrap_or_else(|| "Not saved yet".into()),
                        );
                    let (dot, text) = if scene.project_path.is_some() {
                        (ACCENT, "Saved project · autosave on")
                    } else {
                        (WARN, "Unsaved · save to keep your work")
                    };
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;
                        ui.label(RichText::new("●").size(9.0).color(dot));
                        ui.label(small_muted(text));
                    });
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if primary(ui, "Save project", !comparing) {
                        if scene.project_path.is_some() {
                            action(state, "save_project", json!({}));
                        } else {
                            state.project_dialog = Some("save".into());
                        }
                    }
                    ui.menu_button("Project  ▾", |ui| {
                        ui.set_min_width(220.0);
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
                        eyebrow(ui, "Project title");
                        if text_field(
                            ui,
                            &mut scene.project_name,
                            "Untitled project",
                            f32::INFINITY,
                        )
                        .changed()
                        {
                            state.scene_dirty_for_disk = true;
                        }
                        eyebrow(ui, "Display units");
                        let current = scene.units.clone();
                        let mut unit: &str = &current;
                        if segmented(
                            ui,
                            &mut unit,
                            &[
                                ("m", "m".into(), "Meters"),
                                ("cm", "cm".into(), "Centimeters"),
                                ("mm", "mm".into(), "Millimeters"),
                                ("in", "in".into(), "Inches"),
                            ],
                        ) {
                            scene.units = unit.to_string();
                            state.scene_dirty_for_disk = true;
                        }
                        ui.label(small_muted("World coordinates are stored in meters."));
                    });
                    ui.add_space(6.0);
                    let blocked = if comparing {
                        "Return to the live scene first"
                    } else {
                        "Unavailable while a revision is open or an operation runs"
                    };
                    if tool_button(
                        ui,
                        "Redo",
                        history.can_redo() && !comparing && !in_progress,
                        "Redo  ·  Ctrl+Shift+Z",
                        if history.can_redo() {
                            blocked
                        } else {
                            "Nothing to redo"
                        },
                    ) {
                        action(state, "redo", json!({}));
                    }
                    if tool_button(
                        ui,
                        "Undo",
                        history.can_undo() && !comparing && !in_progress,
                        "Undo  ·  Ctrl+Z",
                        if history.can_undo() {
                            blocked
                        } else {
                            "Nothing to undo"
                        },
                    ) {
                        action(state, "undo", json!({}));
                    }
                });
            });
        });
}

fn draw_selection_toolbar(
    ctx: &egui::Context,
    scene: &mut Scene,
    state: &mut UiState,
    comparing: bool,
) {
    egui::TopBottomPanel::top("selection_toolbar")
        .frame(
            egui::Frame::new()
                .fill(Color32::from_rgb(21, 31, 44))
                .inner_margin(egui::Margin::symmetric(16, 7)),
        )
        .show(ctx, |ui| {
            ui.add_enabled_ui(!comparing, |ui| {
                ui.horizontal_wrapped(|ui| {
                    eyebrow(ui, "Select");
                    if segmented(
                        ui,
                        &mut state.select_mode,
                        &[
                            (SmartSelectMode::SubPart, "1  Part".into(), "Part (1) · click selects one connected piece"),
                            (SmartSelectMode::SmartSurface, "2  Surface".into(), "Surface (2) · click selects a surface bounded by sharp edges"),
                            (SmartSelectMode::WholeLayer, "3  Layer".into(), "Layer (3) · click selects the entire object"),
                            (SmartSelectMode::Brush, "4  Brush".into(), "Brush (4) · drag to paint a selection; Ctrl+drag erases"),
                        ],
                    ) {
                        state.hover_preview_seed_face = None;
                        state.measure_active = false;
                    }
                    if state.select_mode == SmartSelectMode::SmartSurface {
                        ui.label(small_muted("Crease"));
                        if ui
                            .add(egui::Slider::new(&mut state.crease_angle_deg, 5.0..=85.0).suffix("°"))
                            .on_hover_text("Edges sharper than this angle stop the surface fill")
                            .changed()
                        {
                            state.hover_preview_seed_face = None;
                        }
                    } else if state.select_mode == SmartSelectMode::Brush {
                        ui.label(small_muted("Radius"));
                        ui.add(egui::Slider::new(&mut state.selection_radius, 0.05..=2.5))
                            .on_hover_text("Brush radius  ·  [ and ] adjust");
                        state.sculpt.radius = state.selection_radius;
                    }
                    ui.separator();
                    ui.checkbox(&mut state.multi_select, "Multi-select").on_hover_text(
                        "Add each click to the selection; click a selected region to remove it.\nWithout it: Shift+click adds, Ctrl+click removes.",
                    );
                    ui.separator();
                    let has_faces = scene.objects.iter().any(|o| !o.mesh.selected_faces.is_empty());
                    let has_any = scene.has_any_selection();
                    if tool_button(ui, "Grow", has_faces, "Grow the selection by one ring  ·  ]", "Select faces first") {
                        for o in &mut scene.objects {
                            if !o.mesh.selected_faces.is_empty() {
                                o.mesh.grow_selection();
                            }
                        }
                        update_pending_anchor_from_selection(scene);
                        state.scene_dirty_for_disk = true;
                    }
                    if tool_button(ui, "Shrink", has_faces, "Shrink the selection by one ring  ·  [", "Select faces first") {
                        for o in &mut scene.objects {
                            if !o.mesh.selected_faces.is_empty() {
                                o.mesh.shrink_selection();
                            }
                        }
                        update_pending_anchor_from_selection(scene);
                        state.scene_dirty_for_disk = true;
                    }
                    if tool_button(ui, "Clear", has_any, "Clear the selection  ·  Esc", "Nothing selected") {
                        scene.clear_all_selections();
                        scene.selected_object = None;
                        state.scene_dirty_for_disk = true;
                    }
                    ui.add_space(6.0);
                    match selection_summary(scene) {
                        Some(summary) => {
                            ui.label(RichText::new("●").size(9.0).color(ACCENT));
                            truncated(ui, RichText::new(&summary).small().color(TEXT), &summary);
                        }
                        None => {
                            ui.label(small_muted("Nothing selected"));
                        }
                    }
                });
            });
        });
}

fn draw_status_bar(ctx: &egui::Context, scene: &Scene, state: &UiState) {
    egui::TopBottomPanel::bottom("studio_status")
        .frame(
            egui::Frame::new()
                .fill(HEADER)
                .inner_margin(egui::Margin::symmetric(16, 6)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let recent = state
                    .last_agent_activity
                    .is_some_and(|t| t.elapsed().as_secs() < 30);
                let agent_text = match state.last_agent_activity {
                    Some(t) if recent => format!("Agent active · {}", ago(t.elapsed())),
                    Some(t) => format!("Agent idle · last seen {}", ago(t.elapsed())),
                    None => "Waiting for an agent".into(),
                };
                ui.label(
                    RichText::new("●")
                        .size(9.0)
                        .color(if recent { ACCENT } else { SUBTLE }),
                );
                ui.label(small_muted(agent_text)).on_hover_ui(|ui| {
                    ui.set_max_width(420.0);
                    eyebrow(ui, "Recent agent activity");
                    if state.activity.is_empty() {
                        ui.label(small_muted("Agent operations appear here."));
                    }
                    for line in state.activity.iter().rev().take(10) {
                        ui.label(small_muted(line));
                    }
                });
                if let Some((name, started)) = &state.busy_operation {
                    ui.separator();
                    ui.spinner();
                    ui.label(
                        RichText::new(format!("{name} · {:.0}s", started.elapsed().as_secs_f32()))
                            .small()
                            .color(TEXT),
                    );
                }
                ui.separator();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let faces: usize = scene.objects.iter().map(|o| o.mesh.faces.len()).sum();
                    ui.label(small_muted(format!(
                        "{}   ·   {} faces   ·   {}",
                        plural(scene.objects.len(), "part", "parts"),
                        faces,
                        scene.units
                    )));
                    ui.separator();
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        truncated(
                            ui,
                            RichText::new(&state.status_message).small(),
                            &state.status_message,
                        );
                    });
                });
            });
        });
}

fn draw_scene_panel(
    ctx: &egui::Context,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    renderer: &mut Renderer3D,
    state: &mut UiState,
    comparing: bool,
) {
    use crate::scene::{SavedSelection, SavedView};
    egui::SidePanel::left("scene_panel")
        .default_width(280.0)
        .width_range(230.0..=400.0)
        .frame(egui::Frame::new().fill(PANEL).inner_margin(14))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Scene");
                ui.label(small_muted(plural(scene.objects.len(), "part", "parts")));
            });
            ui.add_space(2.0);
            segmented(
                ui,
                &mut state.scene_tab,
                &[
                    (0, "Parts".into(), "Browse, show and hide parts"),
                    (
                        1,
                        "Inspect".into(),
                        "Section plane, ghosting and measurement",
                    ),
                    (2, "Saved".into(), "Saved selections and camera views"),
                ],
            );
            ui.add_space(4.0);

            match state.scene_tab {
                0 => scene_parts_tab(ui, scene, camera, state, comparing),
                1 => scene_inspect_tab(ui, scene, renderer, state),
                _ => {
                    egui::ScrollArea::vertical()
                        .id_salt("saved_scroll")
                        .show(ui, |ui| {
                            eyebrow(ui, "Saved selections");
                            ui.horizontal(|ui| {
                                let w = (ui.available_width() - 70.0).max(80.0);
                                let r =
                                    text_field(ui, &mut state.selection_name, "Selection name", w);
                                let can = scene.has_any_selection() && !comparing;
                                let enter =
                                    r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                if tool_button(
                                    ui,
                                    "Save",
                                    can,
                                    "Save the current selection",
                                    "Select a region first",
                                ) || (enter && can)
                                {
                                    let targets = scene
                                        .collect_current_target_sections(state.selection_radius);
                                    let name = if state.selection_name.trim().is_empty() {
                                        format!("Selection {}", scene.saved_selections.len() + 1)
                                    } else {
                                        state.selection_name.trim().into()
                                    };
                                    scene
                                        .saved_selections
                                        .push(SavedSelection { name, targets });
                                    state.selection_name.clear();
                                    state.scene_dirty_for_disk = true;
                                }
                            });
                            if scene.saved_selections.is_empty() {
                                ui.label(small_muted(
                                    "Save a selection to come back to the same regions later.",
                                ));
                            }
                            let mut restore = None;
                            let mut remove = None;
                            for (i, s) in scene.saved_selections.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let w = ui.available_width() - 34.0;
                                    if ui
                                        .add_enabled(
                                            !comparing,
                                            egui::Button::new(&s.name).min_size(egui::vec2(w, 0.0)),
                                        )
                                        .on_hover_text(format!(
                                            "Restore {}",
                                            plural(s.targets.len(), "region", "regions")
                                        ))
                                        .clicked()
                                    {
                                        restore = Some(s.targets.clone());
                                    }
                                    if quiet_button(ui, "×", "Delete saved selection") {
                                        remove = Some(i);
                                    }
                                });
                            }
                            if let Some(targets) = restore {
                                match scene.restore_targets(&targets) {
                                    Ok(()) => {
                                        update_pending_anchor_from_selection(scene);
                                        state.scene_dirty_for_disk = true;
                                    }
                                    Err(e) => state.last_error = Some(e),
                                }
                            }
                            if let Some(i) = remove {
                                scene.saved_selections.remove(i);
                                state.scene_dirty_for_disk = true;
                            }

                            ui.add_space(14.0);
                            eyebrow(ui, "Saved views");
                            ui.horizontal(|ui| {
                                let w = (ui.available_width() - 70.0).max(80.0);
                                let r = text_field(ui, &mut state.view_name, "View name", w);
                                let enter =
                                    r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                if tool_button(ui, "Save", true, "Save the current camera", "")
                                    || enter
                                {
                                    let name = if state.view_name.trim().is_empty() {
                                        format!("View {}", scene.saved_views.len() + 1)
                                    } else {
                                        state.view_name.trim().into()
                                    };
                                    scene.saved_views.push(SavedView {
                                        name,
                                        camera: camera.clone(),
                                    });
                                    state.view_name.clear();
                                    state.scene_dirty_for_disk = true;
                                }
                            });
                            if scene.saved_views.is_empty() {
                                ui.label(small_muted("Save camera angles you want to revisit."));
                            }
                            let mut remove = None;
                            for (i, v) in scene.saved_views.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let w = ui.available_width() - 34.0;
                                    if ui
                                        .add(
                                            egui::Button::new(&v.name).min_size(egui::vec2(w, 0.0)),
                                        )
                                        .on_hover_text("Jump to this view")
                                        .clicked()
                                    {
                                        let aspect = camera.aspect;
                                        *camera = v.camera.clone();
                                        camera.aspect = aspect;
                                    }
                                    if quiet_button(ui, "×", "Delete saved view") {
                                        remove = Some(i);
                                    }
                                });
                            }
                            if let Some(i) = remove {
                                scene.saved_views.remove(i);
                                state.scene_dirty_for_disk = true;
                            }
                        });
                }
            }
        });
}

fn scene_parts_tab(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    state: &mut UiState,
    comparing: bool,
) {
    search_field(ui, &mut state.outliner_search, "Search parts…");
    let has_selection = scene.has_any_selection();
    let any_hidden = scene.objects.iter().any(|o| !o.visible);
    ui.horizontal(|ui| {
        if tool_button(
            ui,
            "Show all",
            any_hidden,
            "Make every part visible",
            "Every part is visible",
        ) {
            for o in &mut scene.objects {
                o.visible = true;
            }
            state.scene_dirty_for_disk = true;
        }
        if tool_button(
            ui,
            "Isolate",
            has_selection,
            "Hide everything except the selection",
            "Select a part first",
        ) {
            let selected = scene.selected_object;
            for (i, o) in scene.objects.iter_mut().enumerate() {
                o.visible = selected == Some(i) || !o.mesh.selected_faces.is_empty();
            }
            state.scene_dirty_for_disk = true;
        }
        if tool_button(
            ui,
            if has_selection {
                "Frame selection"
            } else {
                "Frame all"
            },
            true,
            "Fit the camera  ·  F",
            "",
        ) {
            frame_selection(scene, camera);
        }
    });
    ui.add_space(2.0);

    let mut open_counts: HashMap<u64, usize> = HashMap::new();
    for c in scene
        .comments
        .iter()
        .filter(|c| !c.resolved && !c.is_general)
    {
        let mut ids: Vec<u64> = c.targets.iter().map(|t| t.object_id).collect();
        ids.push(c.object_id);
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            *open_counts.entry(id).or_default() += 1;
        }
    }
    let selected_idx = scene.selected_object;
    let search = state.outliner_search.to_lowercase();
    let mut clicked = None;
    let mut frame = false;
    let mut matches = 0;
    egui::ScrollArea::vertical()
        .id_salt("scene_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            for (i, obj) in scene.objects.iter_mut().enumerate() {
                if !obj.name.to_lowercase().contains(&search) {
                    continue;
                }
                matches += 1;
                let selected = selected_idx == Some(i) || !obj.mesh.selected_faces.is_empty();
                let open = open_counts.get(&obj.id).copied().unwrap_or(0);
                let row = ui
                    .push_id(obj.id, |ui| {
                        part_row(
                            ui,
                            &obj.name,
                            obj.mesh.faces.len(),
                            open,
                            obj.visible,
                            selected,
                        )
                    })
                    .inner;
                if row.toggle_visibility {
                    obj.visible = !obj.visible;
                    state.scene_dirty_for_disk = true;
                } else if row.clicked || row.double_clicked {
                    clicked = Some(i);
                    frame |= row.double_clicked;
                }
            }
            if matches == 0 {
                ui.add_space(12.0);
                ui.label(small_muted(if scene.objects.is_empty() {
                    "This scene is empty. Ask your agent to add or import geometry."
                } else {
                    "No parts match your search."
                }));
            }
        });
    if let Some(i) = clicked {
        if !comparing {
            scene.selected_object = Some(i);
            let o = &scene.objects[i];
            let (min, max) = o.mesh.bounds();
            let local = (min + max) * 0.5;
            scene.pending_anchor = Some((
                o.id,
                o.transform.matrix().transform_point3(local),
                local,
                Vec3::Y,
            ));
            state.scene_dirty_for_disk = true;
        }
        if frame {
            frame_selection(scene, camera);
        }
    }
}

fn scene_inspect_tab(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    renderer: &mut Renderer3D,
    state: &mut UiState,
) {
    egui::ScrollArea::vertical()
        .id_salt("inspect_scroll")
        .show(ui, |ui| {
            eyebrow(ui, "Display");
            ui.checkbox(&mut renderer.ghost_unselected, "Ghost unselected parts")
                .on_hover_text("Fade everything except the selection");
            ui.add_space(8.0);
            eyebrow(ui, "Section plane");
            ui.checkbox(&mut state.clip_enabled, "Cut the model with a plane");
            if state.clip_enabled {
                ui.horizontal(|ui| {
                    segmented(
                        ui,
                        &mut state.clip_axis,
                        &[
                            (0, "X".into(), "Cut along X"),
                            (1, "Y".into(), "Cut along Y"),
                            (2, "Z".into(), "Cut along Z"),
                        ],
                    );
                    ui.checkbox(&mut state.clip_flip, "Flip");
                });
                ui.add(
                    egui::DragValue::new(&mut state.clip_offset)
                        .speed(0.05)
                        .prefix("Position  ")
                        .suffix(" m"),
                )
                .on_hover_text("Drag to move the plane");
            }
            let sign = if state.clip_flip { -1.0 } else { 1.0 };
            scene.clip_plane = if state.clip_enabled {
                let mut p = [0.0, 0.0, 0.0, -state.clip_offset * sign];
                p[state.clip_axis] = sign;
                Some(p)
            } else {
                None
            };
            ui.add_space(8.0);
            eyebrow(ui, "Measure");
            ui.checkbox(&mut state.measure_active, "Pick measurement points");
            ui.label(small_muted(
                "Two points give a distance; a third gives the angle at the middle point.",
            ));
            if let Some(text) =
                measurement_text(&state.measure_points, scene.unit_factor(), &scene.units)
            {
                card().inner_margin(10).show(ui, |ui| {
                    ui.label(RichText::new(&text).strong().color(ACCENT));
                    ui.horizontal(|ui| {
                        if ui
                            .button("Add to comment")
                            .on_hover_text("Insert this measurement into the comment draft")
                            .clicked()
                        {
                            if !state.comment_draft.is_empty() {
                                state.comment_draft.push('\n');
                            }
                            state.comment_draft.push_str(&format!(
                                "Measurement: {text}. World points (m): {:?}. ",
                                state
                                    .measure_points
                                    .iter()
                                    .map(|p| p.to_array())
                                    .collect::<Vec<_>>()
                            ));
                            state.review_tab = 0;
                        }
                        if ui.button("Clear points").clicked() {
                            state.measure_points.clear();
                        }
                    });
                });
            } else if !state.measure_points.is_empty() && ui.button("Clear points").clicked() {
                state.measure_points.clear();
            }
            ui.add_space(8.0);
            eyebrow(ui, "Agent activity");
            if state.activity.is_empty() {
                ui.label(small_muted("Agent operations appear here."));
            }
            for line in state.activity.iter().rev().take(8) {
                ui.label(small_muted(line));
            }
        });
}

fn draw_review_panel(
    ctx: &egui::Context,
    scene: &mut Scene,
    state: &mut UiState,
    comparing: bool,
    view_changed: &mut bool,
) {
    use serde_json::json;
    egui::SidePanel::right("review_panel")
        .default_width(390.0)
        .width_range(320.0..=500.0)
        .frame(egui::Frame::new().fill(PANEL).inner_margin(16))
        .show(ctx, |ui| {
            let open = scene.comments.iter().filter(|c| !c.resolved).count();
            let to_review = scene
                .revisions
                .iter()
                .filter(|r| r.status == RevisionStatus::ReadyForReview)
                .count();
            ui.horizontal(|ui| {
                ui.heading("Review");
                if open > 0 {
                    pill(ui, &format!("{open} open"), WARN);
                }
                if to_review > 0 {
                    pill(ui, &format!("{to_review} to review"), ACCENT);
                }
            });
            ui.add_space(2.0);
            segmented(
                ui,
                &mut state.review_tab,
                &[
                    (
                        0,
                        format!("Comments  {}", scene.comments.len()),
                        "Your feedback, anchored in 3D",
                    ),
                    (
                        1,
                        if to_review > 0 {
                            format!("Revisions  {}  ●", scene.revisions.len())
                        } else {
                            format!("Revisions  {}", scene.revisions.len())
                        },
                        "Compare agent changes before you accept them",
                    ),
                ],
            );
            ui.add_space(4.0);
            if let Some((name, started)) = state.busy_operation.clone() {
                card()
                    .stroke(egui::Stroke::new(1.0_f32, INFO.gamma_multiply(0.5)))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new(name).strong());
                            ui.label(small_muted(format!(
                                "{:.1}s",
                                started.elapsed().as_secs_f32()
                            )));
                        });
                        if ui
                            .button("Cancel operation")
                            .on_hover_text(
                                "Discard this operation's result; your live scene stays unchanged",
                            )
                            .clicked()
                        {
                            action(state, "cancel_operation", json!({}));
                        }
                    });
                ui.add_space(4.0);
            }
            if state.review_tab == 0 {
                comments_tab(ui, scene, state, comparing);
            } else {
                revisions_tab(ui, ctx, scene, state, comparing, view_changed);
            }
        });
}

fn comments_tab(ui: &mut egui::Ui, scene: &mut Scene, state: &mut UiState, comparing: bool) {
    use serde_json::json;
    if !comparing {
        if let Some(r) = scene
            .revisions
            .iter()
            .rev()
            .find(|r| r.status == RevisionStatus::ReadyForReview)
        {
            let (label, id) = (r.label.clone(), r.id);
            card()
                .stroke(egui::Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.6)))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.set_max_width(ui.available_width() - 110.0);
                            eyebrow(ui, &format!("Revision {id:02} ready"));
                            truncated(ui, semibold(&label, 15.0), &label);
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if primary(ui, "Review", true) {
                                state.review_tab = 1;
                            }
                        });
                    });
                });
            ui.add_space(4.0);
        }
    }

    // Composer
    card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let sections = scene.collect_current_target_sections(state.selection_radius);
        ui.horizontal(|ui| {
            eyebrow(ui, "New comment");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if sections.is_empty() {
                    pill(ui, "Scene-wide", MUTED);
                } else {
                    pill(ui, &plural(sections.len(), "region", "regions"), ACCENT);
                }
            });
        });
        if sections.is_empty() {
            ui.label(small_muted("Click a part in the viewport to comment on it, or write feedback for the whole scene."));
        } else {
            // One row per part, however many regions it contributes.
            let mut groups: Vec<(u64, String, usize, usize)> = Vec::new();
            for t in &sections {
                match groups.iter_mut().find(|g| g.0 == t.object_id) {
                    Some(g) => {
                        g.2 += 1;
                        g.3 += t.selected_faces.len();
                    }
                    None => groups.push((t.object_id, t.object_name.clone(), 1, t.selected_faces.len())),
                }
            }
            let mut remove = None;
            for (id, name, regions, faces) in groups.iter().take(4) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("●").size(8.0).color(ACCENT));
                    let detail = match (*regions, *faces) {
                        (_, 0) => "whole part".to_string(),
                        (1, f) => plural(f, "face", "faces"),
                        (r, f) => format!("{} · {}", plural(r, "region", "regions"), plural(f, "face", "faces")),
                    };
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if quiet_button(ui, "×", "Remove this part from the selection") {
                            remove = Some(*id);
                        }
                        ui.label(small_muted(&detail));
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            truncated(ui, RichText::new(name).small(), name);
                        });
                    });
                });
            }
            if groups.len() > 4 {
                ui.label(small_muted(format!("+ {} more", plural(groups.len() - 4, "part", "parts"))));
            }
            if let Some(id) = remove {
                if let Some(o) = scene.objects.iter_mut().find(|o| o.id == id) {
                    o.mesh.selected_faces.clear();
                    o.mesh.bump_version();
                }
                if scene.active_object().map(|o| o.id) == Some(id) {
                    scene.selected_object =
                        scene.objects.iter().position(|o| !o.mesh.selected_faces.is_empty());
                }
                update_pending_anchor_from_selection(scene);
                state.scene_dirty_for_disk = true;
            }
        }
        let draft = ui.add(
            egui::TextEdit::multiline(&mut state.comment_draft)
                .margin(egui::vec2(10.0, 8.0))
                .hint_text(hint(if sections.is_empty() {
                    "Describe a change to the whole scene…"
                } else {
                    "What should change here? Describe the result you want…"
                }))
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        let submit_key = draft.has_focus()
            && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
        egui::CollapsingHeader::new(RichText::new("Dimensions & constraints").small().color(MUTED))
            .id_salt("constraints")
            .show(ui, |ui| {
                text_field(
                    ui,
                    &mut state.constraint_draft,
                    "e.g. opening +20 mm; keep pillars equal",
                    f32::INFINITY,
                );
            });
        let enabled = !state.comment_draft.trim().is_empty() && !comparing;
        let mut post: Option<bool> = None; // Some(scene_wide)
        ui.horizontal(|ui| {
            if primary(
                ui,
                if sections.is_empty() { "Post scene feedback" } else { "Comment on selection" },
                enabled,
            ) || (submit_key && enabled)
            {
                post = Some(sections.is_empty());
            }
            if !sections.is_empty()
                && ui
                    .add_enabled(enabled, egui::Button::new("Scene-wide"))
                    .on_hover_text("Post as feedback for the whole scene instead")
                    .clicked()
            {
                post = Some(true);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("Ctrl+Enter").small().color(SUBTLE));
            });
        });
        if let Some(scene_wide) = post {
            let mut text = state.comment_draft.trim().to_string();
            if !state.constraint_draft.trim().is_empty() {
                text.push_str(&format!("\nConstraints: {}", state.constraint_draft.trim()));
            }
            if scene_wide {
                scene.add_general_comment(text);
            } else {
                scene.add_comment_for_selection(text, state.selection_radius);
            }
            state.comment_draft.clear();
            state.constraint_draft.clear();
            state.comment_filter = FILTER_ACTIVE;
            state.scene_dirty_for_disk = true;
        }
    });
    ui.add_space(6.0);

    // Filters
    let count =
        |f: &dyn Fn(&SpatialComment) -> bool| scene.comments.iter().filter(|c| f(c)).count();
    let active = count(&|c| !c.resolved);
    let review = count(&|c| !c.resolved && c.status == CommentStatus::ReadyForReview);
    let accepted = count(&|c| effective_status(c) == CommentStatus::Accepted);
    let all = scene.comments.len();
    ui.horizontal(|ui| {
        segmented(
            ui,
            &mut state.comment_filter,
            &[
                (
                    FILTER_ACTIVE,
                    format!("Active {active}"),
                    "Open, in progress and ready for review",
                ),
                (
                    FILTER_NEEDS_REVIEW,
                    format!("Review {review}"),
                    "Agent work waiting for your decision",
                ),
                (
                    FILTER_ACCEPTED,
                    format!("Done {accepted}"),
                    "Accepted feedback",
                ),
                (FILTER_ALL, format!("All {all}"), "Everything"),
            ],
        );
    });
    if all > 3 {
        search_field(ui, &mut state.comment_search, "Search feedback…");
    }

    let search = state.comment_search.to_lowercase();
    let mut focus = None;
    let mut actions: Vec<(&'static str, serde_json::Value)> = Vec::new();
    let mut toggle_expand = None;
    let mut start_reply = None;
    let mut cancel_reply = false;
    let mut post_reply = None;
    let mut open_revision = false;
    let scroll_target = state.scroll_to_comment.take();
    egui::ScrollArea::vertical()
        .id_salt("feedback_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let mut shown = 0;
            for c in &scene.comments {
                let status = effective_status(c);
                let visible = match state.comment_filter {
                    FILTER_ACTIVE => !c.resolved,
                    FILTER_NEEDS_REVIEW => !c.resolved && c.status == CommentStatus::ReadyForReview,
                    FILTER_ACCEPTED => status == CommentStatus::Accepted,
                    _ => true,
                };
                if !visible {
                    continue;
                }
                if !search.is_empty()
                    && !format!("#{} {} {} {}", c.id, c.comment, c.object_name, c.agent_reply.as_deref().unwrap_or(""))
                        .to_lowercase()
                        .contains(&search)
                {
                    continue;
                }
                shown += 1;
                let is_active = scene.active_comment_id == Some(c.id);
                let expanded = state.expanded_comments.contains(&c.id);
                let needs_reattach = scene.comment_needs_reattachment(c);
                let revision_open = scene.revisions.iter().any(|r| {
                    r.comment_id == Some(c.id) && r.status == RevisionStatus::InProgress
                });
                let revision_busy = scene.revisions.iter().any(|r| {
                    r.comment_id == Some(c.id)
                        && matches!(r.status, RevisionStatus::InProgress | RevisionStatus::ReadyForReview)
                });
                let ready_revision = scene
                    .revisions
                    .iter()
                    .find(|r| r.comment_id == Some(c.id) && r.status == RevisionStatus::ReadyForReview)
                    .map(|r| r.id);
                let response = ui.push_id(c.id, |ui| {
                    card()
                        .stroke(egui::Stroke::new(
                            1.0_f32,
                            if is_active { ACCENT } else { BORDER },
                        ))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                ui.label(semibold(format!("#{:02}", c.id), 13.0).color(MUTED));
                                ui.label(small_muted(if c.is_general { "Scene" } else { "Region" }));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    pill(ui, status.label(), comment_status_color(status));
                                });
                            });
                            let rows = if expanded { None } else { Some(5) };
                            let mut elided = clamped_text(ui, &c.comment, 15.0, 22.0, TEXT, rows);
                            if !c.is_general {
                                let (short, full) = comment_targets(c);
                                truncated(ui, RichText::new(format!("◎  {short}")).small().color(SUBTLE), &full);
                            }
                            if needs_reattach {
                                egui::Frame::new()
                                    .fill(WARN.gamma_multiply(0.1))
                                    .corner_radius(6)
                                    .inner_margin(8)
                                    .show(ui, |ui| {
                                        ui.set_width(ui.available_width());
                                        ui.label(RichText::new("Geometry changed · this comment needs a new region").small().color(WARN));
                                        if ui
                                            .add_enabled(scene.has_any_selection() && !comparing, egui::Button::new("Attach current selection"))
                                            .on_disabled_hover_text("Select the intended region in the viewport first")
                                            .clicked()
                                        {
                                            actions.push(("reattach_comment", json!({"comment_id": c.id})));
                                        }
                                    });
                            }
                            if let Some(reply) = &c.agent_reply {
                                ui.add_space(2.0);
                                egui::Frame::new()
                                    .fill(Color32::from_rgb(23, 32, 45))
                                    .corner_radius(6)
                                    .inner_margin(10)
                                    .show(ui, |ui| {
                                        ui.set_width(ui.available_width());
                                        eyebrow(ui, "Agent update");
                                        let rows = if expanded { None } else { Some(4) };
                                        elided |= clamped_text(ui, reply, 14.0, 21.0, Color32::from_rgb(214, 222, 233), rows);
                                    });
                            }
                            let replies = if expanded || c.replies.len() <= 2 {
                                &c.replies[..]
                            } else {
                                elided = true;
                                &c.replies[c.replies.len() - 2..]
                            };
                            for reply in replies {
                                ui.label(
                                    RichText::new(format!("{}: {}", reply.author, reply.text))
                                        .size(13.0)
                                        .line_height(Some(19.0))
                                        .color(MUTED),
                                );
                            }
                            if (elided || expanded)
                                && ui
                                    .link(RichText::new(if expanded { "Show less" } else { "Show more" }).small())
                                    .clicked()
                            {
                                toggle_expand = Some(c.id);
                            }
                            ui.add_space(2.0);
                            ui.horizontal_wrapped(|ui| {
                                if let Some(rid) = ready_revision {
                                    if primary(ui, "Review changes", !comparing) {
                                        actions.push(("preview_revision", json!({"revision_id": rid, "before": false})));
                                        open_revision = true;
                                    }
                                }
                                if !c.is_general
                                    && ui
                                        .add_enabled(!comparing && !needs_reattach, egui::Button::new("View"))
                                        .on_hover_text("Select and frame this region")
                                        .on_disabled_hover_text(if needs_reattach {
                                            "Reattach this comment first"
                                        } else {
                                            "Return to the live scene first"
                                        })
                                        .clicked()
                                {
                                    focus = Some(c.id);
                                }
                                let accept_label = if c.resolved { "Reopen" } else { "Accept" };
                                if ui
                                    .add_enabled(!comparing && !revision_open, egui::Button::new(accept_label))
                                    .on_hover_text(if c.resolved {
                                        "Mark this feedback as open again"
                                    } else if ready_revision.is_some() {
                                        "Accept the agent's revision for this comment"
                                    } else {
                                        "Mark this feedback as done"
                                    })
                                    .on_disabled_hover_text("The agent is still working on this")
                                    .clicked()
                                {
                                    if let Some(rid) = ready_revision {
                                        actions.push(("accept_revision", json!({"revision_id": rid})));
                                    } else {
                                        actions.push((
                                            "update_comment",
                                            json!({"comment_id": c.id, "status": if c.resolved { "open" } else { "accepted" }}),
                                        ));
                                    }
                                }
                                if ui.button("Reply").clicked() {
                                    start_reply = Some(c.id);
                                }
                                ui.menu_button("…", |ui| {
                                    if ui
                                        .add_enabled(!revision_busy, egui::Button::new(RichText::new("Delete comment").color(DANGER)))
                                        .on_disabled_hover_text("A revision for this comment is open")
                                        .clicked()
                                    {
                                        actions.push(("delete_comment", json!({"comment_id": c.id})));
                                        ui.close_menu();
                                    }
                                });
                            });
                            if state.reply_to == Some(c.id) {
                                let r = ui.add(
                                    egui::TextEdit::multiline(&mut state.reply_draft)
                                        .hint_text(hint("Add a reply…"))
                                        .margin(egui::vec2(8.0, 6.0))
                                        .desired_rows(2)
                                        .desired_width(f32::INFINITY),
                                );
                                let send_key = r.has_focus()
                                    && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
                                ui.horizontal(|ui| {
                                    let ok = !state.reply_draft.trim().is_empty();
                                    if primary(ui, "Post reply", ok) || (send_key && ok) {
                                        post_reply = Some(c.id);
                                    }
                                    if ui.button("Cancel").clicked() {
                                        cancel_reply = true;
                                    }
                                });
                            }
                        })
                        .response
                });
                if scroll_target == Some(c.id) {
                    response.inner.scroll_to_me(Some(egui::Align::TOP));
                }
                ui.add_space(6.0);
            }
            if shown == 0 {
                ui.add_space(16.0);
                if scene.comments.is_empty() {
                    eyebrow(ui, "Start a conversation");
                    ui.label(RichText::new("Select a part in the viewport and describe how you'd like it to change. Your agent reads these comments and responds with revisions.").color(MUTED));
                } else if !search.is_empty() {
                    ui.label(small_muted("No feedback matches your search."));
                } else {
                    ui.label(small_muted(match state.comment_filter {
                        FILTER_NEEDS_REVIEW => "Nothing is waiting for your review.",
                        FILTER_ACCEPTED => "No accepted feedback yet.",
                        _ => "All feedback has been accepted.",
                    }));
                }
            }
        });
    for (tool, args) in actions {
        action(state, tool, args);
    }
    if open_revision {
        state.review_tab = 1;
    }
    if let Some(id) = toggle_expand {
        if !state.expanded_comments.remove(&id) {
            state.expanded_comments.insert(id);
        }
    }
    if let Some(id) = start_reply {
        state.reply_to = Some(id);
        state.reply_draft.clear();
    }
    if cancel_reply {
        state.reply_to = None;
        state.reply_draft.clear();
    }
    if let Some(id) = post_reply {
        action(
            state,
            "update_comment",
            json!({"comment_id": id, "reply": state.reply_draft.trim(), "author": "You"}),
        );
        state.reply_draft.clear();
        state.reply_to = None;
    }
    if let Some(id) = focus {
        state.focus_comment = Some(id);
    }
}

fn revisions_tab(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    scene: &mut Scene,
    state: &mut UiState,
    comparing: bool,
    view_changed: &mut bool,
) {
    use serde_json::json;
    ui.horizontal(|ui| {
        if ui
            .checkbox(&mut scene.highlight_changes, "Highlight changed parts")
            .changed()
        {
            *view_changed = true;
        }
        if comparing {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if primary(ui, "Back to live", true) {
                    scene.preview_revision = None;
                    *view_changed = true;
                }
            });
        }
    });
    let mut go_to_comment = None;
    egui::ScrollArea::vertical()
        .id_salt("revision_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if scene.revisions.is_empty() {
                card().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    eyebrow(ui, "No revisions yet");
                    ui.label(RichText::new("When your agent changes the model it submits a revision here. You can compare before and after, then accept or revert it.").color(MUTED));
                });
            }
            let revision_ids: Vec<u64> = scene.revisions.iter().rev().map(|r| r.id).collect();
            for rid in revision_ids {
                let changed = {
                    let r = scene.revisions.iter().find(|r| r.id == rid).unwrap();
                    if r.status == RevisionStatus::InProgress {
                        Vec::new()
                    } else {
                        scene.changed_objects(rid)
                    }
                };
                let r = scene.revisions.iter().find(|r| r.id == rid).unwrap();
                let previewing = scene.preview_revision.filter(|(id, _)| *id == rid);
                let (status_text, status_color) = revision_status(r.status);
                card()
                    .stroke(egui::Stroke::new(
                        1.0_f32,
                        if previewing.is_some() {
                            ACCENT
                        } else if r.status == RevisionStatus::ReadyForReview {
                            ACCENT.gamma_multiply(0.45)
                        } else {
                            BORDER
                        },
                    ))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            eyebrow(ui, &format!("Revision {:02}", r.id));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                pill(ui, status_text, status_color);
                            });
                        });
                        ui.label(semibold(&r.label, 17.0).line_height(Some(23.0)));
                        if let Some(id) = r.comment_id {
                            if ui
                                .link(RichText::new(format!("For comment #{id:02}")).small())
                                .on_hover_text("Show the comment")
                                .clicked()
                            {
                                go_to_comment = Some(id);
                            }
                        }
                        if !r.summary.is_empty() {
                            clamped_text(ui, &r.summary, 14.5, 21.0, TEXT, None);
                        }
                        if r.status != RevisionStatus::InProgress {
                            let names: Vec<String> = changed
                                .iter()
                                .filter_map(|id| scene.objects.iter().find(|o| o.id == *id).map(|o| o.name.clone()))
                                .collect();
                            ui.label(RichText::new(plural(changed.len(), "changed part", "changed parts")).small().color(ACCENT))
                                .on_hover_text(if names.is_empty() { "No parts changed".into() } else { names.join("\n") });
                        }
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label(small_muted("Compare"));
                            let mut choice = match previewing {
                                Some((_, true)) => 1,
                                Some((_, false)) => 2,
                                None => 0,
                            };
                            if segmented(
                                ui,
                                &mut choice,
                                &[
                                    (0, "Live".into(), "The current scene"),
                                    (1, "Before".into(), "Geometry before this revision"),
                                    (2, "After".into(), "Geometry after this revision"),
                                ],
                            ) {
                                match choice {
                                    0 => scene.preview_revision = None,
                                    n => action(state, "preview_revision", json!({"revision_id": rid, "before": n == 1})),
                                }
                                *view_changed = true;
                            }
                        });
                        if r.status == RevisionStatus::ReadyForReview {
                            ui.horizontal(|ui| {
                                if primary(ui, "Accept revision", true) {
                                    action(state, "accept_revision", json!({"revision_id": rid}));
                                    *view_changed = true;
                                }
                                if ui
                                    .button("Revert")
                                    .on_hover_text("Restore the geometry from before this revision")
                                    .clicked()
                                {
                                    action(state, "revert_revision", json!({"revision_id": rid}));
                                    *view_changed = true;
                                }
                            });
                        } else if r.status == RevisionStatus::InProgress
                            && ui
                                .button("Cancel revision")
                                .on_hover_text("Restore its starting geometry between agent operations")
                                .clicked()
                        {
                            action(state, "cancel_revision", json!({"revision_id": rid}));
                            *view_changed = true;
                        }
                        let shots = r.screenshots.clone();
                        if !shots.is_empty() {
                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                let w = if shots.len() == 1 {
                                    ui.available_width()
                                } else {
                                    (ui.available_width() - ui.spacing().item_spacing.x) / 2.0
                                };
                                for path in &shots {
                                    match load_screenshot(ctx, state, path) {
                                        Ok(texture) => {
                                            let resp = ui
                                                .add(
                                                    egui::Image::new((texture.id(), texture.size_vec2()))
                                                        .max_width(w)
                                                        .max_height(200.0)
                                                        .corner_radius(6)
                                                        .sense(egui::Sense::click()),
                                                )
                                                .on_hover_text("Click to enlarge")
                                                .on_hover_cursor(egui::CursorIcon::ZoomIn);
                                            if resp.clicked() {
                                                state.lightbox = Some(path.clone());
                                            }
                                        }
                                        Err(_) => {
                                            ui.label(small_muted("Screenshot unavailable")).on_hover_text(path);
                                        }
                                    }
                                }
                            });
                        }
                    });
                ui.add_space(8.0);
            }
        });
    if let Some(id) = go_to_comment {
        state.review_tab = 0;
        state.scroll_to_comment = Some(id);
        if state.comment_filter != FILTER_ALL {
            state.comment_filter = FILTER_ALL;
        }
    }
}

fn load_screenshot(
    ctx: &egui::Context,
    state: &mut UiState,
    path: &str,
) -> Result<egui::TextureHandle, String> {
    state
        .screenshot_cache
        .entry(path.to_owned())
        .or_insert_with(|| {
            let image = image::open(path)
                .map_err(|e| e.to_string())?
                .thumbnail(1600, 1200)
                .into_rgba8();
            let pixels = egui::ColorImage::from_rgba_unmultiplied(
                [image.width() as usize, image.height() as usize],
                image.as_raw(),
            );
            Ok(ctx.load_texture(path, pixels, egui::TextureOptions::LINEAR))
        })
        .clone()
}

fn draw_viewport(
    ctx: &egui::Context,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    renderer: &mut Renderer3D,
    state: &mut UiState,
    comparing: bool,
    view_changed: &mut bool,
) {
    use serde_json::json;
    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
        state.viewport_rect = ui.max_rect();
        let rect = ui.max_rect();

        // View controls (top-left)
        egui::Area::new("viewport_controls".into())
            .fixed_pos(rect.left_top() + egui::vec2(14.0, 14.0))
            .show(ctx, |ui| {
                overlay_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        segmented(
                            ui,
                            &mut renderer.shading,
                            &[
                                (ViewportShading::StudioLit, "Lit".into(), "Materials and studio lighting"),
                                (ViewportShading::SculptClay, "Clay".into(), "Neutral clay to judge form"),
                                (ViewportShading::WireframeOnly, "Wire".into(), "Wireframe only"),
                            ],
                        );
                        ui.checkbox(&mut renderer.show_wireframe_overlay, "Edges");
                        ui.checkbox(&mut renderer.show_grid, "Grid");
                        ui.menu_button(if camera.orthographic { "Ortho  ▾" } else { "Persp  ▾" }, |ui| {
                            ui.set_min_width(170.0);
                            if ui.button("Front").clicked() {
                                camera.set_view_front();
                                ui.close_menu();
                            }
                            if ui.button("Right").clicked() {
                                camera.set_view_right();
                                ui.close_menu();
                            }
                            if ui.button("Top").clicked() {
                                camera.set_view_top();
                                ui.close_menu();
                            }
                            ui.separator();
                            ui.checkbox(&mut camera.orthographic, "Orthographic");
                            ui.separator();
                            if ui.button("Frame selection   F").clicked() {
                                frame_selection(scene, camera);
                                ui.close_menu();
                            }
                            if ui.button("Frame all").clicked() {
                                frame_all(scene, camera);
                                ui.close_menu();
                            }
                        })
                        .response
                        .on_hover_text("Camera views and projection");
                    });
                });
            });

        // Comparison banner (top-centre)
        let mut top = rect.top() + 14.0;
        if let Some((rid, before)) = scene.preview_revision {
            let label = scene
                .revisions
                .iter()
                .find(|r| r.id == rid)
                .map(|r| r.label.clone())
                .unwrap_or_default();
            let resp = egui::Area::new("compare_banner".into())
                .pivot(egui::Align2::CENTER_TOP)
                .fixed_pos(egui::pos2(rect.center().x, rect.top() + 76.0))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(OVERLAY)
                        .stroke(egui::Stroke::new(1.5_f32, ACCENT))
                        .corner_radius(10)
                        .inner_margin(egui::Margin::symmetric(14, 10))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 0.0;
                                    eyebrow(ui, &format!("Comparing revision {rid:02} · read-only"));
                                    ui.set_max_width(260.0);
                                    truncated(ui, semibold(&label, 14.0), &label);
                                });
                                ui.add_space(8.0);
                                let mut choice = before;
                                if segmented(
                                    ui,
                                    &mut choice,
                                    &[(true, "Before".into(), ""), (false, "After".into(), "")],
                                ) {
                                    action(state, "preview_revision", json!({"revision_id": rid, "before": choice}));
                                    *view_changed = true;
                                }
                                if primary(ui, "Back to live", true) {
                                    scene.preview_revision = None;
                                    *view_changed = true;
                                }
                            });
                        });
                });
            top = resp.response.rect.bottom() + 8.0;
        } else {
            top += 50.0;
        }

        // Error toast
        if let Some(error) = state.last_error.clone() {
            egui::Area::new("error_toast".into())
                .pivot(egui::Align2::CENTER_TOP)
                .fixed_pos(egui::pos2(rect.center().x, top))
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(DANGER_BG)
                        .stroke(egui::Stroke::new(1.0_f32, DANGER.gamma_multiply(0.5)))
                        .corner_radius(8)
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .show(ui, |ui| {
                            ui.set_max_width((rect.width() - 80.0).clamp(240.0, 560.0));
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("!").strong().color(DANGER));
                                ui.label(RichText::new(&error).color(Color32::from_rgb(255, 205, 198)));
                                if quiet_button(ui, "×", "Dismiss") {
                                    state.last_error = None;
                                }
                            });
                        });
                });
        }

        // Comment pins
        let viewport = Vec2::new(rect.width(), rect.height());
        if !comparing {
            for c in &scene.comments {
                if c.resolved || c.is_general || scene.comment_needs_reattachment(c) {
                    continue;
                }
                let Some(p) = camera.world_to_screen(scene.current_comment_anchor(c), viewport) else {
                    continue;
                };
                let pos = rect.left_top() + egui::vec2(p.x, p.y);
                if !rect.shrink(20.0).contains(pos) {
                    continue;
                }
                let active = scene.active_comment_id == Some(c.id);
                let color = comment_status_color(c.status);
                egui::Area::new(egui::Id::new(("comment_pin", c.id)))
                    .pivot(egui::Align2::CENTER_BOTTOM)
                    .fixed_pos(pos)
                    .show(ctx, |ui| {
                        let text = RichText::new(format!("#{:02}", c.id))
                            .font(medium_font(12.5))
                            .color(if active { ON_ACCENT } else { color });
                        let resp = ui
                            .add(
                                egui::Button::new(text)
                                    .fill(if active { ACCENT } else { OVERLAY })
                                    .stroke(egui::Stroke::new(1.2_f32, color))
                                    .corner_radius(12)
                                    .min_size(egui::vec2(0.0, 24.0)),
                            )
                            .on_hover_ui(|ui| {
                                ui.set_max_width(300.0);
                                ui.label(RichText::new(c.comment.clone()));
                                ui.label(small_muted(format!("{} · click to open", c.status.label())));
                            });
                        if resp.clicked() {
                            state.focus_comment = Some(c.id);
                            state.scroll_to_comment = Some(c.id);
                            state.review_tab = 0;
                            if state.comment_filter == FILTER_ACCEPTED {
                                state.comment_filter = FILTER_ACTIVE;
                            }
                            state.comment_search.clear();
                        }
                    });
            }
        }

        // Measurement overlay
        let painter = ui.painter().with_clip_rect(rect);
        let mut screen_points = Vec::new();
        for (i, &p) in state.measure_points.iter().enumerate() {
            if let Some(p) = camera.world_to_screen(p, viewport) {
                let p = rect.left_top() + egui::vec2(p.x, p.y);
                screen_points.push(p);
                painter.circle_filled(p, 5.0, ACCENT);
                painter.text(
                    p + egui::vec2(9.0, -9.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{}", i + 1),
                    FontId::proportional(12.0),
                    ACCENT,
                );
            }
        }
        for pair in screen_points.windows(2) {
            painter.line_segment([pair[0], pair[1]], egui::Stroke::new(2.0_f32, ACCENT));
        }

        // Contextual hint (bottom-left)
        let mode_hint = if comparing {
            "Comparison is read-only · go back to live to select".to_string()
        } else if state.measure_active {
            "Measure · click surfaces to place points".to_string()
        } else {
            match state.select_mode {
                SmartSelectMode::SubPart => "Click a connected piece to select it",
                SmartSelectMode::SmartSurface => "Click a surface · the crease angle sets its edges",
                SmartSelectMode::WholeLayer => "Click an object to select all of it",
                SmartSelectMode::Brush => "Drag to paint · Ctrl+drag erases · [ ] radius",
            }
            .to_string()
        };
        egui::Area::new("viewport_hint".into())
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(egui::pos2(rect.left() + 14.0, rect.bottom() - 14.0))
            .show(ctx, |ui| {
                overlay_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            ui.label(RichText::new(mode_hint).small().color(TEXT));
                            ui.label(small_muted("Middle-drag or Alt+drag orbit · Right-drag pan · Scroll zoom · F frame"));
                        });
                        if ui
                            .add(egui::Button::new(RichText::new("?").strong()).min_size(egui::vec2(28.0, 28.0)))
                            .on_hover_text("Keyboard shortcuts  ·  F1")
                            .clicked()
                        {
                            state.show_shortcuts = !state.show_shortcuts;
                        }
                    });
                });
            });
    });
}

fn draw_dialogs(ctx: &egui::Context, state: &mut UiState) {
    use serde_json::json;
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    if let Some(mode) = state.project_dialog.clone() {
        let mut open = true;
        let (title, verb, help) = match mode.as_str() {
            "open" => (
                "Open project",
                "Open project",
                "Loads geometry, feedback, saved views and revision history.",
            ),
            "recover" => (
                "Recover autosave",
                "Recover autosave",
                "Loads the most recent .autosave copy of this project.",
            ),
            _ => (
                "Save project",
                "Save project",
                "Saves geometry, feedback, saved views and revision history. Each save keeps a .bak copy.",
            ),
        };
        egui::Window::new(title)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(480.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                eyebrow(ui, "Project file (.spatial.json)");
                let r = text_field(ui, &mut state.project_file, "path/to/project.spatial.json", f32::INFINITY);
                if !r.has_focus() && ui.memory(|m| m.focused().is_none()) {
                    r.request_focus();
                }
                ui.label(small_muted(help));
                ui.add_space(4.0);
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let ok = !state.project_file.trim().is_empty();
                ui.horizontal(|ui| {
                    if primary(ui, verb, ok) || (enter && ok) {
                        action(
                            state,
                            if mode == "save" { "save_project" } else { "open_project" },
                            json!({"path": state.project_file.trim(), "recover": mode == "recover"}),
                        );
                        state.project_dialog = None;
                    }
                    if ui.button("Cancel").clicked() {
                        state.project_dialog = None;
                    }
                });
            });
        if !open || escape {
            state.project_dialog = None;
        }
    }

    if state.show_io_modal {
        let mut open = true;
        egui::Window::new("Export model")
            .open(&mut open)
            .default_width(460.0)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                eyebrow(ui, "Format");
                let ext = std::path::Path::new(state.io_file_path.trim())
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase())
                    .unwrap_or_default();
                let mut chosen: &str = &ext;
                if segmented(
                    ui,
                    &mut chosen,
                    &[
                        ("obj", "OBJ".into(), "Wavefront OBJ"),
                        ("stl", "STL".into(), "STL for 3D printing"),
                        ("glb", "GLB".into(), "Binary glTF 2.0"),
                    ],
                ) {
                    state.io_file_path = std::path::Path::new(state.io_file_path.trim())
                        .with_extension(chosen)
                        .to_string_lossy()
                        .into_owned();
                }
                eyebrow(ui, "Destination");
                let r = text_field(
                    ui,
                    &mut state.io_file_path,
                    "model_export.obj",
                    f32::INFINITY,
                );
                if !r.has_focus() && ui.memory(|m| m.focused().is_none()) {
                    r.request_focus();
                }
                let valid = matches!(ext.as_str(), "obj" | "stl" | "glb");
                if !valid && !state.io_file_path.trim().is_empty() {
                    ui.label(
                        RichText::new("Use a .obj, .stl or .glb file name")
                            .small()
                            .color(WARN),
                    );
                }
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                ui.horizontal(|ui| {
                    if primary(ui, "Export scene", valid) || (enter && valid) {
                        action(
                            state,
                            "export_model",
                            json!({"path": state.io_file_path.trim()}),
                        );
                        state.show_io_modal = false;
                    }
                    if ui.button("Cancel").clicked() {
                        state.show_io_modal = false;
                    }
                });
            });
        if !open || escape {
            state.show_io_modal = false;
        }
    }

    if let Some(path) = state.lightbox.clone() {
        let screen = ctx.screen_rect();
        egui::Area::new("lightbox_scrim".into())
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                let (rect, resp) = ui.allocate_exact_size(screen.size(), egui::Sense::click());
                ui.painter()
                    .rect_filled(rect, 0, Color32::from_black_alpha(200));
                if let Some(Ok(texture)) = state.screenshot_cache.get(&path) {
                    let size = texture.size_vec2();
                    let scale = ((screen.width() * 0.88) / size.x)
                        .min((screen.height() * 0.85) / size.y)
                        .min(2.0);
                    let img_rect = egui::Rect::from_center_size(screen.center(), size * scale);
                    ui.painter().image(
                        texture.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    ui.painter().text(
                        egui::pos2(img_rect.left(), img_rect.bottom() + 10.0),
                        egui::Align2::LEFT_TOP,
                        format!("{path}   ·   click or press Esc to close"),
                        FontId::proportional(12.0),
                        MUTED,
                    );
                }
                if resp.clicked() {
                    state.lightbox = None;
                }
            });
        if escape {
            state.lightbox = None;
        }
    }

    if state.show_shortcuts {
        let mut open = true;
        egui::Window::new("Keyboard shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(400.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                for (group, rows) in [
                    (
                        "Select",
                        &[
                            ("1  2  3  4", "Part · Surface · Layer · Brush"),
                            ("Tab", "Cycle selection mode"),
                            ("Shift + click", "Add to selection"),
                            ("Ctrl + click", "Remove from selection"),
                            ("A", "Select all faces of the active part"),
                            ("]  /  [", "Grow / shrink selection (brush radius)"),
                            ("Esc  or  C", "Clear selection"),
                        ][..],
                    ),
                    (
                        "Navigate",
                        &[
                            ("Middle-drag  /  Alt + drag", "Orbit"),
                            ("Right-drag  /  Shift + middle", "Pan"),
                            ("Scroll", "Zoom"),
                            ("F", "Frame selection"),
                        ][..],
                    ),
                    (
                        "Project",
                        &[
                            ("Ctrl + S", "Save project"),
                            ("Ctrl + Z", "Undo"),
                            ("Ctrl + Shift + Z", "Redo"),
                            ("Ctrl + Enter", "Post comment or reply"),
                            ("F1  or  ?", "Toggle this panel"),
                        ][..],
                    ),
                ] {
                    eyebrow(ui, group);
                    egui::Grid::new(group)
                        .num_columns(2)
                        .spacing([18.0, 6.0])
                        .show(ui, |ui| {
                            for (keys, what) in rows {
                                ui.label(
                                    RichText::new(*keys).font(medium_font(13.0)).color(ACCENT),
                                );
                                ui.label(RichText::new(*what).size(13.5));
                                ui.end_row();
                            }
                        });
                    ui.add_space(8.0);
                }
            });
        if !open || (escape && !state.modal_open()) {
            state.show_shortcuts = false;
        }
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
