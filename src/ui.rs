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
    /// Drag a screen-space rectangle; every visible face inside it is selected.
    Box,
    /// Click single polygons; with Path on, a click fills the run of faces from the last pick.
    Face,
    /// A world-space 3D box with handles; everything inside it is selected from any angle.
    Volume,
}

impl SmartSelectMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::SubPart => "Part",
            Self::SmartSurface => "Surface",
            Self::WholeLayer => "Layer",
            Self::Box => "Box",
            Self::Face => "Face",
            Self::Volume => "3D box",
        }
    }

    /// Modes driven by dragging rather than by a hover-previewed click.
    pub fn is_drag(self) -> bool {
        matches!(self, Self::Box)
    }
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
    /// Last time an agent polled `wait_for_feedback`.
    pub agent_listening: Option<std::time::Instant>,
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
    /// Mode to return to when Ctrl+C toggles the box selector off.
    pub prev_select_mode: SmartSelectMode,
    /// Box selection also picks faces hidden behind others.
    pub box_through: bool,
    /// Rectangle being dragged by the box selector, in egui points.
    pub box_rect: Option<egui::Rect>,
    /// The 3D box is being dragged (a handle or a size field) this frame.
    pub volume_dragging: bool,
    /// Box edits waiting for the drag to end before scheduling a save.
    pub volume_drag_changed: bool,
    /// Comment targets for the current selection, keyed by `selection_stamp`. Collecting them
    /// walks every selected face, so it only happens when the selection actually changes.
    pub target_sections: Option<(u64, std::sync::Arc<Vec<crate::scene::CommentTargetSection>>)>,
    /// Face mode: a click fills the shortest run of faces from the previous pick.
    pub face_path: bool,
    /// Outliner groups ("Group/Part" names) the user has folded.
    pub collapsed_groups: HashSet<String>,
    /// World point the camera should centre on next frame.
    pub frame_point: Option<Vec3>,
    /// Last face picked in Face mode: (object id, face index).
    pub last_face_pick: Option<(u64, usize)>,
    /// Free-flight navigation is active (cursor captured, WASD moves).
    pub fly_active: bool,
    /// Requests the app to enter/leave fly mode (needs the window to capture the cursor).
    pub fly_toggle_requested: bool,
    /// Free-flight speed in scene units per second.
    pub fly_speed: f32,
    /// The "Start a new project" confirmation is open.
    pub confirm_new_project: bool,
    /// Title typed into the new-project confirmation.
    pub new_project_name: String,
    /// Command palette (Ctrl+K).
    pub palette_open: bool,
    pub palette_query: String,
    pub palette_index: usize,
    /// New comments go to the agent immediately instead of waiting as drafts.
    pub auto_send: bool,
    /// Reference images attached to the comment being written.
    pub draft_attachments: Vec<String>,
    /// Revision whose "Request changes" note is being written.
    pub request_changes_for: Option<u64>,
    pub request_notes: String,
    /// Before/after wipe position while comparing (None = single side).
    pub compare_split: Option<f32>,
    /// Focus the comment composer on the next frame.
    pub focus_composer: bool,
    /// Path being typed to attach an image (None = field hidden).
    pub attach_path: Option<String>,
    /// Asset library tab.
    pub library: LibraryUi,
}

/// State of the asset library tab. The listing is shared so cloning `UiState` stays cheap.
#[derive(Clone, Default)]
pub struct LibraryUi {
    pub sections: std::sync::Arc<Vec<crate::library::Section>>,
    /// When the library folder was last read; `None` forces a rescan.
    pub scanned: Option<std::time::Instant>,
    pub search: String,
    /// Folded folders: "Section" or "Section/Subsection".
    pub collapsed: HashSet<String>,
    /// The full-window library view is open.
    pub open: bool,
    /// Section (and subsection) shown in the full view; `None` shows everything.
    pub filter: Option<(String, Option<String>)>,
    /// The "save to library" form is open.
    pub save_open: bool,
    /// The form saves the selection instead of the whole model.
    pub save_selection: bool,
    pub asset_name: String,
    pub section: String,
    pub subsection: String,
    /// Folder being named: the parent section (None = a new section) and the typed name.
    pub new_folder: Option<(Option<String>, String)>,
    /// Asset whose delete button was pressed once; a second press within a few seconds deletes.
    pub confirm_delete: Option<(String, std::time::Instant)>,
    /// The export form: `Some(Some(reference))` exports one asset, `Some(None)` every asset
    /// the view is showing.
    pub export_scope: Option<Option<String>>,
    pub export: ExportUi,
}

/// Export choices shared by the library export form and the scene export dialog.
#[derive(Clone, Default)]
pub struct ExportUi {
    /// The remembered choices have been read from the library folder.
    pub loaded: bool,
    pub directory: String,
    pub blender: bool,
    /// Keep every part as its own object instead of merging by material.
    pub separate: bool,
    pub collision: crate::gltf::Collision,
    /// Triangle budget as typed; empty exports every triangle.
    pub budget: String,
}

impl ExportUi {
    pub fn max_triangles(&self) -> Option<usize> {
        self.budget.trim().replace([',', '_', ' '], "").parse().ok().filter(|&n: &usize| n > 0)
    }
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            busy_operation: None,
            actions: Vec::new(),
            last_error: None,
            last_agent_activity: None,
            agent_listening: None,
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
            prev_select_mode: SmartSelectMode::SubPart,
            box_through: false,
            box_rect: None,
            volume_dragging: false,
            volume_drag_changed: false,
            target_sections: None,
            face_path: false,
            collapsed_groups: HashSet::new(),
            frame_point: None,
            last_face_pick: None,
            fly_active: false,
            fly_toggle_requested: false,
            fly_speed: 3.0,
            confirm_new_project: false,
            new_project_name: String::new(),
            palette_open: false,
            palette_query: String::new(),
            palette_index: 0,
            auto_send: false,
            draft_attachments: Vec::new(),
            request_changes_for: None,
            request_notes: String::new(),
            compare_split: None,
            focus_composer: false,
            attach_path: None,
            library: LibraryUi::default(),
        }
    }
}

impl UiState {
    /// True while a modal surface (dialog or lightbox) owns Escape and pointer input.
    pub fn modal_open(&self) -> bool {
        self.project_dialog.is_some()
            || self.show_io_modal
            || self.lightbox.is_some()
            || self.palette_open
            || self.library.open
    }
}

/// Fingerprint of everything `Scene::collect_current_target_sections` depends on.
fn selection_stamp(scene: &Scene, selection_radius: f32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    scene.selected_object.hash(&mut h);
    selection_radius.to_bits().hash(&mut h);
    scene
        .pending_anchor
        .map(|(id, world, local, normal)| {
            (
                id,
                [world, local, normal].map(|v| v.to_array().map(f32::to_bits)),
            )
        })
        .hash(&mut h);
    for o in &scene.objects {
        o.id.hash(&mut h);
        o.name.hash(&mut h);
        o.mesh.version.hash(&mut h);
        o.mesh.selected_faces.len().hash(&mut h);
        o.transform.matrix().to_cols_array().map(f32::to_bits).hash(&mut h);
    }
    // Order-independent: the labels live in a hash map.
    let labels = scene.region_labels.iter().fold(0u64, |acc, entry| {
        let mut e = std::collections::hash_map::DefaultHasher::new();
        entry.hash(&mut e);
        acc.wrapping_add(e.finish())
    });
    labels.hash(&mut h);
    h.finish()
}

/// Comment targets for the current selection, recomputed only when the selection changes.
/// While the 3D box is being dragged the previous result is reused, so a drag over a dense
/// scene is not slowed down by regrouping hundreds of thousands of faces every frame.
fn current_target_sections(
    scene: &Scene,
    state: &mut UiState,
) -> std::sync::Arc<Vec<crate::scene::CommentTargetSection>> {
    if let (true, Some((_, sections))) = (state.volume_dragging, &state.target_sections) {
        return sections.clone();
    }
    let stamp = selection_stamp(scene, state.selection_radius);
    match &state.target_sections {
        Some((key, sections)) if *key == stamp => sections.clone(),
        _ => {
            let sections =
                std::sync::Arc::new(scene.collect_current_target_sections(state.selection_radius));
            state.target_sections = Some((stamp, sections.clone()));
            sections
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

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

pub const ACCENT: Color32 = Color32::from_rgb(103, 226, 195);
const MUTED: Color32 = Color32::from_rgb(158, 169, 186);
const SUBTLE: Color32 = Color32::from_rgb(110, 122, 140);
const TEXT: Color32 = Color32::from_rgb(232, 237, 244);
const CARD: Color32 = Color32::from_rgb(26, 32, 43);
const CARD_HI: Color32 = Color32::from_rgb(33, 41, 54);
const BORDER: Color32 = Color32::from_rgb(39, 48, 62);
const PANEL: Color32 = Color32::from_rgb(18, 23, 32);
const HEADER: Color32 = Color32::from_rgb(13, 17, 24);
const FIELD: Color32 = Color32::from_rgb(11, 15, 22);
const OVERLAY: Color32 = Color32::from_rgba_premultiplied(15, 19, 27, 236);
const ON_ACCENT: Color32 = Color32::from_rgb(8, 30, 27);
const SELECTED_ROW: Color32 = Color32::from_rgb(27, 52, 52);
const HOVER_ROW: Color32 = Color32::from_rgb(26, 33, 45);
const WARN: Color32 = Color32::from_rgb(245, 187, 113);
const INFO: Color32 = Color32::from_rgb(130, 176, 255);
const DANGER: Color32 = Color32::from_rgb(255, 160, 150);
const DANGER_BG: Color32 = Color32::from_rgb(64, 35, 41);

use egui_phosphor::regular as icon;

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
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
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
    style.visuals.window_fill = CARD;
    style.visuals.extreme_bg_color = FIELD;
    style.visuals.faint_bg_color = CARD;
    style.visuals.override_text_color = Some(TEXT);
    style.visuals.hyperlink_color = ACCENT;
    style.visuals.selection.bg_fill = Color32::from_rgb(34, 82, 76);
    style.visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    style.visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 10],
        blur: 32,
        spread: 0,
        color: Color32::from_black_alpha(120),
    };
    style.visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: Color32::from_black_alpha(110),
    };
    let radius = egui::CornerRadius::same(7);
    let widgets = &mut style.visuals.widgets;
    widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
    widgets.noninteractive.corner_radius = radius;
    widgets.inactive.bg_fill = CARD_HI;
    widgets.inactive.weak_bg_fill = CARD_HI;
    widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(44, 54, 70));
    widgets.inactive.corner_radius = radius;
    widgets.hovered.bg_fill = Color32::from_rgb(41, 52, 68);
    widgets.hovered.weak_bg_fill = Color32::from_rgb(38, 48, 63);
    widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(70, 86, 108));
    widgets.hovered.corner_radius = radius;
    widgets.active.bg_fill = Color32::from_rgb(34, 82, 76);
    widgets.active.weak_bg_fill = Color32::from_rgb(34, 82, 76);
    widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, ACCENT);
    widgets.active.corner_radius = radius;
    widgets.open.corner_radius = radius;
    style.visuals.window_stroke = egui::Stroke::new(1.0_f32, BORDER);
    style.visuals.window_corner_radius = egui::CornerRadius::same(12);
    style.visuals.menu_corner_radius = egui::CornerRadius::same(9);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.interact_size.y = 30.0;
    style.spacing.window_margin = egui::Margin::same(18);
    style.spacing.menu_margin = egui::Margin::same(6);
    style.spacing.slider_rail_height = 4.0;
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(19.0, egui::FontFamily::Name("Studio Semibold".into())),
    );
    style
        .text_styles
        .insert(egui::TextStyle::Body, FontId::proportional(14.5));
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

/// Square icon button for toolbars. `selected` draws it as the active tool.
fn icon_button(
    ui: &mut egui::Ui,
    glyph: &str,
    tip: &str,
    selected: bool,
    enabled: bool,
) -> egui::Response {
    let size = egui::vec2(34.0, 34.0);
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    let hovered = enabled && resp.hovered();
    let p = ui.painter();
    if selected {
        p.rect_filled(rect, 8, ACCENT.gamma_multiply(0.16));
        p.rect_stroke(
            rect,
            8,
            egui::Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.55)),
            egui::StrokeKind::Inside,
        );
    } else if hovered {
        p.rect_filled(rect, 8, Color32::from_rgb(40, 50, 66));
    }
    let color = if !enabled {
        Color32::from_rgb(72, 82, 98)
    } else if selected {
        ACCENT
    } else if hovered {
        TEXT
    } else {
        MUTED
    };
    p.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        glyph,
        FontId::proportional(18.0),
        color,
    );
    if tip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tip)
    }
}

/// Icon followed by text, for buttons and menu entries.
fn icon_text(glyph: &str, text: &str) -> String {
    format!("{glyph}  {text}")
}

/// iOS-style switch with a clickable label.
fn toggle(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    let switch = egui::vec2(30.0, 17.0);
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        FontId::proportional(14.0),
        TEXT,
    );
    let gap = if label.is_empty() { 0.0 } else { 9.0 };
    let size = egui::vec2(
        switch.x + gap + galley.size().x,
        switch.y.max(galley.size().y).max(22.0),
    );
    let (rect, mut resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(resp.id, *on);
    let sw = egui::Rect::from_min_size(
        egui::pos2(rect.left(), rect.center().y - switch.y * 0.5),
        switch,
    );
    let off_bg = Color32::from_rgb(46, 56, 72);
    let on_bg = Color32::from_rgb(84, 186, 160);
    let mix = |a: u8, b: u8| egui::lerp(a as f32..=b as f32, t) as u8;
    let bg = Color32::from_rgb(
        mix(off_bg.r(), on_bg.r()),
        mix(off_bg.g(), on_bg.g()),
        mix(off_bg.b(), on_bg.b()),
    );
    let p = ui.painter();
    p.rect_filled(sw, 8.5, bg);
    let r = switch.y * 0.5 - 2.5;
    let cx = egui::lerp(sw.left() + r + 2.5..=sw.right() - r - 2.5, t);
    p.circle_filled(
        egui::pos2(cx, sw.center().y),
        r,
        if *on { ON_ACCENT } else { MUTED },
    );
    if resp.hovered() {
        p.rect_stroke(
            sw,
            8.5,
            egui::Stroke::new(1.0_f32, Color32::from_white_alpha(28)),
            egui::StrokeKind::Outside,
        );
    }
    if !label.is_empty() {
        p.galley(
            egui::pos2(sw.right() + gap, rect.center().y - galley.size().y * 0.5),
            galley,
            TEXT,
        );
    }
    resp
}

/// Thin vertical divider for toolbars.
fn vdivider(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, 22.0), egui::Sense::hover());
    ui.painter().line_segment(
        [rect.center_top(), rect.center_bottom()],
        egui::Stroke::new(1.0_f32, BORDER),
    );
}

/// Horizontal divider for vertical toolbars.
fn hdivider(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(34.0, 9.0), egui::Sense::hover());
    ui.painter().line_segment(
        [rect.left_center() + egui::vec2(6.0, 0.0), rect.right_center() - egui::vec2(6.0, 0.0)],
        egui::Stroke::new(1.0_f32, BORDER),
    );
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
    if let Some(v) = &c.volume {
        // A 3D box target: name it by its label and what it covers, not by region count.
        let mut names: Vec<&str> = Vec::new();
        for t in &c.targets {
            if !names.contains(&t.object_name.as_str()) {
                names.push(&t.object_name);
            }
        }
        let what = if names.is_empty() { "empty space".to_string() } else { names.join(", ") };
        let size = v.half_extents * 2.0;
        let short = if v.label.is_empty() {
            format!("3D box · {what}")
        } else {
            format!("{} · {what}", v.label)
        };
        let full = format!(
            "3D box {:.2} × {:.2} × {:.2} at [{:.2}, {:.2}, {:.2}]\n{what}",
            size.x, size.y, size.z, v.center.x, v.center.y, v.center.z
        );
        return (short, full);
    }
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
        if let Some(v) = &scene.selection_volume {
            return Some(if v.label.is_empty() {
                "3D box · empty space".to_string()
            } else {
                format!("3D box · {} · empty space", v.label)
            });
        }
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
    swatch: [f32; 3],
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
        HOVER_ROW
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
        SUBTLE
    } else {
        Color32::from_rgb(72, 82, 98)
    };
    painter.text(
        c,
        egui::Align2::CENTER_CENTER,
        if visible { icon::EYE } else { icon::EYE_SLASH },
        FontId::proportional(15.0),
        eye_color,
    );
    // Material colour swatch.
    let sw = egui::Rect::from_center_size(egui::pos2(rect.left() + 41.0, rect.center().y), egui::vec2(10.0, 10.0));
    let to8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
    let mut col = Color32::from_rgb(to8(swatch[0]), to8(swatch[1]), to8(swatch[2]));
    if !visible {
        col = col.gamma_multiply(0.35);
    }
    painter.rect_filled(sw, 3, col);
    painter.rect_stroke(sw, 3, egui::Stroke::new(1.0_f32, Color32::from_black_alpha(90)), egui::StrokeKind::Inside);

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

    let name_left = rect.left() + 54.0;
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

/// Representative display colour: base tint times the average vertex colour.
fn swatch_color(obj: &crate::scene::SceneObject) -> [f32; 3] {
    let colors = &obj.mesh.colors;
    if colors.is_empty() {
        return obj.color;
    }
    let step = (colors.len() / 64).max(1);
    let mut sum = [0.0f32; 3];
    let mut n = 0.0;
    for c in colors.iter().step_by(step) {
        for k in 0..3 {
            sum[k] += c[k];
        }
        n += 1.0;
    }
    [
        obj.color[0] * sum[0] / n,
        obj.color[1] * sum[1] / n,
        obj.color[2] * sum[2] / n,
    ]
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
        scene.invalidate_mesh_caches();
        renderer.invalidate_cache();
    }

    // Image files dropped on the window become comment attachments.
    let dropped: Vec<std::path::PathBuf> = ctx.input(|i| {
        i.raw
            .dropped_files
            .iter()
            .filter_map(|f| f.path.clone())
            .collect()
    });
    for path in dropped {
        let is_image = path
            .extension()
            .map(|e| matches!(e.to_string_lossy().to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp"))
            .unwrap_or(false);
        if is_image {
            state.draft_attachments.push(path.to_string_lossy().into_owned());
            state.review_tab = 0;
            state.status_message = "Image attached to your next comment".into();
        } else {
            state.last_error = Some(format!("Only images can be attached: {}", path.display()));
        }
    }

    draw_dialogs(ctx, scene, state);
    draw_library_view(ctx, scene, camera, state, comparing);
    draw_command_palette(ctx, scene, camera, renderer, history, state, comparing, in_progress);

    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::K)) {
        state.palette_open = !state.palette_open;
        state.palette_query.clear();
        state.palette_index = 0;
    }
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::N)) && !comparing {
        state.confirm_new_project = true;
    }
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
                .stroke(egui::Stroke::new(1.0_f32, Color32::from_rgb(28, 35, 46)))
                .inner_margin(egui::Margin::symmetric(14, 8)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Brand mark: a cube outline with a lit core.
                let (mark, _) = ui.allocate_exact_size(egui::vec2(26.0, 30.0), egui::Sense::hover());
                let c = mark.center();
                let p = ui.painter();
                p.rect_filled(egui::Rect::from_center_size(c, egui::vec2(22.0, 22.0)), 6, ACCENT.gamma_multiply(0.14));
                p.text(c, egui::Align2::CENTER_CENTER, icon::CUBE_FOCUS, FontId::proportional(17.0), ACCENT);
                ui.label(semibold("Spatial Studio", 15.0));
                vdivider(ui);

                // Project menu: title, save state and file actions.
                let (dot, status) = if scene.project_path.is_some() {
                    (ACCENT, "Saved · autosave on")
                } else {
                    (WARN, "Not saved yet")
                };
                let title = format!("{}  {}", scene.project_name, icon::CARET_DOWN);
                let menu = ui.menu_button(semibold(title, 14.0), |ui| {
                    ui.set_min_width(250.0);
                    let item = |ui: &mut egui::Ui, glyph: &str, text: &str, keys: &str, enabled: bool| {
                        ui.add_enabled(enabled, egui::Button::new(icon_text(glyph, text)).shortcut_text(keys))
                            .clicked()
                    };
                    if item(ui, icon::FILE_PLUS, "New project…", "Ctrl+N", !comparing) {
                        state.confirm_new_project = true;
                        ui.close_menu();
                    }
                    if item(ui, icon::FOLDER_OPEN, "Open project…", "", true) {
                        state.project_dialog = Some("open".into());
                        ui.close_menu();
                    }
                    if item(ui, icon::FLOPPY_DISK, "Save project as…", "", true) {
                        state.project_dialog = Some("save".into());
                        ui.close_menu();
                    }
                    if item(ui, icon::CLOCK_COUNTER_CLOCKWISE, "Recover autosave…", "", true) {
                        state.project_dialog = Some("recover".into());
                        ui.close_menu();
                    }
                    if item(ui, icon::EXPORT, "Export model…", "", true) {
                        state.show_io_modal = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    eyebrow(ui, "Project title");
                    if text_field(ui, &mut scene.project_name, "Untitled project", f32::INFINITY).changed() {
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
                menu.response.on_hover_text(
                    scene.project_path.clone().unwrap_or_else(|| "Not saved yet".into()),
                );
                ui.label(RichText::new("●").size(8.0).color(dot));
                ui.label(small_muted(status));

                // Live agent state, where the eye already is.
                let (text, color, active) = agent_presence(scene, state);
                if color != SUBTLE {
                    ui.add_space(6.0);
                    let resp = egui::Frame::new()
                        .fill(color.gamma_multiply(0.12))
                        .stroke(egui::Stroke::new(1.0_f32, color.gamma_multiply(0.45)))
                        .corner_radius(12)
                        .inner_margin(egui::Margin::symmetric(10, 3))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                if active {
                                    // A soft pulse while the agent is busy.
                                    let t = ui.input(|i| i.time);
                                    let pulse = 0.55 + 0.45 * ((t * 3.2).sin() as f32 * 0.5 + 0.5);
                                    ui.label(RichText::new(icon::ROBOT).size(14.0).color(color.gamma_multiply(pulse)));
                                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
                                } else {
                                    ui.label(RichText::new(icon::ROBOT).size(14.0).color(color));
                                }
                                ui.label(RichText::new(&text).size(12.5).color(color));
                            });
                        })
                        .response;
                    if resp.interact(egui::Sense::click()).on_hover_text("Open the review panel").clicked() {
                        state.review_tab = if text.starts_with("Revision") { 1 } else { 0 };
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            !comparing,
                            egui::Button::new(
                                RichText::new(icon_text(icon::FLOPPY_DISK, "Save"))
                                    .font(medium_font(14.0))
                                    .color(ON_ACCENT),
                            )
                            .fill(ACCENT),
                        )
                        .on_hover_text("Save project  ·  Ctrl+S")
                        .clicked()
                    {
                        if scene.project_path.is_some() {
                            action(state, "save_project", json!({}));
                        } else {
                            state.project_dialog = Some("save".into());
                        }
                    }
                    ui.add_space(4.0);
                    let blocked = !comparing && !in_progress;
                    if icon_button(
                        ui,
                        icon::ARROW_CLOCKWISE,
                        if history.can_redo() { "Redo  ·  Ctrl+Shift+Z" } else { "Nothing to redo" },
                        false,
                        history.can_redo() && blocked,
                    )
                    .clicked()
                    {
                        action(state, "redo", json!({}));
                    }
                    if icon_button(
                        ui,
                        icon::ARROW_COUNTER_CLOCKWISE,
                        if history.can_undo() { "Undo  ·  Ctrl+Z" } else { "Nothing to undo" },
                        false,
                        history.can_undo() && blocked,
                    )
                    .clicked()
                    {
                        action(state, "undo", json!({}));
                    }
                    ui.add_space(8.0);
                    // Command palette trigger, centred in the remaining space.
                    let w = ui.available_width().min(340.0);
                    let pad = ((ui.available_width() - w) * 0.5).max(0.0);
                    ui.add_space(pad);
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 32.0), egui::Sense::click());
                    let hovered = resp.hovered();
                    let p = ui.painter();
                    p.rect(
                        rect,
                        8,
                        FIELD,
                        egui::Stroke::new(1.0_f32, if hovered { Color32::from_rgb(64, 78, 98) } else { BORDER }),
                        egui::StrokeKind::Inside,
                    );
                    p.text(
                        rect.left_center() + egui::vec2(12.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        icon_text(icon::MAGNIFYING_GLASS, "Search commands…"),
                        FontId::proportional(13.5),
                        SUBTLE,
                    );
                    if rect.width() > 230.0 {
                        p.text(
                            rect.right_center() - egui::vec2(12.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            "Ctrl K",
                            medium_font(12.0),
                            SUBTLE,
                        );
                    }
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        state.palette_open = true;
                        state.palette_query.clear();
                        state.palette_index = 0;
                    }
                });
            });
        });
}

/// What the agent is doing right now, for the header pill and the status bar:
/// (label, colour, whether it is actively doing something).
fn agent_presence(scene: &Scene, state: &UiState) -> (String, Color32, bool) {
    let recent = state
        .last_agent_activity
        .is_some_and(|t| t.elapsed().as_secs() < 30);
    let listening = state
        .agent_listening
        .is_some_and(|t| t.elapsed().as_secs_f32() < 3.0);
    let in_progress = scene
        .revisions
        .iter()
        .rev()
        .find(|r| r.status == RevisionStatus::InProgress);
    let ready = scene
        .revisions
        .iter()
        .any(|r| r.status == RevisionStatus::ReadyForReview);
    if let Some((name, _)) = &state.busy_operation {
        return (format!("Agent · {name}"), ACCENT, true);
    }
    if let Some(r) = in_progress {
        let label: String = r.label.chars().take(34).collect();
        return (format!("Agent working · {label}"), ACCENT, true);
    }
    match state.last_agent_activity {
        _ if ready => ("Revision ready for your review".into(), WARN, false),
        Some(t) if recent => (format!("Agent working · {}", ago(t.elapsed())), ACCENT, true),
        _ if listening => ("Agent listening for feedback".into(), INFO, false),
        Some(t) => (format!("Agent idle · last seen {}", ago(t.elapsed())), SUBTLE, false),
        None => ("No agent connected".into(), SUBTLE, false),
    }
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
                let drafts = scene.draft_count();
                let (agent_text, color, _) = agent_presence(scene, state);
                ui.label(RichText::new(icon::ROBOT).size(13.0).color(color));
                ui.label(small_muted(agent_text)).on_hover_ui(|ui| {
                    ui.set_max_width(420.0);
                    eyebrow(ui, "Recent agent activity");
                    if state.activity.is_empty() {
                        ui.label(small_muted("Agent operations appear here."));
                    }
                    for line in state.activity.iter().rev().take(10) {
                        ui.label(small_muted(line));
                    }
                    ui.separator();
                    ui.label(small_muted("Agents call wait_for_feedback to pick up comments as soon as you send them."));
                });
                if drafts > 0 {
                    ui.separator();
                    ui.label(RichText::new(format!("{} {} not sent", icon::PAPER_PLANE_TILT, plural(drafts, "comment", "comments"))).small().color(WARN));
                }
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
            // Four tabs do not fit the panel with labels, so only the open tab shows its name.
            let tab = state.scene_tab;
            let label = |i: usize, glyph: &str, text: &str| {
                if tab == i { icon_text(glyph, text) } else { glyph.to_string() }
            };
            segmented(
                ui,
                &mut state.scene_tab,
                &[
                    (0, label(0, icon::TREE_STRUCTURE, "Parts"), "Parts · browse, show and hide parts"),
                    (
                        1,
                        label(1, icon::SLIDERS, "Inspect"),
                        "Inspect · section plane, ghosting and measurement",
                    ),
                    (2, label(2, icon::BOOKMARK_SIMPLE, "Saved"), "Saved · selections and camera views"),
                    (3, label(3, icon::BOOKS, "Library"), "Library · reusable assets shared by every project"),
                ],
            );
            ui.add_space(4.0);

            match state.scene_tab {
                0 => scene_parts_tab(ui, scene, camera, state, comparing),
                1 => scene_inspect_tab(ui, scene, renderer, state),
                3 => scene_library_tab(ui, scene, camera, state, comparing),
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
    let has_selection = scene.has_any_selection();
    let any_hidden = scene.objects.iter().any(|o| !o.visible);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let w = (ui.available_width() - 3.0 * 36.0).max(60.0);
        ui.allocate_ui(egui::vec2(w, 32.0), |ui| {
            search_field(ui, &mut state.outliner_search, "Search parts…");
        });
        if icon_button(ui, icon::EYE, if any_hidden { "Show every part" } else { "Every part is visible" }, false, any_hidden).clicked() {
            for o in &mut scene.objects {
                o.visible = true;
            }
            state.scene_dirty_for_disk = true;
        }
        if icon_button(ui, icon::CUBE_FOCUS, if has_selection { "Isolate · hide everything except the selection" } else { "Select a part to isolate it" }, false, has_selection).clicked() {
            let selected = scene.selected_object;
            for (i, o) in scene.objects.iter_mut().enumerate() {
                o.visible = selected == Some(i) || !o.mesh.selected_faces.is_empty();
            }
            state.scene_dirty_for_disk = true;
        }
        if icon_button(ui, icon::FRAME_CORNERS, if has_selection { "Frame selection  ·  F" } else { "Frame everything  ·  F" }, false, true).clicked() {
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
            // Parts named "Group/Part" are listed under a collapsible "Group" heading.
            let mut order: Vec<(Option<String>, Vec<usize>)> = Vec::new();
            for (i, obj) in scene.objects.iter().enumerate() {
                if !obj.name.to_lowercase().contains(&search) {
                    continue;
                }
                matches += 1;
                let group = obj
                    .name
                    .split_once('/')
                    .map(|(g, _)| g.trim().to_string())
                    .filter(|g| !g.is_empty());
                match (&group, order.iter_mut().find(|(g, _)| g.is_some() && *g == group)) {
                    (Some(_), Some((_, list))) => list.push(i),
                    _ => order.push((group, vec![i])),
                }
            }
            for (group, indices) in order {
                let grouped = group.is_some() && indices.len() > 1;
                let mut collapsed = false;
                if let (true, Some(name)) = (grouped, &group) {
                    collapsed = state.collapsed_groups.contains(name) && search.is_empty();
                    let any_visible = indices.iter().any(|&i| scene.objects[i].visible);
                    let faces: usize = indices.iter().map(|&i| scene.objects[i].mesh.faces.len()).sum();
                    let (rect, row) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 28.0),
                        egui::Sense::click(),
                    );
                    let eye_rect = egui::Rect::from_center_size(
                        egui::pos2(rect.left() + 17.0, rect.center().y),
                        egui::vec2(28.0, 24.0),
                    );
                    let eye = ui
                        .interact(eye_rect, row.id.with("group_eye"), egui::Sense::click())
                        .on_hover_text(if any_visible { "Hide this group" } else { "Show this group" });
                    let p = ui.painter();
                    if row.hovered() {
                        p.rect_filled(rect, 6, HOVER_ROW);
                    }
                    p.text(
                        eye_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        if any_visible { icon::EYE } else { icon::EYE_SLASH },
                        FontId::proportional(15.0),
                        if eye.hovered() { TEXT } else { SUBTLE },
                    );
                    p.text(
                        egui::pos2(rect.left() + 40.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        if collapsed { icon::CARET_RIGHT } else { icon::CARET_DOWN },
                        FontId::proportional(12.0),
                        MUTED,
                    );
                    p.text(
                        egui::pos2(rect.left() + 58.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        name,
                        medium_font(13.5),
                        TEXT,
                    );
                    p.text(
                        egui::pos2(rect.right() - 8.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        format!("{} · {}", indices.len(), compact_count(faces)),
                        FontId::proportional(11.5),
                        SUBTLE,
                    );
                    if eye.clicked() {
                        for &i in &indices {
                            scene.objects[i].visible = !any_visible;
                        }
                        state.scene_dirty_for_disk = true;
                    } else if row.clicked() {
                        if !state.collapsed_groups.remove(name) {
                            state.collapsed_groups.insert(name.clone());
                        }
                    }
                }
                if collapsed {
                    continue;
                }
                for i in indices {
                    let obj = &mut scene.objects[i];
                    let selected = selected_idx == Some(i) || !obj.mesh.selected_faces.is_empty();
                    let open = open_counts.get(&obj.id).copied().unwrap_or(0);
                    let shown = if grouped {
                        obj.name.split_once('/').map(|(_, n)| n.trim()).unwrap_or(&obj.name).to_string()
                    } else {
                        obj.name.clone()
                    };
                    let row = ui
                        .push_id(obj.id, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 0.0;
                                if grouped {
                                    ui.add_space(14.0);
                                }
                                part_row(
                                    ui,
                                    &shown,
                                    obj.mesh.faces.len(),
                                    open,
                                    obj.visible,
                                    selected,
                                    swatch_color(obj),
                                )
                            })
                            .inner
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

/// Text field with a menu of existing names to pick from.
fn name_with_choices(ui: &mut egui::Ui, value: &mut String, placeholder: &str, choices: &[String]) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let width = (ui.available_width() - 56.0).max(60.0);
        text_field(ui, value, placeholder, width);
        ui.add_enabled_ui(!choices.is_empty(), |ui| {
            ui.menu_button(icon::CARET_DOWN, |ui| {
                for choice in choices {
                    if ui.button(choice).clicked() {
                        *value = choice.clone();
                        ui.close_menu();
                    }
                }
            })
            .response
            .on_hover_text("Pick an existing one")
            .on_disabled_hover_text("Nothing to pick yet; type a new name");
        });
    });
}

/// Small glyph buttons painted at the right edge of `rect`. Returns the pressed one and the
/// x where the buttons start.
fn row_buttons(
    ui: &egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    buttons: &[(&str, &str, Color32)],
) -> (Option<usize>, f32) {
    let mut pressed = None;
    let mut left = rect.right() - 2.0;
    for (i, (glyph, tip, color)) in buttons.iter().enumerate().rev() {
        let button = egui::Rect::from_min_max(
            egui::pos2(left - 26.0, rect.center().y - 13.0),
            egui::pos2(left, rect.center().y + 13.0),
        );
        left -= 26.0;
        let r = ui
            .interact(button, id.with(("button", i)), egui::Sense::click())
            .on_hover_text(*tip);
        if r.hovered() {
            ui.painter().rect_filled(button, 6, Color32::from_rgb(40, 50, 66));
        }
        ui.painter().text(
            button.center(),
            egui::Align2::CENTER_CENTER,
            *glyph,
            FontId::proportional(15.0),
            if r.hovered() && *color == SUBTLE { TEXT } else { *color },
        );
        if r.clicked() {
            pressed = Some(i);
        }
    }
    (pressed, left)
}

/// Section / subsection row. `collapsed` draws a fold caret (the tab's tree); `selected`
/// highlights it (the library view's navigation). Returns (row clicked, pressed button).
fn folder_row(
    ui: &mut egui::Ui,
    indent: f32,
    collapsed: Option<bool>,
    selected: bool,
    glyph: &str,
    name: &str,
    count: usize,
    buttons: &[(&str, &str, Color32)],
) -> (bool, Option<usize>) {
    let (rect, row) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::click());
    if selected {
        ui.painter().rect_filled(rect, 6, SELECTED_ROW);
    } else if row.hovered() {
        ui.painter().rect_filled(rect, 6, HOVER_ROW);
    }
    let (pressed, right) = row_buttons(ui, rect, row.id, buttons);
    let p = ui.painter().with_clip_rect(rect);
    let mut x = rect.left() + 6.0 + indent;
    if let Some(collapsed) = collapsed {
        p.text(
            egui::pos2(x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            if collapsed { icon::CARET_RIGHT } else { icon::CARET_DOWN },
            FontId::proportional(12.0),
            MUTED,
        );
        x += 18.0;
    }
    p.text(
        egui::pos2(x, rect.center().y),
        egui::Align2::LEFT_CENTER,
        glyph,
        FontId::proportional(15.0),
        if selected { ACCENT } else if indent > 0.0 { SUBTLE } else { MUTED },
    );
    let count_text = p.text(
        egui::pos2(right - 6.0, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        count.to_string(),
        FontId::proportional(11.5),
        SUBTLE,
    );
    p.with_clip_rect(egui::Rect::from_min_max(rect.min, egui::pos2(count_text.left() - 6.0, rect.bottom())))
        .text(
            egui::pos2(x + 22.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            name,
            if indent > 0.0 { FontId::proportional(13.0) } else { medium_font(13.5) },
            TEXT,
        );
    (row.clicked() && pressed.is_none(), pressed)
}

/// Inline "name a new folder" row. `Some(true)` = create, `Some(false)` = cancel.
fn new_folder_row(ui: &mut egui::Ui, draft: &mut String, placeholder: &str) -> Option<bool> {
    let mut result = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let width = (ui.available_width() - 110.0).max(60.0);
        let field = text_field(ui, draft, placeholder, width);
        let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let ready = !draft.trim().is_empty();
        if (tool_button(ui, "Create", ready, "Create it", "Type a name first") || (enter && ready))
            && ready
        {
            result = Some(true);
        }
        if quiet_button(ui, "×", "Cancel") {
            result = Some(false);
        }
    });
    result
}

/// The "new section" / "new subsection" field when it belongs under `parent`. Returns true
/// when the library changed.
fn library_new_folder(
    ui: &mut egui::Ui,
    state: &mut UiState,
    root: &std::path::Path,
    parent: Option<&str>,
) -> bool {
    let Some((draft_parent, draft)) = &mut state.library.new_folder else {
        return false;
    };
    if draft_parent.as_deref() != parent {
        return false;
    }
    let placeholder = if parent.is_some() { "Subsection name" } else { "Section name" };
    match new_folder_row(ui, draft, placeholder) {
        Some(true) => {
            let created = match parent {
                Some(section) => crate::library::create_section(root, section, Some(draft)),
                None => crate::library::create_section(root, draft, None),
            };
            match created {
                Ok(_) => state.library.new_folder = None,
                Err(e) => state.last_error = Some(e),
            }
            true
        }
        Some(false) => {
            state.library.new_folder = None;
            false
        }
        None => false,
    }
}

fn library_thumbnail(
    ctx: &egui::Context,
    state: &mut UiState,
    asset: &crate::library::AssetInfo,
) -> Option<egui::TextureHandle> {
    let path = asset.thumbnail.as_ref()?;
    // Saving an asset again changes `modified`, so a replaced thumbnail is reloaded.
    let key = format!("{path}#{}", asset.modified);
    state
        .screenshot_cache
        .entry(key.clone())
        .or_insert_with(|| {
            let image = image::open(path).map_err(|e| e.to_string())?.into_rgba8();
            let pixels = egui::ColorImage::from_rgba_unmultiplied(
                [image.width() as usize, image.height() as usize],
                image.as_raw(),
            );
            Ok(ctx.load_texture(key, pixels, egui::TextureOptions::LINEAR))
        })
        .clone()
        .ok()
}

/// Rereads the library folder when it has not been read for a moment: agents and other
/// windows can change it.
fn library_refresh(state: &mut UiState, root: &std::path::Path) {
    if state.library.scanned.is_none_or(|t| t.elapsed().as_secs_f32() > 2.0) {
        state.library.sections = std::sync::Arc::new(crate::library::list(root));
        state.library.scanned = Some(std::time::Instant::now());
    }
}

fn library_matches(asset: &crate::library::AssetInfo, search: &str) -> bool {
    search.is_empty()
        || asset.name.to_lowercase().contains(search)
        || asset.description.to_lowercase().contains(search)
        || asset.tags.iter().any(|t| t.to_lowercase().contains(search))
}

/// Where a new asset lands: the picked surface point, otherwise the centre of the view.
fn library_insert_point(scene: &Scene, camera: &OrbitCamera) -> Vec3 {
    match scene.pending_anchor {
        Some((_, world, _, _)) if scene.objects.iter().any(|o| !o.mesh.selected_faces.is_empty()) => world,
        _ => camera.target,
    }
}

/// The group every listed part shares ("Bookshelf/…"), if there is one.
fn shared_group<'a>(scene: &'a Scene, parts: &[usize]) -> Option<&'a str> {
    let group = |i: usize| {
        scene.objects[i].name.split_once('/').map(|(g, _)| g.trim()).filter(|g| !g.is_empty())
    };
    let first = group(*parts.first()?)?;
    parts.iter().all(|&i| group(i) == Some(first)).then_some(first)
}

/// Opens the save form with a suggested name. The whole model is the default: one asset
/// for the bookshelf, not for a shelf board of it.
fn library_open_save(scene: &Scene, state: &mut UiState) {
    let whole = crate::library::visible_parts(scene);
    let project = scene.project_name.trim();
    state.library.save_selection = false;
    state.library.asset_name = if !project.is_empty() && !project.to_lowercase().starts_with("untitled") {
        project.to_string()
    } else if let Some(group) = shared_group(scene, &whole) {
        group.to_string()
    } else if whole.len() == 1 {
        scene.objects[whole[0]].name.clone()
    } else {
        String::new()
    };
    if state.library.section.is_empty() {
        if let Some(first) = state.library.sections.first() {
            state.library.section = first.name.clone();
            state.library.subsection =
                first.subsections.first().map(|s| s.name.clone()).unwrap_or_default();
        }
    }
    state.library.save_open = true;
}

/// The form that saves the whole model, or the selection, as one asset.
fn library_save_form(
    ui: &mut egui::Ui,
    scene: &Scene,
    state: &mut UiState,
    sections: &[crate::library::Section],
    root: &std::path::Path,
) {
    use crate::library;
    let whole = library::visible_parts(scene);
    let selection = library::selected_parts(scene);
    card().show(ui, |ui| {
        eyebrow(ui, "Save as one asset");
        segmented(
            ui,
            &mut state.library.save_selection,
            &[
                (
                    false,
                    format!("Whole model · {}", plural(whole.len(), "part", "parts")),
                    "Everything visible becomes a single asset",
                ),
                (
                    true,
                    format!("Selection · {}", selection.len()),
                    "Only the selected parts become the asset",
                ),
            ],
        );
        let parts = if state.library.save_selection { &selection } else { &whole };
        let width = ui.available_width();
        text_field(ui, &mut state.library.asset_name, "Asset name, e.g. Oak bookshelf", width - 18.0);
        let section_names: Vec<String> = sections.iter().map(|s| s.name.clone()).collect();
        name_with_choices(ui, &mut state.library.section, "Section, e.g. Furniture", &section_names);
        let typed_section = state.library.section.trim().to_lowercase();
        let section = sections.iter().find(|s| s.name.to_lowercase() == typed_section);
        let sub_names: Vec<String> = section
            .map(|s| s.subsections.iter().map(|sub| sub.name.clone()).collect())
            .unwrap_or_default();
        name_with_choices(ui, &mut state.library.subsection, "Subsection, e.g. Bookshelves", &sub_names);

        let name = state.library.asset_name.trim().to_string();
        let typed_sub = state.library.subsection.trim().to_lowercase();
        let exists = section
            .and_then(|s| s.subsections.iter().find(|sub| sub.name.to_lowercase() == typed_sub))
            .is_some_and(|sub| sub.assets.iter().any(|a| a.name.to_lowercase() == name.to_lowercase()));
        let ready = !parts.is_empty()
            && !name.is_empty()
            && !typed_section.is_empty()
            && !typed_sub.is_empty();
        if exists {
            ui.label(RichText::new("An asset with this name is already there.").size(12.0).color(WARN));
        }
        ui.horizontal(|ui| {
            let disabled_tip = if !parts.is_empty() {
                "Fill in the name, section and subsection"
            } else if state.library.save_selection {
                "Select the parts to save"
            } else {
                "There is nothing visible to save"
            };
            let save = if exists {
                tool_button(ui, "Replace it", ready, "Overwrite the asset in the library", disabled_tip)
            } else {
                primary(ui, "Save to library", ready)
            };
            if quiet_button(ui, "Cancel", "") {
                state.library.save_open = false;
            }
            if save {
                let saved = library::save_asset(
                    root,
                    scene,
                    parts,
                    library::SaveOptions {
                        section: &state.library.section,
                        subsection: &state.library.subsection,
                        name: &state.library.asset_name,
                        description: "",
                        tags: Vec::new(),
                        overwrite: exists,
                    },
                );
                match saved {
                    Ok(info) => {
                        // Without a GPU adapter the asset simply has no picture.
                        let _ = library::render_thumbnail(&info);
                        state.status_message = format!("Saved '{}' to the library", info.reference());
                        state.library.filter = Some((info.section.clone(), Some(info.subsection.clone())));
                        state.library.section = info.section;
                        state.library.subsection = info.subsection;
                        state.library.save_open = false;
                        state.library.scanned = None;
                    }
                    Err(e) => state.last_error = Some(e),
                }
            }
        });
    });
}

/// Reads the remembered export choices once, then opens the export form.
fn library_open_export(state: &mut UiState, root: &std::path::Path, asset: Option<String>) {
    load_export_choices(&mut state.library.export, root);
    state.library.save_open = false;
    state.library.export_scope = Some(asset);
}

fn load_export_choices(export: &mut ExportUi, root: &std::path::Path) {
    if std::mem::replace(&mut export.loaded, true) {
        return;
    }
    if let Some(prefs) = crate::library::load_export_prefs(root) {
        export.directory = prefs.directory;
        export.blender = prefs.target == "blender";
        export.separate = !prefs.merge;
        export.collision = crate::gltf::Collision::from_name(&prefs.collision).unwrap_or_default();
        export.budget = prefs.max_triangles.map(|n| n.to_string()).unwrap_or_default();
    }
}

/// Merge, collision and triangle-budget controls for a glTF export.
fn export_options(ui: &mut egui::Ui, export: &mut ExportUi, godot: bool, id: &str) {
    let mut merge = !export.separate;
    toggle(ui, &mut merge, "Merge parts that share a material")
        .on_hover_text("One mesh with a surface per material: far fewer draw calls in a game. Turn it off to keep every part as its own object.");
    export.separate = !merge;
    if godot {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Collision").color(MUTED));
            egui::ComboBox::from_id_salt(id)
                .selected_text(export.collision.label())
                .show_ui(ui, |ui| {
                    for choice in crate::gltf::Collision::ALL {
                        ui.selectable_value(&mut export.collision, choice, choice.label());
                    }
                })
                .response
                .on_hover_text("Godot turns these into a StaticBody3D when it imports the file. Exact mesh suits static scenery only.");
        });
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new("Triangle budget").color(MUTED));
        text_field(ui, &mut export.budget, "all triangles", 110.0)
            .on_hover_text("Simplify dense parts so the model has about this many triangles, e.g. 20000. Leave empty to keep them all. The original is not changed.");
    });
}

/// Where a folder sits in a Godot project, as a line under the destination field.
fn godot_folder_note(ui: &mut egui::Ui, directory: &str) {
    if directory.trim().is_empty() {
        return;
    }
    match crate::gltf::godot_resource_path(&std::path::Path::new(directory.trim()).join("x")) {
        Some((project, res)) => {
            let name = project.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            ui.label(small_muted(format!("Godot project “{name}” · {}", res.trim_end_matches('x'))));
        }
        None => {
            ui.label(
                RichText::new("This folder is not inside a Godot project (no project.godot above it). The files are still written.")
                    .size(12.0)
                    .color(WARN),
            );
        }
    }
}

/// Exports one asset, or every asset the view is showing, for Godot or Blender.
fn library_export_form(
    ui: &mut egui::Ui,
    state: &mut UiState,
    sections: &[crate::library::Section],
    root: &std::path::Path,
    search: &str,
) {
    use crate::library;
    let Some(scope) = state.library.export_scope.clone() else {
        return;
    };
    let assets: Vec<&library::AssetInfo> = sections
        .iter()
        .flat_map(|section| section.subsections.iter().flat_map(|sub| sub.assets.iter()))
        .filter(|asset| match &scope {
            Some(reference) => asset.reference() == *reference,
            None => {
                library_matches(asset, search)
                    && state.library.filter.as_ref().is_none_or(|(section, sub)| {
                        asset.section == *section && sub.as_ref().is_none_or(|sub| asset.subsection == *sub)
                    })
            }
        })
        .collect();
    card().show(ui, |ui| {
        let title = match (&scope, assets.as_slice()) {
            (Some(_), [asset]) => format!("Export “{}”", asset.name),
            _ => format!("Export {}", plural(assets.len(), "asset", "assets")),
        };
        eyebrow(ui, &title);
        let export = &mut state.library.export;
        segmented(
            ui,
            &mut export.blender,
            &[
                (false, "Godot".into(), "snake_case .glb files ready for a Godot project"),
                (true, "Blender".into(), ".glb files with the original names; File > Import > glTF 2.0"),
            ],
        );
        let godot = !export.blender;
        let width = ui.available_width();
        text_field(
            ui,
            &mut export.directory,
            if godot { "Folder in your Godot project, e.g. ~/game/assets/library" } else { "Destination folder" },
            width - 18.0,
        );
        let directory = expand_home(&export.directory);
        if godot {
            godot_folder_note(ui, &directory);
        }
        export_options(ui, export, godot, "library_export_collision");
        ui.label(small_muted(if godot {
            "One .glb per asset in section/subsection folders. Godot imports them when its window gets focus, and re-imports after every export."
        } else {
            "One .glb per asset in Section/Subsection folders."
        }));
        let ready = !assets.is_empty() && !directory.is_empty();
        let mut run = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            run = primary(ui, "Export", ready);
            cancel = quiet_button(ui, "Cancel", "");
        });
        if cancel {
            state.library.export_scope = None;
        }
        if run {
            let export = state.library.export.clone();
            let options = library::ExportOptions {
                target: if export.blender { library::ExportTarget::Blender } else { library::ExportTarget::Godot },
                collision: export.collision,
                merge: Some(!export.separate),
                scale: 1.0,
                max_triangles: export.max_triangles(),
            };
            let path = std::path::Path::new(&directory);
            if let Err(e) = library::check_export_paths(assets.iter().copied(), path, options.target) {
                state.last_error = Some(e);
                return;
            }
            let (mut done, mut triangles, mut failed, mut note) = (0, 0, Vec::new(), None);
            for asset in &assets {
                match library::export_asset(asset, path, &options) {
                    Ok((_, report)) => {
                        done += 1;
                        triangles += report.triangles;
                        note = note.or(report.warnings.into_iter().next());
                    }
                    Err(e) => failed.push(format!("{}: {e}", asset.name)),
                }
            }
            library::save_export_prefs(
                root,
                &library::ExportPrefs {
                    directory: export.directory.trim().to_string(),
                    target: options.target.name().into(),
                    collision: options.collision.name().into(),
                    merge: !export.separate,
                    max_triangles: options.max_triangles,
                },
            );
            state.status_message = format!(
                "Exported {} ({} triangles) to {directory}{}",
                plural(done, "asset", "assets"),
                compact_count(triangles),
                note.map(|n| format!(" · {n}")).unwrap_or_default()
            );
            if failed.is_empty() {
                state.library.export_scope = None;
            } else {
                state.last_error = Some(format!("Could not export {}", failed.join("; ")));
            }
        }
    });
}

/// `~/game` -> `/home/me/game`.
fn expand_home(path: &str) -> String {
    let path = path.trim();
    match (path.strip_prefix('~'), std::env::var("HOME")) {
        (Some(rest), Ok(home)) if rest.is_empty() || rest.starts_with('/') => format!("{home}{rest}"),
        _ => path.to_string(),
    }
}

fn library_asset_tip(ui: &mut egui::Ui, asset: &crate::library::AssetInfo) {
    ui.label(semibold(&asset.name, 13.5));
    if !asset.description.is_empty() {
        ui.label(&asset.description);
    }
    ui.label(small_muted(format!(
        "{} · {:.2} × {:.2} × {:.2}",
        plural(asset.parts, "part", "parts"),
        asset.size[0],
        asset.size[1],
        asset.size[2]
    )));
    if !asset.tags.is_empty() {
        ui.label(small_muted(asset.tags.join(", ")));
    }
    ui.label(small_muted("Double-click to add it to the scene"));
}

/// The delete button of an asset: (glyph, tooltip, colour) and whether it is armed.
fn library_delete_button(state: &UiState, reference: &str) -> ((&'static str, &'static str, Color32), bool) {
    let armed = state
        .library
        .confirm_delete
        .as_ref()
        .is_some_and(|(r, at)| r == reference && at.elapsed().as_secs_f32() < 4.0);
    if armed {
        ((icon::TRASH, "Click again to move this asset to the library trash", DANGER), true)
    } else {
        ((icon::TRASH, "Delete from the library", SUBTLE), false)
    }
}

/// Applies an asset's add / delete press. Returns true when the library changed.
#[allow(clippy::too_many_arguments)]
fn library_asset_actions(
    ui: &egui::Ui,
    state: &mut UiState,
    root: &std::path::Path,
    asset: &crate::library::AssetInfo,
    insert_at: Vec3,
    comparing: bool,
    add: bool,
    delete: bool,
    armed: bool,
) -> bool {
    if add && !comparing {
        action(
            state,
            "library_insert_asset",
            serde_json::json!({ "asset": asset.reference(), "position": insert_at.to_array() }),
        );
        // Show the model where the asset just landed.
        state.library.open = false;
    }
    if delete {
        if armed {
            state.library.confirm_delete = None;
            match crate::library::delete_asset(root, asset) {
                Ok(()) => state.status_message = format!("Moved '{}' to the library trash", asset.name),
                Err(e) => state.last_error = Some(e),
            }
            return true;
        }
        state.library.confirm_delete = Some((asset.reference(), std::time::Instant::now()));
        ui.ctx().request_repaint_after(std::time::Duration::from_secs(4));
    }
    false
}

fn paint_thumbnail(ui: &egui::Ui, state: &mut UiState, asset: &crate::library::AssetInfo, rect: egui::Rect, glyph_size: f32) {
    ui.painter().rect_filled(rect, 6, FIELD);
    match library_thumbnail(ui.ctx(), state, asset) {
        Some(texture) => egui::Image::new(&texture).corner_radius(6).paint_at(ui, rect),
        None => {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                icon::CUBE,
                FontId::proportional(glyph_size),
                SUBTLE,
            );
        }
    }
}

fn scene_library_tab(
    ui: &mut egui::Ui,
    scene: &Scene,
    camera: &OrbitCamera,
    state: &mut UiState,
    comparing: bool,
) {
    use crate::library;
    let root = library::root();
    library_refresh(state, &root);
    let sections = state.library.sections.clone();

    let full = egui::vec2(ui.available_width(), 0.0);
    if ui
        .add(egui::Button::new(icon_text(icon::SQUARES_FOUR, "Open library")).min_size(full))
        .on_hover_text("Browse every asset with large previews")
        .clicked()
    {
        state.library.open = true;
    }
    if state.library.save_open && !state.library.open {
        library_save_form(ui, scene, state, &sections, &root);
    } else if ui
        .add_enabled(
            !scene.objects.is_empty(),
            egui::Button::new(icon_text(icon::PLUS, "Save model to library")).min_size(full),
        )
        .on_hover_text("Keep this model, or the selected parts, as one reusable asset")
        .on_disabled_hover_text("The scene is empty")
        .clicked()
    {
        library_open_save(scene, state);
    }
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let w = (ui.available_width() - 36.0).max(60.0);
        ui.allocate_ui(egui::vec2(w, 32.0), |ui| {
            search_field(ui, &mut state.library.search, "Search assets…");
        });
        if icon_button(ui, icon::FOLDER_PLUS, "New section", false, true).clicked() {
            state.library.new_folder = Some((None, String::new()));
        }
    });
    let mut rescan = library_new_folder(ui, state, &root, None);

    let search = state.library.search.trim().to_lowercase();
    let insert_at = library_insert_point(scene, camera);
    let mut shown = 0;
    egui::ScrollArea::vertical()
        .id_salt("library_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            for section in sections.iter() {
                let total: usize = section
                    .subsections
                    .iter()
                    .map(|s| s.assets.iter().filter(|a| library_matches(a, &search)).count())
                    .sum();
                if !search.is_empty() && total == 0 {
                    continue;
                }
                let empty = section.subsections.iter().all(|s| s.assets.is_empty());
                let folded = search.is_empty() && state.library.collapsed.contains(&section.name);
                let mut buttons = vec![(icon::FOLDER_PLUS, "New subsection", SUBTLE)];
                if empty {
                    buttons.push((icon::TRASH, "Delete this empty section", SUBTLE));
                }
                let glyph = if folded { icon::FOLDER } else { icon::FOLDER_OPEN };
                let (clicked, pressed) =
                    folder_row(ui, 0.0, Some(folded), false, glyph, &section.name, total, &buttons);
                if clicked && !state.library.collapsed.remove(&section.name) {
                    state.library.collapsed.insert(section.name.clone());
                }
                match pressed {
                    Some(0) => {
                        state.library.collapsed.remove(&section.name);
                        state.library.new_folder = Some((Some(section.name.clone()), String::new()));
                    }
                    Some(_) => {
                        if let Err(e) = library::delete_section(&root, &section.name, None) {
                            state.last_error = Some(e);
                        }
                        rescan = true;
                    }
                    None => {}
                }
                if folded {
                    continue;
                }
                rescan |= library_new_folder(ui, state, &root, Some(&section.name));
                if section.subsections.is_empty() && search.is_empty() {
                    ui.horizontal(|ui| {
                        ui.add_space(30.0);
                        ui.label(small_muted("No subsections yet."));
                    });
                }
                for sub in &section.subsections {
                    let assets: Vec<&library::AssetInfo> =
                        sub.assets.iter().filter(|a| library_matches(a, &search)).collect();
                    if !search.is_empty() && assets.is_empty() {
                        continue;
                    }
                    let key = format!("{}/{}", section.name, sub.name);
                    let folded = search.is_empty() && state.library.collapsed.contains(&key);
                    let buttons: &[(&str, &str, Color32)] = if sub.assets.is_empty() {
                        &[(icon::TRASH, "Delete this empty subsection", SUBTLE)]
                    } else {
                        &[]
                    };
                    let glyph = if folded { icon::FOLDER } else { icon::FOLDER_OPEN };
                    let (clicked, pressed) =
                        folder_row(ui, 16.0, Some(folded), false, glyph, &sub.name, assets.len(), buttons);
                    if clicked && !state.library.collapsed.remove(&key) {
                        state.library.collapsed.insert(key);
                    }
                    if pressed.is_some() {
                        if let Err(e) = library::delete_section(&root, &section.name, Some(&sub.name)) {
                            state.last_error = Some(e);
                        }
                        rescan = true;
                    }
                    if folded {
                        continue;
                    }
                    for asset in assets {
                        shown += 1;
                        rescan |= library_asset_row(ui, state, &root, asset, insert_at, comparing);
                    }
                }
            }
            if shown == 0 && sections.is_empty() {
                ui.add_space(12.0);
                ui.label(small_muted(
                    "Your library is empty. Save this model as an asset, or ask your agent to build some. Assets stay available in every project.",
                ));
            } else if shown == 0 && !search.is_empty() {
                ui.add_space(12.0);
                ui.label(small_muted("No assets match your search."));
            }
        });
    if rescan {
        state.library.scanned = None;
    }
}

/// One asset in the tab's list: thumbnail, name, counts, and add / delete buttons. Returns
/// true when the library changed.
fn library_asset_row(
    ui: &mut egui::Ui,
    state: &mut UiState,
    root: &std::path::Path,
    asset: &crate::library::AssetInfo,
    insert_at: Vec3,
    comparing: bool,
) -> bool {
    let (delete, armed) = library_delete_button(state, &asset.reference());
    let (rect, row) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 58.0), egui::Sense::click());
    if row.hovered() {
        ui.painter().rect_filled(rect, 6, HOVER_ROW);
    }
    let add_tip = if comparing { "Return to the live scene to add assets" } else { "Add to the scene" };
    let (pressed, right) = row_buttons(
        ui,
        rect,
        row.id,
        &[(icon::PLUS, add_tip, if comparing { SUBTLE } else { ACCENT }), delete],
    );
    let thumb = egui::Rect::from_min_size(
        egui::pos2(rect.left() + 28.0, rect.center().y - 24.0),
        egui::vec2(64.0, 48.0),
    );
    paint_thumbnail(ui, state, asset, thumb, 20.0);
    let text_left = thumb.right() + 10.0;
    let p = ui.painter().with_clip_rect(egui::Rect::from_min_max(
        egui::pos2(text_left, rect.top()),
        egui::pos2(right - 4.0, rect.bottom()),
    ));
    p.text(
        egui::pos2(text_left, rect.center().y - 9.0),
        egui::Align2::LEFT_CENTER,
        &asset.name,
        medium_font(13.5),
        TEXT,
    );
    p.text(
        egui::pos2(text_left, rect.center().y + 10.0),
        egui::Align2::LEFT_CENTER,
        format!("{} · {} faces", plural(asset.parts, "part", "parts"), compact_count(asset.faces)),
        FontId::proportional(11.5),
        SUBTLE,
    );
    let row = row.on_hover_ui(|ui| library_asset_tip(ui, asset));
    let add = pressed == Some(0) || (row.double_clicked() && pressed.is_none());
    library_asset_actions(ui, state, root, asset, insert_at, comparing, add, pressed == Some(1), armed)
}

const LIBRARY_CARD: egui::Vec2 = egui::vec2(216.0, 244.0);

/// One asset in the library view: a large preview with its name, counts and actions.
/// Returns true when the library changed.
fn library_asset_card(
    ui: &mut egui::Ui,
    state: &mut UiState,
    root: &std::path::Path,
    asset: &crate::library::AssetInfo,
    insert_at: Vec3,
    comparing: bool,
) -> bool {
    let (delete, armed) = library_delete_button(state, &asset.reference());
    let (rect, card) = ui.allocate_exact_size(LIBRARY_CARD, egui::Sense::click());
    let hovered = ui.rect_contains_pointer(rect);
    ui.painter().rect(
        rect,
        10,
        if hovered { CARD_HI } else { CARD },
        egui::Stroke::new(1.0_f32, if hovered { ACCENT.gamma_multiply(0.45) } else { BORDER }),
        egui::StrokeKind::Inside,
    );
    let inner = rect.shrink(8.0);
    let thumb = egui::Rect::from_min_size(inner.min, egui::vec2(inner.width(), inner.width() * 0.75));
    paint_thumbnail(ui, state, asset, thumb, 40.0);
    let p = ui.painter().with_clip_rect(inner);
    p.text(
        egui::pos2(inner.left() + 2.0, thumb.bottom() + 16.0),
        egui::Align2::LEFT_CENTER,
        &asset.name,
        medium_font(14.0),
        TEXT,
    );
    p.text(
        egui::pos2(inner.left() + 2.0, thumb.bottom() + 35.0),
        egui::Align2::LEFT_CENTER,
        format!("{} · {} faces", plural(asset.parts, "part", "parts"), compact_count(asset.faces)),
        FontId::proportional(11.5),
        SUBTLE,
    );
    let strip = egui::Rect::from_min_max(egui::pos2(inner.left(), inner.bottom() - 28.0), inner.max);
    let add_rect = egui::Rect::from_min_size(strip.min, egui::vec2(124.0, 28.0));
    let add = ui
        .interact(add_rect, card.id.with("add"), if comparing { egui::Sense::hover() } else { egui::Sense::click() })
        .on_hover_text(if comparing {
            "Return to the live scene to add assets"
        } else {
            "Add a copy to the scene"
        });
    let tint = if comparing { SUBTLE } else { ACCENT };
    ui.painter().rect(
        add_rect,
        7,
        tint.gamma_multiply(if add.hovered() { 0.26 } else { 0.14 }),
        egui::Stroke::new(1.0_f32, tint.gamma_multiply(0.5)),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        add_rect.center(),
        egui::Align2::CENTER_CENTER,
        icon_text(icon::PLUS, "Add to scene"),
        medium_font(12.5),
        tint,
    );
    let (pressed, _) = row_buttons(
        ui,
        strip,
        card.id,
        &[(icon::EXPORT, "Export for Godot or Blender", SUBTLE), delete],
    );
    let card = card.on_hover_ui(|ui| library_asset_tip(ui, asset));
    let add = add.clicked() || (card.double_clicked() && pressed.is_none());
    if pressed == Some(0) {
        library_open_export(state, root, Some(asset.reference()));
    }
    library_asset_actions(ui, state, root, asset, insert_at, comparing, add, pressed == Some(1), armed)
}

/// The full-window library: sections on the left, a grid of asset previews on the right.
fn draw_library_view(
    ctx: &egui::Context,
    scene: &Scene,
    camera: &OrbitCamera,
    state: &mut UiState,
    comparing: bool,
) {
    if !state.library.open {
        return;
    }
    use crate::library;
    let root = library::root();
    library_refresh(state, &root);
    let sections = state.library.sections.clone();
    // A filter can outlive its folder (deleted here, by an agent or in a file manager).
    if let Some((section, sub)) = &state.library.filter {
        let found = sections.iter().find(|s| s.name == *section).is_some_and(|s| {
            sub.as_ref().is_none_or(|sub| s.subsections.iter().any(|x| x.name == *sub))
        });
        if !found {
            state.library.filter = None;
        }
    }
    let total: usize = sections.iter().flat_map(|s| &s.subsections).map(|s| s.assets.len()).sum();
    let insert_at = library_insert_point(scene, camera);
    let screen = ctx.screen_rect();
    // Escape first closes whatever is open inside the view.
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    let mut close = false;
    let mut rescan = false;

    egui::Area::new("library_view".into())
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let (rect, scrim) = ui.allocate_exact_size(screen.size(), egui::Sense::click());
            ui.painter().rect_filled(rect, 0, Color32::from_black_alpha(170));
            let panel = egui::Rect::from_center_size(
                screen.center(),
                egui::vec2((screen.width() - 64.0).min(1280.0), screen.height() - 72.0),
            );
            // Clicks on the panel's own background must not fall through to the scrim.
            ui.interact(panel, ui.id().with("library_panel"), egui::Sense::click());
            ui.painter().rect(
                panel,
                14,
                PANEL,
                egui::Stroke::new(1.0_f32, BORDER),
                egui::StrokeKind::Inside,
            );
            close |= scrim.clicked();

            let inner = panel.shrink(18.0);
            let mut top = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(egui::Rect::from_min_size(inner.min, egui::vec2(inner.width(), 34.0)))
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            top.label(RichText::new(icon::BOOKS).size(22.0).color(ACCENT));
            top.heading("Asset library");
            top.label(small_muted(plural(total, "asset", "assets")));
            top.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                close |= icon_button(ui, icon::X, "Close  ·  Esc", false, true).clicked();
                if primary(ui, &icon_text(icon::PLUS, "Save current model"), !scene.objects.is_empty())
                {
                    state.library.export_scope = None;
                    library_open_save(scene, state);
                }
                if tool_button(
                    ui,
                    &icon_text(icon::EXPORT, "Export"),
                    total > 0,
                    "Export the assets shown here for Godot or Blender",
                    "There is nothing to export yet",
                ) {
                    state.library.save_open = false;
                    library_open_export(state, &root, None);
                }
                ui.allocate_ui(egui::vec2(280.0, 32.0), |ui| {
                    search_field(ui, &mut state.library.search, "Search assets…");
                });
            });
            let search = state.library.search.trim().to_lowercase();

            let body = egui::Rect::from_min_max(egui::pos2(inner.left(), inner.top() + 48.0), inner.max);
            ui.painter().line_segment(
                [egui::pos2(panel.left(), body.top() - 8.0), egui::pos2(panel.right(), body.top() - 8.0)],
                egui::Stroke::new(1.0_f32, BORDER),
            );
            let nav_rect = egui::Rect::from_min_max(body.min, egui::pos2(body.left() + 230.0, body.bottom()));
            let grid_rect = egui::Rect::from_min_max(egui::pos2(nav_rect.right() + 18.0, body.top()), body.max);
            ui.painter().line_segment(
                [
                    egui::pos2(nav_rect.right() + 9.0, body.top() - 8.0),
                    egui::pos2(nav_rect.right() + 9.0, panel.bottom()),
                ],
                egui::Stroke::new(1.0_f32, BORDER),
            );

            // ---- Sections
            let mut nav = ui.new_child(
                egui::UiBuilder::new().max_rect(nav_rect).layout(egui::Layout::top_down(egui::Align::Min)),
            );
            egui::ScrollArea::vertical()
                .id_salt("library_nav")
                .auto_shrink([false, false])
                .show(&mut nav, |ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    let (clicked, pressed) = folder_row(
                        ui,
                        0.0,
                        None,
                        state.library.filter.is_none(),
                        icon::SQUARES_FOUR,
                        "All assets",
                        total,
                        &[(icon::FOLDER_PLUS, "New section", SUBTLE)],
                    );
                    if clicked {
                        state.library.filter = None;
                    }
                    if pressed.is_some() {
                        state.library.new_folder = Some((None, String::new()));
                    }
                    rescan |= library_new_folder(ui, state, &root, None);
                    ui.add_space(6.0);
                    for section in sections.iter() {
                        let count: usize = section.subsections.iter().map(|s| s.assets.len()).sum();
                        let selected = state.library.filter == Some((section.name.clone(), None));
                        let mut buttons = vec![(icon::FOLDER_PLUS, "New subsection", SUBTLE)];
                        if count == 0 {
                            buttons.push((icon::TRASH, "Delete this empty section", SUBTLE));
                        }
                        let (clicked, pressed) =
                            folder_row(ui, 0.0, None, selected, icon::FOLDER, &section.name, count, &buttons);
                        if clicked {
                            state.library.filter = Some((section.name.clone(), None));
                        }
                        match pressed {
                            Some(0) => {
                                state.library.new_folder = Some((Some(section.name.clone()), String::new()));
                            }
                            Some(_) => {
                                if let Err(e) = library::delete_section(&root, &section.name, None) {
                                    state.last_error = Some(e);
                                }
                                rescan = true;
                            }
                            None => {}
                        }
                        rescan |= library_new_folder(ui, state, &root, Some(&section.name));
                        for sub in &section.subsections {
                            let filter = Some((section.name.clone(), Some(sub.name.clone())));
                            let buttons: &[(&str, &str, Color32)] = if sub.assets.is_empty() {
                                &[(icon::TRASH, "Delete this empty subsection", SUBTLE)]
                            } else {
                                &[]
                            };
                            let (clicked, pressed) = folder_row(
                                ui,
                                20.0,
                                None,
                                state.library.filter == filter,
                                icon::FOLDER_OPEN,
                                &sub.name,
                                sub.assets.len(),
                                buttons,
                            );
                            if clicked {
                                state.library.filter = filter;
                            }
                            if pressed.is_some() {
                                if let Err(e) = library::delete_section(&root, &section.name, Some(&sub.name)) {
                                    state.last_error = Some(e);
                                }
                                rescan = true;
                            }
                        }
                    }
                    ui.add_space(12.0);
                    let path = root.to_string_lossy();
                    truncated(ui, small_muted(format!("Stored in {path}")), &path);
                });

            // ---- Previews
            let mut grid = ui.new_child(
                egui::UiBuilder::new().max_rect(grid_rect).layout(egui::Layout::top_down(egui::Align::Min)),
            );
            egui::ScrollArea::vertical()
                .id_salt("library_grid")
                .auto_shrink([false, false])
                .show(&mut grid, |ui| {
                    if state.library.save_open {
                        ui.scope(|ui| {
                            ui.set_max_width(420.0);
                            library_save_form(ui, scene, state, &sections, &root);
                        });
                        ui.add_space(10.0);
                    } else if state.library.export_scope.is_some() {
                        ui.scope(|ui| {
                            ui.set_max_width(460.0);
                            library_export_form(ui, state, &sections, &root, &search);
                        });
                        ui.add_space(10.0);
                    }
                    let mut shown = 0;
                    for section in sections.iter() {
                        if state.library.filter.as_ref().is_some_and(|(name, _)| *name != section.name) {
                            continue;
                        }
                        for sub in &section.subsections {
                            if state
                                .library
                                .filter
                                .as_ref()
                                .is_some_and(|(_, only)| only.as_ref().is_some_and(|only| *only != sub.name))
                            {
                                continue;
                            }
                            let assets: Vec<&library::AssetInfo> =
                                sub.assets.iter().filter(|a| library_matches(a, &search)).collect();
                            if assets.is_empty() && (!search.is_empty() || state.library.filter.is_none()) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                ui.label(semibold(&section.name, 15.0));
                                ui.label(RichText::new(icon::CARET_RIGHT).size(11.0).color(SUBTLE));
                                ui.label(RichText::new(&sub.name).size(15.0).color(MUTED));
                            });
                            if assets.is_empty() {
                                ui.label(small_muted("No assets here yet."));
                            }
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(12.0, 12.0);
                                for asset in assets {
                                    shown += 1;
                                    rescan |= library_asset_card(ui, state, &root, asset, insert_at, comparing);
                                }
                            });
                            ui.add_space(14.0);
                        }
                    }
                    if shown == 0 && !state.library.save_open {
                        ui.add_space(40.0);
                        ui.vertical_centered(|ui| {
                            ui.label(RichText::new(icon::BOOKS).size(40.0).color(SUBTLE));
                            ui.label(semibold(
                                if total == 0 { "Your library is empty" } else { "Nothing to show here" },
                                16.0,
                            ));
                            ui.label(small_muted(if total == 0 {
                                "Save the current model as an asset, or ask your agent to build some.\nAssets stay available in every project."
                            } else if !search.is_empty() {
                                "No assets match your search."
                            } else {
                                "This section has no assets yet."
                            }));
                        });
                    }
                });
        });

    if rescan {
        state.library.scanned = None;
    }
    if escape {
        // The palette draws above this view and takes Escape itself.
        if state.palette_open {
        } else if state.library.new_folder.take().is_some() {
        } else if state.library.save_open {
            state.library.save_open = false;
        } else if state.library.export_scope.take().is_some() {
        } else {
            close = true;
        }
    }
    if close {
        state.library.open = false;
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
            toggle(ui, &mut renderer.xray_selection, "Show selection through geometry")
                .on_hover_text("Selection outlines stay visible behind other parts");
            toggle(ui, &mut renderer.ghost_unselected, "Ghost unselected parts")
                .on_hover_text("Fade everything except the selection");
            ui.add_space(8.0);
            eyebrow(ui, "Section plane");
            toggle(ui, &mut state.clip_enabled, "Cut the model with a plane");
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
                    toggle(ui, &mut state.clip_flip, "Flip");
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
            toggle(ui, &mut state.measure_active, "Pick measurement points");
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
            let drafts = scene.draft_count();
            if drafts > 0 && !comparing {
                card()
                    .fill(Color32::from_rgb(40, 36, 30))
                    .stroke(egui::Stroke::new(1.0_f32, WARN.gamma_multiply(0.45)))
                    .inner_margin(12)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(icon::PAPER_PLANE_TILT).size(18.0).color(WARN));
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 0.0;
                                ui.label(semibold(format!("{} ready to send", plural(drafts, "comment", "comments")), 14.0));
                                ui.label(small_muted("The agent picks them up as one batch."));
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if primary(ui, "Send to agent", true) {
                                    scene.send_drafts();
                                    state.scene_dirty_for_disk = true;
                                    state.status_message = format!("Sent {} to the agent", plural(drafts, "comment", "comments"));
                                }
                            });
                        });
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
        let has_faces = scene.objects.iter().any(|o| !o.mesh.selected_faces.is_empty());
        // A 3D box over empty space is a target of its own.
        let volume_only = scene.selection_volume.is_some() && !has_faces;
        let has_target = has_faces || volume_only || scene.active_object().is_some();
        let sections = if volume_only {
            Default::default()
        } else {
            current_target_sections(scene, state)
        };
        ui.horizontal(|ui| {
            eyebrow(ui, "New comment");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if volume_only {
                    pill(ui, "3D box", ACCENT);
                } else if !has_target {
                    pill(ui, "Scene-wide", MUTED);
                } else if !sections.is_empty() {
                    pill(ui, &plural(sections.len(), "region", "regions"), ACCENT);
                }
            });
        });
        if volume_only {
            ui.label(small_muted("The box covers no geometry, so the comment is about that space: describe what should be added there."));
        } else if !has_target {
            ui.label(small_muted("Click a part in the viewport to comment on it, or write feedback for the whole scene."));
        } else {
            // One row per part, however many regions it contributes.
            let mut groups: Vec<(u64, String, usize, usize)> = Vec::new();
            for t in sections.iter() {
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
            // Individual regions: name them for the agent, or drop one.
            let regions: Vec<_> = sections.iter().filter(|t| !t.selected_faces.is_empty()).collect();
            if regions.len() > 1 || regions.iter().any(|t| !t.label.is_empty()) || regions.len() == 1 {
                egui::CollapsingHeader::new(
                    RichText::new(format!("Name or remove {}", if regions.len() == 1 { "this region".to_string() } else { plural(regions.len(), "region", "regions") }))
                        .small()
                        .color(MUTED),
                )
                .id_salt("region_names")
                .show(ui, |ui| {
                    const SHOWN: usize = 12;
                    let mut drop_region: Option<(u64, Vec<usize>)> = None;
                    for (i, t) in regions.iter().take(SHOWN).enumerate() {
                        let key = (t.object_id, t.selected_faces[0]);
                        let mut label = scene.region_labels.get(&key).cloned().unwrap_or_default();
                        ui.horizontal(|ui| {
                            ui.label(small_muted(format!("{:>2}", i + 1)));
                            let w = (ui.available_width() - 96.0).max(60.0);
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut label)
                                    .hint_text(hint(&format!("{} · {}", t.object_name, plural(t.selected_faces.len(), "face", "faces"))))
                                    .desired_width(w),
                            );
                            if resp.changed() {
                                if label.trim().is_empty() {
                                    scene.region_labels.remove(&key);
                                } else {
                                    scene.region_labels.insert(key, label.clone());
                                }
                            }
                            if quiet_button(ui, icon::CROSSHAIR, "Show this region") {
                                state.frame_point = Some(t.world_anchor);
                            }
                            if quiet_button(ui, "×", "Remove this region from the selection") {
                                drop_region = Some((t.object_id, t.selected_faces.clone()));
                            }
                        });
                    }
                    if regions.len() > SHOWN {
                        ui.label(small_muted(format!("+ {} more", regions.len() - SHOWN)));
                    }
                    if let Some((id, faces)) = drop_region {
                        if let Some(o) = scene.objects.iter_mut().find(|o| o.id == id) {
                            for f in &faces {
                                o.mesh.selected_faces.remove(f);
                            }
                            o.mesh.bump_version();
                        }
                        scene.region_labels.remove(&(id, faces[0]));
                        update_pending_anchor_from_selection(scene);
                        state.scene_dirty_for_disk = true;
                    }
                });
            }
        }
        let draft = ui.add(
            egui::TextEdit::multiline(&mut state.comment_draft)
                .margin(egui::vec2(10.0, 8.0))
                .hint_text(hint(if volume_only {
                    "What should be added or changed inside this box?"
                } else if !has_target {
                    "Describe a change to the whole scene…"
                } else {
                    "What should change here? Describe the result you want…"
                }))
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        if std::mem::take(&mut state.focus_composer) {
            draft.request_focus();
        }
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
        // Reference images for the agent.
        let mut remove_attachment = None;
        if !state.draft_attachments.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for (i, path) in state.draft_attachments.clone().iter().enumerate() {
                    let resp = match load_screenshot(ui.ctx(), state, path) {
                        Ok(t) => ui.add(
                            egui::Image::new((t.id(), t.size_vec2()))
                                .max_height(54.0)
                                .max_width(86.0)
                                .corner_radius(6)
                                .sense(egui::Sense::click()),
                        ),
                        Err(_) => ui.add(egui::Button::new(RichText::new(icon::IMAGE).size(20.0)).min_size(egui::vec2(54.0, 54.0))),
                    };
                    let file = std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                    if resp.on_hover_text(format!("{file}\nClick to remove")).clicked() {
                        remove_attachment = Some(i);
                    }
                }
            });
        }
        if let Some(i) = remove_attachment {
            state.draft_attachments.remove(i);
        }
        if let Some(path) = state.attach_path.as_mut() {
            let mut add = false;
            ui.horizontal(|ui| {
                let w = ui.available_width() - 60.0;
                let r = text_field(ui, path, "/path/to/reference.png", w);
                add = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                add |= ui.button("Add").clicked();
            });
            if add {
                let p = path.trim().to_string();
                if !p.is_empty() {
                    match crate::project::absolute_path(&p) {
                        Ok(abs) if std::path::Path::new(&abs).is_file() => {
                            state.draft_attachments.push(abs);
                            state.attach_path = None;
                        }
                        _ => state.last_error = Some(format!("No image at {p}")),
                    }
                }
            }
        }
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            if ui
                .add(egui::Button::new(RichText::new(icon::CAMERA).size(15.0)).frame(false).min_size(egui::vec2(28.0, 26.0)))
                .on_hover_text("Attach a snapshot of the current view")
                .clicked()
            {
                let dir = attachments_dir(scene);
                let _ = std::fs::create_dir_all(&dir);
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let path = dir.join(format!("snapshot-{stamp}.png")).to_string_lossy().into_owned();
                action(state, "capture_viewport", serde_json::json!({"path": path, "width": 1280, "height": 800}));
                state.draft_attachments.push(path);
            }
            if ui
                .add(egui::Button::new(RichText::new(icon::PAPERCLIP).size(15.0)).frame(false).min_size(egui::vec2(28.0, 26.0)))
                .on_hover_text("Attach a reference image by path (or drop image files on the window)")
                .clicked()
            {
                state.attach_path = if state.attach_path.is_some() { None } else { Some(String::new()) };
            }
            ui.add_space(6.0);
            toggle(ui, &mut state.auto_send, "Send immediately").on_hover_text(
                "On: each comment goes to the agent right away.\nOff: comments wait as drafts until you press Send to agent.",
            );
        });
        let enabled = !state.comment_draft.trim().is_empty() && !comparing;
        let mut post: Option<bool> = None; // Some(scene_wide)
        ui.horizontal(|ui| {
            let verb = if state.auto_send { "Send" } else { "Add" };
            if primary(
                ui,
                &if has_target { format!("{verb} comment") } else { format!("{verb} scene comment") },
                enabled,
            ) || (submit_key && enabled)
            {
                post = Some(!has_target);
            }
            if has_target
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
            let id = if scene_wide {
                Some(scene.add_general_comment(text).id)
            } else {
                scene.add_comment_for_selection(text, state.selection_radius).map(|c| c.id)
            };
            if let Some(c) = id.and_then(|id| scene.comments.iter_mut().find(|c| c.id == id)) {
                c.draft = true;
                c.attachments = std::mem::take(&mut state.draft_attachments);
            }
            if state.auto_send {
                scene.send_drafts();
            }
            state.attach_path = None;
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
    if all > 0 {
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
    }
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
    let mut lightbox = None;
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
                                    if c.draft {
                                        pill(ui, "Draft · not sent", WARN);
                                    } else {
                                        pill(ui, status.label(), comment_status_color(status));
                                    }
                                });
                            });
                            let rows = if expanded { None } else { Some(5) };
                            let mut elided = clamped_text(ui, &c.comment, 15.0, 22.0, TEXT, rows);
                            if !c.is_general {
                                let (short, full) = comment_targets(c);
                                truncated(ui, RichText::new(format!("{}  {short}", icon::CROSSHAIR)).small().color(SUBTLE), &full);
                            }
                            if !c.attachments.is_empty() {
                                ui.horizontal_wrapped(|ui| {
                                    for path in &c.attachments {
                                        if let Ok(t) = load_screenshot(ui.ctx(), state, path) {
                                            if ui
                                                .add(egui::Image::new((t.id(), t.size_vec2())).max_height(48.0).max_width(76.0).corner_radius(5).sense(egui::Sense::click()))
                                                .on_hover_cursor(egui::CursorIcon::ZoomIn)
                                                .clicked()
                                            {
                                                lightbox = Some(path.clone());
                                            }
                                        } else {
                                            ui.label(RichText::new(icon::IMAGE).color(SUBTLE)).on_hover_text(path);
                                        }
                                    }
                                });
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
    if lightbox.is_some() {
        state.lightbox = lightbox;
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
        if toggle(ui, &mut scene.highlight_changes, "Highlight changed parts").changed() {
            *view_changed = true;
        }
        if comparing {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if primary(ui, "Back to live", true) {
                    scene.preview_revision = None;
                    state.compare_split = None;
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
                        if r.geometry_pruned {
                            ui.label(small_muted("History only · older geometry is dropped to keep the project small"));
                        } else {
                            ui.horizontal(|ui| {
                                ui.label(small_muted("Compare"));
                                let mut choice = match previewing {
                                    Some(_) if state.compare_split.is_some() => 3,
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
                                        (3, icon::SQUARE_HALF.into(), "Split view · drag the divider between before and after"),
                                        (2, "After".into(), "Geometry after this revision"),
                                    ],
                                ) {
                                    state.compare_split = (choice == 3).then_some(0.5);
                                    match choice {
                                        0 => scene.preview_revision = None,
                                        n => action(state, "preview_revision", json!({"revision_id": rid, "before": n == 1})),
                                    }
                                    *view_changed = true;
                                }
                            });
                        }
                        if let Some(notes) = &r.change_request {
                            egui::Frame::new()
                                .fill(WARN.gamma_multiply(0.08))
                                .corner_radius(6)
                                .inner_margin(8)
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    eyebrow(ui, "You asked for changes");
                                    ui.label(RichText::new(notes).size(13.5).color(TEXT));
                                });
                        }
                        if r.status == RevisionStatus::ReadyForReview {
                            if state.request_changes_for == Some(rid) {
                                ui.add(
                                    egui::TextEdit::multiline(&mut state.request_notes)
                                        .hint_text(hint("What should be different? The agent gets this as a reply."))
                                        .margin(egui::vec2(8.0, 6.0))
                                        .desired_rows(2)
                                        .desired_width(f32::INFINITY),
                                );
                                ui.horizontal(|ui| {
                                    if primary(ui, "Revert & send notes", !state.request_notes.trim().is_empty()) {
                                        action(state, "request_changes", json!({"revision_id": rid, "notes": state.request_notes.trim()}));
                                        *view_changed = true;
                                    }
                                    if ui.button("Cancel").clicked() {
                                        state.request_changes_for = None;
                                    }
                                });
                            } else {
                                ui.horizontal(|ui| {
                                    if primary(ui, &icon_text(icon::CHECK, "Accept"), true) {
                                        action(state, "accept_revision", json!({"revision_id": rid}));
                                        *view_changed = true;
                                    }
                                    if ui
                                        .button(icon_text(icon::CHAT_TEXT, "Request changes"))
                                        .on_hover_text("Revert this revision and tell the agent what to change")
                                        .clicked()
                                    {
                                        state.request_changes_for = Some(rid);
                                        state.request_notes.clear();
                                    }
                                    if ui
                                        .button(icon::ARROW_U_UP_LEFT)
                                        .on_hover_text("Revert without notes")
                                        .clicked()
                                    {
                                        action(state, "revert_revision", json!({"revision_id": rid}));
                                        *view_changed = true;
                                    }
                                });
                            }
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

fn attachments_dir(scene: &Scene) -> std::path::PathBuf {
    match &scene.project_path {
        Some(p) => {
            let p = std::path::Path::new(p);
            let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "project".into());
            let stem = stem.trim_end_matches(".spatial").to_string();
            p.parent().unwrap_or(std::path::Path::new(".")).join(format!("{stem}.attachments"))
        }
        None => std::path::PathBuf::from("/tmp/rust_modeler_attachments"),
    }
}

fn load_screenshot(
    ctx: &egui::Context,
    state: &mut UiState,
    path: &str,
) -> Result<egui::TextureHandle, String> {
    // A snapshot may still be rendering; retry until the file exists.
    if matches!(state.screenshot_cache.get(path), Some(Err(_))) && !std::path::Path::new(path).exists() {
        state.screenshot_cache.remove(path);
        return Err("Not rendered yet".into());
    }
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
    if scene.preview_revision.is_none() {
        state.compare_split = None;
    }
    renderer.compare_split = state.compare_split;
    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
        state.viewport_rect = ui.max_rect();
        let rect = ui.max_rect();
        if let Some(p) = state.frame_point.take() {
            camera.target = p;
            *view_changed = true;
        }

        let was_dragging_volume = std::mem::take(&mut state.volume_dragging);
        viewport_tool_strip(ctx, rect, scene, camera, state, comparing);
        viewport_context_bar(ctx, rect, scene, state, comparing);
        if !comparing && !state.fly_active {
            viewport_volume(ctx, ui, rect, scene, camera, state);
        }
        if was_dragging_volume && !state.volume_dragging {
            // Drag finished: save once and let the comment panel catch up with the selection.
            state.scene_dirty_for_disk |= std::mem::take(&mut state.volume_drag_changed);
            ctx.request_repaint();
        }
        viewport_view_controls(ctx, rect, scene, camera, renderer, state);
        viewport_axis_gizmo(ctx, rect, camera);
        viewport_selection_chip(ctx, rect, scene, state, comparing);

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
                                let mut choice = match (state.compare_split.is_some(), before) {
                                    (true, _) => 1,
                                    (false, true) => 0,
                                    (false, false) => 2,
                                };
                                if segmented(
                                    ui,
                                    &mut choice,
                                    &[
                                        (0, "Before".into(), "Geometry before the revision"),
                                        (1, icon_text(icon::SQUARE_HALF, "Split"), "Drag the divider to wipe between before and after"),
                                        (2, "After".into(), "Geometry after the revision"),
                                    ],
                                ) {
                                    state.compare_split = (choice == 1).then_some(0.5);
                                    if choice != 1 {
                                        action(state, "preview_revision", json!({"revision_id": rid, "before": choice == 0}));
                                    }
                                    *view_changed = true;
                                }
                                if primary(ui, "Back to live", true) {
                                    scene.preview_revision = None;
                                    state.compare_split = None;
                                    *view_changed = true;
                                }
                            });
                        });
                });
            top = resp.response.rect.bottom() + 8.0;
            if let Some(split) = state.compare_split {
                // Wipe divider: drag to reveal before (left) and after (right).
                let x = rect.left() + rect.width() * split;
                let handle = egui::Rect::from_center_size(egui::pos2(x, rect.center().y), egui::vec2(28.0, rect.height()));
                let drag = ui.interact(handle, egui::Id::new("compare_split"), egui::Sense::drag());
                if drag.dragged() {
                    if let Some(p) = drag.interact_pointer_pos() {
                        state.compare_split = Some(((p.x - rect.left()) / rect.width()).clamp(0.02, 0.98));
                    }
                }
                let painter = ui.painter().with_clip_rect(rect);
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(2.0_f32, Color32::from_white_alpha(220)),
                );
                let knob = egui::pos2(x, rect.center().y);
                painter.circle(knob, 15.0, OVERLAY, egui::Stroke::new(1.5_f32, Color32::WHITE));
                painter.text(knob, egui::Align2::CENTER_CENTER, icon::ARROWS_OUT_LINE_HORIZONTAL, FontId::proportional(15.0), TEXT);
                for (text, pos, align) in [
                    ("BEFORE", egui::pos2(x - 12.0, rect.bottom() - 60.0), egui::Align2::RIGHT_BOTTOM),
                    ("AFTER", egui::pos2(x + 12.0, rect.bottom() - 60.0), egui::Align2::LEFT_BOTTOM),
                ] {
                    let g = painter.layout_no_wrap(text.into(), medium_font(11.5), TEXT);
                    let r = align.anchor_size(pos, g.size()).expand2(egui::vec2(8.0, 4.0));
                    painter.rect_filled(r, 6, OVERLAY);
                    painter.galley(r.min + egui::vec2(8.0, 4.0), g, TEXT);
                }
                if drag.hovered() || drag.dragged() {
                    ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
            }
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
            // Project every pin, then merge the ones that would overlap into one badge.
            let mut clusters: Vec<(egui::Pos2, Vec<&SpatialComment>)> = Vec::new();
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
                match clusters.iter_mut().find(|(q, _)| q.distance(pos) < 34.0) {
                    Some((_, list)) => list.push(c),
                    None => clusters.push((pos, vec![c])),
                }
            }
            for (pos, list) in &clusters {
                let active_in = list.iter().position(|c| scene.active_comment_id == Some(c.id));
                let lead = list[active_in.unwrap_or(0)];
                let active = active_in.is_some();
                let color = comment_status_color(lead.status);
                egui::Area::new(egui::Id::new(("comment_pin", list[0].id)))
                    .pivot(egui::Align2::CENTER_BOTTOM)
                    .fixed_pos(*pos)
                    .show(ctx, |ui| {
                        let label = if list.len() > 1 {
                            format!("#{:02} +{}", lead.id, list.len() - 1)
                        } else {
                            format!("#{:02}", lead.id)
                        };
                        let text = RichText::new(label)
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
                                for c in list.iter().take(6) {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(RichText::new(format!("#{:02}", c.id)).color(comment_status_color(c.status)));
                                        ui.label(RichText::new(c.comment.clone()));
                                    });
                                }
                                if list.len() > 6 {
                                    ui.label(small_muted(format!("+ {} more", list.len() - 6)));
                                }
                                ui.label(small_muted(if list.len() > 1 {
                                    "Click to step through these comments".to_string()
                                } else {
                                    format!("{} · click to open", lead.status.label())
                                }));
                            });
                        if resp.clicked() {
                            // A cluster steps to the next comment each click.
                            let next = match active_in {
                                Some(i) => list[(i + 1) % list.len()],
                                None => lead,
                            };
                            state.focus_comment = Some(next.id);
                            state.scroll_to_comment = Some(next.id);
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
        if let Some(r) = state.box_rect {
            let r = r.intersect(rect);
            painter.rect_filled(r, 2.0, ACCENT.gamma_multiply(0.10));
            painter.rect_stroke(r, 2.0, egui::Stroke::new(1.5_f32, ACCENT), egui::StrokeKind::Inside);
        }
        if state.fly_active {
            let c = rect.center();
            let stroke = egui::Stroke::new(1.5_f32, Color32::from_white_alpha(220));
            let shadow = egui::Stroke::new(3.5_f32, Color32::from_black_alpha(140));
            for s in [shadow, stroke] {
                painter.line_segment([c - egui::vec2(9.0, 0.0), c - egui::vec2(3.0, 0.0)], s);
                painter.line_segment([c + egui::vec2(3.0, 0.0), c + egui::vec2(9.0, 0.0)], s);
                painter.line_segment([c - egui::vec2(0.0, 9.0), c - egui::vec2(0.0, 3.0)], s);
                painter.line_segment([c + egui::vec2(0.0, 3.0), c + egui::vec2(0.0, 9.0)], s);
            }
        }
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
        let mode_hint = if state.fly_active {
            format!(
                "Fly · WASD move · Q/E down/up · Shift fast · Ctrl slow · Scroll speed ({:.1} {}/s) · Click selects at the crosshair · Esc exits",
                state.fly_speed,
                scene.units
            )
        } else if comparing {
            "Comparison is read-only · go back to live to select".to_string()
        } else if state.measure_active {
            "Measure · click surfaces to place points".to_string()
        } else {
            match state.select_mode {
                SmartSelectMode::SubPart => "Click a connected piece to select it",
                SmartSelectMode::SmartSurface => "Click a surface · the crease angle sets its edges",
                SmartSelectMode::WholeLayer => "Click an object to select all of it",
                SmartSelectMode::Box => "Drag boxes over each place to change · Shift adds · Ctrl removes · each separate patch becomes its own AI target",
                SmartSelectMode::Face => "Click polygons one at a time · Ctrl+click removes · turn on Path to fill between two clicks",
                SmartSelectMode::Volume => "Click a surface to place the 3D box · drag coloured handles to resize, the centre to move · it can sit in empty space",
            }
            .to_string()
        };
        let chip_visible = !comparing
            && !state.fly_active
            && (scene.has_any_selection() || scene.selection_volume.is_some());
        if !chip_visible || state.measure_active || state.select_mode.is_drag() {
        egui::Area::new("viewport_hint".into())
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(egui::pos2(rect.left() + 66.0, rect.bottom() - 12.0))
            .show(ctx, |ui| {
                ui.set_max_width((rect.width() - 220.0).max(200.0));
                egui::Frame::new()
                    .fill(OVERLAY)
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            ui.label(RichText::new(icon::INFO).size(13.0).color(SUBTLE));
                            ui.add(egui::Label::new(RichText::new(mode_hint).small().color(MUTED)).truncate())
                                .on_hover_text("Middle-drag or Alt+drag orbit · Right-drag pan · Scroll zoom · F frame · ` fly");
                        });
                    });
            });
        }
    });
}

fn mode_icon(mode: SmartSelectMode) -> &'static str {
    match mode {
        SmartSelectMode::SubPart => icon::CUBE,
        SmartSelectMode::SmartSurface => icon::POLYGON,
        SmartSelectMode::WholeLayer => icon::STACK,
        SmartSelectMode::Box => icon::SELECTION,
        SmartSelectMode::Face => icon::TRIANGLE,
        SmartSelectMode::Volume => icon::BOUNDING_BOX,
    }
}

const AXIS_COLORS: [Color32; 3] = [
    Color32::from_rgb(240, 96, 96),
    Color32::from_rgb(104, 214, 120),
    Color32::from_rgb(96, 156, 250),
];

/// The 3D box selector: wireframe, size label, six face handles (resize) and a centre handle
/// (move). Handles live in their own egui areas so the viewport does not treat a press on
/// them as a click on the model.
fn viewport_volume(
    ctx: &egui::Context,
    ui: &egui::Ui,
    rect: egui::Rect,
    scene: &mut Scene,
    camera: &OrbitCamera,
    state: &mut UiState,
) {
    use crate::scene::SelectionVolume;
    let Some(mut volume) = scene.selection_volume.clone() else {
        return;
    };
    let placed = volume.clone();
    let viewport = Vec2::new(rect.width(), rect.height());
    let to_screen = |p: Vec3| {
        camera
            .world_to_screen(p, viewport)
            .map(|s| rect.left_top() + egui::vec2(s.x, s.y))
    };
    let painter = ui.painter().with_clip_rect(rect);
    let pts: Vec<Option<egui::Pos2>> = volume.corners().iter().map(|&c| to_screen(c)).collect();
    for (a, b) in SelectionVolume::EDGES {
        if let (Some(a), Some(b)) = (pts[a], pts[b]) {
            painter.line_segment([a, b], egui::Stroke::new(3.5_f32, Color32::from_black_alpha(90)));
            painter.line_segment([a, b], egui::Stroke::new(1.5_f32, ACCENT));
        }
    }
    let size = volume.half_extents * 2.0 * scene.unit_factor();
    if let Some(top) = to_screen(volume.face_handle(1, true)) {
        let text = format!(
            "{}{:.2} × {:.2} × {:.2} {}",
            if volume.label.is_empty() { String::new() } else { format!("{} · ", volume.label) },
            size.x,
            size.y,
            size.z,
            scene.units
        );
        let g = painter.layout_no_wrap(text, medium_font(11.5), TEXT);
        let r = egui::Align2::CENTER_BOTTOM
            .anchor_size(top - egui::vec2(0.0, 16.0), g.size())
            .expand2(egui::vec2(7.0, 3.0));
        painter.rect_filled(r, 6, OVERLAY);
        painter.galley(r.min + egui::vec2(7.0, 3.0), g, TEXT);
    }
    if state.select_mode != SmartSelectMode::Volume || state.measure_active {
        return;
    }

    let pointer_ray = |p: egui::Pos2| {
        camera.screen_ray(Vec2::new(p.x - rect.left(), p.y - rect.top()), viewport)
    };
    let handle = |id: egui::Id, pos: egui::Pos2, color: Color32, radius: f32, tip: &str| {
        egui::Area::new(id)
            .pivot(egui::Align2::CENTER_CENTER)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                let (r, resp) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::drag());
                let hot = resp.hovered() || resp.dragged();
                ui.painter().circle(
                    r.center(),
                    if hot { radius + 1.5 } else { radius },
                    color,
                    egui::Stroke::new(1.5_f32, Color32::WHITE),
                );
                if hot {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
                resp.on_hover_text(tip)
            })
            .inner
    };
    for axis in 0..3 {
        for positive in [false, true] {
            let Some(pos) = to_screen(volume.face_handle(axis, positive)) else {
                continue;
            };
            if !rect.contains(pos) {
                continue;
            }
            let resp = handle(
                egui::Id::new(("volume_face", axis, positive)),
                pos,
                AXIS_COLORS[axis],
                5.5,
                &format!("Drag to resize along {}", ["X", "Y", "Z"][axis]),
            );
            state.volume_dragging |= resp.dragged();
            if let (true, Some(p)) = (resp.dragged(), resp.interact_pointer_pos()) {
                let mut dir = Vec3::ZERO;
                dir[axis] = 1.0;
                let t = pointer_ray(p).closest_t_on_axis(volume.center, dir);
                if t.is_finite() {
                    volume.move_face(axis, positive, volume.center[axis] + t);
                }
            }
        }
    }
    if let Some(pos) = to_screen(volume.center).filter(|p| rect.contains(*p)) {
        let resp = handle(
            egui::Id::new("volume_centre"),
            pos,
            ACCENT,
            6.5,
            "Drag to move the box",
        );
        state.volume_dragging |= resp.dragged();
        if let (true, Some(p)) = (resp.dragged(), resp.interact_pointer_pos()) {
            let ray = pointer_ray(p);
            if let Some(t) = ray.intersect_plane(volume.center, camera.forward()) {
                let hit = ray.at(t);
                if hit.is_finite() {
                    volume.center = hit;
                }
            }
        }
    }
    // A held handle reports a drag every frame; only redo the selection when the box moved.
    // The project is saved once, when the drag ends (see `viewport`).
    if volume != placed {
        scene.selection_volume = Some(volume);
        scene.apply_volume_selection();
        update_pending_anchor_from_selection(scene);
        state.volume_drag_changed = true;
    }
}

/// Vertical tool strip on the viewport's left edge.
fn viewport_tool_strip(
    ctx: &egui::Context,
    rect: egui::Rect,
    scene: &Scene,
    camera: &mut OrbitCamera,
    state: &mut UiState,
    comparing: bool,
) {
    egui::Area::new("viewport_tools".into())
        .fixed_pos(rect.left_top() + egui::vec2(12.0, 12.0))
        .show(ctx, |ui| {
            overlay_frame()
                .inner_margin(egui::Margin::same(4))
                .corner_radius(11)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    ui.vertical(|ui| {
                        for (key, mode, what) in [
                            ("1", SmartSelectMode::SubPart, "Click selects one connected piece"),
                            ("2", SmartSelectMode::SmartSurface, "Click selects a surface bounded by sharp edges"),
                            ("3", SmartSelectMode::WholeLayer, "Click selects the entire object"),
                            ("4", SmartSelectMode::Box, "Drag rectangles over the places to change; Shift adds, Ctrl removes  ·  Ctrl+C"),
                            ("5", SmartSelectMode::Face, "Click single polygons; Path fills the faces between two clicks"),
                            ("6", SmartSelectMode::Volume, "Place a 3D box and resize it; selects everything inside from any angle, and can mark empty space"),
                        ] {
                            let selected = state.select_mode == mode && !state.measure_active;
                            let resp = icon_button(ui, mode_icon(mode), "", selected, !comparing);
                            // Shortcut digit in the corner, so the keys teach themselves.
                            ui.painter().text(
                                resp.rect.right_bottom() - egui::vec2(4.0, 2.0),
                                egui::Align2::RIGHT_BOTTOM,
                                key,
                                FontId::proportional(9.0),
                                if selected { ACCENT } else { SUBTLE },
                            );
                            let resp = resp.on_hover_ui_at_pointer(|ui| {
                                ui.set_max_width(280.0);
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(mode_icon(mode)).size(16.0).color(ACCENT));
                                    ui.label(semibold(format!("{} select", mode.label()), 14.0));
                                    pill(ui, key, MUTED);
                                });
                                ui.label(small_muted(what));
                            });
                            if resp.clicked() {
                                if mode == SmartSelectMode::Box && state.select_mode != SmartSelectMode::Box {
                                    state.prev_select_mode = state.select_mode;
                                }
                                state.select_mode = mode;
                                state.hover_preview_seed_face = None;
                                state.measure_active = false;
                            }
                        }
                        hdivider(ui);
                        if icon_button(ui, icon::RULER, "Measure distances and angles", state.measure_active, !comparing).clicked() {
                            state.measure_active = !state.measure_active;
                        }
                        if icon_button(
                            ui,
                            icon::AIRPLANE_TILT,
                            "Fly  ·  ` (backtick)\nWASD move · Q/E down/up · mouse look · Esc exits",
                            state.fly_active,
                            true,
                        )
                        .clicked()
                        {
                            state.fly_toggle_requested = true;
                        }
                        hdivider(ui);
                        let has_sel = scene.has_any_selection();
                        if icon_button(
                            ui,
                            icon::FRAME_CORNERS,
                            if has_sel { "Frame selection  ·  F" } else { "Frame everything  ·  F" },
                            false,
                            true,
                        )
                        .clicked()
                        {
                            frame_selection(scene, camera);
                        }
                        if icon_button(ui, icon::KEYBOARD, "Keyboard shortcuts  ·  F1", state.show_shortcuts, true).clicked() {
                            state.show_shortcuts = !state.show_shortcuts;
                        }
                    });
                });
        });
}

/// Options for the active selection tool, top-left of the viewport.
fn viewport_context_bar(
    ctx: &egui::Context,
    rect: egui::Rect,
    scene: &mut Scene,
    state: &mut UiState,
    comparing: bool,
) {
    if comparing || state.fly_active {
        return;
    }
    egui::Area::new("viewport_context".into())
        .fixed_pos(rect.left_top() + egui::vec2(66.0, 12.0))
        .show(ctx, |ui| {
            overlay_frame()
                .inner_margin(egui::Margin::symmetric(10, 5))
                .corner_radius(11)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        let (glyph, label) = if state.measure_active {
                            (icon::RULER, "Measure")
                        } else {
                            (mode_icon(state.select_mode), state.select_mode.label())
                        };
                        ui.label(RichText::new(glyph).size(15.0).color(ACCENT));
                        ui.label(semibold(label, 13.5));
                        if !state.measure_active {
                            match state.select_mode {
                                SmartSelectMode::SmartSurface => {
                                    vdivider(ui);
                                    ui.label(small_muted("Crease"));
                                    ui.spacing_mut().slider_width = 90.0;
                                    if ui
                                        .add(egui::Slider::new(&mut state.crease_angle_deg, 5.0..=85.0).suffix("°"))
                                        .on_hover_text("Edges sharper than this angle stop the surface fill")
                                        .changed()
                                    {
                                        state.hover_preview_seed_face = None;
                                    }
                                }
                                SmartSelectMode::Box => {
                                    vdivider(ui);
                                    toggle(ui, &mut state.box_through, "Through")
                                        .on_hover_text("Also pick faces hidden behind other geometry inside the box");
                                }
                                SmartSelectMode::Face => {
                                    vdivider(ui);
                                    toggle(ui, &mut state.face_path, "Path")
                                        .on_hover_text("Each click also selects the shortest run of faces from the previous click");
                                }
                                SmartSelectMode::Volume => {
                                    vdivider(ui);
                                    let factor = scene.unit_factor();
                                    let units = scene.units.clone();
                                    let mut changed = false;
                                    let mut remove = false;
                                    let mut dragging = false;
                                    if let Some(volume) = scene.selection_volume.as_mut() {
                                        ui.label(small_muted("Size"));
                                        for axis in 0..3 {
                                            let mut v = volume.half_extents[axis] * 2.0 * factor;
                                            let resp = ui.add(
                                                egui::DragValue::new(&mut v)
                                                    .speed(0.01 * factor)
                                                    .range(0.01 * factor..=10_000.0 * factor)
                                                    .max_decimals(2)
                                                    .prefix(["X ", "Y ", "Z "][axis]),
                                            );
                                            dragging |= resp.dragged();
                                            if resp.on_hover_text(format!("Box size in {units}")).changed() {
                                                volume.half_extents[axis] = (v / factor * 0.5)
                                                    .max(crate::scene::SelectionVolume::MIN_HALF);
                                                changed = true;
                                            }
                                        }
                                        ui.add(
                                            egui::TextEdit::singleline(&mut volume.label)
                                                .hint_text(hint("Name this region"))
                                                .desired_width(120.0),
                                        )
                                        .on_hover_text("The agent sees this name with your comment");
                                        if quiet_button(ui, icon::TRASH, "Remove the 3D box") {
                                            remove = true;
                                        }
                                    } else {
                                        ui.label(small_muted("Click a surface to place the box"));
                                    }
                                    if remove {
                                        scene.selection_volume = None;
                                        scene.clear_all_selections();
                                        scene.selected_object = None;
                                        state.scene_dirty_for_disk = true;
                                    } else if changed {
                                        scene.apply_volume_selection();
                                        update_pending_anchor_from_selection(scene);
                                        // A dragged size field saves once, when the drag ends.
                                        state.volume_drag_changed |= dragging;
                                        state.scene_dirty_for_disk |= !dragging;
                                    }
                                    state.volume_dragging |= dragging;
                                }
                                _ => {}
                            }
                            if state.select_mode != SmartSelectMode::Volume {
                            vdivider(ui);
                            toggle(ui, &mut state.multi_select, "Add")
                                .on_hover_text("Multi-select: each click adds to the selection; click a selected region to remove it.\nOff: Shift+click adds, Ctrl+click removes.");
                            }
                        }
                    });
                });
        });
}

/// Floating selection summary with grow / shrink / clear, bottom-centre of the viewport.
fn viewport_selection_chip(
    ctx: &egui::Context,
    rect: egui::Rect,
    scene: &mut Scene,
    state: &mut UiState,
    comparing: bool,
) {
    if comparing || state.fly_active || !(scene.has_any_selection() || scene.selection_volume.is_some()) {
        return;
    }
    let Some(summary) = selection_summary(scene) else {
        return;
    };
    egui::Area::new("viewport_selection".into())
        .pivot(egui::Align2::CENTER_BOTTOM)
        .fixed_pos(egui::pos2(rect.center().x, rect.bottom() - 14.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(OVERLAY)
                .stroke(egui::Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.45)))
                .corner_radius(20)
                .inner_margin(egui::Margin { left: 14, right: 6, top: 4, bottom: 4 })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(RichText::new(icon::SELECTION_ALL).size(14.0).color(ACCENT));
                        ui.add_space(2.0);
                        let short: String = if summary.chars().count() > 56 {
                            summary.chars().take(55).collect::<String>() + "…"
                        } else {
                            summary.clone()
                        };
                        ui.label(RichText::new(short).size(13.0).color(TEXT)).on_hover_text(&summary);
                        ui.add_space(6.0);
                        let has_faces = scene.objects.iter().any(|o| !o.mesh.selected_faces.is_empty());
                        let small = |ui: &mut egui::Ui, glyph: &str, tip: &str, enabled: bool| {
                            ui.add_enabled(
                                enabled,
                                egui::Button::new(RichText::new(glyph).size(16.0))
                                    .frame(false)
                                    .min_size(egui::vec2(28.0, 28.0)),
                            )
                            .on_hover_text(tip)
                            .on_disabled_hover_text("Select faces (Part, Surface, Face or Box) to grow or shrink")
                            .clicked()
                        };
                        if small(ui, icon::PLUS_CIRCLE, "Grow selection by one ring  ·  ]", has_faces) {
                            for o in &mut scene.objects {
                                if !o.mesh.selected_faces.is_empty() {
                                    o.mesh.grow_selection();
                                }
                            }
                            update_pending_anchor_from_selection(scene);
                            state.scene_dirty_for_disk = true;
                        }
                        if small(ui, icon::MINUS_CIRCLE, "Shrink selection by one ring  ·  [", has_faces) {
                            for o in &mut scene.objects {
                                if !o.mesh.selected_faces.is_empty() {
                                    o.mesh.shrink_selection();
                                }
                            }
                            update_pending_anchor_from_selection(scene);
                            state.scene_dirty_for_disk = true;
                        }
                        if small(ui, icon::SELECTION_INVERSE, "Invert selection on the selected parts  ·  I", has_faces) {
                            for o in &mut scene.objects {
                                if !o.mesh.selected_faces.is_empty() {
                                    o.mesh.invert_selection();
                                }
                            }
                            scene.selection_volume = None;
                            scene.region_labels.clear();
                            update_pending_anchor_from_selection(scene);
                            state.scene_dirty_for_disk = true;
                        }
                        let mut similar = None;
                        ui.add_enabled_ui(has_faces, |ui| {
                            ui.menu_button(RichText::new(icon::MAGIC_WAND).size(16.0), |ui| {
                                ui.label(small_muted("Also select faces with a similar…"));
                                for (label, by, tip) in [
                                    ("Direction", crate::mesh::SimilarBy::Normal, "Faces pointing the same way, e.g. every sill or every roof slope"),
                                    ("Size", crate::mesh::SimilarBy::Area, "Faces of about the same area"),
                                    ("Colour", crate::mesh::SimilarBy::Color, "Faces painted about the same colour"),
                                ] {
                                    if ui.button(label).on_hover_text(tip).clicked() {
                                        similar = Some(by);
                                        ui.close_menu();
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Select similar faces on the selected parts");
                        });
                        if let Some(by) = similar {
                            let mut added = 0;
                            for o in &mut scene.objects {
                                if !o.mesh.selected_faces.is_empty() {
                                    added += o.mesh.select_similar(by, by.default_tolerance());
                                }
                            }
                            scene.selection_volume = None;
                            update_pending_anchor_from_selection(scene);
                            state.status_message = format!("Added {}", plural(added, "similar face", "similar faces"));
                            state.scene_dirty_for_disk = true;
                        }
                        if small(ui, icon::CHAT_CIRCLE_TEXT, "Write a comment about this selection", true) {
                            state.review_tab = 0;
                            state.focus_composer = true;
                        }
                        if small(ui, icon::X_CIRCLE, "Clear selection  ·  Esc", true) {
                            scene.selection_volume = None;
                            scene.clear_all_selections();
                            scene.selected_object = None;
                            state.scene_dirty_for_disk = true;
                        }
                    });
                });
        });
}

/// Shading, overlays and camera menu, top-right of the viewport.
fn viewport_view_controls(
    ctx: &egui::Context,
    rect: egui::Rect,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    renderer: &mut Renderer3D,
    state: &mut UiState,
) {
    egui::Area::new("viewport_view".into())
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
        .show(ctx, |ui| {
            overlay_frame()
                .inner_margin(egui::Margin::same(4))
                .corner_radius(11)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        for (mode, glyph, tip) in [
                            (ViewportShading::StudioLit, icon::SUN, "Lit · materials and studio lighting"),
                            (ViewportShading::SculptClay, icon::SPHERE, "Clay · neutral material to judge form"),
                            (ViewportShading::WireframeOnly, icon::CUBE_TRANSPARENT, "Wireframe only"),
                        ] {
                            if icon_button(ui, glyph, tip, renderer.shading == mode, true).clicked() {
                                renderer.shading = mode;
                            }
                        }
                        vdivider(ui);
                        if icon_button(ui, icon::POLYGON, "Show edges", renderer.show_wireframe_overlay, true).clicked() {
                            renderer.show_wireframe_overlay = !renderer.show_wireframe_overlay;
                        }
                        if icon_button(ui, icon::GRID_FOUR, "Ground grid and contact shadow", renderer.show_grid, true).clicked() {
                            renderer.show_grid = !renderer.show_grid;
                        }
                        if icon_button(ui, icon::GHOST, "Ghost everything except the selection", renderer.ghost_unselected, true).clicked() {
                            renderer.ghost_unselected = !renderer.ghost_unselected;
                        }
                        if icon_button(ui, icon::KNIFE, "Section plane · cut the model to see inside", state.clip_enabled, true).clicked() {
                            state.clip_enabled = !state.clip_enabled;
                            if state.clip_enabled {
                                state.scene_tab = 1;
                            }
                        }
                        vdivider(ui);
                        ui.menu_button(
                            RichText::new(format!(
                                "{}  {}",
                                if camera.orthographic { "Ortho" } else { "Persp" },
                                icon::CARET_DOWN
                            ))
                            .size(13.0),
                            |ui| {
                                ui.set_min_width(190.0);
                                for (name, f) in [
                                    ("Front", OrbitCamera::set_view_front as fn(&mut OrbitCamera)),
                                    ("Right", OrbitCamera::set_view_right),
                                    ("Top", OrbitCamera::set_view_top),
                                ] {
                                    if ui.button(name).clicked() {
                                        f(camera);
                                        ui.close_menu();
                                    }
                                }
                                ui.separator();
                                toggle(ui, &mut camera.orthographic, "Orthographic");
                                ui.separator();
                                if ui.add(egui::Button::new("Frame selection").shortcut_text("F")).clicked() {
                                    frame_selection(scene, camera);
                                    ui.close_menu();
                                }
                                if ui.button("Frame all").clicked() {
                                    frame_all(scene, camera);
                                    ui.close_menu();
                                }
                            },
                        )
                        .response
                        .on_hover_text("Camera views and projection");
                    });
                });
        });
}

/// Orientation gizmo; click an axis to look along it.
fn viewport_axis_gizmo(ctx: &egui::Context, rect: egui::Rect, camera: &mut OrbitCamera) {
    let radius = 34.0;
    let center = rect.right_bottom() + egui::vec2(-radius - 22.0, -radius - 22.0);
    let area = egui::Rect::from_center_size(center, egui::vec2(radius * 2.0 + 24.0, radius * 2.0 + 24.0));
    egui::Area::new("axis_gizmo".into())
        .fixed_pos(area.min)
        .show(ctx, |ui| {
            let (r, bg) = ui.allocate_exact_size(area.size(), egui::Sense::hover());
            let p = ui.painter();
            if bg.hovered() {
                p.circle_filled(center, radius + 12.0, OVERLAY);
            }
            let view = camera.view_matrix();
            let mut axes: Vec<(Vec3, Color32, &str, f32, f32, f32)> = Vec::new();
            for (dir, color, label, yaw, pitch) in [
                (Vec3::X, Color32::from_rgb(236, 92, 104), "X", std::f32::consts::FRAC_PI_2, 0.0),
                (Vec3::Y, Color32::from_rgb(132, 214, 96), "Y", 0.0, std::f32::consts::FRAC_PI_2 - 0.021),
                (Vec3::Z, Color32::from_rgb(92, 150, 240), "Z", 0.0, 0.0),
                (-Vec3::X, Color32::from_rgb(236, 92, 104), "", -std::f32::consts::FRAC_PI_2, 0.0),
                (-Vec3::Y, Color32::from_rgb(132, 214, 96), "", 0.0, -(std::f32::consts::FRAC_PI_2 - 0.021)),
                (-Vec3::Z, Color32::from_rgb(92, 150, 240), "", std::f32::consts::PI, 0.0),
            ] {
                let v = view.transform_vector3(dir);
                axes.push((v, color, label, yaw, pitch, v.z));
            }
            // Far axes first so near ones draw on top.
            axes.sort_by(|a, b| a.5.total_cmp(&b.5));
            let mut clicked = None;
            for (i, (v, color, label, yaw, pitch, _)) in axes.iter().enumerate() {
                let tip = center + egui::vec2(v.x, -v.y) * radius;
                let positive = !label.is_empty();
                if positive {
                    p.line_segment([center, tip], egui::Stroke::new(2.0_f32, *color));
                }
                let knob = if positive { 9.0 } else { 6.5 };
                let resp = ui.interact(
                    egui::Rect::from_center_size(tip, egui::vec2(knob * 2.0, knob * 2.0)),
                    ui.id().with(("axis", i)),
                    egui::Sense::click(),
                );
                let fill = if positive { *color } else { color.gamma_multiply(0.35) };
                p.circle(tip, knob, fill, egui::Stroke::new(if resp.hovered() { 1.5_f32 } else { 0.0 }, Color32::WHITE));
                if positive {
                    p.text(tip, egui::Align2::CENTER_CENTER, *label, medium_font(11.0), Color32::from_rgb(16, 18, 24));
                }
                if resp.clicked() {
                    clicked = Some((*yaw, *pitch));
                }
                resp.on_hover_text("Look along this axis");
            }
            if let Some((yaw, pitch)) = clicked {
                camera.yaw = yaw;
                camera.pitch = pitch;
            }
            let _ = r;
        });
}

#[derive(Clone, Copy)]
enum Cmd {
    Mode(SmartSelectMode),
    Measure,
    Fly,
    FrameSelection,
    FrameAll,
    ViewFront,
    ViewRight,
    ViewTop,
    Ortho,
    Shading(ViewportShading),
    Edges,
    Grid,
    Ghost,
    Section,
    Undo,
    Redo,
    Save,
    SaveAs,
    Open,
    New,
    Recover,
    Export,
    SendDrafts,
    AutoSend,
    Comment,
    CommentsTab,
    RevisionsTab,
    Library,
    BackToLive,
    ClearSelection,
    Shortcuts,
}

#[allow(clippy::too_many_arguments)]
fn draw_command_palette(
    ctx: &egui::Context,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    renderer: &mut Renderer3D,
    history: &UndoHistory,
    state: &mut UiState,
    comparing: bool,
    in_progress: bool,
) {
    use serde_json::json;
    if !state.palette_open {
        return;
    }
    let drafts = scene.draft_count();
    let editable = !comparing && !in_progress;
    let mut commands: Vec<(&str, String, &str, bool, Cmd)> = vec![
        (icon::CUBE, "Select: Part".into(), "1", !comparing, Cmd::Mode(SmartSelectMode::SubPart)),
        (icon::POLYGON, "Select: Surface".into(), "2", !comparing, Cmd::Mode(SmartSelectMode::SmartSurface)),
        (icon::STACK, "Select: Layer (whole object)".into(), "3", !comparing, Cmd::Mode(SmartSelectMode::WholeLayer)),
        (icon::SELECTION, "Select: Box".into(), "4", !comparing, Cmd::Mode(SmartSelectMode::Box)),
        (icon::TRIANGLE, "Select: Face (single polygons)".into(), "5", !comparing, Cmd::Mode(SmartSelectMode::Face)),
        (icon::BOUNDING_BOX, "Select: 3D box (volume)".into(), "6", !comparing, Cmd::Mode(SmartSelectMode::Volume)),
        (icon::X_CIRCLE, "Clear selection".into(), "Esc", scene.has_any_selection(), Cmd::ClearSelection),
        (icon::CHAT_CIRCLE_TEXT, "Comment on selection".into(), "", !comparing, Cmd::Comment),
        (icon::PAPER_PLANE_TILT, format!("Send {} to the agent", plural(drafts, "draft", "drafts")), "", drafts > 0, Cmd::SendDrafts),
        (icon::LIGHTNING, format!("Send comments immediately: {}", if state.auto_send { "on → off" } else { "off → on" }), "", true, Cmd::AutoSend),
        (icon::CHAT_TEXT, "Show comments".into(), "", true, Cmd::CommentsTab),
        (icon::GIT_DIFF, "Show revisions".into(), "", true, Cmd::RevisionsTab),
        (icon::BOOKS, "Open asset library".into(), "", true, Cmd::Library),
        (icon::ARROW_LEFT, "Back to live scene".into(), "", comparing, Cmd::BackToLive),
        (icon::RULER, "Measure".into(), "", !comparing, Cmd::Measure),
        (icon::AIRPLANE_TILT, "Fly mode".into(), "`", true, Cmd::Fly),
        (icon::FRAME_CORNERS, "Frame selection".into(), "F", true, Cmd::FrameSelection),
        (icon::CORNERS_OUT, "Frame all".into(), "", true, Cmd::FrameAll),
        (icon::CUBE, "View: Front".into(), "", true, Cmd::ViewFront),
        (icon::CUBE, "View: Right".into(), "", true, Cmd::ViewRight),
        (icon::CUBE, "View: Top".into(), "", true, Cmd::ViewTop),
        (icon::PERSPECTIVE, "Toggle orthographic camera".into(), "", true, Cmd::Ortho),
        (icon::SUN, "Shading: Lit".into(), "", true, Cmd::Shading(ViewportShading::StudioLit)),
        (icon::SPHERE, "Shading: Clay".into(), "", true, Cmd::Shading(ViewportShading::SculptClay)),
        (icon::CUBE_TRANSPARENT, "Shading: Wireframe".into(), "", true, Cmd::Shading(ViewportShading::WireframeOnly)),
        (icon::POLYGON, "Toggle edges".into(), "", true, Cmd::Edges),
        (icon::GRID_FOUR, "Toggle grid and shadow".into(), "", true, Cmd::Grid),
        (icon::GHOST, "Toggle ghosting".into(), "", true, Cmd::Ghost),
        (icon::KNIFE, "Toggle section plane".into(), "", true, Cmd::Section),
        (icon::ARROW_COUNTER_CLOCKWISE, "Undo".into(), "Ctrl+Z", history.can_undo() && editable, Cmd::Undo),
        (icon::ARROW_CLOCKWISE, "Redo".into(), "Ctrl+Shift+Z", history.can_redo() && editable, Cmd::Redo),
        (icon::FLOPPY_DISK, "Save project".into(), "Ctrl+S", !comparing, Cmd::Save),
        (icon::FLOPPY_DISK, "Save project as…".into(), "", true, Cmd::SaveAs),
        (icon::FOLDER_OPEN, "Open project…".into(), "", true, Cmd::Open),
        (icon::FILE_PLUS, "New project…".into(), "Ctrl+N", !comparing, Cmd::New),
        (icon::CLOCK_COUNTER_CLOCKWISE, "Recover autosave…".into(), "", true, Cmd::Recover),
        (icon::EXPORT, "Export model…".into(), "", true, Cmd::Export),
        (icon::KEYBOARD, "Keyboard shortcuts".into(), "F1", true, Cmd::Shortcuts),
    ];
    let query = state.palette_query.to_lowercase();
    commands.retain(|(_, label, _, enabled, _)| {
        *enabled && query.split_whitespace().all(|w| label.to_lowercase().contains(w))
    });
    let count = commands.len();
    if count > 0 {
        state.palette_index = state.palette_index.min(count - 1);
    }
    let (up, down, enter, escape) = ctx.input_mut(|i| {
        (
            i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
            i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
            i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
            i.consume_key(egui::Modifiers::NONE, egui::Key::Escape),
        )
    });
    if escape {
        state.palette_open = false;
        return;
    }
    if down && count > 0 {
        state.palette_index = (state.palette_index + 1) % count;
    }
    if up && count > 0 {
        state.palette_index = (state.palette_index + count - 1) % count;
    }
    let mut run = enter.then(|| commands.get(state.palette_index).map(|c| c.4)).flatten();

    let screen = ctx.screen_rect();
    // Dim the app behind the palette; clicking outside closes it.
    let backdrop = egui::Area::new("palette_backdrop".into())
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let (r, resp) = ui.allocate_exact_size(screen.size(), egui::Sense::click());
            ui.painter().rect_filled(r, 0, Color32::from_black_alpha(110));
            resp.clicked()
        })
        .inner;
    egui::Area::new("command_palette".into())
        .order(egui::Order::Tooltip)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(screen.center().x, screen.top() + 90.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(CARD)
                .stroke(egui::Stroke::new(1.0_f32, Color32::from_rgb(54, 66, 84)))
                .corner_radius(14)
                .shadow(ctx.style().visuals.window_shadow)
                .inner_margin(8)
                .show(ui, |ui| {
                    ui.set_width(560.0);
                    ui.horizontal(|ui| {
                        ui.add_space(6.0);
                        ui.label(RichText::new(icon::MAGNIFYING_GLASS).size(17.0).color(SUBTLE));
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut state.palette_query)
                                .hint_text(hint("Type a command…"))
                                .frame(false)
                                .font(FontId::proportional(16.0))
                                .desired_width(f32::INFINITY)
                                .margin(egui::vec2(4.0, 8.0)),
                        );
                        r.request_focus();
                        if r.changed() {
                            state.palette_index = 0;
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 1.0;
                        if commands.is_empty() {
                            ui.add_space(8.0);
                            ui.label(small_muted("  No matching commands"));
                            ui.add_space(8.0);
                        }
                        for (i, (glyph, label, keys, _, cmd)) in commands.iter().enumerate() {
                            let active = i == state.palette_index;
                            let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 34.0), egui::Sense::click());
                            if resp.hovered() && ui.input(|i| i.pointer.delta().length_sq() > 0.0) {
                                state.palette_index = i;
                            }
                            let p = ui.painter();
                            if active {
                                p.rect_filled(rect, 8, SELECTED_ROW);
                                if up || down {
                                    ui.scroll_to_rect(rect, None);
                                }
                            }
                            p.text(rect.left_center() + egui::vec2(14.0, 0.0), egui::Align2::LEFT_CENTER, *glyph, FontId::proportional(16.0), if active { ACCENT } else { MUTED });
                            p.text(rect.left_center() + egui::vec2(42.0, 0.0), egui::Align2::LEFT_CENTER, label, FontId::proportional(14.5), TEXT);
                            if !keys.is_empty() {
                                p.text(rect.right_center() - egui::vec2(12.0, 0.0), egui::Align2::RIGHT_CENTER, *keys, medium_font(12.0), SUBTLE);
                            }
                            if resp.clicked() {
                                run = Some(*cmd);
                            }
                        }
                    });
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.add_space(8.0);
                        ui.label(small_muted("↑↓ move   ·   Enter run   ·   Esc close"));
                    });
                });
        });
    if backdrop && run.is_none() {
        state.palette_open = false;
    }
    let Some(cmd) = run else { return };
    state.palette_open = false;
    match cmd {
        Cmd::Mode(m) => {
            if m == SmartSelectMode::Box && state.select_mode != SmartSelectMode::Box {
                state.prev_select_mode = state.select_mode;
            }
            state.select_mode = m;
            state.measure_active = false;
            state.hover_preview_seed_face = None;
        }
        Cmd::Measure => state.measure_active = !state.measure_active,
        Cmd::Fly => state.fly_toggle_requested = true,
        Cmd::FrameSelection => frame_selection(scene, camera),
        Cmd::FrameAll => frame_all(scene, camera),
        Cmd::ViewFront => camera.set_view_front(),
        Cmd::ViewRight => camera.set_view_right(),
        Cmd::ViewTop => camera.set_view_top(),
        Cmd::Ortho => camera.orthographic = !camera.orthographic,
        Cmd::Shading(s) => renderer.shading = s,
        Cmd::Edges => renderer.show_wireframe_overlay = !renderer.show_wireframe_overlay,
        Cmd::Grid => renderer.show_grid = !renderer.show_grid,
        Cmd::Ghost => renderer.ghost_unselected = !renderer.ghost_unselected,
        Cmd::Section => {
            state.clip_enabled = !state.clip_enabled;
            state.scene_tab = 1;
        }
        Cmd::Undo => action(state, "undo", json!({})),
        Cmd::Redo => action(state, "redo", json!({})),
        Cmd::Save => {
            if scene.project_path.is_some() {
                action(state, "save_project", json!({}));
            } else {
                state.project_dialog = Some("save".into());
            }
        }
        Cmd::SaveAs => state.project_dialog = Some("save".into()),
        Cmd::Open => state.project_dialog = Some("open".into()),
        Cmd::New => state.confirm_new_project = true,
        Cmd::Recover => state.project_dialog = Some("recover".into()),
        Cmd::Export => state.show_io_modal = true,
        Cmd::SendDrafts => {
            let n = scene.send_drafts();
            state.scene_dirty_for_disk = true;
            state.status_message = format!("Sent {} to the agent", plural(n, "comment", "comments"));
        }
        Cmd::AutoSend => state.auto_send = !state.auto_send,
        Cmd::Comment => {
            state.review_tab = 0;
            state.focus_composer = true;
        }
        Cmd::CommentsTab => state.review_tab = 0,
        Cmd::RevisionsTab => state.review_tab = 1,
        Cmd::Library => {
            state.library.open = true;
            state.library.scanned = None;
        }
        Cmd::BackToLive => {
            scene.preview_revision = None;
            state.compare_split = None;
            scene.invalidate_mesh_caches();
            renderer.invalidate_cache();
        }
        Cmd::ClearSelection => {
            scene.clear_all_selections();
            scene.selected_object = None;
            state.scene_dirty_for_disk = true;
        }
        Cmd::Shortcuts => state.show_shortcuts = true,
    }
}

fn draw_dialogs(ctx: &egui::Context, scene: &Scene, state: &mut UiState) {
    use serde_json::json;
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    if state.confirm_new_project {
        draw_new_project_dialog(ctx, scene, state, escape);
    }

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
                let path = expand_home(&state.io_file_path);
                if ext == "glb" {
                    // The game-ready format: Godot and Blender both import it.
                    let godot = crate::gltf::godot_resource_path(std::path::Path::new(&path));
                    if let Some((_, res)) = &godot {
                        ui.label(small_muted(format!("Godot project file · {res}")));
                    }
                    load_export_choices(&mut state.library.export, &crate::library::root());
                    export_options(ui, &mut state.library.export, true, "scene_export_collision");
                    ui.label(small_muted(
                        "Collision shapes are read by Godot. Save inside a Godot project and it imports the file by itself.",
                    ));
                }
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                ui.horizontal(|ui| {
                    if primary(ui, "Export scene", valid) || (enter && valid) {
                        let export = &state.library.export;
                        let mut args = json!({ "path": path });
                        if ext == "glb" {
                            args["merge"] = json!(!export.separate);
                            args["collision"] = json!(export.collision.name());
                            args["max_triangles"] = json!(export.max_triangles().unwrap_or(0));
                        }
                        action(state, "export_model", args);
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
                            ("1  2  3", "Part · Surface · Layer"),
                            ("4  5  6", "Box · Face · 3D box"),
                            ("3D box", "Click a surface to place it, drag handles to resize"),
                            ("I", "Invert selection"),
                            ("Ctrl + C", "Toggle the box selector"),
                            ("Box: drag", "Select visible faces in a rectangle"),
                            ("Tab", "Cycle selection mode"),
                            ("Shift + click", "Add to selection"),
                            ("Ctrl + click", "Remove from selection"),
                            ("A", "Select all faces of the active part"),
                            ("]  /  [", "Grow / shrink selection"),
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
                            ("`  (backtick)", "Enter / leave fly mode"),
                        ][..],
                    ),
                    (
                        "Fly mode",
                        &[
                            ("W A S D  /  arrows", "Move"),
                            ("Q  /  E  or  Space", "Down / up"),
                            ("Mouse", "Look around"),
                            ("Shift  /  Ctrl", "Fast / slow"),
                            ("Scroll", "Change fly speed"),
                            ("Click", "Select at the crosshair"),
                            ("Esc  /  `  /  right-click", "Leave fly mode"),
                        ][..],
                    ),
                    (
                        "Project",
                        &[
                            ("Ctrl + K", "Command palette"),
                            ("Ctrl + N", "New project"),
                            ("Ctrl + S", "Save project"),
                            ("Ctrl + Z", "Undo"),
                            ("Ctrl + Shift + Z", "Redo"),
                            ("Ctrl + Enter", "Add comment or post reply"),
                            ("Drop images", "Attach references to your next comment"),
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

/// What starting a new project would do to the current one, as shown in the confirmation.
fn new_project_notice(scene: &Scene) -> (bool, String) {
    let pending = scene.revisions.iter().any(|r| {
        matches!(
            r.status,
            RevisionStatus::ReadyForReview | RevisionStatus::InProgress
        )
    });
    if pending {
        return (
            false,
            "Accept, revert or cancel the pending revision before starting a new project.".into(),
        );
    }
    let text = match &scene.project_path {
        Some(path) => format!(
            "\u{201c}{}\u{201d} stays saved at {path}; changes since your last save are in its autosave. \
             Undo (Ctrl+Z) brings it back.",
            scene.project_name
        ),
        None if scene.objects.is_empty() && scene.comments.is_empty() => {
            "The current project is empty, so nothing will be lost.".into()
        }
        None => format!(
            "\u{201c}{}\u{201d} has never been saved. Save it first to keep it as a file; \
             otherwise only Undo (Ctrl+Z) can bring it back, until you close the app.",
            scene.project_name
        ),
    };
    (true, text)
}

fn draw_new_project_dialog(ctx: &egui::Context, scene: &Scene, state: &mut UiState, escape: bool) {
    let (allowed, notice) = new_project_notice(scene);
    let unsaved =
        scene.project_path.is_none() && !(scene.objects.is_empty() && scene.comments.is_empty());
    let mut open = true;
    egui::Window::new("New project")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.label(small_muted(
                "Starts a blank canvas: no objects, feedback, revisions, saved selections or views.",
            ));
            ui.add_space(6.0);
            eyebrow(ui, "Project title");
            let r = text_field(ui, &mut state.new_project_name, "Untitled project", f32::INFINITY);
            if !r.has_focus() && ui.memory(|m| m.focused().is_none()) {
                r.request_focus();
            }
            ui.add_space(4.0);
            let color = if allowed && !unsaved { MUTED } else { WARN };
            ui.label(RichText::new(notice).size(12.0).color(color));
            ui.add_space(6.0);
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if primary(ui, "Start new project", allowed) || (enter && allowed) {
                    action(
                        state,
                        "new_project",
                        serde_json::json!({"name": state.new_project_name.trim()}),
                    );
                    state.new_project_name.clear();
                    state.confirm_new_project = false;
                }
                if unsaved && allowed && ui.button("Save first…").clicked() {
                    state.confirm_new_project = false;
                    state.project_dialog = Some("save".into());
                }
                if ui.button("Cancel").clicked() {
                    state.confirm_new_project = false;
                }
            });
        });
    if !open || escape {
        state.confirm_new_project = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_refreshes_when_the_same_part_gets_a_new_anchor() {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Part", crate::mesh::Mesh::cube(2.0), [0.5; 3]);
        scene.selected_object = Some(0);
        let id = scene.objects[0].id;

        let mut state = UiState::default();
        assert_eq!(
            current_target_sections(&scene, &mut state)[0].world_anchor,
            Vec3::ZERO
        );

        scene.pending_anchor = Some((id, Vec3::X, Vec3::X, Vec3::X));
        assert_eq!(
            current_target_sections(&scene, &mut state)[0].world_anchor,
            Vec3::X
        );
    }
}

#[cfg(test)]
mod library_view_tests {
    use super::*;
    use crate::mesh::Mesh;

    /// Lays out the library view and the Library tab for a few frames: empty, with assets,
    /// with the save form open, filtered and searched.
    #[test]
    fn library_view_lays_out() {
        let _guard = crate::library::TEST_ROOT.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join(format!("rust_modeler_library_view_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        unsafe { std::env::set_var("RUST_MODELER_LIBRARY", &root) };

        let mut scene = Scene::empty();
        scene.project_name = "Oak bookshelf".into();
        scene.add_mesh_object("Bookshelf/Frame", Mesh::cube(1.0), [0.6, 0.4, 0.2]);
        scene.add_mesh_object("Bookshelf/Shelf", Mesh::cube(0.5), [0.6, 0.4, 0.2]);
        let camera = OrbitCamera::default();
        let mut state = UiState::default();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let frame = |scene: &Scene, state: &mut UiState| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::SidePanel::left("scene_panel").show(ctx, |ui| {
                    scene_library_tab(ui, scene, &camera, state, false);
                });
                draw_library_view(ctx, scene, &camera, state, false);
            });
        };

        state.library.open = true;
        frame(&scene, &mut state);
        assert!(state.library.sections.is_empty());

        // The whole model is the default: one asset named after the project.
        library_open_save(&scene, &mut state);
        assert_eq!(state.library.asset_name, "Oak bookshelf");
        assert!(!state.library.save_selection);
        frame(&scene, &mut state);

        let whole = crate::library::visible_parts(&scene);
        for (sub, name) in [("Bookshelves", "Oak bookshelf"), ("Bookshelves", "Pine bookshelf"), ("Chairs", "Stool")] {
            crate::library::save_asset(
                &root,
                &scene,
                &whole,
                crate::library::SaveOptions {
                    section: "Furniture",
                    subsection: sub,
                    name,
                    description: "A test asset",
                    tags: vec!["wood".into()],
                    overwrite: false,
                },
            )
            .unwrap();
        }
        crate::library::create_section(&root, "Props", Some("Empty")).unwrap();
        state.library.scanned = None;
        state.library.save_open = false;
        frame(&scene, &mut state);
        assert_eq!(state.library.sections.len(), 2);
        assert_eq!(state.library.sections[0].subsections[0].assets[0].parts, 2);

        state.library.filter = Some(("Furniture".into(), Some("Chairs".into())));
        state.library.search = "stool".into();
        frame(&scene, &mut state);
        state.library.filter = Some(("Props".into(), None));
        state.library.new_folder = Some((Some("Props".into()), "Lamps".into()));
        frame(&scene, &mut state);
        // The export form, for everything shown and for one asset.
        state.library.new_folder = None;
        state.library.filter = None;
        state.library.search.clear();
        library_open_export(&mut state, &root, None);
        state.library.export.directory = "~/game/assets".into();
        frame(&scene, &mut state);
        state.library.export.blender = true;
        state.library.export_scope = Some(Some("Furniture/Chairs/Stool".into()));
        frame(&scene, &mut state);
        assert_eq!(state.library.export.max_triangles(), None);
        state.library.export.budget = "20,000".into();
        assert_eq!(state.library.export.max_triangles(), Some(20_000));
        state.library.export_scope = None;

        // A filter whose folder is gone falls back to everything.
        state.library.filter = Some(("Gone".into(), None));
        frame(&scene, &mut state);
        assert!(state.library.filter.is_none());
        assert!(state.library.open);

        unsafe { std::env::remove_var("RUST_MODELER_LIBRARY") };
        let _ = std::fs::remove_dir_all(&root);
    }
}
