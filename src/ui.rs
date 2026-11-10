use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::history::UndoHistory;
use crate::io;
use crate::mesh::EditElementMode;
use crate::renderer::{Renderer3D, ViewportShading};
use crate::scene::Scene;
use crate::voxel::{SculptSettings, VoxelBrushOp};
use glam::{Vec2, Vec3};
use std::collections::HashSet;
use std::path::Path;

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

pub struct UiState {
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
            extrude_distance: 0.45,
            inset_factor: 0.28,
            inset_depth: 0.0,
            sculpt: SculptSettings::default(),
            voxel_op: VoxelBrushOp::AddSphere,
            voxel_brush_radius: 0.45,
            voxel_paint_color: [0.88, 0.55, 0.35],
            voxel_symmetry_x: true,
            io_file_path: "model_export.obj".to_string(),
            status_message: "Multi-Select Active — Click multiple parts/surfaces across the model to select them for a single comment (Click again to toggle off)".to_string(),
            show_io_modal: false,
            select_mode: SmartSelectMode::SubPart,
            multi_select: true,
            crease_angle_deg: 32.0,
            selection_radius: 0.45,
            hover_preview_obj_id: None,
            hover_preview_seed_face: None,
            hover_preview_faces: HashSet::new(),
            comment_draft: String::new(),
            show_resolved_comments: true,
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

#[allow(clippy::too_many_arguments)]
pub fn draw_ui(
    ctx: &egui::Context,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    _gizmo: &mut GizmoState,
    renderer: &mut Renderer3D,
    _history: &mut UndoHistory,
    ui_state: &mut UiState,
    hover_world_pt: Option<Vec3>,
) {
    // =========================================================================
    // TOP BAR: SELECTION MODE, SECTION RADIUS, SHADING & CAMERA PRESETS
    // =========================================================================
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.heading("🤖 Rust AI 3D Studio");
            ui.separator();

            // Export Menu (Read-only / Export workflow)
            ui.menu_button("📁 Export", |ui| {
                if ui.button("💾 Export Model To Path...").clicked() {
                    ui_state.show_io_modal = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("📤 Quick Export scene.obj").clicked() {
                    match io::export_obj_file(scene, Path::new("scene.obj")) {
                        Ok(()) => {
                            ui_state.status_message = "Exported scene to scene.obj".to_string()
                        }
                        Err(e) => ui_state.status_message = e,
                    }
                    ui.close_menu();
                }
                if ui.button("📤 Quick Export scene.stl").clicked() {
                    match io::export_stl_file(scene, Path::new("scene.stl")) {
                        Ok(()) => {
                            ui_state.status_message = "Exported scene to scene.stl".to_string()
                        }
                        Err(e) => ui_state.status_message = e,
                    }
                    ui.close_menu();
                }
                if ui.button("📤 Quick Export scene.glb (glTF 2.0)").clicked() {
                    match io::export_glb_file(scene, Path::new("scene.glb")) {
                        Ok(()) => {
                            ui_state.status_message = "Exported scene to scene.glb".to_string()
                        }
                        Err(e) => ui_state.status_message = e,
                    }
                    ui.close_menu();
                }
            });

            ui.separator();

            // Smart Section Selection Tool for AI Comments
            ui.label("Select Mode:");
            if ui
                .selectable_label(
                    ui_state.select_mode == SmartSelectMode::SubPart,
                    "🧩 Sub-Part (1)",
                )
                .on_hover_text(
                    "1-Click selects the whole connected 3D component/piece under cursor (e.g., a single tree, gear, column, crystal, step, or airship).\n• Shift+Click: Add another sub-part\n• Ctrl+Click: Subtract sub-part",
                )
                .clicked()
            {
                ui_state.select_mode = SmartSelectMode::SubPart;
                ui_state.hover_preview_seed_face = None;
            }
            if ui
                .selectable_label(
                    ui_state.select_mode == SmartSelectMode::SmartSurface,
                    "📐 Smart Surface (2)",
                )
                .on_hover_text(
                    "1-Click flood-fills a contiguous wall, roof slope, step top, or smooth surface bounded by sharp edges (Crease Angle).\n• Shift+Click: Add surface patch\n• Ctrl+Click: Subtract surface patch",
                )
                .clicked()
            {
                ui_state.select_mode = SmartSelectMode::SmartSurface;
                ui_state.hover_preview_seed_face = None;
            }
            if ui
                .selectable_label(
                    ui_state.select_mode == SmartSelectMode::WholeLayer,
                    "📦 Whole Layer (3)",
                )
                .on_hover_text("1-Click selects the entire object layer")
                .clicked()
            {
                ui_state.select_mode = SmartSelectMode::WholeLayer;
                ui_state.hover_preview_seed_face = None;
            }
            if ui
                .selectable_label(
                    ui_state.select_mode == SmartSelectMode::Brush,
                    "🖌 Paint Brush (4)",
                )
                .on_hover_text(
                    "Click & drag to paint faces within the Brush Radius (Shift+Drag to add, Ctrl+Drag to erase)",
                )
                .clicked()
            {
                ui_state.select_mode = SmartSelectMode::Brush;
                ui_state.hover_preview_seed_face = None;
            }

            if ui_state.select_mode == SmartSelectMode::SmartSurface {
                ui.separator();
                ui.label("Crease:");
                if ui
                    .add(
                        egui::Slider::new(&mut ui_state.crease_angle_deg, 5.0..=85.0)
                            .suffix("°")
                            .fixed_decimals(0),
                    )
                    .on_hover_text(
                        "Max angle between adjacent faces before flood-fill stops at a sharp edge",
                    )
                    .changed()
                {
                    ui_state.hover_preview_seed_face = None;
                }
            } else if ui_state.select_mode == SmartSelectMode::Brush {
                ui.separator();
                ui.label("Radius:");
                if ui
                    .add(
                        egui::Slider::new(&mut ui_state.selection_radius, 0.05..=2.50)
                            .fixed_decimals(2)
                            .show_value(true),
                    )
                    .changed()
                {
                    ui_state.sculpt.radius = ui_state.selection_radius;
                }
            }

            ui.separator();
            ui.checkbox(&mut ui_state.multi_select, "➕ Multi-Select")
                .on_hover_text(
                    "When checked, clicking multiple sub-parts or surfaces across any object adds them to the selection (clicking an already-selected sub-part unselects it) so you can attach a single comment to multiple parts!",
                );

            if ui
                .button("➕ Grow (])")
                .on_hover_text("Expand current selection to neighboring connected faces (Shortcut: ])")
                .clicked()
            {
                let mut changed = false;
                for obj in &mut scene.objects {
                    if !obj.mesh.selected_faces.is_empty() {
                        changed |= obj.mesh.grow_selection();
                    }
                }
                if changed {
                    update_pending_anchor_from_selection(scene);
                    ui_state.scene_dirty_for_disk = true;
                }
            }
            if ui
                .button("➖ Shrink ([)")
                .on_hover_text("Contract outer boundary ring of selected faces (Shortcut: [)")
                .clicked()
            {
                let mut changed = false;
                for obj in &mut scene.objects {
                    if !obj.mesh.selected_faces.is_empty() {
                        changed |= obj.mesh.shrink_selection();
                    }
                }
                if changed {
                    update_pending_anchor_from_selection(scene);
                    ui_state.scene_dirty_for_disk = true;
                }
            }
            if ui
                .button("🧹 Clear")
                .on_hover_text("Clear all selected sections and pins across the scene (Shortcut: Esc)")
                .clicked()
            {
                scene.clear_all_selections();
                scene.active_comment_id = None;
                ui_state.scene_dirty_for_disk = true;
            }

            ui.separator();

            // Viewport Shading & Camera controls
            ui.label("Shading:");
            ui.selectable_value(&mut renderer.shading, ViewportShading::StudioLit, "Lit");
            ui.selectable_value(&mut renderer.shading, ViewportShading::SculptClay, "Clay");
            ui.selectable_value(
                &mut renderer.shading,
                ViewportShading::WireframeOnly,
                "Wire",
            );
            ui.checkbox(&mut renderer.show_wireframe_overlay, "Edges");
            ui.checkbox(&mut renderer.show_grid, "Grid");

            ui.separator();
            if ui.button("Front").clicked() {
                camera.set_view_front();
            }
            if ui.button("Right").clicked() {
                camera.set_view_right();
            }
            if ui.button("Top").clicked() {
                camera.set_view_top();
            }
            let proj_label = if camera.orthographic { "Ortho" } else { "Persp" };
            if ui.button(proj_label).clicked() {
                camera.orthographic = !camera.orthographic;
            }
            if ui.button("🎯 Focus (F)").clicked() {
                if let Some((_id, wp, _lp, _wn)) = scene.pending_anchor {
                    camera.target = wp;
                } else if let Some(obj) = scene.active_object() {
                    camera.target = obj.transform.translation;
                }
            }
        });
    });

    // =========================================================================
    // EXPORT MODAL WINDOW
    // =========================================================================
    if ui_state.show_io_modal {
        egui::Window::new("📁 Export 3D Model (.obj, .stl, .glb)")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Destination file path:");
                ui.text_edit_singleline(&mut ui_state.io_file_path);
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    if ui.button("📤 Export Scene to File").clicked() {
                        let path = Path::new(&ui_state.io_file_path);
                        let ext = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("obj")
                            .to_lowercase();
                        let res = match ext.as_str() {
                            "stl" => io::export_stl_file(scene, path),
                            "glb" | "gltf" => io::export_glb_file(scene, path),
                            _ => io::export_obj_file(scene, path),
                        };
                        match res {
                            Ok(()) => {
                                ui_state.status_message =
                                    format!("Exported scene to '{}'", ui_state.io_file_path);
                                ui_state.show_io_modal = false;
                            }
                            Err(e) => ui_state.status_message = e,
                        }
                    }

                    if ui.button("Close").clicked() {
                        ui_state.show_io_modal = false;
                    }
                });
            });
    }

    // =========================================================================
    // BOTTOM STATUS BAR
    // =========================================================================
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let unresolved_count = scene.comments.iter().filter(|c| !c.resolved).count();
            ui.label(
                egui::RichText::new(format!(
                    "🤖 AI-Driven Mode | 💬 {unresolved_count} Open Comment(s)"
                ))
                .strong()
                .color(egui::Color32::from_rgb(110, 215, 255)),
            );
            ui.separator();
            if let Some(obj) = scene.active_object() {
                ui.label(format!(
                    "Selected: '{}' (ID {}) | Highlighted Faces: {} / {}",
                    obj.name,
                    obj.id,
                    obj.mesh.selected_faces.len(),
                    obj.mesh.faces.len(),
                ));
            } else {
                ui.label("Click any part of the 3D model to select a section and leave a comment");
            }
            if let Some(pt) = hover_world_pt {
                ui.separator();
                ui.label(format!("Cursor 3D: ({:.2}, {:.2}, {:.2})", pt.x, pt.y, pt.z));
            }
            ui.separator();
            ui.label(&ui_state.status_message);
        });
    });

    // =========================================================================
    // LEFT PANEL: SCENE PARTS & VISIBILITY OUTLINER (READ-ONLY)
    // =========================================================================
    egui::SidePanel::left("left_outliner_panel")
        .default_width(245.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("🗂 Scene Parts");
            ui.label(
                egui::RichText::new(
                    "Click a part in the list or directly in the 3D viewport to inspect or select it.",
                )
                .small()
                .weak(),
            );
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if ui.small_button("👁 Show All").clicked() {
                    for obj in &mut scene.objects {
                        obj.visible = true;
                    }
                }
                if ui.small_button("🔍 Isolate Selected").clicked() {
                    if let Some(sel_idx) = scene.selected_object {
                        for (i, obj) in scene.objects.iter_mut().enumerate() {
                            obj.visible = i == sel_idx;
                        }
                    }
                }
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("outliner_scroll")
                .show(ui, |ui| {
                    let mut clicked_idx = None;
                    for (i, obj) in scene.objects.iter_mut().enumerate() {
                        let has_faces = !obj.mesh.selected_faces.is_empty();
                        let is_sel = scene.selected_object == Some(i) || has_faces;
                        let open_comments_on_obj = scene
                            .comments
                            .iter()
                            .filter(|c| {
                                !c.resolved
                                    && (c.object_id == obj.id
                                        || c.targets.iter().any(|t| t.object_id == obj.id))
                            })
                            .count();

                        ui.horizontal(|ui| {
                            let vis_icon = if obj.visible { "👁" } else { "🚫" };
                            if ui
                                .small_button(vis_icon)
                                .on_hover_text("Toggle part visibility")
                                .clicked()
                            {
                                obj.visible = !obj.visible;
                            }
                            let mut label = obj.name.clone();
                            if has_faces {
                                label = format!("✦ {} ({})", label, obj.mesh.selected_faces.len());
                            }
                            if open_comments_on_obj > 0 {
                                label = format!("{} 💬{}", label, open_comments_on_obj);
                            }
                            if ui.selectable_label(is_sel, label).clicked() {
                                clicked_idx = Some(i);
                            }
                        });
                    }
                    if let Some(idx) = clicked_idx {
                        scene.selected_object = Some(idx);
                        if let Some(obj) = scene.objects.get(idx) {
                            let (min_b, max_b) = obj.mesh.bounds();
                            let lc = (min_b + max_b) * 0.5;
                            let wc = obj.transform.matrix().transform_point3(lc);
                            scene.pending_anchor = Some((obj.id, wc, lc, Vec3::Y));
                        }
                        ui_state.scene_dirty_for_disk = true;
                    }
                });
        });

    // =========================================================================
    // RIGHT PANEL: SPATIAL SELECTION & AI AGENT COMMENTS WORKFLOW
    // =========================================================================
    egui::SidePanel::right("right_comments_panel")
        .default_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("💬 AI Agent Comments");
            ui.label(
                egui::RichText::new(
                    "Select any section of the 3D model by clicking or brushing over faces, then leave a comment for the AI agent.",
                )
                .small()
                .weak(),
            );
            ui.separator();

            // -----------------------------------------------------------------
            // 1. CURRENT SELECTION & NEW COMMENT COMPOSER (MULTI-SECTION OR GENERAL)
            // -----------------------------------------------------------------
            egui::CollapsingHeader::new("💬 Leave a Comment for AI Agent")
                .default_open(true)
                .show(ui, |ui| {
                    let target_sections =
                        scene.collect_current_target_sections(ui_state.selection_radius);
                    let has_selected_obj = !target_sections.is_empty();

                    if has_selected_obj {
                        let total_sel_faces: usize =
                            target_sections.iter().map(|t| t.selected_faces.len()).sum();
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Selected Sections ({} part(s), {} faces):",
                                    target_sections.len(),
                                    total_sel_faces
                                ))
                                .strong(),
                            );
                        });

                        let mut remove_obj_id: Option<u64> = None;
                        for t in &target_sections {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "• {} (ID {}) — {} faces",
                                        t.object_name,
                                        t.object_id,
                                        t.selected_faces.len()
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(255, 195, 95)),
                                );
                                if ui
                                    .small_button("✖")
                                    .on_hover_text("Remove this part from the current selection")
                                    .clicked()
                                {
                                    remove_obj_id = Some(t.object_id);
                                }
                            });
                        }

                        if let Some(rid) = remove_obj_id {
                            if let Some(obj) = scene.objects.iter_mut().find(|o| o.id == rid) {
                                obj.mesh.clear_selection();
                            }
                            if scene
                                .objects
                                .iter()
                                .all(|o| o.mesh.selected_faces.is_empty())
                            {
                                scene.selected_object = None;
                                scene.pending_anchor = None;
                            } else {
                                update_pending_anchor_from_selection(scene);
                            }
                            ui_state.scene_dirty_for_disk = true;
                        }

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Refine:").small().weak());
                            if ui
                                .small_button("➕ Grow (])")
                                .on_hover_text("Expand selection to neighboring connected faces")
                                .clicked()
                            {
                                let mut changed = false;
                                for obj in &mut scene.objects {
                                    if !obj.mesh.selected_faces.is_empty() {
                                        changed |= obj.mesh.grow_selection();
                                    }
                                }
                                if changed {
                                    update_pending_anchor_from_selection(scene);
                                    ui_state.scene_dirty_for_disk = true;
                                }
                            }
                            if ui
                                .small_button("➖ Shrink ([)")
                                .on_hover_text("Contract outer ring of selected faces")
                                .clicked()
                            {
                                let mut changed = false;
                                for obj in &mut scene.objects {
                                    if !obj.mesh.selected_faces.is_empty() {
                                        changed |= obj.mesh.shrink_selection();
                                    }
                                }
                                if changed {
                                    update_pending_anchor_from_selection(scene);
                                    ui_state.scene_dirty_for_disk = true;
                                }
                            }
                            if ui
                                .small_button("📦 Whole Active Layer")
                                .on_hover_text("Select all faces of the active object layer")
                                .clicked()
                            {
                                if let Some(active) = scene.active_object_mut() {
                                    active.mesh.select_all(EditElementMode::Face);
                                }
                                update_pending_anchor_from_selection(scene);
                                ui_state.scene_dirty_for_disk = true;
                            }
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Target:").strong());
                            ui.label(
                                egui::RichText::new("🌐 General Scene (Whole Model)")
                                    .color(egui::Color32::from_rgb(175, 145, 255)),
                            );
                        });
                        ui.label(
                            egui::RichText::new(
                                "Tip: You can add a general comment below, or click multiple parts on the 3D model to target specific sections.",
                            )
                            .small()
                            .weak(),
                        );
                    }

                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Instruction / Feedback for AI Agent:").strong());
                    let text_resp = ui.add(
                        egui::TextEdit::multiline(&mut ui_state.comment_draft)
                            .desired_rows(3)
                            .hint_text(
                                "e.g., Make these selected towers taller, or (general) Add a bridge on the south side...",
                            ),
                    );

                    let submit_shortcut = text_resp.has_focus()
                        && ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl);

                    ui.add_space(4.0);
                    let can_submit = !ui_state.comment_draft.trim().is_empty();

                    ui.horizontal_wrapped(|ui| {
                        if has_selected_obj {
                            let btn_lbl = if target_sections.len() > 1 {
                                format!(
                                    "📍 Add Comment ({} Selections)",
                                    target_sections.len()
                                )
                            } else {
                                "📍 Add Section Comment".to_string()
                            };
                            if ui
                                .add_enabled(can_submit, egui::Button::new(btn_lbl))
                                .on_hover_text(
                                    "Attach this comment to all currently selected 3D sections/parts (Ctrl+Enter)",
                                )
                                .clicked()
                                || (can_submit && submit_shortcut)
                            {
                                let text = ui_state.comment_draft.trim().to_string();
                                let rad = ui_state.selection_radius;
                                if let Some(added) = scene.add_comment_for_selection(text, rad) {
                                    let total_f: usize = added
                                        .targets
                                        .iter()
                                        .map(|t| t.selected_faces.len())
                                        .sum();
                                    ui_state.comment_draft.clear();
                                    ui_state.status_message = format!(
                                        "Added Comment #{} across {} selection(s) ({} total faces) — Ready for AI Agent!",
                                        added.id,
                                        added.targets.len().max(1),
                                        total_f
                                    );
                                    ui_state.scene_dirty_for_disk = true;
                                }
                            }
                        }

                        let gen_btn_label = if has_selected_obj {
                            "🌐 Add as General"
                        } else {
                            "🌐 Add General Comment"
                        };
                        if ui
                            .add_enabled(can_submit, egui::Button::new(gen_btn_label))
                            .on_hover_text(
                                "Add a general scene-wide comment not tied to a single 3D surface pin",
                            )
                            .clicked()
                            || (!has_selected_obj && can_submit && submit_shortcut)
                        {
                            let text = ui_state.comment_draft.trim().to_string();
                            let added = scene.add_general_comment(text);
                            ui_state.comment_draft.clear();
                            ui_state.status_message = format!(
                                "Added General Scene Comment #{} — Ready for AI Agent!",
                                added.id
                            );
                            ui_state.scene_dirty_for_disk = true;
                        }

                        if has_selected_obj && ui.button("Clear Selection").clicked() {
                            scene.clear_all_selections();
                            scene.selected_object = None;
                            ui_state.scene_dirty_for_disk = true;
                        }
                    });
                });

            ui.separator();

            // -----------------------------------------------------------------
            // 2. SPATIAL COMMENTS LIST
            // -----------------------------------------------------------------
            ui.horizontal(|ui| {
                let open_cnt = scene.comments.iter().filter(|c| !c.resolved).count();
                let total_cnt = scene.comments.len();
                ui.label(
                    egui::RichText::new(format!("📌 Comments ({open_cnt} open / {total_cnt} total)"))
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut ui_state.show_resolved_comments, "Show Resolved");
                });
            });

            if scene.comments.is_empty() {
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new(
                        "No comments yet. Select a section on the 3D model and add a comment above so the AI agent can see where and what to modify!",
                    )
                    .weak(),
                );
            } else {
                let mut focus_id: Option<u64> = None;
                let mut toggle_resolve_id: Option<u64> = None;
                let mut delete_id: Option<u64> = None;

                egui::ScrollArea::vertical()
                    .id_salt("comments_scroll")
                    .show(ui, |ui| {
                        for c in &scene.comments {
                            if c.resolved && !ui_state.show_resolved_comments {
                                continue;
                            }
                            let is_general = c.is_general || c.object_id == 0;
                            let is_active = scene.active_comment_id == Some(c.id);
                            let border_col = if c.resolved {
                                egui::Color32::from_rgb(75, 110, 85)
                            } else if is_active {
                                egui::Color32::from_rgb(65, 220, 255)
                            } else if is_general {
                                egui::Color32::from_rgb(165, 130, 250)
                            } else {
                                egui::Color32::from_rgb(235, 140, 55)
                            };

                            egui::Frame::group(ui.style())
                                .stroke(egui::Stroke::new(1.5_f32, border_col))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let badge = if c.resolved {
                                            format!("✔ #{}", c.id)
                                        } else if is_general {
                                            format!("🌐 #{}", c.id)
                                        } else {
                                            format!("📍 #{}", c.id)
                                        };
                                        ui.label(
                                            egui::RichText::new(badge)
                                                .strong()
                                                .color(border_col),
                                        );
                                        let title = if is_general {
                                            "General Scene"
                                        } else {
                                            c.object_name.as_str()
                                        };
                                        ui.label(egui::RichText::new(title).strong());
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui
                                                    .small_button("🗑")
                                                    .on_hover_text("Delete comment")
                                                    .clicked()
                                                {
                                                    delete_id = Some(c.id);
                                                }
                                                let res_lbl = if c.resolved {
                                                    "↺ Reopen"
                                                } else {
                                                    "✔ Resolve"
                                                };
                                                if ui.small_button(res_lbl).clicked() {
                                                    toggle_resolve_id = Some(c.id);
                                                }
                                                if !is_general
                                                    && ui
                                                        .small_button("🎯 View")
                                                        .on_hover_text(
                                                            "Highlight this comment's selected faces and focus camera",
                                                        )
                                                        .clicked()
                                                {
                                                    focus_id = Some(c.id);
                                                }
                                            },
                                        );
                                    });

                                    ui.label(&c.comment);

                                    if !is_general {
                                        if c.targets.len() > 1 {
                                            let total_f: usize = c
                                                .targets
                                                .iter()
                                                .map(|t| t.selected_faces.len())
                                                .sum();
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "📍 {} selections ({} total faces)",
                                                    c.targets.len(),
                                                    total_f
                                                ))
                                                .small()
                                                .weak(),
                                            );
                                            for t in &c.targets {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "   • {} (ID {}): {} faces at ({:.1}, {:.1}, {:.1})",
                                                        t.object_name,
                                                        t.object_id,
                                                        t.selected_faces.len(),
                                                        t.world_anchor.x,
                                                        t.world_anchor.y,
                                                        t.world_anchor.z
                                                    ))
                                                    .small()
                                                    .weak(),
                                                );
                                            }
                                        } else {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "📍 ({:.2}, {:.2}, {:.2}) | {} faces | r={:.2}",
                                                    c.world_anchor.x,
                                                    c.world_anchor.y,
                                                    c.world_anchor.z,
                                                    c.selected_faces.len(),
                                                    c.selection_radius
                                                ))
                                                .small()
                                                .weak(),
                                            );
                                        }
                                    }

                                    if let Some(reply) = &c.agent_reply {
                                        ui.add_space(2.0);
                                        ui.label(
                                            egui::RichText::new(format!("🤖 AI: {reply}"))
                                                .small()
                                                .color(egui::Color32::from_rgb(130, 230, 175)),
                                        );
                                    }
                                });
                            ui.add_space(4.0);
                        }
                    });

                if let Some(cid) = focus_id {
                    if let Some(wp) = scene.focus_comment(cid) {
                        camera.target = wp;
                        ui_state.status_message =
                            format!("Focused Comment #{cid} and restored its section selection.");
                        ui_state.scene_dirty_for_disk = true;
                    }
                }
                if let Some(cid) = toggle_resolve_id {
                    if let Some(c) = scene.comments.iter_mut().find(|c| c.id == cid) {
                        c.resolved = !c.resolved;
                        ui_state.scene_dirty_for_disk = true;
                    }
                }
                if let Some(cid) = delete_id {
                    scene.comments.retain(|c| c.id != cid);
                    if scene.active_comment_id == Some(cid) {
                        scene.active_comment_id = None;
                    }
                    ui_state.scene_dirty_for_disk = true;
                }
            }
        });

    // =========================================================================
    // FLOATING 3D VIEWPORT COMMENT PIN BADGES
    // =========================================================================
    let screen_rect = ctx.screen_rect();
    let vp_size = Vec2::new(screen_rect.width(), screen_rect.height());
    let mut clicked_pin_id: Option<u64> = None;

    for c in &scene.comments {
        if c.is_general || c.object_id == 0 || (c.resolved && !ui_state.show_resolved_comments) {
            continue;
        }
        let pin_dir = (c.surface_normal * 0.65 + Vec3::Y * 0.75).normalize_or_zero();
        let stem_len = (c.selection_radius * 1.15).clamp(0.45, 1.4);
        let tip_world = c.world_anchor + pin_dir * stem_len;

        if let Some(screen_pt) = camera.world_to_screen(tip_world, vp_size) {
            // Only draw floating pin badge inside central viewport area (away from side panels)
            if screen_pt.x > 255.0
                && screen_pt.x < vp_size.x - 350.0
                && screen_pt.y > 42.0
                && screen_pt.y < vp_size.y - 32.0
            {
                let is_active = scene.active_comment_id == Some(c.id);
                let short_text: String = c.comment.chars().take(28).collect();
                let suffix = if c.comment.chars().count() > 28 {
                    "…"
                } else {
                    ""
                };
                let pill_label = format!("💬 #{}: {}{}", c.id, short_text, suffix);

                egui::Area::new(egui::Id::new(("viewport_comment_pin", c.id)))
                    .fixed_pos(egui::pos2(screen_pt.x - 16.0, screen_pt.y - 24.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        let bg = if c.resolved {
                            egui::Color32::from_rgba_unmultiplied(35, 65, 45, 220)
                        } else if is_active {
                            egui::Color32::from_rgba_unmultiplied(15, 115, 155, 240)
                        } else {
                            egui::Color32::from_rgba_unmultiplied(165, 75, 20, 230)
                        };
                        let btn = egui::Button::new(
                            egui::RichText::new(pill_label)
                                .small()
                                .strong()
                                .color(egui::Color32::WHITE),
                        )
                        .fill(bg)
                        .corner_radius(10.0);

                        if ui
                            .add(btn)
                            .on_hover_text(format!(
                                "Comment #{} on '{}'\n{}",
                                c.id, c.object_name, c.comment
                            ))
                            .clicked()
                        {
                            clicked_pin_id = Some(c.id);
                        }
                    });
            }
        }
    }

    if let Some(cid) = clicked_pin_id {
        if let Some(wp) = scene.focus_comment(cid) {
            camera.target = wp;
            ui_state.status_message = format!("Selected Comment #{cid}");
            ui_state.scene_dirty_for_disk = true;
        }
    }
}
