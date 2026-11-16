use crate::camera::OrbitCamera;
use crate::history::UndoHistory;
use crate::io;
use crate::mesh::{EditElementMode, Mesh};
use crate::renderer::{ViewportShading, render_scene_headless_to_png};
use crate::scene::{InteractionMode, ProceduralPattern, Scene};
use crate::ui::UiState;
use crate::voxel::{
    SculptBrush, SculptSettings, VoxelBrushOp, VoxelGrid, VoxelMeshingMode, apply_sculpt_stroke,
    voxel_remesh,
};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const IPC_SOCKET_PATH: &str = "/tmp/rust_modeler_ipc.sock";
pub const FALLBACK_SCENE_PATH: &str = "/tmp/rust_modeler_scene.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub tool: String,
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub ok: bool,
    pub message: String,
    #[serde(default)]
    pub data: Value,
}

impl IpcResponse {
    pub fn success(message: impl Into<String>, data: Value) -> Self {
        Self {
            ok: true,
            message: message.into(),
            data,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: message.into(),
            data: Value::Null,
        }
    }
}

pub(crate) fn parse_vec3(v: &Value, default: Vec3) -> Vec3 {
    if let Some(arr) = v.as_array() {
        if arr.len() >= 3 {
            let x = arr[0].as_f64().unwrap_or(default.x as f64) as f32;
            let y = arr[1].as_f64().unwrap_or(default.y as f64) as f32;
            let z = arr[2].as_f64().unwrap_or(default.z as f64) as f32;
            return Vec3::new(x, y, z);
        }
    }
    default
}

pub(crate) fn parse_color(v: &Value, default: [f32; 3]) -> [f32; 3] {
    if let Some(arr) = v.as_array() {
        if arr.len() >= 3 {
            return [
                (arr[0].as_f64().unwrap_or(default[0] as f64) as f32).clamp(0.0, 1.0),
                (arr[1].as_f64().unwrap_or(default[1] as f64) as f32).clamp(0.0, 1.0),
                (arr[2].as_f64().unwrap_or(default[2] as f64) as f32).clamp(0.0, 1.0),
            ];
        }
    }
    default
}

pub(crate) fn resolve_object_index(scene: &Scene, args: &Value) -> Option<usize> {
    if let Some(id) = args.get("object_id").and_then(|v| v.as_u64()) {
        return scene.objects.iter().position(|o| o.id == id);
    }
    if let Some(name) = args.get("name").and_then(|v| v.as_str()) {
        let lower = name.to_lowercase();
        if let Some(idx) = scene
            .objects
            .iter()
            .position(|o| o.name.to_lowercase() == lower)
        {
            return Some(idx);
        }
    }
    scene.selected_object
}

pub fn summarize_scene(
    scene: &Scene,
    camera: &OrbitCamera,
    shading: ViewportShading,
    show_wireframe: bool,
    show_grid: bool,
) -> Value {
    let objects: Vec<Value> = scene
        .objects
        .iter()
        .enumerate()
        .map(|(idx, obj)| {
            let mut min_p = Vec3::splat(f32::MAX);
            let mut max_p = Vec3::splat(f32::MIN);
            for &p in &obj.mesh.positions {
                min_p = min_p.min(p);
                max_p = max_p.max(p);
            }
            if obj.mesh.positions.is_empty() {
                min_p = Vec3::ZERO;
                max_p = Vec3::ZERO;
            }
            json!({
                "index": idx,
                "id": obj.id,
                "name": obj.name,
                "visible": obj.visible,
                "selected": scene.selected_object == Some(idx),
                "is_voxel": obj.voxel_grid.is_some(),
                "vertices": obj.mesh.positions.len(),
                "faces": obj.mesh.faces.len(),
                "world_bounds_min": crate::agent_tools::world_bounds(obj).map(|b| b.0.to_array()),
                "world_bounds_max": crate::agent_tools::world_bounds(obj).map(|b| b.1.to_array()),
                "selected_faces": obj.mesh.selected_faces.len(),
                "selected_edges": obj.mesh.selected_edges.len(),
                "selected_vertices": obj.mesh.selected_vertices.len(),
                "position": obj.transform.translation.to_array(),
                "rotation_deg": obj.transform.rotation_deg.to_array(),
                "scale": obj.transform.scale.to_array(),
                "color": obj.color,
                "roughness": obj.roughness,
                "metallic": obj.metallic,
                "flat_shading": obj.flat_shading,
                "pattern": obj.pattern.label(),
                "secondary_color": obj.secondary_color,
                "texture_scale": obj.texture_scale,
                "emission_color": obj.emission_color,
                "emission_strength": obj.emission_strength,
                "local_bounds_min": min_p.to_array(),
                "local_bounds_max": max_p.to_array(),
            })
        })
        .collect();

    let unresolved_comments: Vec<Value> = scene
        .comments
        .iter()
        .filter(|c| !c.resolved)
        .map(|c| {
            json!({
                "id": c.id,
                "is_general": c.is_general || (c.object_id == 0 && c.volume.is_none()),
                "object_id": c.object_id,
                "object_name": c.object_name,
                "comment": c.comment,
                "world_anchor": c.world_anchor.to_array(),
                "local_anchor": c.local_anchor.to_array(),
                "surface_normal": c.surface_normal.to_array(),
                "selection_radius": c.selection_radius,
                "selected_faces_count": c.selected_faces.len(),
                "local_bounds_min": c.local_bounds_min.to_array(),
                "local_bounds_max": c.local_bounds_max.to_array(),
                "targets_count": c.targets.len().max(if c.is_general { 0 } else { 1 }),
                "targets": c.targets,
                "resolved": c.resolved,
                "agent_reply": c.agent_reply,
            })
        })
        .collect();

    let current_targets = scene.collect_current_target_sections(0.45);
    let active_selection = scene.active_object().map(|obj| {
        let bounds_info =
            obj.mesh
                .selected_faces_bounds_and_normal()
                .map(|(c, n, min_b, max_b)| {
                    json!({
                        "local_centroid": c.to_array(),
                        "local_normal": n.to_array(),
                        "local_bounds_min": min_b.to_array(),
                        "local_bounds_max": max_b.to_array(),
                    })
                });
        json!({
            "object_id": obj.id,
            "object_name": obj.name,
            "selected_faces_count": obj.mesh.selected_faces.len(),
            "total_faces": obj.mesh.faces.len(),
            "pending_anchor": scene.pending_anchor.map(|(oid, wp, lp, wn)| json!({
                "object_id": oid,
                "world_point": wp.to_array(),
                "local_point": lp.to_array(),
                "world_normal": wn.to_array(),
            })),
            "selected_section_bounds": bounds_info,
            "selected_sections_count": current_targets.len(),
            "selected_sections": current_targets,
        })
    });

    json!({
        "mode": scene.mode.label(),
        "selected_object_index": scene.selected_object,
        "object_count": scene.objects.len(),
        "objects": objects,
        "unresolved_comments_count": unresolved_comments.len(),
        "unresolved_comments": unresolved_comments,
        "active_selection": active_selection,
        "project_name": scene.project_name,
        "units": scene.units,
        "project_path": scene.project_path,
        "revision_count": scene.revisions.len(),
        "camera": {
            "target": camera.target.to_array(),
            "distance": camera.distance,
            "yaw_deg": camera.yaw.to_degrees(),
            "pitch_deg": camera.pitch.to_degrees(),
            "orthographic": camera.orthographic,
        },
        "viewport": {
            "shading": format!("{shading:?}"),
            "wireframe_overlay": show_wireframe,
            "show_grid": show_grid,
        }
    })
}

pub fn allowed_during_operation(tool: &str) -> bool {
    is_read_only(tool)
        || matches!(
            tool,
            "update_comment" | "reattach_comment" | "delete_comment" | "save_project"
        )
}

pub fn is_geometry_tool(tool: &str) -> bool {
    matches!(
        tool,
        "clear_scene"
            | "add_primitive"
            | "transform_object"
            | "set_material"
            | "duplicate_object"
            | "delete_object"
            | "edit_mesh"
            | "sculpt_stroke"
            | "voxel_brush"
            | "voxel_remesh"
            | "import_model"
            | "mirror_object"
            | "array_object"
            | "add_lathe"
            | "add_sweep"
            | "boolean"
    ) || crate::agent_tools::GEOMETRY.contains(&tool)
        || crate::library::GEOMETRY.contains(&tool)
}

/// Clear per-project UI state (hover, clipping, measurements, comment focus, screenshots)
/// when a different project replaces the scene.
fn reset_project_ui(ui_state: &mut UiState) {
    ui_state.hover_preview_faces.clear();
    ui_state.hover_preview_obj_id = None;
    ui_state.hover_preview_seed_face = None;
    ui_state.volume_dragging = false;
    ui_state.volume_drag_changed = false;
    ui_state.clip_enabled = false;
    ui_state.measure_active = false;
    ui_state.measure_points.clear();
    ui_state.screenshot_cache.clear();
    ui_state.focus_comment = None;
    ui_state.scroll_to_comment = None;
    ui_state.expanded_comments.clear();
    ui_state.reply_to = None;
    ui_state.reply_draft.clear();
    ui_state.comment_draft.clear();
    ui_state.lightbox = None;
}

pub fn is_read_only(tool: &str) -> bool {
    matches!(
        tool,
        "get_scene_info"
            | "get_comments"
            | "get_revisions"
            | "capture_viewport"
            | "capture_studio"
            | "get_feedback_state"
            | "wait_for_feedback"
    ) || crate::agent_tools::READ_ONLY.contains(&tool)
        || crate::library::READ_ONLY.contains(&tool)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_ipc_command(
    req: &IpcRequest,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    history: &mut UndoHistory,
    ui_state: &mut UiState,
    shading: &mut ViewportShading,
    show_wireframe: &mut bool,
    show_grid: &mut bool,
) -> IpcResponse {
    let start = std::time::Instant::now();
    let response = dispatch(
        req,
        scene,
        camera,
        history,
        ui_state,
        shading,
        show_wireframe,
        show_grid,
    );
    if !is_read_only(&req.tool) {
        // Some commands replace meshes or edit them without bumping their versions.
        ui_state.target_sections = None;
        ui_state.activity.push(format!(
            "{} · {} · {:.2}s",
            req.tool,
            if response.ok { "Done" } else { "Failed" },
            start.elapsed().as_secs_f32()
        ));
        if ui_state.activity.len() > 24 {
            ui_state.activity.remove(0);
        }
        if response.ok {
            ui_state.scene_dirty_for_disk = true;
            ui_state.last_error = None;
        }
    }
    if !response.ok {
        ui_state.last_error = Some(response.message.clone());
    }
    response
}

/// Executes an MCP tool command against the scene and state.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    req: &IpcRequest,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    history: &mut UndoHistory,
    ui_state: &mut UiState,
    shading: &mut ViewportShading,
    show_wireframe: &mut bool,
    show_grid: &mut bool,
) -> IpcResponse {
    let args = &req.args;
    if scene
        .revisions
        .iter()
        .any(|r| r.status == crate::review::RevisionStatus::ReadyForReview)
        && (is_geometry_tool(&req.tool)
            || matches!(
                req.tool.as_str(),
                "undo" | "redo" | "open_project" | "new_project"
            ))
    {
        return IpcResponse::error(
            "Accept or revert the pending revision before changing geometry",
        );
    }
    if scene.preview_revision.is_some()
        && !is_read_only(&req.tool)
        && !matches!(
            req.tool.as_str(),
            "accept_revision"
                | "revert_revision"
                | "request_changes"
                | "cancel_revision"
                | "preview_revision"
        )
    {
        return IpcResponse::error("Return to the live scene before editing");
    }
    if scene
        .revisions
        .iter()
        .any(|r| r.status == crate::review::RevisionStatus::InProgress)
        && matches!(
            req.tool.as_str(),
            "undo" | "redo" | "open_project" | "new_project"
        )
    {
        return IpcResponse::error("Finish or cancel the active revision first");
    }

    match req.tool.as_str() {
        "save_project" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| scene.project_path.clone());
            let Some(path) = path.filter(|p| !p.trim().is_empty()) else {
                return IpcResponse::error("Choose a project file path");
            };
            let path = match crate::project::absolute_path(&path) {
                Ok(p) => p,
                Err(e) => return IpcResponse::error(e),
            };
            let old_path = scene.project_path.clone();
            scene.project_path = Some(path.clone());
            match crate::project::save(Path::new(&path), scene, camera) {
                Ok(()) => {
                    ui_state.status_message = format!("Saved {path}");
                    IpcResponse::success("Project saved", json!({"path":path}))
                }
                Err(e) => {
                    scene.project_path = old_path;
                    IpcResponse::error(e)
                }
            }
        }
        "open_project" => {
            let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                return IpcResponse::error("Missing project path");
            };
            let path = match crate::project::absolute_path(path) {
                Ok(p) => p,
                Err(e) => return IpcResponse::error(e),
            };
            let source = if args
                .get("recover")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                crate::project::autosave_path(Path::new(&path))
            } else {
                Path::new(&path).to_path_buf()
            };
            match crate::project::load(&source) {
                Ok(mut project) => {
                    history.record("Open project", scene);
                    project.scene.project_path = Some(path.clone());
                    project.scene.project_generation = scene.project_generation + 1;
                    *scene = project.scene;
                    *camera = project.camera;
                    reset_project_ui(ui_state);
                    ui_state.status_message = format!("Opened {path}");
                    IpcResponse::success("Project opened", json!({"path":path}))
                }
                Err(e) => IpcResponse::error(e),
            }
        }
        "new_project" => {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            history.record("New project", scene);
            let previous = scene.start_new_project(name);
            let aspect = camera.aspect;
            *camera = OrbitCamera::default();
            camera.aspect = aspect;
            reset_project_ui(ui_state);
            ui_state.status_message = format!("Started {}", scene.project_name);
            IpcResponse::success(
                "Started a new, unsaved project. Undo restores the previous one.",
                json!({
                    "project_name": scene.project_name,
                    "previous_project_name": previous.project_name,
                    "previous_project_path": previous.project_path,
                }),
            )
        }
        "begin_revision" => {
            let label = args
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or("Agent revision")
                .to_string();
            let before = scene.clone();
            match scene.begin_revision(label, args.get("comment_id").and_then(|v| v.as_u64())) {
                Ok(id) => {
                    history.record("Agent revision", &before);
                    IpcResponse::success("Revision started", json!({"revision_id":id}))
                }
                Err(e) => IpcResponse::error(e),
            }
        }
        "finish_revision" => {
            let id = args
                .get("revision_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let screenshots = args
                .get("screenshots")
                .and_then(|v| v.as_array())
                .map(|v| {
                    v.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            match scene.finish_revision(
                id,
                args.get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                screenshots,
            ) {
                Ok(()) => IpcResponse::success("Ready for review", json!({"revision_id":id})),
                Err(e) => IpcResponse::error(e),
            }
        }
        "accept_revision" | "revert_revision" | "cancel_revision" => {
            let id = args
                .get("revision_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let before = scene.clone();
            match scene.review_revision(id, req.tool == "accept_revision") {
                Ok(()) => {
                    if req.tool != "accept_revision" {
                        history.record("Revert revision", &before);
                    }
                    IpcResponse::success("Revision reviewed", json!({"revision_id":id}))
                }
                Err(e) => IpcResponse::error(e),
            }
        }
        "preview_revision" => {
            if args.get("live").and_then(|v| v.as_bool()).unwrap_or(false) {
                scene.preview_revision = None;
            } else {
                let id = args
                    .get("revision_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                if !scene.revisions.iter().any(|r| r.id == id) {
                    return IpcResponse::error("Revision not found");
                }
                scene.preview_revision = Some((
                    id,
                    args.get("before").and_then(|v| v.as_bool()).unwrap_or(true),
                ));
            }
            IpcResponse::success("Updated comparison", json!({}))
        }
        "get_revisions" => {
            let revisions: Vec<_> = scene.revisions.iter().map(|r| json!({"id":r.id,"label":r.label,"comment_id":r.comment_id,"status":r.status,"summary":r.summary,"screenshots":r.screenshots,"changed_objects":scene.changed_objects(r.id)})).collect();
            IpcResponse::success("Revision history", json!({"revisions":revisions}))
        }
        "update_comment" => {
            let id = args.get("comment_id").and_then(|v| v.as_u64()).unwrap_or(0);
            if args.get("status").is_some()
                && scene.revisions.iter().any(|r| {
                    r.comment_id == Some(id)
                        && matches!(
                            r.status,
                            crate::review::RevisionStatus::InProgress
                                | crate::review::RevisionStatus::ReadyForReview
                        )
                })
            {
                return IpcResponse::error(
                    "Use the linked revision to update this comment's review status",
                );
            }
            if let Some(status) = args.get("status") {
                let Ok(status) =
                    serde_json::from_value::<crate::scene::CommentStatus>(status.clone())
                else {
                    return IpcResponse::error("Invalid comment status");
                };
                if let Err(e) = scene.set_comment_status(id, status) {
                    return IpcResponse::error(e);
                }
            }
            let Some(c) = scene.comments.iter_mut().find(|c| c.id == id) else {
                return IpcResponse::error("Comment not found");
            };
            if let Some(text) = args
                .get("reply")
                .and_then(|v| v.as_str())
                .filter(|t| !t.trim().is_empty())
            {
                c.replies.push(crate::scene::CommentReply {
                    author: args
                        .get("author")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Agent")
                        .into(),
                    text: text.into(),
                });
            }
            IpcResponse::success("Comment updated", json!({"comment":c}))
        }
        "delete_comment" => {
            let id = args.get("comment_id").and_then(|v| v.as_u64()).unwrap_or(0);
            if scene.revisions.iter().any(|r| {
                r.comment_id == Some(id)
                    && matches!(
                        r.status,
                        crate::review::RevisionStatus::InProgress
                            | crate::review::RevisionStatus::ReadyForReview
                    )
            }) {
                return IpcResponse::error(
                    "Review the linked revision before deleting its comment",
                );
            }
            if !scene.comments.iter().any(|c| c.id == id) {
                return IpcResponse::error("Comment not found");
            }
            scene.comments.retain(|c| c.id != id);
            if scene.active_comment_id == Some(id) {
                scene.active_comment_id = None;
            }
            IpcResponse::success("Comment deleted", json!({}))
        }
        "reattach_comment" => {
            let id = args.get("comment_id").and_then(|v| v.as_u64()).unwrap_or(0);
            match scene.reattach_comment(id, ui_state.selection_radius) {
                Ok(()) => IpcResponse::success("Comment reattached", json!({})),
                Err(e) => IpcResponse::error(e),
            }
        }

        "get_scene_info" => {
            let info = summarize_scene(scene, camera, *shading, *show_wireframe, *show_grid);
            IpcResponse::success("Retrieved scene info", info)
        }

        "clear_scene" => {
            history.record("MCP: Clear Scene", scene);
            scene.objects.clear();
            scene.selected_object = None;
            ui_state.status_message = "MCP cleared scene".to_string();
            IpcResponse::success("Cleared all objects from scene", json!({}))
        }

        "add_primitive" => {
            let shape = args
                .get("shape")
                .and_then(|v| v.as_str())
                .unwrap_or("cube")
                .to_lowercase();
            let size = args.get("size").and_then(|v| v.as_f64()).unwrap_or(1.8) as f32;
            let radius = args
                .get("radius")
                .and_then(|v| v.as_f64())
                .unwrap_or((size * 0.55) as f64) as f32;
            let height = args.get("height").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
            let subdivisions = args
                .get("subdivisions")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize;
            let color = parse_color(&args["color"], [0.78, 0.80, 0.85]);
            let pos = parse_vec3(&args["position"], Vec3::ZERO);
            let rot = parse_vec3(&args["rotation_deg"], Vec3::ZERO);
            let scale = parse_vec3(&args["scale"], Vec3::ONE);

            history.record(format!("MCP: Add {shape}"), scene);

            let idx = match shape.as_str() {
                "cube" | "box" => {
                    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Cube");
                    scene.add_mesh_object(name, Mesh::cube(size), color)
                }
                "sphere" | "uv_sphere" => {
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Sphere");
                    let rings = subdivisions.max(8);
                    let sectors = (subdivisions * 2).max(12);
                    scene.add_mesh_object(name, Mesh::sphere(radius, rings, sectors), color)
                }
                "cylinder" => {
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Cylinder");
                    scene.add_mesh_object(
                        name,
                        Mesh::cylinder(radius, height, subdivisions.max(8)),
                        color,
                    )
                }
                "plane" | "grid" => {
                    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Plane");
                    scene.add_mesh_object(name, Mesh::plane(size, subdivisions.max(1)), color)
                }
                "torus" => {
                    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Torus");
                    let minor_r = args
                        .get("minor_radius")
                        .and_then(|v| v.as_f64())
                        .unwrap_or((radius * 0.35) as f64) as f32;
                    scene.add_mesh_object(
                        name,
                        Mesh::torus(
                            radius,
                            minor_r,
                            subdivisions.max(12),
                            (subdivisions / 2).max(8),
                        ),
                        color,
                    )
                }
                "voxel_sphere" | "voxel" => {
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Voxel Clay");
                    let res = subdivisions.clamp(12, 64);
                    let grid = VoxelGrid::new_sphere(res, size.max(2.5), radius);
                    scene.add_voxel_object(name, grid, color)
                }
                other => {
                    return IpcResponse::error(format!(
                        "Unknown primitive shape '{other}'. Supported: cube, sphere, cylinder, plane, torus, voxel_sphere"
                    ));
                }
            };

            if let Some(obj) = scene.objects.get_mut(idx) {
                obj.transform.translation = pos;
                obj.transform.rotation_deg = rot;
                obj.transform.scale = scale;
                let id = obj.id;
                let name = obj.name.clone();
                ui_state.status_message = format!("MCP added '{name}' (ID {id})");
                IpcResponse::success(
                    format!("Added {shape} '{name}' with object_id={id}"),
                    json!({ "object_id": id, "index": idx, "name": name }),
                )
            } else {
                IpcResponse::error("Failed to create object")
            }
        }

        "mirror_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No matching or active object found");
            };
            let axis = match crate::modeling::axis_index(
                args.get("axis").and_then(|v| v.as_str()).unwrap_or("x"),
            ) {
                Ok(a) => a,
                Err(e) => return IpcResponse::error(e),
            };
            let merge = args.get("merge").and_then(|v| v.as_bool()).unwrap_or(true);
            history.record("MCP: Mirror", scene);
            let mirrored = crate::modeling::mirrored(&scene.objects[idx].mesh, axis);
            let id = if merge {
                let obj = &mut scene.objects[idx];
                crate::modeling::append(&mut obj.mesh, &mirrored);
                obj.voxel_grid = None;
                obj.id
            } else {
                scene.selected_object = Some(idx);
                let Some(new_idx) = scene.duplicate_selected() else {
                    return IpcResponse::error("Could not copy the object");
                };
                let src = scene.objects[idx].clone();
                let obj = &mut scene.objects[new_idx];
                obj.name = format!("{} (mirror)", src.name);
                obj.transform = src.transform;
                obj.mesh = mirrored;
                obj.voxel_grid = None;
                obj.id
            };
            let obj = scene.objects.iter().find(|o| o.id == id).unwrap();
            ui_state.status_message = format!("Mirrored '{}'", obj.name);
            IpcResponse::success(
                format!("Mirrored across local {} into '{}'", ["X", "Y", "Z"][axis], obj.name),
                json!({"object_id": id, "faces": obj.mesh.faces.len()}),
            )
        }

        "array_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No matching or active object found");
            };
            let count = args.get("count").and_then(|v| v.as_u64()).unwrap_or(3).clamp(2, 128) as usize;
            let offset = parse_vec3(&args["offset"], Vec3::new(1.0, 0.0, 0.0));
            let rot = parse_vec3(&args["rotation_deg"], Vec3::ZERO);
            let step_scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let merge = args.get("merge").and_then(|v| v.as_bool()).unwrap_or(false);
            if !offset.is_finite() || !rot.is_finite() || !(step_scale.is_finite() && step_scale > 0.0) {
                return IpcResponse::error("Array offset, rotation and scale must be finite");
            }
            history.record("MCP: Array", scene);
            let step = crate::modeling::array_step(offset, rot, step_scale);
            let mut ids = vec![scene.objects[idx].id];
            if merge {
                let base = scene.objects[idx].mesh.clone();
                let obj = &mut scene.objects[idx];
                let mut m = glam::Mat4::IDENTITY;
                for _ in 1..count {
                    m = step * m;
                    crate::modeling::append(&mut obj.mesh, &crate::modeling::transformed(&base, m));
                }
                obj.voxel_grid = None;
            } else {
                // Copies follow the step in the source object's local frame.
                let src = scene.objects[idx].clone();
                let world = src.transform.matrix();
                let mut m = glam::Mat4::IDENTITY;
                for i in 1..count {
                    m = step * m;
                    let (s, r, t) = (world * m).to_scale_rotation_translation();
                    scene.selected_object = Some(idx);
                    let Some(new_idx) = scene.duplicate_selected() else { break };
                    let obj = &mut scene.objects[new_idx];
                    obj.name = format!("{} {}", src.name, i + 1);
                    obj.transform.translation = t;
                    obj.transform.scale = s;
                    let (x, y, z) = r.to_euler(glam::EulerRot::XYZ);
                    obj.transform.rotation_deg = Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                    ids.push(obj.id);
                }
            }
            ui_state.status_message = format!("Arrayed '{}' ×{count}", scene.objects[idx].name);
            IpcResponse::success(
                format!("Created an array of {count}"),
                json!({"object_ids": ids, "merged": merge}),
            )
        }

        "add_lathe" | "add_sweep" => {
            let color = parse_color(&args["color"], [0.78, 0.80, 0.85]);
            let points2 = |v: &Value| -> Vec<glam::Vec2> {
                v.as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|p| {
                                let p = p.as_array()?;
                                Some(glam::Vec2::new(p.first()?.as_f64()? as f32, p.get(1)?.as_f64()? as f32))
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let cap = args.get("cap").and_then(|v| v.as_bool()).unwrap_or(true);
            let result = if req.tool == "add_lathe" {
                crate::modeling::lathe(
                    &points2(&args["profile"]),
                    args.get("segments").and_then(|v| v.as_u64()).unwrap_or(32) as usize,
                    args.get("angle_deg").and_then(|v| v.as_f64()).unwrap_or(360.0) as f32,
                    cap,
                    color,
                )
            } else {
                let path: Vec<Vec3> = args["path"]
                    .as_array()
                    .map(|a| a.iter().map(|p| parse_vec3(p, Vec3::NAN)).collect())
                    .unwrap_or_default();
                let section = if args["profile"].is_array() {
                    points2(&args["profile"])
                } else {
                    crate::modeling::circle_section(
                        args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
                        args.get("sides").and_then(|v| v.as_u64()).unwrap_or(12) as usize,
                    )
                };
                crate::modeling::sweep(
                    &path,
                    &section,
                    args.get("closed").and_then(|v| v.as_bool()).unwrap_or(false),
                    cap,
                    args.get("twist_deg").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    args.get("end_scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    color,
                )
            };
            let mesh = match result {
                Ok(m) => m,
                Err(e) => return IpcResponse::error(e),
            };
            history.record(format!("MCP: {}", req.tool), scene);
            let default_name = if req.tool == "add_lathe" { "Lathe" } else { "Sweep" };
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or(default_name);
            let idx = scene.add_mesh_object(name, mesh, color);
            let obj = &mut scene.objects[idx];
            obj.transform.translation = parse_vec3(&args["position"], Vec3::ZERO);
            obj.transform.rotation_deg = parse_vec3(&args["rotation_deg"], Vec3::ZERO);
            ui_state.status_message = format!("Added '{}'", obj.name);
            IpcResponse::success(
                format!("Added '{}' (object_id={})", obj.name, obj.id),
                json!({"object_id": obj.id, "name": obj.name, "vertices": obj.mesh.positions.len(), "faces": obj.mesh.faces.len()}),
            )
        }

        "boolean" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No matching or active target object");
            };
            let Some(tool_id) = args.get("tool_object_id").and_then(|v| v.as_u64()) else {
                return IpcResponse::error("Missing 'tool_object_id'");
            };
            let Some(tidx) = scene.objects.iter().position(|o| o.id == tool_id) else {
                return IpcResponse::error("Tool object not found");
            };
            if tidx == idx {
                return IpcResponse::error("Pick two different objects");
            }
            let op = match crate::modeling::BooleanOp::parse(
                args.get("operation").and_then(|v| v.as_str()).unwrap_or("subtract"),
            ) {
                Ok(op) => op,
                Err(e) => return IpcResponse::error(e),
            };
            let target = &scene.objects[idx];
            let tool = &scene.objects[tidx];
            let mesh = match crate::modeling::boolean(
                &target.mesh,
                target.transform.matrix(),
                &tool.mesh,
                tool.transform.matrix(),
                op,
                tool.color,
            ) {
                Ok(m) => m,
                Err(e) => return IpcResponse::error(e),
            };
            history.record("MCP: Boolean", scene);
            let target_id = scene.objects[idx].id;
            let obj = &mut scene.objects[idx];
            obj.mesh = mesh;
            obj.voxel_grid = None;
            obj.flat_shading = true;
            if !args.get("keep_tool").and_then(|v| v.as_bool()).unwrap_or(false) {
                scene.objects.remove(tidx);
            }
            scene.selected_object = scene.objects.iter().position(|o| o.id == target_id);
            let obj = scene.objects.iter().find(|o| o.id == target_id).unwrap();
            ui_state.status_message = format!("Boolean {:?} on '{}'", op, obj.name);
            IpcResponse::success(
                format!("Applied {:?} to '{}'", op, obj.name),
                json!({"object_id": target_id, "vertices": obj.mesh.positions.len(), "faces": obj.mesh.faces.len()}),
            )
        }

        "transform_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No matching or active object found");
            };
            history.record("MCP: Transform Object", scene);
            let obj = &mut scene.objects[idx];

            if !args["position"].is_null() {
                obj.transform.translation =
                    parse_vec3(&args["position"], obj.transform.translation);
            }
            if !args["translate_delta"].is_null() {
                obj.transform.translation += parse_vec3(&args["translate_delta"], Vec3::ZERO);
            }
            if !args["rotation_deg"].is_null() {
                obj.transform.rotation_deg =
                    parse_vec3(&args["rotation_deg"], obj.transform.rotation_deg);
            }
            if !args["rotate_delta_deg"].is_null() {
                obj.transform.rotation_deg += parse_vec3(&args["rotate_delta_deg"], Vec3::ZERO);
            }
            if !args["scale"].is_null() {
                obj.transform.scale = parse_vec3(&args["scale"], obj.transform.scale);
            }
            if let Some(new_name) = args.get("new_name").and_then(|v| v.as_str()) {
                obj.name = new_name.to_string();
            }
            if let Some(vis) = args.get("visible").and_then(|v| v.as_bool()) {
                obj.visible = vis;
            }

            let data = json!({
                "object_id": obj.id,
                "name": obj.name,
                "position": obj.transform.translation.to_array(),
                "rotation_deg": obj.transform.rotation_deg.to_array(),
                "scale": obj.transform.scale.to_array(),
            });
            ui_state.status_message = format!("MCP transformed '{}'", obj.name);
            IpcResponse::success(format!("Updated transform for '{}'", obj.name), data)
        }

        "set_material" | "texture_and_bake" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No matching or active object found");
            };
            history.record("MCP: Set Material & Texture", scene);
            let obj = &mut scene.objects[idx];

            if !args["color"].is_null() {
                obj.color = parse_color(&args["color"], obj.color);
            }
            if let Some(r) = args.get("roughness").and_then(|v| v.as_f64()) {
                obj.roughness = (r as f32).clamp(0.05, 1.0);
            }
            if let Some(m) = args.get("metallic").and_then(|v| v.as_f64()) {
                obj.metallic = (m as f32).clamp(0.0, 1.0);
            }
            if let Some(flat) = args.get("flat_shading").and_then(|v| v.as_bool()) {
                obj.flat_shading = flat;
            }
            if let Some(pat_str) = args.get("pattern").and_then(|v| v.as_str()) {
                obj.pattern = ProceduralPattern::from_name(pat_str);
            }
            if !args["secondary_color"].is_null() {
                obj.secondary_color = parse_color(&args["secondary_color"], obj.secondary_color);
            }
            if let Some(scale) = args.get("texture_scale").and_then(|v| v.as_f64()) {
                obj.texture_scale = (scale as f32).clamp(0.1, 30.0);
            }
            if !args["emission_color"].is_null() {
                obj.emission_color = parse_color(&args["emission_color"], obj.emission_color);
            }
            if let Some(em_s) = args.get("emission_strength").and_then(|v| v.as_f64()) {
                obj.emission_strength = (em_s as f32).clamp(0.0, 5.0);
            }

            if args
                .get("bake_ao")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                let ao_s = args
                    .get("ao_strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.55) as f32;
                let wear_s = args
                    .get("wear_strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.45) as f32;
                let crevice = parse_color(&args["crevice_color"], [0.16, 0.16, 0.20]);
                let wear = parse_color(&args["wear_color"], [1.0, 0.96, 0.88]);
                obj.mesh.bake_ao_and_curvature(ao_s, wear_s, crevice, wear);
            }

            if !args["top_coat_color"].is_null() || !args["bottom_color"].is_null() {
                let bot = parse_color(&args["bottom_color"], [0.65, 0.65, 0.68]);
                let top = parse_color(&args["top_color"], [1.0, 1.0, 1.0]);
                let g_blend = args
                    .get("gradient_blend")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.35) as f32;
                let coat = if !args["top_coat_color"].is_null() {
                    Some(parse_color(&args["top_coat_color"], [0.30, 0.68, 0.36]))
                } else {
                    None
                };
                let thresh = args
                    .get("top_coat_threshold")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.45) as f32;
                let strength = args
                    .get("top_coat_strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.65) as f32;
                obj.mesh
                    .paint_gradient_and_normal_coat(bot, top, g_blend, coat, thresh, strength);
            }

            obj.mesh.bump_version();
            ui_state.status_message = format!("MCP textured '{}'", obj.name);

            IpcResponse::success(
                format!("Updated material & texturing for '{}'", obj.name),
                json!({
                    "object_id": obj.id,
                    "color": obj.color,
                    "roughness": obj.roughness,
                    "metallic": obj.metallic,
                    "flat_shading": obj.flat_shading,
                    "pattern": obj.pattern.label(),
                    "secondary_color": obj.secondary_color,
                    "texture_scale": obj.texture_scale,
                    "emission_color": obj.emission_color,
                    "emission_strength": obj.emission_strength,
                }),
            )
        }

        "clear_selection" => {
            let faces: usize = scene.objects.iter().map(|o| o.mesh.selected_faces.len()).sum();
            let had_object = scene.selected_object.is_some();
            scene.clear_all_selections();
            scene.selected_object = None;
            scene.active_comment_id = None;
            ui_state.hover_preview_faces.clear();
            ui_state.hover_preview_obj_id = None;
            ui_state.status_message = "Selection cleared".into();
            IpcResponse::success(
                "Cleared the selection",
                json!({"cleared_faces": faces, "had_selected_object": had_object}),
            )
        }

        "select_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("Object not found");
            };
            scene.selected_object = Some(idx);
            let obj = &scene.objects[idx];
            IpcResponse::success(
                format!("Selected object '{}' (id={})", obj.name, obj.id),
                json!({ "object_id": obj.id, "index": idx, "name": obj.name }),
            )
        }

        "duplicate_object" => {
            if let Some(idx) = resolve_object_index(scene, args) {
                scene.selected_object = Some(idx);
            }
            history.record("MCP: Duplicate Object", scene);
            if let Some(new_idx) = scene.duplicate_selected() {
                let obj = &scene.objects[new_idx];
                IpcResponse::success(
                    format!("Duplicated object as '{}' (id={})", obj.name, obj.id),
                    json!({ "object_id": obj.id, "index": new_idx, "name": obj.name }),
                )
            } else {
                IpcResponse::error("No object selected to duplicate")
            }
        }

        "delete_object" => {
            if let Some(idx) = resolve_object_index(scene, args) {
                scene.selected_object = Some(idx);
            }
            history.record("MCP: Delete Object", scene);
            if scene.delete_selected_object() {
                IpcResponse::success("Deleted object", json!({}))
            } else {
                IpcResponse::error("No object found to delete")
            }
        }

        "edit_mesh" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No active or matching object to edit");
            };
            scene.selected_object = Some(idx);

            let op = args
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();

            history.record(format!("MCP Mesh Edit: {op}"), scene);
            let obj = &mut scene.objects[idx];
            let mut op_detail = String::new();

            // Optional face selection helper before running the operation
            if let Some(face_arr) = args.get("select_faces").and_then(|v| v.as_array()) {
                obj.mesh.selected_faces.clear();
                for val in face_arr {
                    if let Some(fi) = val.as_u64() {
                        let fi = fi as usize;
                        if fi < obj.mesh.faces.len() {
                            obj.mesh.selected_faces.insert(fi);
                        }
                    }
                }
                obj.mesh.bump_version();
            } else if !args["select_faces_by_normal"].is_null() {
                let target_n =
                    parse_vec3(&args["select_faces_by_normal"], Vec3::Y).normalize_or_zero();
                let min_dot = args.get("min_dot").and_then(|v| v.as_f64()).unwrap_or(0.65) as f32;
                obj.mesh.selected_faces.clear();
                for fi in 0..obj.mesh.faces.len() {
                    if obj.mesh.face_normal(fi).dot(target_n) >= min_dot {
                        obj.mesh.selected_faces.insert(fi);
                    }
                }
                obj.mesh.bump_version();
            }

            match op.as_str() {
                "select_by_normal" => {}
                "select_all" => {
                    obj.mesh.select_all(EditElementMode::Face);
                }
                "clear_selection" => {
                    obj.mesh.clear_selection();
                }
                "extrude" | "extrude_faces" => {
                    let distance = args
                        .get("distance")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.45) as f32;
                    if obj.mesh.selected_faces.is_empty() {
                        return IpcResponse::error(
                            "No faces selected to extrude. Pass `select_faces` or `select_faces_by_normal` (e.g. [0, 1, 0]).",
                        );
                    }
                    obj.mesh.extrude_selected_faces(distance);
                    obj.voxel_grid = None;
                }
                "inset" | "inset_faces" => {
                    let factor = args
                        .get("inset_factor")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.28) as f32;
                    let depth = args.get("depth").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    if obj.mesh.selected_faces.is_empty() {
                        return IpcResponse::error(
                            "No faces selected to inset. Pass `select_faces` or `select_faces_by_normal`.",
                        );
                    }
                    obj.mesh.inset_selected_faces(factor, depth);
                    obj.voxel_grid = None;
                }
                "subdivide" | "loop_cut" => {
                    obj.mesh.subdivide_or_loop_cut();
                    obj.voxel_grid = None;
                }
                "translate_selected" => {
                    let delta = parse_vec3(&args["delta"], Vec3::ZERO);
                    obj.mesh.translate_selected(delta, EditElementMode::Face);
                    obj.voxel_grid = None;
                }
                "scale_selected" => {
                    let scale = parse_vec3(&args["scale"], Vec3::ONE);
                    obj.mesh.scale_selected(scale, EditElementMode::Face);
                    obj.voxel_grid = None;
                }
                "rotate_selected" => {
                    let deg = parse_vec3(&args["rotation_deg"], Vec3::ZERO);
                    let q = Quat::from_euler(
                        glam::EulerRot::XYZ,
                        deg.x.to_radians(),
                        deg.y.to_radians(),
                        deg.z.to_radians(),
                    );
                    obj.mesh.rotate_selected(q, EditElementMode::Face);
                    obj.voxel_grid = None;
                }
                "delete_selected" => {
                    obj.mesh.delete_selected(EditElementMode::Face);
                    obj.voxel_grid = None;
                }
                "bevel" | "chamfer" => {
                    let width = args.get("width").and_then(|v| v.as_f64()).unwrap_or(0.05) as f32;
                    if let Err(e) = crate::modeling::bevel_selected(&mut obj.mesh, width) {
                        return IpcResponse::error(e);
                    }
                    obj.voxel_grid = None;
                }
                other => match crate::agent_tools::edit_mesh_op(other, &mut obj.mesh, args) {
                    Some(Ok(detail)) => {
                        obj.voxel_grid = None;
                        op_detail = detail;
                    }
                    Some(Err(e)) => return IpcResponse::error(e),
                    None => {
                        return IpcResponse::error(format!(
                            "Unknown edit_mesh operation '{other}'. Supported: select_by_normal, select_all, clear_selection, extrude, inset, bevel, subdivide, translate_selected, scale_selected, rotate_selected, delete_selected, {}",
                            crate::agent_tools::EXTRA_EDIT_OPS
                        ));
                    }
                },
            }

            ui_state.status_message = format!("MCP mesh edit '{op}' on '{}'", obj.name);
            IpcResponse::success(
                if op_detail.is_empty() {
                    format!("Executed mesh edit '{op}' on '{}'", obj.name)
                } else {
                    format!("Executed mesh edit '{op}' on '{}': {op_detail}", obj.name)
                },
                json!({
                    "detail": op_detail,
                    "object_id": obj.id,
                    "vertices": obj.mesh.positions.len(),
                    "faces": obj.mesh.faces.len(),
                    "selected_faces": obj.mesh.selected_faces.len(),
                }),
            )
        }

        "sculpt_stroke" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No active or matching object to sculpt");
            };
            scene.selected_object = Some(idx);
            history.record("MCP: Sculpt Stroke", scene);

            let brush_str = args
                .get("brush")
                .and_then(|v| v.as_str())
                .unwrap_or("draw")
                .to_lowercase();
            let brush = match brush_str.as_str() {
                "draw" | "clay" => SculptBrush::Draw,
                "inflate" => SculptBrush::Inflate,
                "smooth" => SculptBrush::Smooth,
                "flatten" => SculptBrush::Flatten,
                "grab" => SculptBrush::Grab,
                "pinch" => SculptBrush::Pinch,
                "crease" => SculptBrush::Crease,
                _ => SculptBrush::Draw,
            };

            let settings = SculptSettings {
                brush,
                radius: args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
                strength: args
                    .get("strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.45) as f32,
                invert: args
                    .get("invert")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                symmetry_x: args
                    .get("symmetry_x")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                symmetry_y: args
                    .get("symmetry_y")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                symmetry_z: args
                    .get("symmetry_z")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                dyntopo_enabled: args
                    .get("dyntopo")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                dyntopo_detail_size: args
                    .get("detail_size")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.15) as f32,
                voxel_remesh_resolution: 32,
            };

            let point = parse_vec3(&args["point"], Vec3::new(0.0, 1.0, 0.0));
            let normal = parse_vec3(
                &args["normal"],
                if point.length_squared() > 1e-4 {
                    point.normalize()
                } else {
                    Vec3::Y
                },
            );
            let grab_delta = parse_vec3(&args["grab_delta"], Vec3::ZERO);

            let obj = &mut scene.objects[idx];
            if let Some(pts) = args.get("points").and_then(|v| v.as_array()) {
                for pt_val in pts {
                    let p = parse_vec3(pt_val, point);
                    let n = if p.length_squared() > 1e-4 {
                        p.normalize()
                    } else {
                        normal
                    };
                    apply_sculpt_stroke(&mut obj.mesh, p, n, grab_delta, &settings);
                }
            } else {
                apply_sculpt_stroke(&mut obj.mesh, point, normal, grab_delta, &settings);
            }
            obj.voxel_grid = None;

            ui_state.status_message = format!("MCP sculpted '{}' with {:?}", obj.name, brush);
            IpcResponse::success(
                format!("Applied {:?} sculpt stroke on '{}'", brush, obj.name),
                json!({
                    "object_id": obj.id,
                    "vertices": obj.mesh.positions.len(),
                    "faces": obj.mesh.faces.len(),
                }),
            )
        }

        "voxel_brush" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No active or matching object for voxel brush");
            };
            scene.selected_object = Some(idx);
            history.record("MCP: Voxel Brush", scene);

            let op_str = args
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("add_sphere")
                .to_lowercase();
            let op = match op_str.as_str() {
                "add_sphere" => VoxelBrushOp::AddSphere,
                "subtract_sphere" | "carve_sphere" => VoxelBrushOp::SubtractSphere,
                "add_box" => VoxelBrushOp::AddBox,
                "subtract_box" | "carve_box" => VoxelBrushOp::SubtractBox,
                "smooth" | "smooth_field" => VoxelBrushOp::SmoothField,
                "paint" | "paint_color" => VoxelBrushOp::PaintColor,
                _ => VoxelBrushOp::AddSphere,
            };

            let center = parse_vec3(&args["center"], Vec3::ZERO);
            let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let color = parse_color(&args["color"], [0.85, 0.65, 0.45]);
            let sym_x = args
                .get("symmetry_x")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let obj = &mut scene.objects[idx];
            if obj.voxel_grid.is_none() {
                obj.voxel_grid = Some(VoxelGrid::new_sphere(32, 3.2, 1.15));
            }
            if let Some(grid) = &mut obj.voxel_grid {
                if let Some(mode_str) = args.get("meshing_mode").and_then(|v| v.as_str()) {
                    grid.meshing_mode = if mode_str.eq_ignore_ascii_case("blocky") {
                        VoxelMeshingMode::BlockyVoxels
                    } else {
                        VoxelMeshingMode::SmoothSurfaceNets
                    };
                }
                if let Some(centers) = args.get("centers").and_then(|v| v.as_array()) {
                    for c_val in centers {
                        let c = parse_vec3(c_val, center);
                        grid.apply_brush(op, c, radius, color, sym_x);
                    }
                } else {
                    grid.apply_brush(op, center, radius, color, sym_x);
                }
            }
            obj.rebuild_voxel_mesh();

            ui_state.status_message = format!("MCP applied voxel {:?} to '{}'", op, obj.name);
            IpcResponse::success(
                format!("Applied voxel {:?} to '{}'", op, obj.name),
                json!({
                    "object_id": obj.id,
                    "vertices": obj.mesh.positions.len(),
                    "faces": obj.mesh.faces.len(),
                }),
            )
        }

        "voxel_remesh" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return IpcResponse::error("No active or matching object to remesh");
            };
            scene.selected_object = Some(idx);
            let res = args
                .get("resolution")
                .and_then(|v| v.as_u64())
                .unwrap_or(32) as usize;
            history.record("MCP: Voxel Remesh", scene);

            let obj = &mut scene.objects[idx];
            obj.mesh = voxel_remesh(&obj.mesh, res);
            obj.voxel_grid = None;

            ui_state.status_message = format!("MCP voxel-remeshed '{}' at {}³", obj.name, res);
            IpcResponse::success(
                format!("Voxel-remeshed '{}' at resolution {}", obj.name, res),
                json!({
                    "object_id": obj.id,
                    "vertices": obj.mesh.positions.len(),
                    "faces": obj.mesh.faces.len(),
                }),
            )
        }

        "set_camera_and_viewport" => {
            if !args["target"].is_null() {
                camera.target = parse_vec3(&args["target"], camera.target);
            }
            if let Some(d) = args.get("distance").and_then(|v| v.as_f64()) {
                camera.distance = (d as f32).clamp(0.4, 200.0);
            }
            if let Some(yaw) = args.get("yaw_deg").and_then(|v| v.as_f64()) {
                camera.yaw = (yaw as f32).to_radians();
            }
            if let Some(pitch) = args.get("pitch_deg").and_then(|v| v.as_f64()) {
                let max_p = std::f32::consts::FRAC_PI_2 - 0.02;
                camera.pitch = (pitch as f32).to_radians().clamp(-max_p, max_p);
            }
            if let Some(ortho) = args.get("orthographic").and_then(|v| v.as_bool()) {
                camera.orthographic = ortho;
            }
            if !args["eye"].is_null() {
                let eye = parse_vec3(&args["eye"], camera.eye());
                let look_at = if args["look_at"].is_null() {
                    eye + camera.forward()
                } else {
                    parse_vec3(&args["look_at"], camera.target)
                };
                camera.look_from(eye, look_at);
            } else if !args["look_at"].is_null() {
                let look_at = parse_vec3(&args["look_at"], camera.target);
                camera.look_from(camera.eye(), look_at);
            }
            if let Some(preset) = args.get("preset").and_then(|v| v.as_str()) {
                match preset.to_lowercase().as_str() {
                    "front" => camera.set_view_front(),
                    "right" => camera.set_view_right(),
                    "top" => camera.set_view_top(),
                    _ => {}
                }
            }
            if let Some(sh) = args.get("shading").and_then(|v| v.as_str()) {
                match sh.to_lowercase().as_str() {
                    "lit" | "studio" => *shading = ViewportShading::StudioLit,
                    "clay" | "sculpt" => *shading = ViewportShading::SculptClay,
                    "wire" | "wireframe" => *shading = ViewportShading::WireframeOnly,
                    _ => {}
                }
            }
            if let Some(wf) = args.get("wireframe_overlay").and_then(|v| v.as_bool()) {
                *show_wireframe = wf;
            }
            if let Some(gr) = args.get("show_grid").and_then(|v| v.as_bool()) {
                *show_grid = gr;
            }
            if let Some(mode_str) = args.get("mode").and_then(|v| v.as_str()) {
                scene.mode = match mode_str.to_lowercase().as_str() {
                    "object" => InteractionMode::Object,
                    "edit" | "edit_mesh" | "face" => {
                        InteractionMode::EditMesh(EditElementMode::Face)
                    }
                    "vertex" => InteractionMode::EditMesh(EditElementMode::Vertex),
                    "edge" => InteractionMode::EditMesh(EditElementMode::Edge),
                    "sculpt" => InteractionMode::Sculpt,
                    "voxel" => InteractionMode::Voxel,
                    _ => scene.mode,
                };
            }

            IpcResponse::success(
                "Updated camera and viewport settings",
                summarize_scene(scene, camera, *shading, *show_wireframe, *show_grid),
            )
        }

        "undo" => {
            if let Some(lbl) = history.undo(scene) {
                ui_state.status_message = format!("MCP Undid: {lbl}");
                IpcResponse::success(format!("Undid action: {lbl}"), json!({ "undone": lbl }))
            } else {
                IpcResponse::error("Nothing to undo")
            }
        }

        "redo" => {
            if let Some(lbl) = history.redo(scene) {
                ui_state.status_message = format!("MCP Redid: {lbl}");
                IpcResponse::success(format!("Redid action: {lbl}"), json!({ "redone": lbl }))
            } else {
                IpcResponse::error("Nothing to redo")
            }
        }

        "export_model" => {
            let Some(path_str) = args.get("path").and_then(|v| v.as_str()) else {
                return IpcResponse::error("Missing 'path' argument for export_model");
            };
            let path = Path::new(path_str);
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("obj")
                .to_lowercase();
            if matches!(ext.as_str(), "glb" | "gltf") {
                // Game-ready glTF: chosen parts (default: everything visible), with options.
                let chosen = ["all", "object_ids", "names", "group"].iter().any(|k| !args[*k].is_null())
                    || args["selection"].as_bool().unwrap_or(false);
                let indices = if chosen {
                    match crate::library::parts_from_args(scene, args) {
                        Ok(indices) => indices,
                        Err(e) => return IpcResponse::error(e),
                    }
                } else {
                    crate::library::visible_parts(scene)
                };
                let parts: Vec<&crate::scene::SceneObject> =
                    indices.iter().map(|&i| &scene.objects[i]).collect();
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Model");
                // Inside a Godot project the file is written the way Godot's importer needs it.
                let options = crate::gltf::GltfOptions {
                    root_name: stem.to_string(),
                    godot: crate::gltf::godot_resource_path(path).is_some(),
                    ..Default::default()
                };
                let report = match options.with_args(args).and_then(|o| crate::gltf::export_glb(&parts, path, &o)) {
                    Ok(report) => report,
                    Err(e) => return IpcResponse::error(e),
                };
                let mut data = report.to_json();
                data["path"] = json!(path_str);
                let mut message = format!(
                    "Exported {} part(s) to {} ({} triangles, {} material(s))",
                    report.parts,
                    path.display(),
                    report.triangles,
                    report.materials
                );
                if report.textures > 0 {
                    message.push_str(&format!(", {} px texture", report.texture_size));
                }
                if let Some((_, res)) = crate::gltf::godot_resource_path(path) {
                    data["godot_path"] = json!(res);
                    message.push_str(&format!(" · {res}"));
                }
                ui_state.status_message = message.clone();
                return IpcResponse::success(message, data);
            }
            let res = match ext.as_str() {
                "stl" => io::export_stl_file(scene, path),
                _ => io::export_obj_file(scene, path),
            };
            match res {
                Ok(()) => {
                    ui_state.status_message = format!("MCP exported to {}", path.display());
                    IpcResponse::success(
                        format!("Exported scene to {}", path.display()),
                        json!({ "path": path_str }),
                    )
                }
                Err(e) => IpcResponse::error(e),
            }
        }

        "import_model" => {
            let Some(path_str) = args.get("path").and_then(|v| v.as_str()) else {
                return IpcResponse::error("Missing 'path' argument for import_model");
            };
            let path = Path::new(path_str);
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("obj")
                .to_lowercase();
            let res = match ext.as_str() {
                "stl" => io::import_stl_file(path),
                "glb" | "gltf" => io::import_glb_file(path),
                _ => io::import_obj_file(path),
            };
            match res {
                Ok(mesh) => {
                    history.record("MCP: Import Model", scene);
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .or_else(|| path.file_stem().and_then(|s| s.to_str()))
                        .unwrap_or("Imported Model");
                    let color = if matches!(ext.as_str(), "glb" | "gltf") {
                        [1.0; 3]
                    } else {
                        [0.80, 0.82, 0.85]
                    };
                    let idx = scene.add_mesh_object(name, mesh, color);
                    let obj = &scene.objects[idx];
                    ui_state.status_message = format!("MCP imported '{}'", obj.name);
                    IpcResponse::success(
                        format!("Imported '{}' from {}", obj.name, path.display()),
                        json!({
                            "object_id": obj.id,
                            "name": obj.name,
                            "vertices": obj.mesh.positions.len(),
                            "faces": obj.mesh.faces.len(),
                        }),
                    )
                }
                Err(e) => IpcResponse::error(e),
            }
        }

        "capture_studio" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("/tmp/spatial_studio.png");
            let width = args
                .get("width")
                .and_then(|v| v.as_u64())
                .unwrap_or(1440)
                .clamp(1100, 2560) as u32;
            let height = args
                .get("height")
                .and_then(|v| v.as_u64())
                .unwrap_or(960)
                .clamp(700, 1600) as u32;
            let tab = args.get("tab").and_then(|v| v.as_u64()).unwrap_or(0).min(1) as usize;
            // Optional before/after wipe of a revision.
            let mut compare_scene = None;
            let mut split = None;
            if let Some(rid) = args.get("compare_revision").and_then(|v| v.as_u64()) {
                if !scene.revisions.iter().any(|r| r.id == rid && r.can_compare()) {
                    return IpcResponse::error("Revision not found or has no geometry to compare");
                }
                let mut s = scene.clone();
                s.preview_revision = Some((rid, false));
                compare_scene = Some(s);
                split = Some(args.get("split").and_then(|v| v.as_f64()).unwrap_or(0.5).clamp(0.02, 0.98) as f32);
            }
            match crate::renderer::render_headless(
                camera,
                compare_scene.as_ref().unwrap_or(scene),
                *shading,
                *show_wireframe,
                *show_grid,
                width,
                height,
                Path::new(path),
                Some(tab),
                split,
            ) {
                Ok(()) => IpcResponse::success(
                    "Studio captured",
                    json!({"path":path,"width":width,"height":height}),
                ),
                Err(e) => IpcResponse::error(e),
            }
        }
        "capture_viewport" => {
            let path_str = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("/tmp/rust_modeler_viewport.png");
            let width = args.get("width").and_then(|v| v.as_u64()).unwrap_or(960) as u32;
            let height = args.get("height").and_then(|v| v.as_u64()).unwrap_or(640) as u32;

            let mut snap_cam = camera.clone();
            snap_cam.aspect = width as f32 / height.max(1) as f32;
            // frame=scene|selection|object|comment aims the camera at that region first.
            match crate::agent_tools::frame_bounds(scene, args) {
                Ok(Some(bounds)) => crate::agent_tools::frame_camera(
                    &mut snap_cam,
                    bounds,
                    args.get("padding").and_then(|v| v.as_f64()).unwrap_or(1.25) as f32,
                ),
                Ok(None) => {}
                Err(e) => return IpcResponse::error(e),
            }
            if !args["target"].is_null() {
                snap_cam.target = parse_vec3(&args["target"], snap_cam.target);
            }
            if let Some(d) = args.get("distance").and_then(|v| v.as_f64()) {
                snap_cam.distance = (d as f32).clamp(0.4, 200.0);
            }
            if let Some(yaw) = args.get("yaw_deg").and_then(|v| v.as_f64()) {
                snap_cam.yaw = (yaw as f32).to_radians();
            }
            if let Some(pitch) = args.get("pitch_deg").and_then(|v| v.as_f64()) {
                let max_p = std::f32::consts::FRAC_PI_2 - 0.02;
                snap_cam.pitch = (pitch as f32).to_radians().clamp(-max_p, max_p);
            }

            let path = Path::new(path_str);
            match render_scene_headless_to_png(
                &snap_cam,
                scene,
                *shading,
                *show_wireframe,
                *show_grid,
                width,
                height,
                path,
            ) {
                Ok(()) => {
                    ui_state.status_message =
                        format!("Captured viewport screenshot to {}", path.display());
                    IpcResponse::success(
                        format!(
                            "Rendered {}x{} viewport screenshot to {}",
                            width,
                            height,
                            path.display()
                        ),
                        json!({
                            "path": path_str,
                            "width": width,
                            "height": height,
                        }),
                    )
                }
                Err(e) => IpcResponse::error(e),
            }
        }

        "get_comments" => {
            let include_resolved = args
                .get("include_resolved")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let include_drafts = args
                .get("include_drafts")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let comments: Vec<Value> = scene
                .comments
                .iter()
                .filter(|c| include_resolved || !c.resolved)
                .filter(|c| include_drafts || !c.draft)
                .map(|c| {
                    json!({
                        "id": c.id,
                        "is_general": c.is_general || (c.object_id == 0 && c.volume.is_none()),
                        "volume": c.volume.as_ref().map(|v| json!({
                            "label": v.label,
                            "center": v.center.to_array(),
                            "size": (v.half_extents * 2.0).to_array(),
                            "min": v.min().to_array(),
                            "max": v.max().to_array(),
                            "space": "world",
                            "contains_geometry": !scene.faces_in_volume(v).is_empty(),
                        })),
                        "object_id": c.object_id,
                        "object_name": c.object_name,
                        "comment": c.comment,
                        "world_anchor": scene.current_comment_anchor(c).to_array(),
                        "local_anchor": c.local_anchor.to_array(),
                        "surface_normal": c.surface_normal.to_array(),
                        "selection_radius": c.selection_radius,
                        "selected_faces_count": c.selected_faces.len(),
                        "selected_faces": c.targets.first().and_then(|t| scene.target_faces(t)).unwrap_or_else(|| c.selected_faces.clone()),
                        "local_bounds_min": c.local_bounds_min.to_array(),
                        "local_bounds_max": c.local_bounds_max.to_array(),
                        "targets_count": c.targets.len().max(if c.is_general { 0 } else { 1 }),
                        "targets": scene.resolved_targets(c),
                        "status": c.status,
                        "needs_reattachment": scene.comment_needs_reattachment(c),
                        "replies": c.replies,
                        "resolved": c.resolved,
                        "agent_reply": c.agent_reply,
                        "draft": c.draft,
                        "attachments": c.attachments,
                    })
                })
                .collect();

            IpcResponse::success(
                format!("Retrieved {} comment(s)", comments.len()),
                json!({
                    "count": comments.len(),
                    "comments": comments,
                    "feedback_batch": scene.feedback_batch,
                    "drafts_pending": scene.draft_count(),
                }),
            )
        }

        "get_feedback_state" => {
            let open = scene
                .comments
                .iter()
                .filter(|c| !c.draft && !c.resolved && c.status == crate::scene::CommentStatus::Open)
                .count();
            IpcResponse::success(
                "Feedback state",
                json!({
                    "feedback_batch": scene.feedback_batch,
                    "open_comments": open,
                    "drafts_pending": scene.draft_count(),
                    "revision_pending": scene.revisions.iter().any(|r| matches!(r.status, crate::review::RevisionStatus::InProgress | crate::review::RevisionStatus::ReadyForReview)),
                }),
            )
        }

        "wait_for_feedback" => {
            // Blocking waits are handled by the MCP/CLI client so the GUI never stalls.
            IpcResponse::error("wait_for_feedback must be called through the MCP server or --cmd")
        }

        "send_comments" => {
            let n = scene.send_drafts();
            IpcResponse::success(
                format!("Sent {n} comment(s) to the agent"),
                json!({"sent": n, "feedback_batch": scene.feedback_batch}),
            )
        }

        "request_changes" => {
            let id = args.get("revision_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let notes = args.get("notes").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let before = scene.clone();
            match scene.request_revision_changes(id, notes) {
                Ok(()) => {
                    if ui_state.request_changes_for == Some(id) {
                        ui_state.request_changes_for = None;
                        ui_state.request_notes.clear();
                    }
                    history.record("Request changes", &before);
                    IpcResponse::success(
                        "Revision reverted; the comment is open again with your notes",
                        json!({"revision_id": id, "feedback_batch": scene.feedback_batch}),
                    )
                }
                Err(e) => IpcResponse::error(e),
            }
        }

        "add_comment" => {
            let Some(comment_text) = args.get("comment").and_then(|v| v.as_str()) else {
                return IpcResponse::error("Missing 'comment' string argument");
            };
            let force_general = args
                .get("is_general")
                .or_else(|| args.get("general"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if force_general {
                let added = scene.add_general_comment(comment_text);
                ui_state.status_message = format!("Added General Comment #{}", added.id);
                return IpcResponse::success(
                    format!("Added General Scene Comment #{}", added.id),
                    json!({
                        "comment_id": added.id,
                        "is_general": true,
                        "object_id": 0,
                        "object_name": added.object_name,
                    }),
                );
            }

            if let Some(idx) = resolve_object_index(scene, args) {
                scene.selected_object = Some(idx);
            }
            let radius = args
                .get("radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(ui_state.selection_radius as f64) as f32;

            if !args["local_point"].is_null() {
                if let Some(obj) = scene.active_object_mut() {
                    let lp = parse_vec3(&args["local_point"], Vec3::ZERO);
                    let avg_scale = ((obj.transform.scale.x.abs()
                        + obj.transform.scale.y.abs()
                        + obj.transform.scale.z.abs())
                        / 3.0)
                        .max(0.01);
                    obj.mesh
                        .select_faces_in_sphere(lp, radius / avg_scale, None, false, false);
                    let wp = obj.transform.matrix().transform_point3(lp);
                    let oid = obj.id;
                    scene.pending_anchor = Some((oid, wp, lp, Vec3::Y));
                }
            }

            if let Some(added) = scene.add_comment_for_selection(comment_text, radius) {
                ui_state.status_message =
                    format!("Added Comment #{} on '{}'", added.id, added.object_name);
                IpcResponse::success(
                    format!(
                        "Added Comment #{} on '{}' ({} faces)",
                        added.id,
                        added.object_name,
                        added.selected_faces.len()
                    ),
                    json!({
                        "comment_id": added.id,
                        "is_general": false,
                        "object_id": added.object_id,
                        "object_name": added.object_name,
                        "world_anchor": added.world_anchor.to_array(),
                        "local_anchor": added.local_anchor.to_array(),
                        "selected_faces_count": added.selected_faces.len(),
                        "local_bounds_min": added.local_bounds_min.to_array(),
                        "local_bounds_max": added.local_bounds_max.to_array(),
                    }),
                )
            } else {
                // Fall back to a general scene comment when no object is selected
                let added = scene.add_general_comment(comment_text);
                ui_state.status_message = format!("Added General Comment #{}", added.id);
                IpcResponse::success(
                    format!("Added General Scene Comment #{}", added.id),
                    json!({
                        "comment_id": added.id,
                        "is_general": true,
                        "object_id": 0,
                        "object_name": added.object_name,
                    }),
                )
            }
        }

        "select_comment_region" | "focus_comment" => {
            let Some(cid) = args
                .get("comment_id")
                .or_else(|| args.get("id"))
                .and_then(|v| v.as_u64())
            else {
                return IpcResponse::error("Missing 'comment_id' integer argument");
            };
            if let Some(wp) = scene.focus_comment(cid) {
                if args
                    .get("focus_camera")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
                {
                    camera.target = wp;
                }
                let comment = scene.comments.iter().find(|c| c.id == cid).cloned();
                ui_state.status_message = format!("AI focused Comment #{cid}");
                IpcResponse::success(
                    format!(
                        "Selected object and restored highlighted face region for Comment #{cid}"
                    ),
                    json!({
                        "comment_id": cid,
                        "world_anchor": wp.to_array(),
                        "comment": comment,
                    }),
                )
            } else {
                IpcResponse::error(format!(
                    "Comment #{cid} could not be focused; it may need reattachment"
                ))
            }
        }

        "resolve_comment" => {
            let Some(cid) = args
                .get("comment_id")
                .or_else(|| args.get("id"))
                .and_then(|v| v.as_u64())
            else {
                return IpcResponse::error("Missing 'comment_id' integer argument");
            };
            let resolved = args
                .get("resolved")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let reply = args
                .get("reply")
                .or_else(|| args.get("agent_reply"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            if scene.revisions.iter().any(|r| {
                r.comment_id == Some(cid)
                    && matches!(
                        r.status,
                        crate::review::RevisionStatus::InProgress
                            | crate::review::RevisionStatus::ReadyForReview
                    )
            }) {
                return IpcResponse::error(
                    "Submit and review the linked revision instead of resolving its comment directly",
                );
            }
            if let Some(c) = scene.comments.iter_mut().find(|c| c.id == cid) {
                c.resolved = resolved;
                c.status = if resolved {
                    crate::scene::CommentStatus::Accepted
                } else {
                    crate::scene::CommentStatus::Open
                };
                if let Some(r) = reply {
                    c.agent_reply = Some(r);
                }
                let updated = c.clone();
                // Clear face highlight if resolved
                if resolved {
                    if !updated.targets.is_empty() {
                        for t in &updated.targets {
                            if let Some(obj) =
                                scene.objects.iter_mut().find(|o| o.id == t.object_id)
                            {
                                obj.mesh.clear_selection();
                            }
                        }
                    } else if let Some(obj) =
                        scene.objects.iter_mut().find(|o| o.id == updated.object_id)
                    {
                        obj.mesh.clear_selection();
                    }
                    scene.pending_anchor = None;
                }
                ui_state.status_message = format!(
                    "AI {} Comment #{cid}",
                    if resolved { "resolved" } else { "updated" }
                );
                IpcResponse::success(
                    format!("Updated Comment #{cid} (resolved={resolved})"),
                    json!({
                        "comment_id": updated.id,
                        "object_id": updated.object_id,
                        "resolved": updated.resolved,
                        "agent_reply": updated.agent_reply,
                    }),
                )
            } else {
                IpcResponse::error(format!("Comment #{cid} not found"))
            }
        }

        other => crate::agent_tools::dispatch(
            req,
            scene,
            camera,
            history,
            ui_state,
            shading,
            show_wireframe,
            show_grid,
        )
        .or_else(|| crate::library::dispatch(req, scene, history, ui_state))
        .unwrap_or_else(|| IpcResponse::error(format!("Unknown MCP tool '{other}'"))),
    }
}
