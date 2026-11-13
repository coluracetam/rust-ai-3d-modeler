//! Agent-facing tools beyond the core dispatcher: geometry queries, face selection, direct mesh
//! creation, extra modelling operations, per-face colour, multi-view captures, health checks
//! and batching. `dispatch` returns `None` for tools it does not know.
use crate::camera::OrbitCamera;
use crate::history::UndoHistory;
use crate::ipc::{IpcRequest, IpcResponse, parse_color, parse_vec3, resolve_object_index};
use crate::mesh::{Mesh, SimilarBy};
use crate::modeling_ops as ops;
use crate::renderer::{ViewportShading, render_scene_headless_to_png};
use crate::scene::{Scene, SceneObject, SelectionVolume};
use crate::ui::UiState;
use glam::{Mat4, Vec2, Vec3};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::Path;

pub const READ_ONLY: &[&str] = &["query_geometry", "measure", "check_mesh", "capture_views"];
pub const GEOMETRY: &[&str] = &[
    "add_mesh",
    "add_extrude",
    "add_loft",
    "deform_object",
    "align_object",
    "scatter_object",
    "paint_faces",
];
/// Tools worth running off the GUI thread.
pub const HEAVY: &[&str] = &[
    "add_mesh",
    "add_extrude",
    "add_loft",
    "deform_object",
    "scatter_object",
    "paint_faces",
    "capture_views",
    "check_mesh",
    "batch",
];

fn f32_arg(args: &Value, key: &str, default: f32) -> f32 {
    args.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
}

fn bool_arg(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn usize_arg(args: &Value, key: &str, default: usize) -> usize {
    args.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
}

fn axis_arg(args: &Value, key: &str, default: usize) -> Result<usize, String> {
    match args.get(key).and_then(|v| v.as_str()) {
        Some(s) => crate::modeling::axis_index(s),
        None => Ok(default),
    }
}

fn round3(v: Vec3) -> [f32; 3] {
    [
        (v.x * 1e4).round() / 1e4,
        (v.y * 1e4).round() / 1e4,
        (v.z * 1e4).round() / 1e4,
    ]
}

pub fn world_bounds(obj: &SceneObject) -> Option<(Vec3, Vec3)> {
    if obj.mesh.positions.is_empty() {
        return None;
    }
    let model = obj.transform.matrix();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &p in &obj.mesh.positions {
        let w = model.transform_point3(p);
        min = min.min(w);
        max = max.max(w);
    }
    Some((min, max))
}

fn faces_world_bounds(obj: &SceneObject, faces: impl Iterator<Item = usize>) -> Option<(Vec3, Vec3)> {
    let model = obj.transform.matrix();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for fi in faces {
        let Some(face) = obj.mesh.faces.get(fi) else { continue };
        for &vi in face {
            let w = model.transform_point3(obj.mesh.positions[vi]);
            min = min.min(w);
            max = max.max(w);
            any = true;
        }
    }
    any.then_some((min, max))
}

fn merge(a: Option<(Vec3, Vec3)>, b: Option<(Vec3, Vec3)>) -> Option<(Vec3, Vec3)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
        (a, b) => a.or(b),
    }
}

/// World bounds an image should frame, from `frame`:
/// "scene" (default), "selection", "object" (object_id / name) or "comment" (comment_id).
pub fn frame_bounds(scene: &Scene, args: &Value) -> Result<Option<(Vec3, Vec3)>, String> {
    let scene_bounds = || {
        scene
            .objects
            .iter()
            .filter(|o| o.visible)
            .fold(None, |acc, o| merge(acc, world_bounds(o)))
    };
    let what = args.get("frame").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
    match what.as_str() {
        "" => Ok(None),
        "scene" | "all" => Ok(scene_bounds()),
        "selection" => {
            let mut b = None;
            for o in &scene.objects {
                if !o.mesh.selected_faces.is_empty() {
                    b = merge(b, faces_world_bounds(o, o.mesh.selected_faces.iter().copied()));
                }
            }
            if b.is_none() {
                b = scene.selection_volume.as_ref().map(|v| (v.min(), v.max()));
            }
            if b.is_none() {
                b = scene.active_object().and_then(world_bounds);
            }
            b.map(Some).ok_or_else(|| "frame=selection but nothing is selected".to_string())
        }
        "object" => {
            let idx = resolve_object_index(scene, args).ok_or("frame=object needs object_id or name")?;
            Ok(world_bounds(&scene.objects[idx]))
        }
        "comment" => {
            let id = args
                .get("comment_id")
                .and_then(|v| v.as_u64())
                .ok_or("frame=comment needs comment_id")?;
            let c = scene
                .comments
                .iter()
                .find(|c| c.id == id)
                .ok_or(format!("Comment #{id} not found"))?;
            let mut b = c.volume.as_ref().map(|v| (v.min(), v.max()));
            for t in scene.resolved_targets(c) {
                if let Some(o) = scene.objects.iter().find(|o| o.id == t.object_id) {
                    b = merge(
                        b,
                        if t.selected_faces.is_empty() {
                            world_bounds(o)
                        } else {
                            faces_world_bounds(o, t.selected_faces.iter().copied())
                        },
                    );
                }
            }
            Ok(b.or_else(scene_bounds))
        }
        other => Err(format!("Unknown frame '{other}'. Use scene, selection, object or comment.")),
    }
}

/// Aim `camera` so `bounds` fills the view, keeping its current viewing direction.
pub fn frame_camera(camera: &mut OrbitCamera, bounds: (Vec3, Vec3), padding: f32) {
    camera.target = (bounds.0 + bounds.1) * 0.5;
    let radius = ((bounds.1 - bounds.0).length() * 0.5).max(0.01);
    let half = (camera.fov_y_radians * 0.5)
        .min(((camera.fov_y_radians * 0.5).tan() * camera.aspect).atan());
    camera.distance = (radius / half.sin().max(0.1) * padding).clamp(0.05, 400.0);
}

fn view_direction(name: &str) -> Option<Vec3> {
    Some(match name {
        "front" => Vec3::new(0.0, 0.0, 1.0),
        "back" => Vec3::new(0.0, 0.0, -1.0),
        "right" => Vec3::new(1.0, 0.0, 0.0),
        "left" => Vec3::new(-1.0, 0.0, 0.0),
        "top" => Vec3::new(0.0, 1.0, 0.02),
        "bottom" => Vec3::new(0.0, -1.0, 0.02),
        "iso" | "iso_front_right" => Vec3::new(1.0, 0.75, 1.0),
        "iso_front_left" => Vec3::new(-1.0, 0.75, 1.0),
        "iso_back_right" => Vec3::new(1.0, 0.75, -1.0),
        "iso_back_left" | "iso_back" => Vec3::new(-1.0, 0.75, -1.0),
        _ => return None,
    })
}

fn object_report(obj: &SceneObject) -> Value {
    let r = ops::check_mesh(&obj.mesh);
    let mut issues: Vec<String> = Vec::new();
    if r.boundary_edges > 0 {
        issues.push(format!("{} open boundary edges (holes); edit_mesh fill_holes closes them", r.boundary_edges));
    }
    if r.non_manifold_edges > 0 {
        issues.push(format!("{} edges shared by more than two faces", r.non_manifold_edges));
    }
    if r.degenerate_faces > 0 {
        issues.push(format!("{} zero-area faces; edit_mesh weld removes them", r.degenerate_faces));
    }
    if r.inconsistent_edges > 0 {
        issues.push(format!("{} edges where neighbouring faces are wound in opposite senses (some faces are flipped); select them and use edit_mesh flip_normals", r.inconsistent_edges));
    }
    if r.inside_out {
        issues.push("faces are wound inside-out; edit_mesh flip_normals fixes it".into());
    }
    json!({
        "object_id": obj.id,
        "name": obj.name,
        "faces": obj.mesh.faces.len(),
        "vertices": obj.mesh.positions.len(),
        "watertight": r.watertight,
        "inside_out": r.inside_out,
        "boundary_edges": r.boundary_edges,
        "non_manifold_edges": r.non_manifold_edges,
        "degenerate_faces": r.degenerate_faces,
        "inconsistent_winding_edges": r.inconsistent_edges,
        "connected_pieces": r.components,
        "signed_volume": r.signed_volume,
        "issues": issues,
    })
}

fn selection_op(args: &Value) -> Result<(bool, bool), String> {
    match args.get("op").and_then(|v| v.as_str()).unwrap_or("replace") {
        "replace" | "set" => Ok((false, false)),
        "add" => Ok((true, false)),
        "subtract" | "remove" => Ok((false, true)),
        other => Err(format!("Unknown op '{other}'. Use replace, add or subtract.")),
    }
}

fn points3(v: &Value) -> Vec<Vec3> {
    v.as_array()
        .map(|a| a.iter().map(|p| parse_vec3(p, Vec3::NAN)).collect())
        .unwrap_or_default()
}

fn points2(v: &Value) -> Vec<Vec2> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let p = p.as_array()?;
                    Some(Vec2::new(p.first()?.as_f64()? as f32, p.get(1)?.as_f64()? as f32))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn place_new_object(scene: &mut Scene, idx: usize, args: &Value) -> Value {
    let obj = &mut scene.objects[idx];
    obj.transform.translation = parse_vec3(&args["position"], Vec3::ZERO);
    obj.transform.rotation_deg = parse_vec3(&args["rotation_deg"], Vec3::ZERO);
    obj.transform.scale = parse_vec3(&args["scale"], Vec3::ONE);
    if let Some(flat) = args.get("flat_shading").and_then(|v| v.as_bool()) {
        obj.flat_shading = flat;
    }
    let b = world_bounds(obj).unwrap_or((Vec3::ZERO, Vec3::ZERO));
    json!({
        "object_id": obj.id,
        "name": obj.name,
        "vertices": obj.mesh.positions.len(),
        "faces": obj.mesh.faces.len(),
        "world_bounds_min": round3(b.0),
        "world_bounds_max": round3(b.1),
    })
}

/// Extra `edit_mesh` operations. `None` means the operation is not one of these.
pub fn edit_mesh_op(op: &str, mesh: &mut Mesh, args: &Value) -> Option<Result<String, String>> {
    Some(match op {
        "bridge" => ops::bridge_selected(mesh).map(|n| format!("bridged with {n} quads")),
        "fill_holes" | "fill" => Ok(format!("filled {} hole(s)", ops::fill_holes(mesh))),
        "weld" | "merge_vertices" => {
            let removed = ops::weld(mesh, f32_arg(args, "tolerance", 1e-4));
            Ok(format!("merged {removed} duplicate vertices"))
        }
        "flip_normals" => {
            let selected = mesh.selected_faces.clone();
            for (fi, face) in mesh.faces.iter_mut().enumerate() {
                if selected.is_empty() || selected.contains(&fi) {
                    face.reverse();
                }
            }
            mesh.recalculate_normals();
            mesh.bump_version();
            Ok("flipped face winding".into())
        }
        "smooth" => {
            ops::smooth_vertices(mesh, f32_arg(args, "strength", 0.5), usize_arg(args, "iterations", 2));
            Ok("relaxed vertices".into())
        }
        "subdivide_smooth" | "catmull_clark" => {
            ops::subdivide_smooth(mesh, usize_arg(args, "levels", 1)).map(|_| "subdivided and smoothed".into())
        }
        "decimate" => {
            let (before, after) = ops::decimate(mesh, f32_arg(args, "ratio", 0.5));
            Ok(format!("reduced {before} faces to {after}"))
        }
        "invert_selection" => {
            mesh.invert_selection();
            Ok(format!("{} faces selected", mesh.selected_faces.len()))
        }
        "grow_selection" => {
            for _ in 0..usize_arg(args, "steps", 1).clamp(1, 64) {
                mesh.grow_selection();
            }
            Ok(format!("{} faces selected", mesh.selected_faces.len()))
        }
        "shrink_selection" => {
            for _ in 0..usize_arg(args, "steps", 1).clamp(1, 64) {
                mesh.shrink_selection();
            }
            Ok(format!("{} faces selected", mesh.selected_faces.len()))
        }
        _ => return None,
    })
}

pub const EXTRA_EDIT_OPS: &str = "bridge, fill_holes, weld, flip_normals, smooth, subdivide_smooth, decimate, invert_selection, grow_selection, shrink_selection";

#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    req: &IpcRequest,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    history: &mut UndoHistory,
    ui_state: &mut UiState,
    shading: &mut ViewportShading,
    show_wireframe: &mut bool,
    show_grid: &mut bool,
) -> Option<IpcResponse> {
    let args = &req.args;
    macro_rules! tri {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => return Some(IpcResponse::error(e.to_string())),
            }
        };
    }
    Some(match req.tool.as_str() {
        // ------------------------------------------------------------------ queries
        "query_geometry" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active object; pass object_id or name"));
            };
            let obj = &scene.objects[idx];
            let model = obj.transform.matrix();
            let normal_mat = model.inverse().transpose();
            let explicit: Option<Vec<usize>> = args.get("faces").and_then(|v| v.as_array()).map(|a| {
                a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()
            });
            let scope = args.get("scope").and_then(|v| v.as_str()).unwrap_or(
                if explicit.is_some() {
                    "faces"
                } else if obj.mesh.selected_faces.is_empty() {
                    "object"
                } else {
                    "selection"
                },
            );
            let mut faces: Vec<usize> = match scope {
                "faces" => explicit.unwrap_or_default(),
                "selection" => obj.mesh.selected_faces.iter().copied().collect(),
                "object" | "all" => (0..obj.mesh.faces.len()).collect(),
                other => {
                    return Some(IpcResponse::error(format!(
                        "Unknown scope '{other}'. Use object, selection or faces."
                    )));
                }
            };
            faces.retain(|&f| f < obj.mesh.faces.len());
            faces.sort_unstable();
            let total = faces.len();
            let limit = usize_arg(args, "max_faces", 60).min(2000);
            let offset = usize_arg(args, "offset", 0);
            let area: f32 = faces.iter().map(|&f| obj.mesh.face_area(f)).sum();
            let bounds = faces_world_bounds(obj, faces.iter().copied());
            let listed: Vec<Value> = faces
                .iter()
                .skip(offset)
                .take(limit)
                .map(|&fi| {
                    let mut entry = json!({
                        "index": fi,
                        "center": round3(model.transform_point3(obj.mesh.face_center(fi))),
                        "normal": round3(normal_mat.transform_vector3(obj.mesh.face_normal(fi)).normalize_or_zero()),
                        "area": obj.mesh.face_area(fi),
                        "corners": obj.mesh.faces[fi].len(),
                    });
                    if bool_arg(args, "include_vertices", false) {
                        entry["vertices"] = json!(
                            obj.mesh.faces[fi]
                                .iter()
                                .map(|&vi| round3(model.transform_point3(obj.mesh.positions[vi])))
                                .collect::<Vec<_>>()
                        );
                        entry["vertex_indices"] = json!(obj.mesh.faces[fi]);
                    }
                    entry
                })
                .collect();
            // Regions: connected patches of the queried faces.
            let mut probe = obj.mesh.clone();
            probe.selected_faces = faces.iter().copied().collect();
            let regions: Vec<Value> = probe
                .selected_components()
                .iter()
                .take(64)
                .map(|group| {
                    let b = faces_world_bounds(obj, group.iter().copied()).unwrap_or((Vec3::ZERO, Vec3::ZERO));
                    let n = group
                        .iter()
                        .map(|&f| obj.mesh.face_normal(f) * obj.mesh.face_area(f))
                        .sum::<Vec3>();
                    json!({
                        "faces": group.len(),
                        "first_face": group[0],
                        "bounds_min": round3(b.0),
                        "bounds_max": round3(b.1),
                        "center": round3((b.0 + b.1) * 0.5),
                        "average_normal": round3(normal_mat.transform_vector3(n).normalize_or_zero()),
                    })
                })
                .collect();
            let wb = world_bounds(obj).unwrap_or((Vec3::ZERO, Vec3::ZERO));
            let (lmin, lmax) = obj.mesh.bounds();
            IpcResponse::success(
                format!("{} face(s) of '{}' ({scope})", total, obj.name),
                json!({
                    "object_id": obj.id,
                    "name": obj.name,
                    "space": "world",
                    "units": scene.units,
                    "scope": scope,
                    "object_world_bounds_min": round3(wb.0),
                    "object_world_bounds_max": round3(wb.1),
                    "object_size": round3(wb.1 - wb.0),
                    "object_local_bounds_min": round3(lmin),
                    "object_local_bounds_max": round3(lmax),
                    "total_object_faces": obj.mesh.faces.len(),
                    "face_count": total,
                    "total_area": area,
                    "bounds_min": bounds.map(|b| round3(b.0)),
                    "bounds_max": bounds.map(|b| round3(b.1)),
                    "regions": regions,
                    "faces": listed,
                    "faces_truncated": total > offset + limit,
                }),
            )
        }

        "measure" => {
            if let Some(points) = args.get("points").and_then(|v| v.as_array()) {
                let pts: Vec<Vec3> = points.iter().map(|p| parse_vec3(p, Vec3::NAN)).collect();
                if pts.len() < 2 || pts.iter().any(|p| !p.is_finite()) {
                    return Some(IpcResponse::error("points needs at least two [x, y, z] entries"));
                }
                let segments: Vec<f32> = pts.windows(2).map(|w| w[0].distance(w[1])).collect();
                let total: f32 = segments.iter().sum();
                return Some(IpcResponse::success(
                    format!("Distance {total:.4} {}", scene.units),
                    json!({"distance": total, "segments": segments, "delta": round3(pts[pts.len() - 1] - pts[0])}),
                ));
            }
            let Some(a) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("Pass object_id (or name) and other_object_id, or points"));
            };
            let other = args.get("other_object_id").and_then(|v| v.as_u64());
            let Some(b) = other.and_then(|id| scene.objects.iter().position(|o| o.id == id)) else {
                return Some(IpcResponse::error("other_object_id not found"));
            };
            let (Some(ba), Some(bb)) = (world_bounds(&scene.objects[a]), world_bounds(&scene.objects[b])) else {
                return Some(IpcResponse::error("One of the objects has no geometry"));
            };
            // Per axis: positive = gap between the boxes, negative = overlap depth.
            let gap = (ba.0 - bb.1).max(bb.0 - ba.1);
            let overlapping = gap.max_element() < 0.0;
            let (ca, cb) = ((ba.0 + ba.1) * 0.5, (bb.0 + bb.1) * 0.5);
            IpcResponse::success(
                format!(
                    "'{}' and '{}' {}",
                    scene.objects[a].name,
                    scene.objects[b].name,
                    if overlapping { "overlap (bounding boxes)" } else { "do not overlap" }
                ),
                json!({
                    "space": "world",
                    "a": {"object_id": scene.objects[a].id, "bounds_min": round3(ba.0), "bounds_max": round3(ba.1), "center": round3(ca)},
                    "b": {"object_id": scene.objects[b].id, "bounds_min": round3(bb.0), "bounds_max": round3(bb.1), "center": round3(cb)},
                    "center_distance": ca.distance(cb),
                    "center_delta": round3(cb - ca),
                    "axis_gap": round3(gap),
                    "bounding_boxes_overlap": overlapping,
                    "box_distance": gap.max(Vec3::ZERO).length(),
                }),
            )
        }

        "check_mesh" => {
            let single = if args.get("object_id").is_some() || args.get("name").is_some() {
                match resolve_object_index(scene, args) {
                    Some(i) => Some(i),
                    None => return Some(IpcResponse::error("Object not found")),
                }
            } else {
                None
            };
            let reports: Vec<Value> = scene
                .objects
                .iter()
                .enumerate()
                .filter(|(i, o)| single.is_none_or(|s| s == *i) && o.visible)
                .map(|(_, o)| object_report(o))
                .collect();
            let mut overlaps = Vec::new();
            let mut isolated = Vec::new();
            if single.is_none() {
                let boxes: Vec<(usize, (Vec3, Vec3))> = scene
                    .objects
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.visible)
                    .filter_map(|(i, o)| world_bounds(o).map(|b| (i, b)))
                    .collect();
                let scene_size = boxes
                    .iter()
                    .fold(None, |acc, (_, b)| merge(acc, Some(*b)))
                    .map(|b| (b.1 - b.0).length())
                    .unwrap_or(1.0);
                let touch = scene_size * 0.002;
                for (n, (i, a)) in boxes.iter().enumerate() {
                    let mut touches_any = boxes.len() <= 1;
                    for (m, (j, b)) in boxes.iter().enumerate() {
                        if n == m {
                            continue;
                        }
                        let gap = (a.0 - b.1).max(b.0 - a.1);
                        if gap.max_element() <= touch {
                            touches_any = true;
                        }
                        let depth = -gap.max_element();
                        if m > n && depth > touch && overlaps.len() < 60 {
                            overlaps.push(json!({
                                "a": scene.objects[*i].id,
                                "a_name": scene.objects[*i].name,
                                "b": scene.objects[*j].id,
                                "b_name": scene.objects[*j].name,
                                "overlap_depth": depth,
                            }));
                        }
                    }
                    if !touches_any {
                        isolated.push(json!({"object_id": scene.objects[*i].id, "name": scene.objects[*i].name}));
                    }
                }
            }
            let problems = reports
                .iter()
                .filter(|r| r["issues"].as_array().is_some_and(|a| !a.is_empty()))
                .count();
            IpcResponse::success(
                format!(
                    "Checked {} part(s): {} with issues, {} bounding-box overlaps, {} isolated",
                    reports.len(),
                    problems,
                    overlaps.len(),
                    isolated.len()
                ),
                json!({
                    "parts": reports,
                    "parts_with_issues": problems,
                    "bounding_box_overlaps": overlaps,
                    "isolated_parts": isolated,
                    "note": "Overlaps compare bounding boxes, so parts that are meant to interlock are listed too. Isolated parts touch no other part's bounding box (they may be floating).",
                }),
            )
        }

        // ---------------------------------------------------------------- selection
        "select_faces" => {
            let (additive, subtract) = tri!(selection_op(args));
            let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("indices").to_lowercase();
            let world = args.get("space").and_then(|v| v.as_str()).unwrap_or("world") != "local";
            let has_object = args.get("object_id").is_some() || args.get("name").is_some();
            // A world box without an object selects across every visible part.
            if mode == "box" && !has_object {
                if args["min"].is_null() || args["max"].is_null() {
                    return Some(IpcResponse::error("mode=box needs min and max [x, y, z]"));
                }
                let volume = SelectionVolume::from_min_max(
                    parse_vec3(&args["min"], Vec3::ZERO),
                    parse_vec3(&args["max"], Vec3::ZERO),
                );
                let hits = scene.faces_in_volume(&volume);
                if !additive && !subtract {
                    scene.clear_all_selections();
                }
                let mut total = 0;
                let mut per_object = Vec::new();
                for (&oi, faces) in &hits {
                    scene.objects[oi].mesh.apply_face_set_selection(faces, true, subtract);
                    total += faces.len();
                    per_object.push(json!({"object_id": scene.objects[oi].id, "name": scene.objects[oi].name, "faces_in_box": faces.len(), "selected_faces": scene.objects[oi].mesh.selected_faces.len()}));
                }
                if bool_arg(args, "show_box", false) {
                    scene.selection_volume = Some(volume);
                }
                crate::ui::update_pending_anchor_from_selection(scene);
                return Some(IpcResponse::success(
                    format!("{total} face(s) inside the box on {} part(s)", hits.len()),
                    json!({"faces_in_box": total, "objects": per_object}),
                ));
            }
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active object; pass object_id or name"));
            };
            scene.selected_object = Some(idx);
            let obj = &mut scene.objects[idx];
            let model = obj.transform.matrix();
            let seed = args.get("seed_face").and_then(|v| v.as_u64()).map(|v| v as usize);
            let need_seed = |seed: Option<usize>, count: usize| -> Result<usize, String> {
                match seed {
                    Some(s) if s < count => Ok(s),
                    Some(s) => Err(format!("seed_face {s} is out of range (the part has {count} faces)")),
                    None => Err("this mode needs seed_face".into()),
                }
            };
            let count = obj.mesh.faces.len();
            let set: HashSet<usize> = match mode.as_str() {
                "indices" | "faces" => args
                    .get("faces")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).filter(|&f| f < count).collect())
                    .unwrap_or_default(),
                "all" => (0..count).collect(),
                "none" | "clear" => {
                    obj.mesh.clear_selection();
                    HashSet::new()
                }
                "box" => {
                    if args["min"].is_null() || args["max"].is_null() {
                        return Some(IpcResponse::error("mode=box needs min and max [x, y, z]"));
                    }
                    let (a, b) = (parse_vec3(&args["min"], Vec3::ZERO), parse_vec3(&args["max"], Vec3::ZERO));
                    obj.mesh.faces_in_world_box(if world { model } else { Mat4::IDENTITY }, a.min(b), a.max(b))
                }
                "sphere" => {
                    let centre = parse_vec3(&args["center"], Vec3::NAN);
                    let radius = f32_arg(args, "radius", 0.0);
                    if !centre.is_finite() || radius <= 0.0 {
                        return Some(IpcResponse::error("mode=sphere needs center [x, y, z] and a positive radius"));
                    }
                    let m = if world { model } else { Mat4::IDENTITY };
                    (0..count)
                        .filter(|&f| m.transform_point3(obj.mesh.face_center(f)).distance(centre) <= radius)
                        .collect()
                }
                "connected" | "part" => obj.mesh.compute_connected_component(tri!(need_seed(seed, count))),
                "surface" => obj
                    .mesh
                    .compute_smart_surface(tri!(need_seed(seed, count)), f32_arg(args, "crease_deg", 32.0)),
                "normal" => {
                    let n = parse_vec3(&args["normal"], Vec3::Y).normalize_or_zero();
                    let local = if world {
                        obj.transform.inverse_matrix().transform_vector3(n).normalize_or_zero()
                    } else {
                        n
                    };
                    let min_dot = f32_arg(args, "min_dot", 0.65);
                    (0..count).filter(|&f| obj.mesh.face_normal(f).dot(local) >= min_dot).collect()
                }
                "similar" => {
                    let by = tri!(SimilarBy::parse(args.get("by").and_then(|v| v.as_str()).unwrap_or("normal")));
                    if obj.mesh.selected_faces.is_empty() {
                        return Some(IpcResponse::error("mode=similar compares against the current selection; select a face first"));
                    }
                    obj.mesh.compute_similar(by, f32_arg(args, "tolerance", by.default_tolerance()))
                }
                "path" => {
                    let (Some(from), Some(to)) = (
                        args.get("from_face").and_then(|v| v.as_u64()),
                        args.get("to_face").and_then(|v| v.as_u64()),
                    ) else {
                        return Some(IpcResponse::error("mode=path needs from_face and to_face"));
                    };
                    match obj.mesh.face_path(from as usize, to as usize) {
                        Some(p) => p.into_iter().collect(),
                        None => return Some(IpcResponse::error("No connected path between those faces")),
                    }
                }
                "invert" => {
                    obj.mesh.invert_selection();
                    obj.mesh.selected_faces.clone()
                }
                "grow" | "shrink" => {
                    for _ in 0..usize_arg(args, "steps", 1).clamp(1, 64) {
                        if mode == "grow" {
                            obj.mesh.grow_selection();
                        } else {
                            obj.mesh.shrink_selection();
                        }
                    }
                    obj.mesh.selected_faces.clone()
                }
                other => {
                    return Some(IpcResponse::error(format!(
                        "Unknown mode '{other}'. Use indices, all, none, box, sphere, connected, surface, normal, similar, path, invert, grow or shrink."
                    )));
                }
            };
            if !matches!(mode.as_str(), "none" | "clear" | "invert" | "grow" | "shrink") {
                obj.mesh.apply_face_set_selection(&set, additive, subtract);
            }
            let selected = obj.mesh.selected_faces.len();
            let bounds = faces_world_bounds(obj, obj.mesh.selected_faces.iter().copied());
            let regions = obj.mesh.selected_components().len();
            let (id, name) = (obj.id, obj.name.clone());
            crate::ui::update_pending_anchor_from_selection(scene);
            ui_state.status_message = format!("Agent selected {selected} face(s) on '{name}'");
            IpcResponse::success(
                format!("{selected} of {count} face(s) selected on '{name}' ({regions} region(s))"),
                json!({
                    "object_id": id,
                    "matched_faces": set.len(),
                    "selected_faces": selected,
                    "regions": regions,
                    "bounds_min": bounds.map(|b| round3(b.0)),
                    "bounds_max": bounds.map(|b| round3(b.1)),
                }),
            )
        }

        // --------------------------------------------------------------- generators
        "add_mesh" => {
            let positions = points3(&args["vertices"]);
            if positions.len() < 3 || positions.iter().any(|p| !p.is_finite()) {
                return Some(IpcResponse::error("vertices needs at least 3 finite [x, y, z] entries"));
            }
            if positions.len() > 2_000_000 {
                return Some(IpcResponse::error("too many vertices (limit 2,000,000)"));
            }
            let mut faces: Vec<Vec<usize>> = Vec::new();
            for (i, f) in args["faces"].as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter().enumerate() {
                let Some(list) = f.as_array() else {
                    return Some(IpcResponse::error(format!("faces[{i}] must be a list of vertex indices")));
                };
                let face: Vec<usize> = list.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
                if face.len() != list.len() || face.len() < 3 {
                    return Some(IpcResponse::error(format!("faces[{i}] needs at least 3 non-negative integer indices")));
                }
                if let Some(bad) = face.iter().find(|&&v| v >= positions.len()) {
                    return Some(IpcResponse::error(format!(
                        "faces[{i}] uses vertex {bad}, but there are only {} vertices (indices start at 0)",
                        positions.len()
                    )));
                }
                faces.push(face);
            }
            if faces.is_empty() {
                return Some(IpcResponse::error("faces must list at least one polygon"));
            }
            let mut mesh = Mesh::new(positions, faces);
            if let Some(cols) = args.get("vertex_colors").and_then(|v| v.as_array()) {
                if cols.len() != mesh.positions.len() {
                    return Some(IpcResponse::error("vertex_colors must have one [r, g, b] per vertex"));
                }
                mesh.colors = cols.iter().map(|c| parse_color(c, [1.0; 3])).collect();
            }
            let report = ops::check_mesh(&mesh);
            if report.inside_out && bool_arg(args, "fix_winding", true) {
                for f in &mut mesh.faces {
                    f.reverse();
                }
            }
            mesh.recalculate_normals();
            let color = parse_color(&args["color"], [0.78, 0.80, 0.85]);
            history.record("MCP: add_mesh", scene);
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Mesh");
            let idx = scene.add_mesh_object(name, mesh, color);
            scene.objects[idx].flat_shading = true;
            let mut data = place_new_object(scene, idx, args);
            data["watertight"] = json!(report.watertight);
            data["winding_fixed"] = json!(report.inside_out && bool_arg(args, "fix_winding", true));
            data["boundary_edges"] = json!(report.boundary_edges);
            IpcResponse::success(
                format!("Added mesh '{}' (object_id={})", data["name"].as_str().unwrap_or(""), data["object_id"]),
                data,
            )
        }

        "add_extrude" => {
            let color = parse_color(&args["color"], [0.78, 0.80, 0.85]);
            let mesh = tri!(ops::extrude_profile(
                &points2(&args["profile"]),
                f32_arg(args, "height", 1.0),
                f32_arg(args, "top_scale", 1.0),
                f32_arg(args, "twist_deg", 0.0),
                usize_arg(args, "segments", if f32_arg(args, "twist_deg", 0.0).abs() > 1e-3 { 16 } else { 1 }),
                bool_arg(args, "cap", true),
                [1.0; 3],
            ));
            history.record("MCP: add_extrude", scene);
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Extrusion");
            let idx = scene.add_mesh_object(name, mesh, color);
            scene.objects[idx].flat_shading = true;
            let data = place_new_object(scene, idx, args);
            IpcResponse::success(format!("Added '{name}' (object_id={})", data["object_id"]), data)
        }

        "add_loft" => {
            let color = parse_color(&args["color"], [0.78, 0.80, 0.85]);
            let sections: Vec<Vec<Vec3>> = args["sections"]
                .as_array()
                .map(|a| a.iter().map(points3).collect())
                .unwrap_or_default();
            let mesh = tri!(ops::loft(
                &sections,
                bool_arg(args, "cap", true),
                bool_arg(args, "closed", false),
                [1.0; 3],
            ));
            history.record("MCP: add_loft", scene);
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Loft");
            let idx = scene.add_mesh_object(name, mesh, color);
            let data = place_new_object(scene, idx, args);
            IpcResponse::success(format!("Added '{name}' (object_id={})", data["object_id"]), data)
        }

        // --------------------------------------------------------------- deformers
        "deform_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active object found"));
            };
            let axis = tri!(axis_arg(args, "axis", 1));
            let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
            let kind = match mode.as_str() {
                "bend" => ops::Deform::Bend {
                    angle_deg: f32_arg(args, "angle_deg", 45.0),
                    toward: tri!(axis_arg(args, "toward", (axis + 1) % 3)),
                },
                "twist" => ops::Deform::Twist { angle_deg: f32_arg(args, "angle_deg", 45.0) },
                "taper" => ops::Deform::Taper { factor: f32_arg(args, "factor", 0.5) },
                other => {
                    return Some(IpcResponse::error(format!("Unknown mode '{other}'. Use bend, twist or taper.")));
                }
            };
            if scene.objects[idx].voxel_grid.is_some() {
                return Some(IpcResponse::error("Deformers work on mesh parts; run voxel_remesh first"));
            }
            history.record(format!("MCP: deform {mode}"), scene);
            let obj = &mut scene.objects[idx];
            if !bool_arg(args, "selected_only", true) {
                obj.mesh.selected_faces.clear();
            }
            let moved = tri!(ops::deform(&mut obj.mesh, axis, kind));
            let b = world_bounds(obj).unwrap_or((Vec3::ZERO, Vec3::ZERO));
            IpcResponse::success(
                format!("Applied {mode} to '{}' ({moved} vertices)", obj.name),
                json!({"object_id": obj.id, "vertices_moved": moved, "world_bounds_min": round3(b.0), "world_bounds_max": round3(b.1)}),
            )
        }

        "align_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active object found"));
            };
            let Some(a) = world_bounds(&scene.objects[idx]) else {
                return Some(IpcResponse::error("The object has no geometry"));
            };
            let target = match args.get("target_object_id").and_then(|v| v.as_u64()) {
                Some(id) => match scene.objects.iter().find(|o| o.id == id).and_then(world_bounds) {
                    Some(b) => Some(b),
                    None => return Some(IpcResponse::error("target_object_id not found or empty")),
                },
                None => None,
            };
            let gap = f32_arg(args, "gap", 0.0);
            let mut delta = Vec3::ZERO;
            let mut used = false;
            for (axis, key) in ["x", "y", "z"].iter().enumerate() {
                let Some(mode) = args.get(*key).and_then(|v| v.as_str()) else { continue };
                used = true;
                let (amin, amax) = (a.0[axis], a.1[axis]);
                if mode == "ground" {
                    delta[axis] = -amin + gap;
                    continue;
                }
                let Some(b) = target else {
                    return Some(IpcResponse::error("Aligning to another part needs target_object_id (only \"ground\" works without it)"));
                };
                let (bmin, bmax) = (b.0[axis], b.1[axis]);
                delta[axis] = match mode {
                    "min" => bmin - amin + gap,
                    "max" => bmax - amax - gap,
                    "center" => (bmin + bmax) * 0.5 - (amin + amax) * 0.5,
                    // Outside the target: touching its far side ("after") or near side ("before").
                    "after" | "above" => bmax - amin + gap,
                    "before" | "below" => bmin - amax - gap,
                    other => {
                        return Some(IpcResponse::error(format!(
                            "Unknown alignment '{other}' for {key}. Use min, center, max, before, after or ground."
                        )));
                    }
                };
            }
            if !used {
                return Some(IpcResponse::error("Pass at least one of x, y, z (min, center, max, before, after, ground)"));
            }
            history.record("MCP: align_object", scene);
            let obj = &mut scene.objects[idx];
            obj.transform.translation += delta;
            let b = world_bounds(obj).unwrap_or(a);
            IpcResponse::success(
                format!("Moved '{}' by [{:.3}, {:.3}, {:.3}]", obj.name, delta.x, delta.y, delta.z),
                json!({"object_id": obj.id, "moved_by": round3(delta), "position": round3(obj.transform.translation), "world_bounds_min": round3(b.0), "world_bounds_max": round3(b.1)}),
            )
        }

        "scatter_object" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active source object (the part to copy)"));
            };
            let Some(tid) = args.get("target_object_id").and_then(|v| v.as_u64()) else {
                return Some(IpcResponse::error("Missing target_object_id (the surface to scatter onto)"));
            };
            let Some(tidx) = scene.objects.iter().position(|o| o.id == tid) else {
                return Some(IpcResponse::error("target_object_id not found"));
            };
            if tidx == idx {
                return Some(IpcResponse::error("Source and target must be different parts"));
            }
            let count = usize_arg(args, "count", 20).clamp(1, 5000);
            let source = &scene.objects[idx];
            if source.mesh.faces.len() * count > 3_000_000 {
                return Some(IpcResponse::error("That scatter would create more than 3,000,000 faces; lower count or decimate the source"));
            }
            let surface = &scene.objects[tidx];
            let faces: HashSet<usize> = if bool_arg(args, "selected_only", !surface.mesh.selected_faces.is_empty()) {
                surface.mesh.selected_faces.clone()
            } else {
                HashSet::new()
            };
            let scale = f32_arg(args, "scale", 1.0);
            let opts = ops::ScatterOptions {
                count,
                seed: args.get("seed").and_then(|v| v.as_u64()).unwrap_or(1),
                align_to_normal: bool_arg(args, "align_to_normal", true),
                random_yaw: bool_arg(args, "random_yaw", true),
                scale_min: f32_arg(args, "scale_min", scale),
                scale_max: f32_arg(args, "scale_max", scale),
                offset: f32_arg(args, "offset", 0.0),
                min_normal_y: args.get("min_normal_y").and_then(|v| v.as_f64()).map(|v| v as f32),
            };
            let transforms = ops::scatter_transforms(&surface.mesh, surface.transform.matrix(), &faces, &opts);
            if transforms.is_empty() {
                return Some(IpcResponse::error("The target has no surface matching those options"));
            }
            // Copies keep the source's own rotation and scale, but not its position.
            let base = Mat4::from_scale_rotation_translation(
                source.transform.scale,
                source.transform.rotation_quat(),
                Vec3::ZERO,
            );
            let template = source.clone();
            history.record("MCP: scatter_object", scene);
            let default_name = format!("{} Scatter", template.name);
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or(&default_name).to_string();
            let mut ids = Vec::new();
            if bool_arg(args, "merge", true) {
                let mut merged = Mesh::default();
                for m in &transforms {
                    crate::modeling::append(&mut merged, &crate::modeling::transformed(&template.mesh, *m * base));
                }
                merged.recalculate_normals();
                merged.bump_version();
                let mut obj = template.clone();
                obj.id = scene.alloc_id();
                obj.name = name.clone();
                obj.mesh = merged;
                obj.voxel_grid = None;
                obj.transform = Default::default();
                ids.push(obj.id);
                scene.objects.push(obj);
            } else {
                for (i, m) in transforms.iter().enumerate() {
                    let mut obj = template.clone();
                    obj.id = scene.alloc_id();
                    obj.name = format!("{name} {}", i + 1);
                    obj.mesh = crate::modeling::transformed(&template.mesh, *m * base);
                    obj.mesh.recalculate_normals();
                    obj.voxel_grid = None;
                    obj.transform = Default::default();
                    ids.push(obj.id);
                    scene.objects.push(obj);
                }
            }
            scene.selected_object = scene.objects.iter().position(|o| Some(o.id) == ids.first().copied());
            IpcResponse::success(
                format!("Scattered {} copies of '{}' onto '{}'", transforms.len(), template.name, scene.objects[tidx].name),
                json!({"copies": transforms.len(), "object_ids": ids, "merged": ids.len() == 1 && transforms.len() > 1}),
            )
        }

        // ------------------------------------------------------------------- colour
        "paint_faces" => {
            let Some(idx) = resolve_object_index(scene, args) else {
                return Some(IpcResponse::error("No matching or active object found"));
            };
            if args["color"].is_null() {
                return Some(IpcResponse::error("Missing color [r, g, b] (0..1)"));
            }
            let obj = &scene.objects[idx];
            if obj.voxel_grid.is_some() {
                return Some(IpcResponse::error("paint_faces works on mesh parts; use voxel_brush paint_color for voxel clay"));
            }
            let faces: HashSet<usize> = match args.get("faces").and_then(|v| v.as_array()) {
                Some(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect(),
                None => obj.mesh.selected_faces.clone(),
            };
            if faces.is_empty() {
                return Some(IpcResponse::error("No faces to paint: pass faces or select some first (select_faces, select_comment_region)"));
            }
            let color = parse_color(&args["color"], [1.0; 3]);
            // Vertex colours multiply the part's base colour; divide it out so the result is exact.
            let vertex_color = if bool_arg(args, "exact", true) {
                [
                    color[0] / obj.color[0].max(0.02),
                    color[1] / obj.color[1].max(0.02),
                    color[2] / obj.color[2].max(0.02),
                ]
            } else {
                color
            };
            history.record("MCP: paint_faces", scene);
            let obj = &mut scene.objects[idx];
            let painted = ops::paint_faces(&mut obj.mesh, &faces, vertex_color, f32_arg(args, "blend", 1.0));
            IpcResponse::success(
                format!("Painted {painted} face(s) on '{}'", obj.name),
                json!({"object_id": obj.id, "painted_faces": painted, "vertices": obj.mesh.positions.len()}),
            )
        }

        // ----------------------------------------------------------------- captures
        "capture_views" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("/tmp/rust_modeler_views.png");
            let names: Vec<String> = match args.get("views").and_then(|v| v.as_array()) {
                Some(a) => a.iter().filter_map(|v| v.as_str()).map(|s| s.to_lowercase()).collect(),
                None => ["iso", "front", "right", "top"].iter().map(|s| s.to_string()).collect(),
            };
            if names.is_empty() || names.len() > 9 {
                return Some(IpcResponse::error("views needs 1 to 9 entries"));
            }
            let cell_w = usize_arg(args, "cell_width", 640).clamp(160, 1600) as u32;
            let cell_h = usize_arg(args, "cell_height", 480).clamp(120, 1200) as u32;
            let mut frame_args = args.clone();
            if frame_args.get("frame").is_none() {
                frame_args["frame"] = json!("scene");
            }
            let Some(bounds) = tri!(frame_bounds(scene, &frame_args)) else {
                return Some(IpcResponse::error("Nothing to capture: the scene has no visible geometry"));
            };
            let cols = match names.len() {
                1 => 1,
                2..=4 => 2,
                _ => 3,
            } as u32;
            let rows = (names.len() as u32).div_ceil(cols);
            let mut sheet = image::RgbaImage::from_pixel(cols * cell_w, rows * cell_h, image::Rgba([18, 23, 32, 255]));
            let mut cells = Vec::new();
            let centre = (bounds.0 + bounds.1) * 0.5;
            for (i, name) in names.iter().enumerate() {
                let Some(dir) = view_direction(name) else {
                    return Some(IpcResponse::error(format!(
                        "Unknown view '{name}'. Use front, back, left, right, top, bottom, iso, iso_front_left, iso_back_right or iso_back_left."
                    )));
                };
                let mut cam = camera.clone();
                cam.aspect = cell_w as f32 / cell_h as f32;
                cam.orthographic = bool_arg(args, "orthographic", false);
                cam.look_from(centre + dir.normalize() * 10.0, centre);
                frame_camera(&mut cam, bounds, f32_arg(args, "padding", 1.15));
                let tmp = format!("{path}.cell{i}.png");
                let rendered = render_scene_headless_to_png(
                    &cam,
                    scene,
                    *shading,
                    *show_wireframe,
                    bool_arg(args, "show_grid", *show_grid),
                    cell_w,
                    cell_h,
                    Path::new(&tmp),
                );
                let img = rendered.and_then(|_| image::open(&tmp).map_err(|e| e.to_string()));
                let _ = std::fs::remove_file(&tmp);
                let img = tri!(img).to_rgba8();
                let (col, row) = (i as u32 % cols, i as u32 / cols);
                image::imageops::replace(&mut sheet, &img, (col * cell_w) as i64, (row * cell_h) as i64);
                cells.push(json!({"view": name, "column": col, "row": row}));
            }
            // Thin separators between cells.
            for c in 1..cols {
                for y in 0..sheet.height() {
                    sheet.put_pixel(c * cell_w, y, image::Rgba([60, 72, 90, 255]));
                }
            }
            for r in 1..rows {
                for x in 0..sheet.width() {
                    sheet.put_pixel(x, r * cell_h, image::Rgba([60, 72, 90, 255]));
                }
            }
            tri!(sheet.save(path));
            IpcResponse::success(
                format!("Rendered {} view(s) into {path}", names.len()),
                json!({
                    "path": path,
                    "width": sheet.width(),
                    "height": sheet.height(),
                    "columns": cols,
                    "rows": rows,
                    "cells": cells,
                    "framed_bounds_min": round3(bounds.0),
                    "framed_bounds_max": round3(bounds.1),
                }),
            )
        }

        // ------------------------------------------------------------------- batch
        "batch" => {
            let Some(commands) = args.get("commands").and_then(|v| v.as_array()) else {
                return Some(IpcResponse::error("Missing commands: [{\"tool\": ..., \"args\": {...}}, ...]"));
            };
            if commands.len() > 200 {
                return Some(IpcResponse::error("A batch can hold at most 200 commands"));
            }
            let stop_on_error = bool_arg(args, "stop_on_error", true);
            let mut results = Vec::new();
            let mut failed = 0;
            for (i, c) in commands.iter().enumerate() {
                let tool = c.get("tool").and_then(|v| v.as_str()).unwrap_or("");
                if matches!(
                    tool,
                    "" | "batch" | "wait_for_feedback" | "launch_gui_window" | "quit_gui" | "cancel_operation"
                ) {
                    results.push(json!({"index": i, "tool": tool, "ok": false, "message": "This tool cannot run inside a batch"}));
                    failed += 1;
                    if stop_on_error {
                        break;
                    }
                    continue;
                }
                let response = crate::ipc::execute_ipc_command(
                    &IpcRequest { tool: tool.to_string(), args: c.get("args").cloned().unwrap_or(json!({})) },
                    scene,
                    camera,
                    history,
                    ui_state,
                    shading,
                    show_wireframe,
                    show_grid,
                );
                let ok = response.ok;
                results.push(json!({"index": i, "tool": tool, "ok": ok, "message": response.message, "data": response.data}));
                if !ok {
                    failed += 1;
                    if stop_on_error {
                        break;
                    }
                }
            }
            let ran = results.len();
            let data = json!({"ran": ran, "requested": commands.len(), "failed": failed, "results": results});
            if failed > 0 && stop_on_error {
                let mut r = IpcResponse::error(format!(
                    "Batch stopped at command {} of {}: {}",
                    ran,
                    commands.len(),
                    data["results"][ran - 1]["message"].as_str().unwrap_or("")
                ));
                r.data = data;
                r
            } else {
                IpcResponse::success(format!("Ran {ran} command(s), {failed} failed"), data)
            }
        }

        _ => return None,
    })
}
