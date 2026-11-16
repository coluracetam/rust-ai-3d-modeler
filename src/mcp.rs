use crate::camera::OrbitCamera;
use crate::history::UndoHistory;
use crate::ipc::{
    FALLBACK_SCENE_PATH, IPC_SOCKET_PATH, IpcRequest, IpcResponse, execute_ipc_command,
};
use crate::renderer::ViewportShading;
use crate::scene::Scene;
use crate::ui::UiState;
use serde_json::{Value, json};
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Sends an `IpcRequest` to the running GUI over `/tmp/rust_modeler_ipc.sock`.
pub fn send_to_live_gui(req: &IpcRequest) -> Option<IpcResponse> {
    let mut stream = UnixStream::connect(IPC_SOCKET_PATH).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(120)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let mut payload = serde_json::to_vec(req).ok()?;
    payload.push(b'\n');
    if stream.write_all(&payload).is_err() || stream.flush().is_err() {
        return Some(IpcResponse::error(
            "Could not send command to the live editor; inspect its state before retrying",
        ));
    }

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return Some(IpcResponse::error(
            "The live editor did not reply. Inspect its state before retrying; the command may have completed.",
        ));
    }
    Some(
        serde_json::from_str::<IpcResponse>(&line)
            .unwrap_or_else(|_| IpcResponse::error("Invalid response from live editor")),
    )
}

pub fn mcp_tool_definitions() -> Value {
    json!([
        { "name":"delete_comment", "description":"Delete a comment after its pending revision has been reviewed.", "inputSchema":{"type":"object","properties":{"comment_id":{"type":"integer"}},"required":["comment_id"]} },
        { "name":"clear_selection", "description":"Clear every face selection, the selected part and the focused comment. Selected parts glow and selected faces are tinted in the viewport and in capture_viewport/capture_studio images, so clear the selection before taking review screenshots. Does not change geometry.", "inputSchema":{"type":"object","properties":{}} },
        { "name":"cancel_operation", "description":"Cancel a running GUI operation and discard its result without changing live geometry.", "inputSchema":{"type":"object","properties":{}} },
        { "name":"capture_studio", "description":"Capture the full studio UI and viewport as a PNG for visual review. With compare_revision, the viewport shows a before/after wipe of that revision (before on the left of `split`, after on the right) - useful as a finish_revision screenshot.", "inputSchema":{"type":"object","properties":{"path":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"},"tab":{"type":"integer","enum":[0,1]},"compare_revision":{"type":"integer"},"split":{"type":"number","description":"Divider position 0..1 (default 0.5)."}}} },
        { "name":"save_project", "description":"Save the full project, camera, comments, selections and revisions atomically with a backup.", "inputSchema":{"type":"object","properties":{"path":{"type":"string"}}} },
        { "name":"new_project", "description":"Start a blank, unsaved project: removes all objects, comments, revisions, saved selections and views, and resets the camera. Undo restores the previous project. Save first if the current project has never been saved.", "inputSchema":{"type":"object","properties":{"name":{"type":"string","description":"Optional project title (default 'Untitled project')."}}} },
        { "name":"open_project", "description":"Open a project, or recover its autosave. Existing work remains undoable.", "inputSchema":{"type":"object","properties":{"path":{"type":"string"},"recover":{"type":"boolean"}},"required":["path"]} },
        { "name":"begin_revision", "description":"Start one named, cancellable geometry revision, optionally linked to a user comment. Finish it for user review.", "inputSchema":{"type":"object","properties":{"label":{"type":"string"},"comment_id":{"type":"integer"}},"required":["label"]} },
        { "name":"finish_revision", "description":"Submit grouped changes for review with a summary and optional screenshot paths. The user can compare, accept or revert.", "inputSchema":{"type":"object","properties":{"revision_id":{"type":"integer"},"summary":{"type":"string"},"screenshots":{"type":"array","items":{"type":"string"}}},"required":["revision_id","summary"]} },
        { "name":"get_revisions", "description":"List revision status, changed object IDs, summaries and screenshot paths.", "inputSchema":{"type":"object","properties":{}} },
        { "name":"preview_revision", "description":"Compare a revision before or after without changing live geometry; live=true exits comparison.", "inputSchema":{"type":"object","properties":{"revision_id":{"type":"integer"},"before":{"type":"boolean"},"live":{"type":"boolean"}}} },
        { "name":"accept_revision", "description":"Accept a ready revision and its linked comment.", "inputSchema":{"type":"object","properties":{"revision_id":{"type":"integer"}},"required":["revision_id"]} },
        { "name":"revert_revision", "description":"Revert a pending revision without deleting comments; rejects conflicts with newer geometry edits.", "inputSchema":{"type":"object","properties":{"revision_id":{"type":"integer"}},"required":["revision_id"]} },
        { "name":"cancel_revision", "description":"Cancel an in-progress revision and restore its starting geometry between tool calls.", "inputSchema":{"type":"object","properties":{"revision_id":{"type":"integer"}},"required":["revision_id"]} },
        { "name":"update_comment", "description":"Update a comment lifecycle status or append a discussion reply.", "inputSchema":{"type":"object","properties":{"comment_id":{"type":"integer"},"status":{"type":"string","enum":["open","in_progress","ready_for_review","accepted"]},"reply":{"type":"string"},"author":{"type":"string"}},"required":["comment_id"]} },
        { "name":"reattach_comment", "description":"Attach an existing comment to the user's current selected regions after topology changes.", "inputSchema":{"type":"object","properties":{"comment_id":{"type":"integer"}},"required":["comment_id"]} },
        {
            "name": "get_scene_info",
            "description": "Inspect the current 3D scene: all objects (IDs, names, transforms, materials, vertex/face counts, bounding boxes), active selection, interaction mode, and camera/viewport state.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "launch_gui_window",
            "description": "Launch the interactive 3D Modeler GUI window if it is not already running, so all MCP operations are visible live on screen.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "clear_scene",
            "description": "Remove all objects from the 3D scene (undoable).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "add_primitive",
            "description": "Create and add a new 3D primitive mesh or volumetric SDF voxel object to the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "string",
                        "enum": ["cube", "sphere", "cylinder", "plane", "torus", "voxel_sphere"],
                        "description": "Primitive type to add."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional object name in the Scene Outliner."
                    },
                    "size": {
                        "type": "number",
                        "description": "Overall size/extent (default 1.8)."
                    },
                    "radius": {
                        "type": "number",
                        "description": "Radius for sphere, cylinder, torus, or voxel_sphere."
                    },
                    "minor_radius": {
                        "type": "number",
                        "description": "Tube radius for torus."
                    },
                    "height": {
                        "type": "number",
                        "description": "Height for cylinder (default 2.0)."
                    },
                    "subdivisions": {
                        "type": "integer",
                        "description": "Segment/ring count or voxel grid resolution (default 16; 12..64 for voxel_sphere)."
                    },
                    "position": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "World position [x, y, z]."
                    },
                    "rotation_deg": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Euler rotation in degrees [rx, ry, rz]."
                    },
                    "scale": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Scale [sx, sy, sz]."
                    },
                    "color": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "RGB base color in [0..1], e.g. [0.85, 0.45, 0.30]."
                    }
                },
                "required": ["shape"]
            }
        },
        {
            "name": "transform_object",
            "description": "Move, rotate, scale, rename, or toggle visibility of a scene object (by object_id, name, or currently selected object).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "Target object ID (optional, defaults to selected object)." },
                    "name": { "type": "string", "description": "Target object name (optional)." },
                    "new_name": { "type": "string", "description": "Rename the object." },
                    "position": { "type": "array", "items": { "type": "number" }, "description": "Absolute position [x, y, z]." },
                    "translate_delta": { "type": "array", "items": { "type": "number" }, "description": "Relative translation offset [dx, dy, dz]." },
                    "rotation_deg": { "type": "array", "items": { "type": "number" }, "description": "Absolute Euler rotation [rx, ry, rz] in degrees." },
                    "rotate_delta_deg": { "type": "array", "items": { "type": "number" }, "description": "Relative Euler rotation offset in degrees." },
                    "scale": { "type": "array", "items": { "type": "number" }, "description": "Scale [sx, sy, sz]." },
                    "visible": { "type": "boolean", "description": "Show or hide object." }
                }
            }
        },
        {
            "name": "set_material",
            "description": "Update an object's PBR material, tri-planar procedural 3D shader pattern, emissive glow, or bake mesh Ambient Occlusion (AO), curvature edge wear, height gradients, and upward-facing moss/snow/weathering coats.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "color": { "type": "array", "items": { "type": "number" }, "description": "Primary RGB color [r, g, b] in 0.0..1.0." },
                    "secondary_color": { "type": "array", "items": { "type": "number" }, "description": "Secondary RGB color [r, g, b] for procedural patterns (mortar, veins, runes, wood rings)." },
                    "roughness": { "type": "number", "description": "Surface roughness in 0.05..1.0." },
                    "metallic": { "type": "number", "description": "Metallic factor in 0.0..1.0." },
                    "flat_shading": { "type": "boolean", "description": "True for faceted low-poly shading, false for smooth normals." },
                    "pattern": {
                        "type": "string",
                        "enum": ["none", "checker", "bricks", "stone_blocks", "wood_grain", "marble", "hex_grid", "rune_circuit"],
                        "description": "Tri-planar 3D procedural texture shader pattern."
                    },
                    "texture_scale": { "type": "number", "description": "Procedural pattern spatial frequency scale in 0.2..20.0 (default 3.0)." },
                    "emission_color": { "type": "array", "items": { "type": "number" }, "description": "Emissive glow RGB color [r, g, b] in 0.0..1.0." },
                    "emission_strength": { "type": "number", "description": "Emissive glow intensity in 0.0..5.0 (0 = no glow)." },
                    "reset_vertex_colors": { "type": "boolean", "description": "Reset per-vertex colors to pure white before baking/painting." },
                    "bake_ao": { "type": "boolean", "description": "Bake 3D mesh concavity Ambient Occlusion and convex ridge edge wear into vertex colors." },
                    "ao_strength": { "type": "number", "description": "Ambient occlusion darkening strength in 0.0..1.0 (default 0.65)." },
                    "wear_strength": { "type": "number", "description": "Convex edge wear highlight strength in 0.0..1.0 (default 0.45)." },
                    "crevice_color": { "type": "array", "items": { "type": "number" }, "description": "RGB tint color in deep crevices/concavities (default [0.15, 0.12, 0.10])." },
                    "wear_color": { "type": "array", "items": { "type": "number" }, "description": "RGB highlight color on sharp convex edges (default [1.32, 1.28, 1.22])." },
                    "bottom_color": { "type": "array", "items": { "type": "number" }, "description": "Vertical gradient bottom RGB color." },
                    "top_color": { "type": "array", "items": { "type": "number" }, "description": "Vertical gradient top RGB color." },
                    "gradient_blend": { "type": "number", "description": "Vertical gradient blend factor in 0.0..1.0 (default 0.5)." },
                    "top_coat_color": { "type": "array", "items": { "type": "number" }, "description": "Upward-facing surface coat RGB color (e.g. moss, snow, dust, gold leaf)." },
                    "top_coat_threshold": { "type": "number", "description": "Normal Y threshold for upward-facing top coat in -1.0..1.0 (default 0.55)." },
                    "top_coat_strength": { "type": "number", "description": "Top coat blend opacity in 0.0..1.0 (default 0.65)." }
                }
            }
        },
        {
            "name": "select_object",
            "description": "Select an object in the scene by object_id or name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" }
                }
            }
        },
        {
            "name": "duplicate_object",
            "description": "Duplicate the selected or specified object.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" }
                }
            }
        },
        {
            "name": "delete_object",
            "description": "Delete the selected or specified object from the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" }
                }
            }
        },
        {
            "name": "edit_mesh",
            "description": "Perform polygonal mesh modeling operations: select faces by index or normal direction, extrude faces, inset faces, bevel (chamfer) the border of the selected region by `width`, loop-cut/subdivide, or translate/rotate/scale/delete selected faces. Also: bridge (join exactly two selected faces of equal corner count with a tube of quads), fill_holes (cap open boundaries), weld (merge duplicate vertices within `tolerance`), flip_normals (selected faces, or all), smooth (relax selected vertices: `strength`, `iterations`), subdivide_smooth (Catmull-Clark rounding, `levels` 1-4), decimate (keep `ratio` of the detail), and invert_selection / grow_selection / shrink_selection (`steps`).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "select_by_normal",
                            "select_all",
                            "clear_selection",
                            "extrude",
                            "inset",
                            "bevel",
                            "subdivide",
                            "translate_selected",
                            "scale_selected",
                            "rotate_selected",
                            "delete_selected",
                            "bridge",
                            "fill_holes",
                            "weld",
                            "flip_normals",
                            "smooth",
                            "subdivide_smooth",
                            "decimate",
                            "invert_selection",
                            "grow_selection",
                            "shrink_selection"
                        ],
                        "description": "Mesh editing operation to perform."
                    },
                    "select_faces": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Optional list of face indices to select prior to executing the operation."
                    },
                    "select_faces_by_normal": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Optional direction vector [nx, ny, nz] to select faces pointing in that direction (e.g. [0, 1, 0] for top faces)."
                    },
                    "min_dot": {
                        "type": "number",
                        "description": "Minimum dot product threshold for select_faces_by_normal (default 0.65)."
                    },
                    "distance": {
                        "type": "number",
                        "description": "Extrusion distance along normal (for 'extrude', default 0.45)."
                    },
                    "width": {
                        "type": "number",
                        "description": "Chamfer width in meters (for 'bevel', default 0.05)."
                    },
                    "inset_factor": {
                        "type": "number",
                        "description": "Inset proportion in 0.05..0.90 (for 'inset', default 0.28)."
                    },
                    "depth": {
                        "type": "number",
                        "description": "Inset depth offset along normal (for 'inset', default 0.0)."
                    },
                    "delta": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Translation vector [dx, dy, dz] for 'translate_selected'."
                    },
                    "scale": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Scale factor [sx, sy, sz] for 'scale_selected'."
                    },
                    "rotation_deg": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Euler angles [rx, ry, rz] in degrees for 'rotate_selected'."
                    },
                    "tolerance": { "type": "number", "description": "Merge distance for 'weld' (default 0.0001)." },
                    "strength": { "type": "number", "description": "0..1 for 'smooth' (default 0.5)." },
                    "iterations": { "type": "integer", "description": "Passes for 'smooth' (default 2)." },
                    "levels": { "type": "integer", "description": "Rounds of 'subdivide_smooth' (default 1, max 4)." },
                    "ratio": { "type": "number", "description": "Fraction of detail to keep for 'decimate' (default 0.5)." },
                    "steps": { "type": "integer", "description": "Rings for grow_selection / shrink_selection (default 1)." }
                },
                "required": ["operation"]
            }
        },
        {
            "name": "sculpt_stroke",
            "description": "Apply a 3D sculpting brush dab or multi-point stroke (draw, inflate, smooth, flatten, grab, pinch, crease) with optional symmetry and Dyntopo auto-tessellation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "brush": {
                        "type": "string",
                        "enum": ["draw", "inflate", "smooth", "flatten", "grab", "pinch", "crease"],
                        "description": "Sculpt brush type."
                    },
                    "point": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Local 3D point [x, y, z] on the mesh surface."
                    },
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": { "type": "number" }
                        },
                        "description": "Optional array of 3D points [[x,y,z], ...] for a continuous stroke."
                    },
                    "normal": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Brush normal direction [nx, ny, nz] (defaults to outward radial direction)."
                    },
                    "grab_delta": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Displacement vector [dx, dy, dz] when using the 'grab' brush."
                    },
                    "radius": { "type": "number", "description": "Brush radius (default 0.5)." },
                    "strength": { "type": "number", "description": "Brush strength in 0.05..1.0 (default 0.45)." },
                    "invert": { "type": "boolean", "description": "Invert brush to carve/push inward." },
                    "symmetry_x": { "type": "boolean", "description": "Mirror stroke across X axis." },
                    "symmetry_y": { "type": "boolean", "description": "Mirror stroke across Y axis." },
                    "symmetry_z": { "type": "boolean", "description": "Mirror stroke across Z axis." },
                    "dyntopo": { "type": "boolean", "description": "Enable Dynamic Topology local subdivision under brush." },
                    "detail_size": { "type": "number", "description": "Dyntopo maximum edge length (default 0.15)." }
                },
                "required": ["brush"]
            }
        },
        {
            "name": "voxel_brush",
            "description": "Apply volumetric SDF Voxel CSG operations (add_sphere, subtract_sphere, add_box, subtract_box, smooth_field, paint_color) and extract either Smooth Surface Nets or Blocky Voxel geometry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "operation": {
                        "type": "string",
                        "enum": ["add_sphere", "subtract_sphere", "add_box", "subtract_box", "smooth_field", "paint_color"]
                    },
                    "center": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Local 3D center [x, y, z] for the voxel operation."
                    },
                    "centers": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": { "type": "number" }
                        },
                        "description": "Optional list of 3D centers [[x,y,z], ...] to apply in batch."
                    },
                    "radius": { "type": "number", "description": "Brush radius (default 0.5)." },
                    "color": { "type": "array", "items": { "type": "number" }, "description": "Voxel RGB color [r, g, b]." },
                    "symmetry_x": { "type": "boolean", "description": "Mirror across X axis." },
                    "meshing_mode": {
                        "type": "string",
                        "enum": ["smooth", "blocky"],
                        "description": "Surface extraction mode: 'smooth' (Dual Surface Nets) or 'blocky' (Minecraft/MagicaVoxel cubes)."
                    }
                },
                "required": ["operation"]
            }
        },
        {
            "name": "voxel_remesh",
            "description": "Remesh an object into a uniform quad mesh at a given voxel grid resolution using Dual Surface Nets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "resolution": { "type": "integer", "description": "Grid resolution in 12..64 (default 32)." }
                }
            }
        },
        {
            "name": "set_camera_and_viewport",
            "description": "Control the 3D viewport camera (orbit angles, target, zoom distance, first-person eye/look_at placement, view presets, ortho/persp), viewport shading mode (lit, clay, wireframe), and interaction mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "array", "items": { "type": "number" }, "description": "Camera look-at target [x, y, z]." },
                    "distance": { "type": "number", "description": "Camera orbit distance." },
                    "yaw_deg": { "type": "number", "description": "Horizontal orbit angle in degrees." },
                    "pitch_deg": { "type": "number", "description": "Vertical orbit angle in degrees (-85..85)." },
                    "eye": { "type": "array", "items": { "type": "number" }, "description": "First-person camera position [x, y, z] (like the GUI fly mode). Applied after target/distance/yaw/pitch." },
                    "look_at": { "type": "array", "items": { "type": "number" }, "description": "Point [x, y, z] to look at from `eye` (or from the current eye if `eye` is omitted). Without it, `eye` keeps the current view direction." },
                    "orthographic": { "type": "boolean", "description": "True for orthographic projection, false for perspective." },
                    "preset": { "type": "string", "enum": ["front", "right", "top"], "description": "Snap camera to standard view." },
                    "shading": { "type": "string", "enum": ["lit", "clay", "wireframe"], "description": "Viewport shading style." },
                    "wireframe_overlay": { "type": "boolean", "description": "Show wireframe edges on top of solid meshes." },
                    "show_grid": { "type": "boolean", "description": "Toggle ground reference grid." },
                    "mode": { "type": "string", "enum": ["object", "edit_mesh", "sculpt", "voxel"], "description": "Switch active editor mode." }
                }
            }
        },
        {
            "name": "undo",
            "description": "Undo the last modeling/sculpting action.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "redo",
            "description": "Redo the previously undone action.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "export_model",
            "description": "Export the current 3D scene to a file (.obj, .stl, or .glb glTF 2.0).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Destination file path ending in .obj, .stl, or .glb." }
                },
                "required": ["path"]
            }
        },
        {
            "name": "import_model",
            "description": "Import a 3D mesh file (.obj, .stl, or .glb) into the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Source file path (.obj, .stl, or .glb)." },
                    "name": { "type": "string", "description": "Optional name for the imported object." }
                },
                "required": ["path"]
            }
        },
        {
            "name": "capture_viewport",
            "description": "Render the 3D scene from the current or specified camera angle to a PNG image file so the AI agent can visually inspect its 3D modeling, composition, proportions, lighting, and textures.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Destination PNG file path (defaults to '/tmp/rust_modeler_viewport.png')." },
                    "width": { "type": "integer", "description": "Image width in pixels (256..2560, default 1024)." },
                    "height": { "type": "integer", "description": "Image height in pixels (256..1600, default 768)." },
                    "target": { "type": "array", "items": { "type": "number" }, "description": "Optional camera look-at target [x, y, z] override for the screenshot." },
                    "distance": { "type": "number", "description": "Optional camera orbit distance override." },
                    "yaw_deg": { "type": "number", "description": "Optional camera horizontal orbit angle in degrees." },
                    "pitch_deg": { "type": "number", "description": "Optional camera vertical orbit angle in degrees." },
                    "shading": { "type": "string", "enum": ["lit", "clay", "wireframe"], "description": "Optional shading mode override." },
                    "wireframe_overlay": { "type": "boolean", "description": "Optional wireframe overlay toggle." },
                    "show_grid": { "type": "boolean", "description": "Optional ground reference grid toggle." },
                    "frame": { "type": "string", "enum": ["scene", "selection", "object", "comment"], "description": "Aim and zoom the camera so this fills the image (keeps the current viewing direction; yaw_deg/pitch_deg still apply): the whole scene, the current selection, one part (object_id/name) or a comment's region (comment_id). Use frame=comment for a close-up of what the user asked about." },
                    "object_id": { "type": "integer" },
                    "name": { "type": "string" },
                    "comment_id": { "type": "integer" },
                    "padding": { "type": "number", "description": "Zoom margin for `frame`, default 1.25 (1.0 = tight)." }
                }
            }
        },
        {
            "name": "get_comments",
            "description": "Retrieve spatial comments left by the user on specific sections of the 3D model, including target object_id, 3D world/local anchor coordinates, surface normal, local bounding box, and selected face indices.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_resolved": { "type": "boolean", "description": "Whether to include resolved comments (default false)." },
                    "include_drafts": { "type": "boolean", "description": "Also include comments the user has not sent yet (default false; do not act on drafts)." }
                }
            }
        },
        { "name":"mirror_object", "description":"Mirror a part across the plane through its local origin normal to `axis`. merge=true (default) adds the reflected half to the same mesh (build half a symmetric part, then mirror); merge=false creates a separate mirrored part.", "inputSchema":{"type":"object","properties":{"object_id":{"type":"integer"},"name":{"type":"string"},"axis":{"type":"string","enum":["x","y","z"]},"merge":{"type":"boolean"}}} },
        { "name":"array_object", "description":"Repeat a part `count` times. Each copy applies the step (rotate by rotation_deg about the part's local origin, scale, then move by `offset`, all in the part's local frame) to the previous one, so offset [1,0,0] makes a row and rotation_deg [0,30,0] with offset [0,0,0] makes a radial ring around the origin. merge=false (default) creates separate parts; merge=true combines them into one mesh.", "inputSchema":{"type":"object","properties":{"object_id":{"type":"integer"},"name":{"type":"string"},"count":{"type":"integer","minimum":2,"maximum":128},"offset":{"type":"array","items":{"type":"number"}},"rotation_deg":{"type":"array","items":{"type":"number"}},"scale":{"type":"number"},"merge":{"type":"boolean"}},"required":["count"]} },
        { "name":"add_lathe", "description":"Create a solid of revolution (vase, column, bowl, wheel, finial) by revolving a profile of [radius, height] points around the local Y axis. Points with radius 0 close the ends.", "inputSchema":{"type":"object","properties":{"profile":{"type":"array","items":{"type":"array","items":{"type":"number"}},"description":"[[radius, height], ...] from bottom to top."},"segments":{"type":"integer"},"angle_deg":{"type":"number","description":"Sweep angle, default 360."},"cap":{"type":"boolean","description":"Close open ends with flat caps (default true)."},"name":{"type":"string"},"color":{"type":"array","items":{"type":"number"}},"position":{"type":"array","items":{"type":"number"}},"rotation_deg":{"type":"array","items":{"type":"number"}}},"required":["profile"]} },
        { "name":"add_sweep", "description":"Create a tube, rail, branch, cable, handle or molding by sweeping a cross-section along a 3D path. Use `radius`+`sides` for a round tube or `profile` ([[x, y], ...], counter-clockwise) for a custom section. Supports closed loops, twist and tapering via end_scale.", "inputSchema":{"type":"object","properties":{"path":{"type":"array","items":{"type":"array","items":{"type":"number"}}},"radius":{"type":"number"},"sides":{"type":"integer"},"profile":{"type":"array","items":{"type":"array","items":{"type":"number"}}},"closed":{"type":"boolean"},"cap":{"type":"boolean"},"twist_deg":{"type":"number"},"end_scale":{"type":"number"},"name":{"type":"string"},"color":{"type":"array","items":{"type":"number"}},"position":{"type":"array","items":{"type":"number"}},"rotation_deg":{"type":"array","items":{"type":"number"}}},"required":["path"]} },
        { "name":"boolean", "description":"Constructive solid geometry between two closed parts: union, subtract (cut tool_object out of the target, e.g. windows, holes, grooves) or intersect. The target (object_id/name) receives the result in its own local space; the tool part is removed unless keep_tool=true. Works best on closed, watertight meshes of moderate size.", "inputSchema":{"type":"object","properties":{"object_id":{"type":"integer"},"name":{"type":"string"},"tool_object_id":{"type":"integer"},"operation":{"type":"string","enum":["union","subtract","intersect"]},"keep_tool":{"type":"boolean"}},"required":["tool_object_id","operation"]} },
        {"name": "query_geometry", "description": "Read real geometry instead of guessing from screenshots. Returns a part's world bounds and size plus, for its selected faces (default when any are selected), listed faces or the whole part: each face's index, world centre, world normal, area and corner count, and the connected regions those faces form with their bounds. Use it after select_comment_region to learn exactly where and how big the commented region is, and to find face indices for select_faces/edit_mesh. Read-only.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "scope": {"type": "string", "enum": ["object", "selection", "faces"]}, "faces": {"type": "array", "items": {"type": "integer"}, "description": "Explicit face indices (scope=faces)."}, "max_faces": {"type": "integer", "description": "Faces to list, default 60 (max 2000)."}, "offset": {"type": "integer", "description": "Skip this many faces first, for paging."}, "include_vertices": {"type": "boolean", "description": "Also return each listed face's corner positions and vertex indices."}}}},
        {"name": "measure", "description": "Measure between two parts (bounding boxes: per-axis gap where negative means overlap depth, centre distance and delta) or along a list of world points. Use it to check clearances, alignment and whether parts touch. Read-only.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "other_object_id": {"type": "integer"}, "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "[[x,y,z], ...] measures the polyline length instead."}}}},
        {"name": "check_mesh", "description": "Health check before finishing a revision. Per part: watertight, inside-out winding, open boundary edges (holes), non-manifold edges, zero-area faces, number of connected pieces and volume, with the edit_mesh operation that fixes each issue. For the whole scene (no object given) also lists parts whose bounding boxes overlap and isolated parts that touch nothing (possibly floating). Read-only.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}}}},
        {"name": "select_faces", "description": "Select faces the way a user would, without knowing indices in advance. Modes: indices (faces), all, none, box (min/max; with no object it selects across every visible part in world space), sphere (center, radius), connected (the whole connected piece containing seed_face), surface (flood fill from seed_face up to crease_deg), normal (faces pointing along `normal`, min_dot), similar (faces like the current selection by normal, area or color), path (from_face to to_face), invert, grow, shrink (steps). op = replace (default), add or subtract. Coordinates are world space unless space=local. The selection then drives edit_mesh, paint_faces, deform_object and scatter_object.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "mode": {"type": "string", "enum": ["indices", "all", "none", "box", "sphere", "connected", "surface", "normal", "similar", "path", "invert", "grow", "shrink"]}, "op": {"type": "string", "enum": ["replace", "add", "subtract"]}, "faces": {"type": "array", "items": {"type": "integer"}}, "min": {"type": "array", "items": {"type": "number"}}, "max": {"type": "array", "items": {"type": "number"}}, "center": {"type": "array", "items": {"type": "number"}}, "radius": {"type": "number"}, "seed_face": {"type": "integer"}, "crease_deg": {"type": "number"}, "normal": {"type": "array", "items": {"type": "number"}}, "min_dot": {"type": "number"}, "by": {"type": "string", "enum": ["normal", "area", "color"]}, "tolerance": {"type": "number", "description": "similar: degrees for normal (default 8), fraction for area (0.15), RGB distance for color (0.08)."}, "from_face": {"type": "integer"}, "to_face": {"type": "integer"}, "steps": {"type": "integer"}, "space": {"type": "string", "enum": ["world", "local"]}, "show_box": {"type": "boolean", "description": "mode=box without an object: also show the box in the user's viewport."}}, "required": ["mode"]}},
        {"name": "add_mesh", "description": "Create a part directly from vertices and polygon faces - use this instead of writing an .obj file and importing it. Faces are lists of 0-based vertex indices (triangles, quads or n-gons, counter-clockwise seen from outside). Inside-out closed meshes are flipped automatically. The result reports whether the mesh is watertight.", "inputSchema": {"type": "object", "properties": {"vertices": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "[[x,y,z], ...] in the part's local space."}, "faces": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}, "description": "[[i0,i1,i2,...], ...]"}, "vertex_colors": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Optional [r,g,b] per vertex (multiplies `color`)."}, "name": {"type": "string"}, "color": {"type": "array", "items": {"type": "number"}}, "position": {"type": "array", "items": {"type": "number"}}, "rotation_deg": {"type": "array", "items": {"type": "number"}}, "scale": {"type": "array", "items": {"type": "number"}}, "flat_shading": {"type": "boolean"}, "fix_winding": {"type": "boolean"}}, "required": ["vertices", "faces"]}},
        {"name": "add_extrude", "description": "Extrude a 2D outline into a solid: floor plans, walls, brackets, gears, letters, plates, any prism. `profile` is [[x,z], ...] on the ground plane (convex or concave, either winding, do not repeat the first point); the solid rises along +Y by `height`. top_scale tapers the top (0.5 = half size, 0.01 = a point), twist_deg rotates it.", "inputSchema": {"type": "object", "properties": {"profile": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "height": {"type": "number"}, "top_scale": {"type": "number"}, "twist_deg": {"type": "number"}, "segments": {"type": "integer", "description": "Steps along the height (default 1, or 16 when twisting)."}, "cap": {"type": "boolean"}, "name": {"type": "string"}, "color": {"type": "array", "items": {"type": "number"}}, "position": {"type": "array", "items": {"type": "number"}}, "rotation_deg": {"type": "array", "items": {"type": "number"}}, "scale": {"type": "array", "items": {"type": "number"}}}, "required": ["profile", "height"]}},
        {"name": "add_loft", "description": "Skin a surface through cross-sections: hulls, bottles, fuselages, limbs, transitions between shapes. `sections` is a list of rings, each [[x,y,z], ...] with the same number of points in the same winding order, listed from one end to the other. cap closes the ends; closed joins the last section back to the first.", "inputSchema": {"type": "object", "properties": {"sections": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}, "cap": {"type": "boolean"}, "closed": {"type": "boolean"}, "name": {"type": "string"}, "color": {"type": "array", "items": {"type": "number"}}, "position": {"type": "array", "items": {"type": "number"}}, "rotation_deg": {"type": "array", "items": {"type": "number"}}, "scale": {"type": "array", "items": {"type": "number"}}}, "required": ["sections"]}},
        {"name": "deform_object", "description": "Bend, twist or taper a part (or only its selected faces) in its local space. bend curves it along `axis` by angle_deg, curling toward the `toward` axis; twist rotates progressively around `axis`; taper scales the cross-section from 1 at the start of `axis` to `factor` at the end. Needs enough segments along the axis to deform smoothly - subdivide first if it looks faceted.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "mode": {"type": "string", "enum": ["bend", "twist", "taper"]}, "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Axis the deformation runs along (default y)."}, "toward": {"type": "string", "enum": ["x", "y", "z"]}, "angle_deg": {"type": "number"}, "factor": {"type": "number"}, "selected_only": {"type": "boolean", "description": "Only move the selected faces (default true when faces are selected)."}}, "required": ["mode"]}},
        {"name": "align_object", "description": "Move a part so its bounding box lines up with another part's (world axes). Per axis give one of: min, center, max (align that side), after / before (place it just outside the target's far / near side, e.g. y=after stacks it on top), or ground (rest on 0, no target needed). `gap` adds clearance.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "target_object_id": {"type": "integer"}, "x": {"type": "string", "enum": ["min", "center", "max", "before", "after", "ground"]}, "y": {"type": "string", "enum": ["min", "center", "max", "before", "after", "ground"]}, "z": {"type": "string", "enum": ["min", "center", "max", "before", "after", "ground"]}, "gap": {"type": "number"}}}},
        {"name": "scatter_object", "description": "Spread copies of a part over another part's surface: leaves, rocks, rivets, grass, roof tiles, studs. Copies are placed at random, area-weighted points on the target (only its selected faces when it has a selection), stood along the surface normal with a random spin. The same seed always gives the same layout. merge=true (default) makes one new part; the source part is kept.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "target_object_id": {"type": "integer"}, "count": {"type": "integer"}, "seed": {"type": "integer"}, "scale": {"type": "number"}, "scale_min": {"type": "number"}, "scale_max": {"type": "number"}, "align_to_normal": {"type": "boolean"}, "random_yaw": {"type": "boolean"}, "offset": {"type": "number", "description": "Lift along the surface normal."}, "min_normal_y": {"type": "number", "description": "Only faces whose world normal Y is at least this (0.5 = upward-facing surfaces)."}, "selected_only": {"type": "boolean"}, "merge": {"type": "boolean"}}, "required": ["target_object_id"]}},
        {"name": "paint_faces", "description": "Give specific faces their own colour (the selected faces, or `faces`) without splitting the part: 'make these panels darker', trim, stripes, a coloured band. The boundary stays crisp. color is the final RGB (0..1); blend < 1 mixes with the existing colour. Note set_material bake_ao/reset_vertex_colors repaint vertex colours, so bake first and paint after.", "inputSchema": {"type": "object", "properties": {"object_id": {"type": "integer"}, "name": {"type": "string"}, "faces": {"type": "array", "items": {"type": "integer"}}, "color": {"type": "array", "items": {"type": "number"}}, "blend": {"type": "number"}, "exact": {"type": "boolean", "description": "Default true: `color` is the final colour. False multiplies the part's base colour instead."}}, "required": ["color"]}},
        {"name": "capture_views", "description": "Render several camera angles into one contact-sheet PNG in a single call (default: iso, front, right, top), framed on the whole scene, the selection, one part or a comment's region. Faster and more reliable for verifying proportions than repeated capture_viewport calls. Views: front, back, left, right, top, bottom, iso, iso_front_left, iso_back_right, iso_back_left. The result lists which view is in which cell. Read-only.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "views": {"type": "array", "items": {"type": "string"}}, "object_id": {"type": "integer"}, "name": {"type": "string"}, "frame": {"type": "string", "enum": ["scene", "selection", "object", "comment"], "description": "Aim the camera so this fills the image: the whole scene, the current selection, one part (object_id/name) or a comment's region (comment_id)."}, "comment_id": {"type": "integer"}, "padding": {"type": "number", "description": "Zoom margin, default 1.15 (1.0 = tight)."}, "cell_width": {"type": "integer"}, "cell_height": {"type": "integer"}, "orthographic": {"type": "boolean"}, "show_grid": {"type": "boolean"}}}},
        {"name": "batch", "description": "Run many tool calls in one round trip: {commands: [{tool, args}, ...]}. Use it for long build sequences (dozens of add/transform/material calls). Stops at the first failure unless stop_on_error=false, and returns every command's result. Cannot contain batch, wait_for_feedback or launch_gui_window.", "inputSchema": {"type": "object", "properties": {"commands": {"type": "array", "items": {"type": "object", "properties": {"tool": {"type": "string"}, "args": {"type": "object"}}, "required": ["tool"]}}, "stop_on_error": {"type": "boolean"}}, "required": ["commands"]}},
        {
            "name": "wait_for_feedback",
            "description": "Block until the user sends comments to the agent (the GUI's 'Send to agent' button) or asks for changes on a revision, then return the open comments with `feedback_batch`. Use it to work in a loop: wait, edit inside a revision, finish_revision, then wait again passing the last feedback_batch as since_batch. Without since_batch it returns immediately if open comments are already waiting. Returns woke=false on timeout; just call again.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since_batch": { "type": "integer", "description": "The feedback_batch you last handled; wait for a newer one." },
                    "timeout_seconds": { "type": "number", "description": "Maximum wait (default 600, max 3600)." }
                }
            }
        },
        {
            "name": "request_changes",
            "description": "Revert a revision that is ready for review and reopen its comment with the user's notes as a reply (what the GUI's 'Request changes' button does).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "revision_id": { "type": "integer" },
                    "notes": { "type": "string", "description": "What should be different next time." }
                },
                "required": ["revision_id"]
            }
        },
        {
            "name": "send_comments",
            "description": "Send all draft comments to the agent as one feedback batch (what the GUI's 'Send to agent' button does).",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "select_comment_region",
            "description": "Select the target object and restore the exact highlighted face selection associated with a user comment so you can run edit_mesh (extrude, inset, translate_selected, scale_selected, delete_selected) or sculpt_stroke directly on that section.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "comment_id": { "type": "integer", "description": "ID of the comment whose selected section should be activated." },
                    "focus_camera": { "type": "boolean", "description": "Whether to center the viewport camera on the comment's 3D pin (default true)." }
                },
                "required": ["comment_id"]
            }
        },
        {
            "name": "resolve_comment",
            "description": "Mark a user's spatial comment as resolved (or reopened) and attach an AI agent reply explaining the 3D changes made.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "comment_id": { "type": "integer", "description": "ID of the comment to resolve." },
                    "resolved": { "type": "boolean", "description": "True to mark resolved (default true), false to reopen." },
                    "reply": { "type": "string", "description": "Summary of the 3D modifications performed by the AI agent." }
                },
                "required": ["comment_id"]
            }
        },
        {
            "name": "library_list",
            "description": "List the asset library: reusable models shared by every project, organised as sections > subsections > assets. Each asset reports its 'asset' reference (Section/Subsection/Name), part and face counts, size and thumbnail path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Only list this section." },
                    "search": { "type": "string", "description": "Only list assets whose name, description or tags contain this text." }
                }
            }
        },
        {
            "name": "library_show",
            "description": "Open the full asset library view in the GUI so the user can browse the previews.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "library_save_asset",
            "description": "Save scene parts to the asset library as one reusable asset. Pass all: true to save the whole model (every visible part) as a single asset, or choose parts with object_ids, names or group; with none of those, the current selection is saved. The asset's base centre becomes its origin. The scene is not changed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Asset name, e.g. 'Oak chair'." },
                    "section": { "type": "string", "description": "Library section, e.g. 'Furniture'. Created if missing." },
                    "subsection": { "type": "string", "description": "Subsection inside the section, e.g. 'Chairs' (default 'General'). Created if missing." },
                    "all": { "type": "boolean", "description": "Save the whole model: every visible part, as one asset." },
                    "object_ids": { "type": "array", "items": { "type": "integer" }, "description": "Parts to save, by id." },
                    "names": { "type": "array", "items": { "type": "string" }, "description": "Parts to save, by name." },
                    "group": { "type": "string", "description": "Save every part named 'Group/…', e.g. 'Chair'." },
                    "description": { "type": "string", "description": "Optional note shown in the library." },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional search tags." },
                    "overwrite": { "type": "boolean", "description": "Replace an asset with the same name (default false)." },
                    "thumbnail": { "type": "boolean", "description": "Render a thumbnail (default true)." }
                },
                "required": ["name", "section"]
            }
        },
        {
            "name": "library_insert_asset",
            "description": "Add a copy of a library asset to the scene. Its parts are named 'AssetName/Part' (a number is added when that group already exists) and one undo step removes them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset": { "type": "string", "description": "'Section/Subsection/Name' from library_list, or just the name when it is unique." },
                    "position": { "type": "array", "items": { "type": "number" }, "description": "World [x, y, z] where the centre of the asset's base goes (default origin)." },
                    "rotation_y_deg": { "type": "number", "description": "Turn around the vertical axis in degrees (default 0)." },
                    "scale": { "type": "number", "description": "Uniform scale factor (default 1)." }
                },
                "required": ["asset"]
            }
        },
        {
            "name": "library_create_section",
            "description": "Create a library section, or a subsection inside it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Section name." },
                    "subsection": { "type": "string", "description": "Subsection to create inside the section." }
                },
                "required": ["section"]
            }
        },
        {
            "name": "library_move_asset",
            "description": "Move a library asset to another section/subsection and/or rename it. Omitted fields keep their current value.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset": { "type": "string", "description": "'Section/Subsection/Name', or just the name when it is unique." },
                    "section": { "type": "string", "description": "Destination section (created if missing)." },
                    "subsection": { "type": "string", "description": "Destination subsection (created if missing)." },
                    "name": { "type": "string", "description": "New asset name." }
                },
                "required": ["asset"]
            }
        },
        {
            "name": "library_delete_asset",
            "description": "Remove an asset from the library. Its files move to the library's .trash folder, so the user can recover them. Only do this when the user asks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "asset": { "type": "string", "description": "'Section/Subsection/Name', or just the name when it is unique." }
                },
                "required": ["asset"]
            }
        },
        {
            "name": "library_delete_section",
            "description": "Delete an empty library subsection, or a section that holds no assets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": { "type": "string", "description": "Section name." },
                    "subsection": { "type": "string", "description": "Delete only this subsection." }
                },
                "required": ["section"]
            }
        }
    ])
}

/// Runs the stdio JSON-RPC 2.0 MCP Server loop.
pub fn run_mcp_stdio_server() {
    let mut fallback_scene = load_fallback_scene().unwrap_or_default();
    let mut fallback_camera = fallback_scene.last_camera.clone().unwrap_or_default();
    let mut fallback_history = UndoHistory::default();
    let mut fallback_ui = UiState::default();
    let mut fallback_shading = ViewportShading::StudioLit;
    let mut fallback_wire = false;
    let mut fallback_grid = true;

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut stdout = io::stdout();

    while let Some((msg, use_content_length)) = read_jsonrpc_message(&mut reader) {
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        // Notifications have no "id" and expect no response
        if id.is_none() {
            continue;
        }
        let req_id = id.unwrap();

        let response = match method {
            "initialize" => {
                let client_proto = msg
                    .pointer("/params/protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("2024-11-05");
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": client_proto,
                        "capabilities": {
                            "tools": {
                                "listChanged": false
                            }
                        },
                        "serverInfo": {
                            "name": "rust-modeler-mcp",
                            "version": "0.1.0"
                        }
                    }
                })
            }

            "ping" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {}
                })
            }

            "tools/list" => {
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": mcp_tool_definitions()
                    }
                })
            }

            "tools/call" => {
                let tool_name = msg
                    .pointer("/params/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = msg
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                let ipc_resp = if tool_name == "launch_gui_window" {
                    launch_gui_if_needed()
                } else if tool_name == "wait_for_feedback" {
                    wait_for_feedback(&args, |req| {
                        send_to_live_gui(req).unwrap_or_else(|| {
                            execute_headless(
                                req,
                                &mut fallback_scene,
                                &mut fallback_camera,
                                &mut fallback_history,
                                &mut fallback_ui,
                                &mut fallback_shading,
                                &mut fallback_wire,
                                &mut fallback_grid,
                            )
                        })
                    })
                } else {
                    let req = IpcRequest {
                        tool: tool_name,
                        args,
                    };
                    if let Some(live_resp) = send_to_live_gui(&req) {
                        live_resp
                    } else {
                        execute_headless(
                            &req,
                            &mut fallback_scene,
                            &mut fallback_camera,
                            &mut fallback_history,
                            &mut fallback_ui,
                            &mut fallback_shading,
                            &mut fallback_wire,
                            &mut fallback_grid,
                        )
                    }
                };

                let text_payload = serde_json::to_string_pretty(&json!({
                    "ok": ipc_resp.ok,
                    "message": ipc_resp.message,
                    "data": ipc_resp.data,
                }))
                .unwrap_or_else(|_| ipc_resp.message.clone());

                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": text_payload
                            }
                        ],
                        "isError": !ipc_resp.ok
                    }
                })
            }

            _ => {
                json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": format!("Method not found: {method}")
                    }
                })
            }
        };

        write_jsonrpc_message(&mut stdout, &response, use_content_length);
    }
}

/// Block until the user sends feedback, then return the open comments.
///
/// Without `since_batch` it returns at once when open comments are waiting; otherwise it waits
/// for the next batch. Polls the editor (or the headless scene file) so the GUI never blocks.
pub fn wait_for_feedback(
    args: &Value,
    mut call: impl FnMut(&IpcRequest) -> IpcResponse,
) -> IpcResponse {
    let timeout = Duration::from_secs_f64(
        args.get("timeout_seconds")
            .and_then(|v| v.as_f64())
            .unwrap_or(600.0)
            .clamp(1.0, 3600.0),
    );
    let since = args.get("since_batch").and_then(|v| v.as_u64());
    let state_req = IpcRequest {
        tool: "get_feedback_state".into(),
        args: json!({}),
    };
    let start = std::time::Instant::now();
    loop {
        let state = call(&state_req);
        if !state.ok {
            return state;
        }
        let batch = state.data["feedback_batch"].as_u64().unwrap_or(0);
        let open = state.data["open_comments"].as_u64().unwrap_or(0);
        let ready = match since {
            Some(since) => batch > since,
            None => open > 0,
        };
        if ready {
            let mut resp = call(&IpcRequest {
                tool: "get_comments".into(),
                args: json!({}),
            });
            if let Some(data) = resp.data.as_object_mut() {
                data.insert("woke".into(), json!(true));
            }
            resp.message = format!("New feedback (batch {batch}). {}", resp.message);
            return resp;
        }
        if start.elapsed() >= timeout {
            return IpcResponse::success(
                "No new feedback yet; call wait_for_feedback again with the same since_batch",
                json!({"woke": false, "feedback_batch": batch, "open_comments": open}),
            );
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn launch_gui_if_needed() -> IpcResponse {
    // Check if GUI is already listening on IPC socket
    let ping_req = IpcRequest {
        tool: "get_scene_info".to_string(),
        args: json!({}),
    };
    if let Some(resp) = send_to_live_gui(&ping_req) {
        return IpcResponse::success(
            "3D Modeler GUI window is already running and connected via live IPC.",
            resp.data,
        );
    }

    let exe: std::path::PathBuf = std::env::current_exe()
        .map(|p| {
            let s = p.to_string_lossy();
            if let Some(stripped) = s.strip_suffix(" (deleted)") {
                std::path::PathBuf::from(stripped)
            } else {
                p
            }
        })
        .unwrap_or_else(|_| "target/debug/rust_modeler".into());
    match Command::new(&exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => {
            let pid = child.id();
            // Wait briefly for the GUI window to initialize its IPC socket
            for _ in 0..25 {
                std::thread::sleep(Duration::from_millis(100));
                if let Some(resp) = send_to_live_gui(&ping_req) {
                    return IpcResponse::success(
                        format!(
                            "Launched 3D Modeler GUI window (PID {pid}) and connected live IPC."
                        ),
                        resp.data,
                    );
                }
            }
            IpcResponse::success(
                format!("Spawned 3D Modeler GUI process (PID {pid})."),
                json!({ "pid": pid }),
            )
        }
        Err(e) => IpcResponse::error(format!("Failed to spawn GUI process: {e}")),
    }
}

/// Serialize headless read/modify/write transactions across CLI and MCP processes.
#[allow(clippy::too_many_arguments)]
pub fn execute_headless(
    req: &IpcRequest,
    scene: &mut Scene,
    camera: &mut OrbitCamera,
    history: &mut UndoHistory,
    ui: &mut UiState,
    shading: &mut ViewportShading,
    wire: &mut bool,
    grid: &mut bool,
) -> IpcResponse {
    let lock = match fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("/tmp/rust_modeler_scene.lock")
    {
        Ok(file) => file,
        Err(e) => return IpcResponse::error(format!("Could not lock scene: {e}")),
    };
    if let Err(e) = lock.lock() {
        return IpcResponse::error(format!("Could not lock scene: {e}"));
    }
    if let Some(disk) = load_fallback_scene() {
        *camera = disk.last_camera.clone().unwrap_or_else(|| camera.clone());
        *scene = disk;
    }
    let mut response = execute_ipc_command(req, scene, camera, history, ui, shading, wire, grid);
    if response.ok && !crate::ipc::is_read_only(&req.tool) {
        if let Err(e) = persist_scene(scene, camera) {
            response =
                IpcResponse::error(format!("Command completed, but persistence failed: {e}"));
        }
    }
    response
}

pub fn load_fallback_scene() -> Option<Scene> {
    if let Ok(text) = fs::read_to_string(FALLBACK_SCENE_PATH) {
        if let Ok(mut scene) = serde_json::from_str::<Scene>(&text) {
            if scene.validate().is_ok() {
                return Some(scene);
            }
        }
    }
    if let Ok(text) = fs::read_to_string(crate::project::backup_path(std::path::Path::new(
        FALLBACK_SCENE_PATH,
    ))) {
        if let Ok(mut scene) = serde_json::from_str::<Scene>(&text) {
            if scene.validate().is_ok() {
                return Some(scene);
            }
        }
    }
    if let Ok(text) = fs::read_to_string("assets/scenes/clockwork_astral_cathedral_scene.json") {
        if let Ok(scene) = serde_json::from_str::<Scene>(&text) {
            save_fallback_scene(&scene);
            return Some(scene);
        }
    }
    None
}

pub fn save_fallback_scene(scene: &Scene) {
    if let Err(error) = try_save_fallback_scene(scene) {
        log::error!("{error}");
    }
}

pub fn try_save_fallback_scene(scene: &Scene) -> Result<(), String> {
    let bytes = serde_json::to_vec(scene).map_err(|e| e.to_string())?;
    crate::project::atomic_write(std::path::Path::new(FALLBACK_SCENE_PATH), &bytes, true)
}

/// Preserve a recoverable project alongside its explicit saved version.
pub fn persist_scene(scene: &mut Scene, camera: &OrbitCamera) -> Result<(), String> {
    scene.last_camera = Some(camera.clone());
    try_save_fallback_scene(scene)?;
    if let Some(path) = &scene.project_path {
        let autosave = crate::project::autosave_path(std::path::Path::new(path));
        crate::project::save(&autosave, scene, camera)?;
    }
    Ok(())
}

/// Reads either a newline-delimited JSON-RPC message or a `Content-Length:` framed JSON-RPC message.
fn read_jsonrpc_message<R: BufRead>(reader: &mut R) -> Option<(Value, bool)> {
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).ok()?;
        if bytes_read == 0 {
            return None; // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            let len: usize = rest.trim().parse().ok()?;
            // Consume remaining header lines until blank line
            loop {
                let mut hdr = String::new();
                if reader.read_line(&mut hdr).ok()? == 0 {
                    return None;
                }
                if hdr.trim().is_empty() {
                    break;
                }
            }
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf).ok()?;
            let val: Value = serde_json::from_slice(&buf).ok()?;
            return Some((val, true));
        }

        if trimmed.starts_with('{') {
            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                return Some((val, false));
            }
        }
    }
}

fn write_jsonrpc_message<W: Write>(writer: &mut W, msg: &Value, use_content_length: bool) {
    if let Ok(bytes) = serde_json::to_vec(msg) {
        if use_content_length {
            let _ = write!(writer, "Content-Length: {}\r\n\r\n", bytes.len());
            let _ = writer.write_all(&bytes);
        } else {
            let _ = writer.write_all(&bytes);
            let _ = writer.write_all(b"\n");
        }
        let _ = writer.flush();
    }
}
