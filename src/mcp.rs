use crate::camera::OrbitCamera;
use crate::history::UndoHistory;
use crate::ipc::{
    execute_ipc_command, IpcRequest, IpcResponse, FALLBACK_SCENE_PATH, IPC_SOCKET_PATH,
};
use crate::renderer::ViewportShading;
use crate::scene::Scene;
use crate::ui::UiState;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Sends an `IpcRequest` to the running GUI over `/tmp/rust_modeler_ipc.sock`.
pub fn send_to_live_gui(req: &IpcRequest) -> Option<IpcResponse> {
    let mut stream = UnixStream::connect(IPC_SOCKET_PATH).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let mut payload = serde_json::to_vec(req).ok()?;
    payload.push(b'\n');
    stream.write_all(&payload).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    if line.trim().is_empty() {
        return None;
    }
    serde_json::from_str::<IpcResponse>(&line).ok()
}

pub fn mcp_tool_definitions() -> Value {
    json!([
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
            "description": "Perform polygonal mesh modeling operations: select faces by index or normal direction, extrude faces, inset faces, loop-cut/subdivide, or translate/rotate/scale/delete selected faces.",
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
                            "subdivide",
                            "translate_selected",
                            "scale_selected",
                            "rotate_selected",
                            "delete_selected"
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
                    }
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
            "description": "Control the 3D viewport camera (orbit angles, target, zoom distance, view presets, ortho/persp), viewport shading mode (lit, clay, wireframe), and interaction mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "array", "items": { "type": "number" }, "description": "Camera look-at target [x, y, z]." },
                    "distance": { "type": "number", "description": "Camera orbit distance." },
                    "yaw_deg": { "type": "number", "description": "Horizontal orbit angle in degrees." },
                    "pitch_deg": { "type": "number", "description": "Vertical orbit angle in degrees (-85..85)." },
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
                    "show_grid": { "type": "boolean", "description": "Optional ground reference grid toggle." }
                }
            }
        },
        {
            "name": "get_comments",
            "description": "Retrieve spatial comments left by the user on specific sections of the 3D model, including target object_id, 3D world/local anchor coordinates, surface normal, local bounding box, and selected face indices.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_resolved": { "type": "boolean", "description": "Whether to include resolved comments (default false)." }
                }
            }
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
        }
    ])
}

/// Runs the stdio JSON-RPC 2.0 MCP Server loop.
pub fn run_mcp_stdio_server() {
    let mut fallback_scene = load_fallback_scene().unwrap_or_default();
    let mut fallback_camera = OrbitCamera::default();
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
                } else {
                    let req = IpcRequest {
                        tool: tool_name,
                        args,
                    };
                    let pre_disk_scene = load_fallback_scene();
                    if let Some(live_resp) = send_to_live_gui(&req) {
                        if !live_resp.ok && live_resp.message.starts_with("Unknown MCP tool") {
                            if let Some(disk_scene) = pre_disk_scene.or_else(load_fallback_scene) {
                                fallback_scene = disk_scene;
                            }
                            let resp = execute_ipc_command(
                                &req,
                                &mut fallback_scene,
                                &mut fallback_camera,
                                &mut fallback_history,
                                &mut fallback_ui,
                                &mut fallback_shading,
                                &mut fallback_wire,
                                &mut fallback_grid,
                            );
                            save_fallback_scene(&fallback_scene);
                            resp
                        } else {
                            // If live GUI is an older binary that doesn't persist new procedural fields yet,
                            // also apply set_material on the disk scene so screenshots & restarts preserve them.
                            if req.tool == "set_material"
                                && live_resp.data.get("pattern").is_none()
                            {
                                if let Some(disk_scene) = pre_disk_scene.or_else(load_fallback_scene) {
                                    fallback_scene = disk_scene;
                                }
                                let resp = execute_ipc_command(
                                    &req,
                                    &mut fallback_scene,
                                    &mut fallback_camera,
                                    &mut fallback_history,
                                    &mut fallback_ui,
                                    &mut fallback_shading,
                                    &mut fallback_wire,
                                    &mut fallback_grid,
                                );
                                save_fallback_scene(&fallback_scene);
                                resp
                            } else {
                                live_resp
                            }
                        }
                    } else {
                        // Execute against fallback scene and persist to /tmp/rust_modeler_scene.json
                        if let Some(disk_scene) = load_fallback_scene() {
                            fallback_scene = disk_scene;
                        }
                        let resp = execute_ipc_command(
                            &req,
                            &mut fallback_scene,
                            &mut fallback_camera,
                            &mut fallback_history,
                            &mut fallback_ui,
                            &mut fallback_shading,
                            &mut fallback_wire,
                            &mut fallback_grid,
                        );
                        save_fallback_scene(&fallback_scene);
                        resp
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
                        format!("Launched 3D Modeler GUI window (PID {pid}) and connected live IPC."),
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

pub fn load_fallback_scene() -> Option<Scene> {
    if let Ok(text) = fs::read_to_string(FALLBACK_SCENE_PATH) {
        if let Ok(scene) = serde_json::from_str::<Scene>(&text) {
            return Some(scene);
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
    if let Ok(json_str) = serde_json::to_string(scene) {
        let _ = fs::write(FALLBACK_SCENE_PATH, json_str);
    }
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
