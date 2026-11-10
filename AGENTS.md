# AGENTS.md — AI Agent Handover & Developer Guide (`rust_modeler`)

This document provides complete architectural context and operational instructions for AI coding agents (**OpenAI Codex**, **Claude Code**, **Google Antigravity**, **Cursor**, etc.) working on or controlling the **Rust AI 3D Modeler & Spatial Review Studio** (`rust_modeler`).

---

## 1. Core Philosophy & What This App Is

`rust_modeler` is an **AI-Agent-Driven 3D Modeling, Voxel Sculpting & Spatial Review Studio** written in Rust (`wgpu` 24 + `egui` 0.31 + `winit` 0.30 + `glam` 0.29).

### Why It Exists
Traditional 3D modeling software requires manual vertex pushing. In `rust_modeler`:
1. **Manual editing tools are intentionally removed from the user GUI.**
2. **The User's role in the GUI** is to inspect the 3D model, select one or more 3D parts/surfaces using **Smart Selection tools** (`🧩 Sub-Part`, `📐 Smart Surface`, `📦 Whole Layer`, `🖌 Paint Brush`) with **Multi-Selection (`➕ Multi-Select`)**, and leave **Spatial 3D Comments** (pinned to exact 3D coordinates, surface normals, and face index sets across one or more objects) or **General Scene Comments**.
3. **The AI Agent's role** is to read the user's spatial/general comments via the built-in MCP server (`get_comments` / `get_scene_info`), modify/sculpt/texture/add 3D geometry accordingly, visually verify the result using `capture_viewport`, and resolve the user's comments via `resolve_comment`.

---

## 2. Standard AI Agent Workflow (Responding to User Comments)

Whenever the user asks you to "check comments", "address my feedback", or "update the 3D model":

1. **Query Open Comments**:
   - Call the MCP tool `get_comments` (or run `./target/debug/rust_modeler --cmd '{"tool":"get_comments","args":{}}'`).
   - Inspect each unresolved comment:
     - If `is_general: true`: The comment applies to the whole scene (e.g., "Add a bridge on the south side" or "Make the lighting warmer").
     - If `is_general: false`: Inspect `targets` (an array of `CommentTargetSection` entries). A single comment can target **one or multiple selections** across the same or different objects! Each target entry contains:
       - `object_id` & `object_name`
       - `world_anchor` & `local_anchor` (`[x, y, z]`)
       - `surface_normal` (`[nx, ny, nz]`)
       - `selected_faces` (exact `usize` polygon face indices highlighted by the user)
       - `local_bounds_min` & `local_bounds_max` (3D bounding box of the selected section in local object space)
2. **Activate / Focus the Commented Region**:
   - Call `select_comment_region` with `{"comment_id": <id>}`. This automatically selects the target object(s) and restores `selected_faces` on every target mesh in the scene.
3. **Perform the 3D Edits**:
   - For polygonal changes on the selected faces: call `edit_mesh` (`extrude`, `inset`, `subdivide`, `translate_selected`, `scale_selected`, `rotate_selected`, `delete_selected`). Note: if you omit `select_faces` in `edit_mesh`, it operates directly on the currently active `selected_faces` restored by `select_comment_region`!
   - For organic/sculpted changes around `local_anchor`: call `sculpt_stroke` (`draw`, `inflate`, `smooth`, `flatten`, `grab`, `pinch`, `crease` with `dyntopo: true`).
   - For volumetric CSG changes: call `voxel_brush` or `voxel_remesh`.
   - For new detailed parts/meshes: you can use `add_primitive` or generate a multi-part `.obj` mesh (via a quick script or procedural generator) and import it via `import_model`, then apply procedural shaders and vertex AO baking via `set_material`.
4. **Visually Verify Your Changes**:
   - Call `capture_viewport` (e.g. `{"path": "/tmp/preview.png", "width": 1280, "height": 800}`) and view the rendered PNG image to confirm the proportions, placement, colors, and lighting look great.
5. **Resolve the Comment**:
   - Call `resolve_comment` with `{"comment_id": <id>, "resolved": true, "reply": "Brief explanation of what you changed"}`.

---

## 3. How to Control the App (MCP & CLI Modes)

The single binary `rust_modeler` supports three execution modes:

1. **Interactive GUI Window (Default)**:
   ```bash
   cargo run
   ```
   - Opens the `wgpu` + `egui` 3D viewport window.
   - Starts a background UNIX domain socket listener at `/tmp/rust_modeler_ipc.sock` (`spawn_ipc_listener_thread` in [`src/main.rs`](src/main.rs)).
   - Automatically loads `/tmp/rust_modeler_scene.json` (or seeds it from [`assets/scenes/clockwork_astral_cathedral_scene.json`](assets/scenes/clockwork_astral_cathedral_scene.json) if `/tmp/rust_modeler_scene.json` does not exist).

2. **MCP Server over `stdio` (`--mcp`)**:
   ```bash
   cargo run --quiet -- --mcp
   ```
   - Implements the JSON-RPC 2.0 Model Context Protocol (`initialize`, `tools/list`, `tools/call`) in [`src/mcp.rs`](src/mcp.rs). Supports both newline-delimited JSON-RPC and `Content-Length` framed stdio streams.
   - If the GUI is running (`/tmp/rust_modeler_ipc.sock` is active), every MCP tool call is forwarded live to the GUI event loop so the user sees changes in real time, and persisted to `/tmp/rust_modeler_scene.json`.
   - If the GUI is not running, the MCP server executes the command headlessly against `/tmp/rust_modeler_scene.json` (including GPU-accelerated headless screenshot rendering for `capture_viewport`!).

3. **One-Shot CLI Command (`--cmd '<json>'`)**:
   ```bash
   ./target/debug/rust_modeler --cmd '{"tool":"get_scene_info","args":{}}'
   ```
   - Useful for shell scripts, tests, or agents without MCP configured. Routes to the live GUI if open, or executes against `/tmp/rust_modeler_scene.json` if headless.

---

## 4. Codebase Architecture & Module Map

| File | Responsibility & Key Types |
| :--- | :--- |
| [`src/main.rs`](src/main.rs) | Entry point (`--mcp`, `--cmd`, or GUI `winit` event loop), `ModelerApp`, viewport mouse/keyboard handling (`handle_viewport_left_press`, `handle_cursor_moved`, `update_hover_preview`, `handle_key_press`), UNIX socket listener thread, and integration unit tests (`#[cfg(test)]`). |
| [`src/scene.rs`](src/scene.rs) | Scene graph (`Scene`, `SceneObject`, `Transform`), procedural material enum (`ProceduralPattern`), raycasting (`Scene::raycast`, `Scene::raycast_active`), and the multi-target spatial comment system (`SpatialComment`, `CommentTargetSection`, `collect_current_target_sections`, `add_comment_for_selection`, `add_general_comment`, `focus_comment`, `clear_all_selections`). |
| [`src/mesh.rs`](src/mesh.rs) | Arbitrary polygonal mesh kernel (`Mesh` with `positions`, `normals`, `colors`, `faces: Vec<Vec<usize>>`, `selected_faces: HashSet<usize>`). Implements:<ul><li>Primitives (`cube`, `sphere`, `cylinder`, `plane`, `torus`)</li><li>Polygonal modeling (`extrude_selected_faces`, `inset_selected_faces`, `subdivide_or_loop_cut`, `translate_selected`, `scale_selected`, `rotate_selected`, `delete_selected`)</li><li>**Smart Selection**: 2mm spatially welded vertex hashing (`Mesh::quantize_pos`), connected component BFS (`compute_connected_component`, `select_connected_component`), crease-angle surface flood-fill (`compute_smart_surface`, `select_smart_surface`), topological ring `grow_selection` & `shrink_selection`, and live hover preview wireframe builder (`build_face_subset_lines`)</li><li>**Vertex Baking**: `bake_ao_and_curvature` (ambient occlusion + edge wear) and `paint_gradient_and_normal_coat` (moss/snow/dust top-coat + vertical gradient)</li></ul> |
| [`src/voxel.rs`](src/voxel.rs) | 3D Sculpting & Volumetric SDF Voxel Engine:<ul><li>Surface sculpting brushes (`SculptBrush`: `Draw`, `Inflate`, `Smooth`, `Flatten`, `Grab`, `Pinch`, `Crease`) with `dyntopo` dynamic edge-splitting tessellation (`apply_sculpt_stroke`)</li><li>Volumetric Signed Distance Field grid (`VoxelGrid`) with CSG brushes (`VoxelBrushOp`), **Smooth Dual Surface Nets** iso-surface extraction (`extract_smooth_surface_nets`), **Blocky Voxel** cube extraction (`extract_blocky_voxels`), and `voxel_remesh`</li></ul> |
| [`src/renderer.rs`](src/renderer.rs) | `wgpu` 3D rendering pipeline (`Renderer3D`):<ul><li>Per-object GPU vertex/wireframe buffer caching (`MeshCacheKey`, `CachedGpuMesh`)</li><li>3-pass rendering: (1) Solid PBR + Procedural Shader triangles (`triangle_pipeline`), (2) Depth-tested grid, selected face wireframes, and electric-cyan **live hover preview wireframe** (`line_depth_pipeline`), (3) Overlay 3D Comment Pin Beacons & brush cursor rings (`line_overlay_pipeline`)</li><li>Offscreen & headless PNG screenshot renderer (`render_to_png_file`, `render_scene_headless_to_png`)</li></ul> |
| [`src/shader.wgsl`](src/shader.wgsl) | WGSL vertex & fragment shader implementing 3-point studio lighting (warm key, cool sky fill, rim), ACES filmic tone mapping, emissive glow, clay matcap shading, and 8 tri-planar 3D procedural textures (`checker`, `bricks`, `stone_blocks`, `wood_grain`, `marble`, `hex_grid`, `rune_circuit`). |
| [`src/ui.rs`](src/ui.rs) | `egui` user interface (`draw_ui`, `UiState`, `SmartSelectMode`):<ul><li>Top Bar: Export menu, Smart Select Mode selector (`🧩 Sub-Part (1)`, `📐 Smart Surface (2)`, `📦 Whole Layer (3)`, `🖌 Paint Brush (4)`), contextual `Crease` / `Radius` slider, `➕ Multi-Select` toggle, `➕ Grow (])`, `➖ Shrink ([)`, `🧹 Clear`, Shading & Camera presets</li><li>Left Panel: Read-only `🗂 Scene Parts` outliner with visibility toggles (`👁`), isolate button, selection indicators (`✦`), and open comment badges (`💬`)</li><li>Right Panel: `💬 AI Agent Comments` composer showing all currently selected sections across the scene (with per-part `✖` removal buttons), `📍 Add Comment` / `🌐 Add General Comment` buttons, and interactive comment cards (`🎯 View`, `✔ Resolve`, `🗑 Delete`)</li><li>Viewport Overlay: Floating clickable 3D comment pill badges (`💬 #id: ...`) projected via `camera.world_to_screen`</li></ul> |
| [`src/ipc.rs`](src/ipc.rs) | Command dispatcher (`execute_ipc_command`, `summarize_scene`, `IpcRequest`, `IpcResponse`) translating JSON tool calls into scene/mesh/voxel/material/comment/screenshot operations. |
| [`src/mcp.rs`](src/mcp.rs) | MCP JSON-RPC 2.0 stdio server (`run_mcp_stdio_server`, `mcp_tool_definitions`), live GUI socket client (`send_to_live_gui`), and scene persistence (`load_fallback_scene`, `save_fallback_scene`). |
| [`src/io.rs`](src/io.rs) | Import and export for Wavefront `.obj`, binary/ASCII `.stl`, and binary `.glb` (glTF 2.0). |
| [`src/camera.rs`](src/camera.rs) | Spherical orbit camera (`OrbitCamera`, `Ray`) with perspective/orthographic projection, `screen_ray` unprojection, and `world_to_screen` projection for 3D UI pins. |
| [`src/history.rs`](src/history.rs) | Bounded snapshot undo/redo stack (`UndoHistory`). |
| [`src/gizmo.rs`](src/gizmo.rs) | 3D axis gizmo state (`GizmoState`). |

---

## 5. Key Implementation Details & Invariants

1. **Spatial Vertex Welding (`Mesh::quantize_pos`)**:
   - Imported `.obj` / `.stl` meshes and extruded faces often duplicate vertices along hard edges so each face has distinct normals.
   - `compute_connected_component`, `compute_smart_surface`, `grow_selection`, and `shrink_selection` in [`src/mesh.rs`](src/mesh.rs) quantize 3D positions to a `2mm` grid (`(p * 500.0).round() as i32`) when building vertex/edge adjacency maps. This guarantees that `🧩 Sub-Part` and `📐 Smart Surface` work reliably even on non-manifold or split-vertex meshes.
2. **State Synchronization Between GUI and MCP**:
   - The active scene is persisted at `/tmp/rust_modeler_scene.json` (`FALLBACK_SCENE_PATH`).
   - Read-only MCP tools (`get_scene_info`, `get_comments`, `capture_viewport`) do **not** overwrite `/tmp/rust_modeler_scene.json` in `ModelerApp::user_event`.
   - Whenever the user selects faces or adds/resolves/deletes comments in the GUI, `mcp::save_fallback_scene(&self.scene)` is called immediately so any external AI agent querying `get_comments` or `get_scene_info` always sees the latest user selections and comments.
3. **Running Tests**:
   - Run `cargo test` (or `cargo test --offline` in sandboxed environments). All 5 test suites in `src/main.rs` cover mesh editing, voxel sculpting + dyntopo + remeshing, `.obj`/`.stl`/`.glb` round-trips, MCP tool execution (including materials, AO baking, spatial & general comments), and smart sub-part / crease-surface / multi-object selection.
