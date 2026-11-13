# AGENTS.md — AI Agent Handover & Developer Guide (`rust_modeler`)

This document provides complete architectural context and operational instructions for AI coding agents (**OpenAI Codex**, **Claude Code**, **Google Antigravity**, **Cursor**, etc.) working on or controlling the **Rust AI 3D Modeler & Spatial Review Studio** (`rust_modeler`).

---

## 1. Core Philosophy & What This App Is

`rust_modeler` is an **AI-Agent-Driven 3D Modeling, Voxel Sculpting & Spatial Review Studio** written in Rust (`wgpu` 24 + `egui` 0.31 + `winit` 0.30 + `glam` 0.29).

### Why It Exists
Traditional 3D modeling software requires manual vertex pushing. In `rust_modeler`:
1. **Manual editing tools are intentionally removed from the user GUI.**
2. **The User's role in the GUI** is to inspect the 3D model, navigate with orbit/pan/zoom or the game-style **Fly mode** (`` ` ``), select one or more 3D parts/surfaces with the viewport tool strip (Part `1`, Surface `2`, Layer `3`, Brush `4`, Box `5` / `Ctrl+C`, with the **Add** multi-select switch), and leave **Spatial 3D Comments** (pinned to exact 3D coordinates, surface normals, and face index sets across one or more objects) or **Scene comments**, optionally with reference images. Comments stay drafts until the user presses **Send to agent**.
3. **The AI Agent's role** is to wait for sent feedback (`wait_for_feedback`), modify/sculpt/texture/add 3D geometry inside a revision, visually verify the result (`capture_viewport`, `capture_studio`), and submit the revision for review.

---

## 2. Standard AI Agent Workflow (Responding to User Comments)

Whenever the user asks you to "check comments", "address my feedback", or "update the 3D model":

1. **Wait for / Query Open Comments**:
   - Call `wait_for_feedback` (no arguments the first time). It returns immediately if sent, open comments are waiting; otherwise it blocks until the user presses **Send to agent** or requests changes on a revision. It returns the same payload as `get_comments` plus `feedback_batch`. After finishing a revision, call it again with `{"since_batch": <last feedback_batch>}` to keep working in a loop. `woke: false` means it timed out; call again.
   - `get_comments` (or `./target/debug/rust_modeler --cmd '{"tool":"get_comments","args":{}}'`) lists sent comments without waiting. Drafts are excluded unless `include_drafts: true`; never act on drafts.
   - `attachments` holds absolute paths of reference images the user attached; view them before modeling.
   - A comment the user reopened with **Request changes** has the user's notes as its latest entry in `replies`.
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
3. **Group the 3D Edits into a Reviewable Revision**:
   - Call `begin_revision` with `{"label": "Short description", "comment_id": <id>}` and retain the returned `revision_id`.
   - Apply the edits below. Calls within this revision form one undo step; only one in-progress or submitted revision can exist at a time.
   - For polygonal changes on the selected faces: call `edit_mesh` (`extrude`, `inset`, `subdivide`, `translate_selected`, `scale_selected`, `rotate_selected`, `delete_selected`). Note: if you omit `select_faces` in `edit_mesh`, it operates directly on the currently active `selected_faces` restored by `select_comment_region`!
   - For organic/sculpted changes around `local_anchor`: call `sculpt_stroke` (`draw`, `inflate`, `smooth`, `flatten`, `grab`, `pinch`, `crease` with `dyntopo: true`).
   - For volumetric CSG changes: call `voxel_brush` or `voxel_remesh`.
   - For chamfered edges: `edit_mesh` with `operation: "bevel"` and `width` chamfers the border of the selected region.
   - For new parts, prefer the built-in generators before writing scripts: `add_lathe` (revolved profiles: vases, columns, bowls), `add_sweep` (rails, pipes, branches, handles, mouldings along a path), `mirror_object` (build half, mirror the rest), `array_object` (rows, rings, spirals) and `boolean` (`subtract` for windows/holes/grooves, `union`, `intersect`). Only for shapes these cannot express, generate an `.obj` and `import_model` it. Then apply procedural shaders and vertex AO baking via `set_material`.
4. **Visually Verify Your Changes**:
   - Call `capture_viewport` (e.g. `{"path": "/tmp/preview.png", "width": 1280, "height": 800}`) and view the rendered PNG image to confirm the proportions, placement, colors, and lighting look great.
   - Call `clear_selection` first: selected parts glow teal and selected faces are tinted in every capture, which reads as a material change in review screenshots.
   - `capture_studio` with `{"compare_revision": <id>, "split": 0.5}` renders a before/after wipe of a finished revision; it makes a good screenshot for the user.
5. **Submit for User Review**:
   - Call `finish_revision` with `{"revision_id": <id>, "summary": "Brief explanation of what changed", "screenshots": ["/tmp/preview.png"]}`.
   - The user can compare Before / Split / After, highlight changed parts, and accept, revert, or **request changes**. The linked comment becomes Ready for review, then Accepted when the revision is accepted. Request changes reverts the geometry, appends the user's notes as a reply, reopens the comment, and wakes `wait_for_feedback`.
   - Use `cancel_revision` to restore an unfinished revision's starting geometry. Finish/cancel the current revision before opening another project. Review submitted work before changing geometry again.
   - If `get_comments` reports `needs_reattachment: true`, do not edit stale face indices. Ask the user to select the intended replacement region, then use `reattach_comment`.
   - `resolve_comment` remains available for legacy workflows without revisions.

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
| [`src/modeling.rs`](src/modeling.rs) | Higher-level modeling used by agent tools: `bevel_selected` (region chamfer), `mirrored`, `array_step`/`transformed`/`append`, `lathe`, `sweep` (rotation-minimising frames), and BSP-tree CSG `boolean` (`BooleanOp::Union`/`Subtract`/`Intersect`, run on a large-stack thread; result in the target's local space). Parts whose bounding boxes do not overlap skip CSG (union joins the meshes, subtract keeps the target). Overlapping parts build full BSP trees for classification but clip only polygons inside the other part's bounding box and gather the result without a final tree rebuild, which keeps fragment counts close to the inputs. `signed_volume` fixes winding of generated solids. |
| [`src/codec.rs`](src/codec.rs) | Serde `with` modules (`vec3s`, `rgbs`, `f32s`, `faces`) that store mesh/voxel arrays as `"<tag>:<base64(deflate(LE bytes))>"` strings; decoders also accept legacy JSON arrays. |
| [`src/mesh.rs`](src/mesh.rs) | Arbitrary polygonal mesh kernel (`Mesh` with `positions`, `normals`, `colors`, `faces: Vec<Vec<usize>>`, `selected_faces: HashSet<usize>`). Implements:<ul><li>Primitives (`cube`, `sphere`, `cylinder`, `plane`, `torus`)</li><li>Polygonal modeling (`extrude_selected_faces`, `inset_selected_faces`, `subdivide_or_loop_cut`, `translate_selected`, `scale_selected`, `rotate_selected`, `delete_selected`)</li><li>**Smart Selection**: 2mm spatially welded vertex hashing (`Mesh::quantize_pos`), connected component BFS (`compute_connected_component`, `select_connected_component`), crease-angle surface flood-fill (`compute_smart_surface`, `select_smart_surface`), topological ring `grow_selection` & `shrink_selection`, and live hover preview wireframe builder (`build_face_subset_lines`)</li><li>**Vertex Baking**: `bake_ao_and_curvature` (ambient occlusion + edge wear) and `paint_gradient_and_normal_coat` (moss/snow/dust top-coat + vertical gradient)</li></ul> |
| [`src/voxel.rs`](src/voxel.rs) | 3D Sculpting & Volumetric SDF Voxel Engine:<ul><li>Surface sculpting brushes (`SculptBrush`: `Draw`, `Inflate`, `Smooth`, `Flatten`, `Grab`, `Pinch`, `Crease`) with `dyntopo` dynamic edge-splitting tessellation (`apply_sculpt_stroke`)</li><li>Volumetric Signed Distance Field grid (`VoxelGrid`) with CSG brushes (`VoxelBrushOp`), **Smooth Dual Surface Nets** iso-surface extraction (`extract_smooth_surface_nets`), **Blocky Voxel** cube extraction (`extract_blocky_voxels`), and `voxel_remesh`</li></ul> |
| [`src/renderer.rs`](src/renderer.rs) | `wgpu` 3D rendering pipeline (`Renderer3D`):<ul><li>Studio backdrop (`vs_backdrop`/`fs_backdrop`) and soft contact shadow (`fs_shadow`, shown with the grid)</li><li>`prepare_objects` + `draw_objects` per pass; `compare_split` renders a revision's before (left) and after (right) with scissor rects. GPU caches are keyed by `(object id, side)` so both sides stay cached.</li><li>Whole-part selection glow (`inspection.y` → `selection_glow`), selected-face tint, and selection outline buffers</li><li>Per-object GPU vertex/wireframe buffer caching (`MeshCacheKey`, `CachedGpuMesh`)</li><li>3-pass rendering: (1) Solid PBR + Procedural Shader triangles (`triangle_pipeline`), (2) Depth-tested grid, selected face wireframes, and electric-cyan **live hover preview wireframe** (`line_depth_pipeline`), (3) Overlay 3D Comment Pin Beacons & brush cursor rings (`line_overlay_pipeline`)</li><li>Offscreen & headless PNG screenshot renderer (`render_to_png_file`, `render_scene_headless_to_png`)</li></ul> |
| [`src/shader.wgsl`](src/shader.wgsl) | WGSL vertex & fragment shader implementing 3-point studio lighting (warm key, cool sky fill, rim), ACES filmic tone mapping, emissive glow, clay matcap shading, and 8 tri-planar 3D procedural textures (`checker`, `bricks`, `stone_blocks`, `wood_grain`, `marble`, `hex_grid`, `rune_circuit`). |
| [`src/ui.rs`](src/ui.rs) | `egui` user interface (`draw_ui`, `UiState`, `SmartSelectMode`), themed in `apply_theme` with Fira Sans and Phosphor icons (`egui_phosphor::regular as icon`). Shared widgets: `icon_button`, `toggle`, `segmented`, `pill`, `card`.<ul><li>Header (`draw_header`): brand, project menu (new/open/save as/recover/export, title, units), command search, undo/redo, Save</li><li>Viewport overlays: `viewport_tool_strip` (modes, measure, fly, frame, shortcuts), `viewport_context_bar` (active tool options), `viewport_view_controls` (shading, edges, grid, ghost, section, camera), `viewport_axis_gizmo` (click to look along an axis), `viewport_selection_chip` (grow/shrink/comment/clear), comment pins, compare banner with the draggable split divider</li><li>Left panel: Parts (search, show all / isolate / frame, rows with visibility, colour swatch, face count and open-comment badge), Inspect, Saved</li><li>Right panel: Send-to-agent card for drafts, comment composer (attachments via drop / path / viewport snapshot, Send immediately switch), comment cards, Revisions tab (Before / Split / After, Accept, Request changes, Revert)</li><li>`draw_command_palette` (`Ctrl+K`, `Cmd` enum)</li></ul> |
| [`src/ipc.rs`](src/ipc.rs) | Command dispatcher (`execute_ipc_command`, `summarize_scene`, `IpcRequest`, `IpcResponse`) translating JSON tool calls into scene/mesh/voxel/material/comment/screenshot operations. |
| [`src/mcp.rs`](src/mcp.rs) | MCP JSON-RPC 2.0 stdio server (`run_mcp_stdio_server`, `mcp_tool_definitions`), live GUI socket client (`send_to_live_gui`), and scene persistence (`load_fallback_scene`, `save_fallback_scene`). |
| [`src/io.rs`](src/io.rs) | Import and export for Wavefront `.obj`, binary/ASCII `.stl`, and binary `.glb` (glTF 2.0). |
| [`src/camera.rs`](src/camera.rs) | Spherical orbit camera (`OrbitCamera`, `Ray`) with perspective/orthographic projection, `screen_ray` unprojection, `world_to_screen` projection for 3D UI pins, and first-person helpers (`fly_look`, `fly_move`, `refocus`, `look_from`). |
| [`src/navigation.rs`](src/navigation.rs) | Free-fly controller (`FlyController`): held-key WASD/QE movement, speed multipliers, scroll speed. `ModelerApp::set_fly` in `main.rs` captures the cursor (Locked, falling back to Confined), reads raw `DeviceEvent::MouseMotion`, and keeps egui from receiving pointer input while flying. |
| [`src/autosave.rs`](src/autosave.rs) | Debounced background scene saving (`SceneSaver`: `request`, `poll`, `flush`). |
| [`src/box_select.rs`](src/box_select.rs) | Box selector: rasterizes faces into a depth/face-id buffer covering only the dragged rectangle (near-plane clipped, section-plane and visibility aware), plus a centre test for sub-pixel faces. `apply_box_selection` handles replace/add/subtract; each disconnected selected patch becomes its own `CommentTargetSection`. |
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
   - Whenever the user selects faces or adds/resolves/deletes comments in the GUI, `ModelerApp::persist` schedules a save through `autosave::SceneSaver`: changes within 400 ms share one write, which runs on a worker thread from a scene snapshot so input and rendering never wait on serialization or `fsync`. Closing the window or `quit_gui` flushes pending writes. While the GUI runs, MCP calls are answered from its in-memory scene over IPC, so agents always see the latest selections and comments.
3. **Running Tests**:
   - Run `cargo test` (or `cargo test --offline` in sandboxed environments). All 5 test suites in `src/main.rs` cover mesh editing, voxel sculpting + dyntopo + remeshing, `.obj`/`.stl`/`.glb` round-trips, MCP tool execution (including materials, AO baking, spatial & general comments), and smart sub-part / crease-surface / multi-object selection.


## 6. Project, Revision, and UI Implementation

- `src/project.rs`: versioned `.spatial.json` projects, atomic write/rename, `.bak` backups, `.autosave` paths, camera and mesh validation. `mcp::persist_scene` updates recovery data and the current camera; explicit Save keeps the named project version. Geometry arrays use `src/codec.rs`; old plain-array files still load.
- `src/review.rs`: revision lifecycle, comparison views (`revision_view`, `display_scene`), changed-object detection, conflict checks, acceptance, reversion and `request_revision_changes`. While in progress a revision holds a full `before` snapshot (needed by cancel). `finish_revision` calls `Revision::compact`, which keeps only changed objects (`partial`, `object_order`); `before_objects`/`after_objects` rebuild full scenes from live parts and `matches_live` checks conflicts on changed parts only. `Scene::prune_revision_geometry` (run on review and in `Scene::validate`, which migrates old projects) drops geometry from reviewed revisions older than `KEEP_REVIEWED_GEOMETRY`, setting `geometry_pruned`. `object_hash` hashes raw data, not JSON. `Scene::preview_revision` is transient; never edit the snapshot displayed by a comparison.
- Feedback batches: `SpatialComment::draft` and `attachments`; `Scene::feedback_batch` increments in `send_drafts` and `request_revision_changes`. `get_feedback_state` is a cheap read-only poll; `mcp::wait_for_feedback` loops on it from the MCP server or `--cmd` process (never inside the GUI thread). The GUI records those polls in `UiState::agent_listening` instead of the activity log.
- `src/review_tests.rs`: project/recovery, comment remapping, topology invalidation, multi-region selection, revision grouping/conflicts, unit conversion, and clipping/picking regression tests.
- Spatial targets store a topology hash plus face signatures. `target_faces` verifies or remaps targets; ambiguous remaps require explicit reattachment. `refreshed_target` returns current coordinates, normals, and indices without rewriting stored comments.
- Performance: hover raycasts, brush painting and fly-mode look run once per frame in `ModelerApp::process_pointer`, never per input event. `Mesh` lazily caches weld adjacency, edge adjacency, bounds and 64-face raycast chunk boxes keyed by `(version, positions.len(), faces.len())`; clones start with an empty cache. Because not every geometry edit bumps `version`, call `Scene::invalidate_mesh_caches` wherever `Renderer3D::invalidate_cache` is called after commands. `[profile.dev]` is optimized because the GUI is normally run from `target/debug`.
- The GUI renders only inside `UiState::viewport_rect`; input coordinates are converted from window pixels to viewport pixels using egui’s current pixels-per-point. Keep camera aspect, render viewport, and egui pin projections aligned with that rectangle.
- `src/viewport_input.rs` routes pointer input to the empty 3D viewport. Do not gate viewport navigation on `egui_ctx.wants_pointer_input()` or `EventResponse::consumed`: the transparent `CentralPanel` makes those true over the canvas. Check the viewport rectangle, topmost layer, active UI gestures, and open menus/dialogs instead. Camera drags must start inside the viewport; panel scrolling stays with egui. Its tests reproduce actual egui panel ownership and cover mouse-wheel/trackpad zoom, overlays, menus/dialogs, fresh cursor events, and display scaling.
- Heavy live commands run on isolated scene copies in a background worker; only successful, uncancelled results are merged into live geometry. Feedback written during the operation is retained. Cancel discards results; it does not forcibly terminate a Rust thread.
- IPC connections are handled independently so cancellation/read-only queries can reach the GUI during long operations. A timeout after connecting to a live GUI must return an error, never trigger headless replay. Headless commands hold a file lock across load/execute/persist.
- GUI project/geometry actions use the same IPC dispatcher as MCP. Keep tool schemas in `mcp_tool_definitions` synchronized with the dispatcher. Read-only tools, including `get_comments`, `get_revisions`, `capture_viewport`, and `capture_studio`, must not rewrite persisted state.


## 7. Repository Boundaries

This repository contains the modeling application, its runtime resources, documentation, and deliberate test fixtures. User-created models, saved projects, exports, per-comment geometry generators, and generated review captures are local project data and must not be committed to application PRs. Keep project files and generated model outputs local; `.gitignore` excludes them. When preparing a PR, inspect untracked files and include only application changes and intentional application resources.
