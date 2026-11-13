# Rust 3D Modeler & Voxel Sculpt Studio MCP Integration

You can directly control the **Rust 3D Modeler** application via the `rust-modeler` MCP server (`rust_modeler --mcp`).
For full architectural details, module layout, and the AI-driven spatial comment workflow, see [`AGENTS.md`](../../../../AGENTS.md) at the repository root.

## Available MCP Tools
- `get_scene_info`: Inspect all objects in the 3D scene, their IDs, names, transforms, materials, vertex/face counts, bounding boxes, camera state, `unresolved_comments` (with multi-selection `targets`), and `active_selection`.
- `get_comments`: Retrieve spatial and general comments left by the user on the 3D model, including `is_general`, `targets` (array of selected sections across one or more objects with `object_id`, `world_anchor`, `local_anchor`, `surface_normal`, `local_bounds_min`/`max`, and `selected_faces` indices).
- `select_comment_region`: Activate the target object(s) and restore all highlighted `selected_faces` of a `comment_id` so `edit_mesh` or `sculpt_stroke` operates directly on the user's selected section(s).
- `resolve_comment`: Mark a `comment_id` as resolved and attach an `agent_reply` explaining the changes made.
- `add_comment`: Programmatically add a section comment or general scene comment (`is_general: true`).
- `launch_gui_window`: Launch the interactive 3D Modeler GUI window if it is not already running.
- `add_primitive`: Create a `cube`, `sphere`, `cylinder`, `plane`, `torus`, or `voxel_sphere` with position, rotation, scale, subdivisions, and RGB color.
- `transform_object`: Move, rotate, scale, rename, or hide/show any object by `object_id` or `name`.
- `set_material`: Update base `color`, `secondary_color`, `roughness`, `metallic`, `flat_shading`, tri-planar 3D procedural shader `pattern` (`none`, `checker`, `bricks`, `stone_blocks`, `wood_grain`, `marble`, `hex_grid`, `rune_circuit`), `texture_scale`, emissive glow (`emission_color`, `emission_strength`), and automatic vertex-color baking (`bake_ao`, `ao_strength`, `wear_strength`, `crevice_color`, `wear_color`, `bottom_color`, `top_color`, `top_coat_color`).
- `edit_mesh`: Perform polygonal modeling operations (`select_by_normal`, `select_all`, `clear_selection`, `extrude`, `inset`, `bevel`, `subdivide`, `translate_selected`, `scale_selected`, `rotate_selected`, `delete_selected`, `bridge`, `fill_holes`, `weld`, `flip_normals`, `smooth`, `subdivide_smooth`, `decimate`, `invert_selection`, `grow_selection`, `shrink_selection`).
- `query_geometry` / `measure` / `check_mesh`: Read real geometry (world bounds, per-face centres, normals and areas, connected regions), distances and gaps between parts, and mesh health (holes, flipped faces, overlaps, floating parts).
- `select_faces`: Select faces by `box`, `sphere`, `connected`, `surface`, `normal`, `similar`, `path`, `invert`, `grow` or `shrink`, with `op` replace/add/subtract.
- `add_mesh` / `add_extrude` / `add_loft`: Create a part from raw vertices and faces, from a 2D outline raised into a solid, or by skinning through cross-sections.
- `deform_object` / `align_object` / `scatter_object`: Bend, twist or taper a part; line parts up or stack them; spread copies over a surface.
- `paint_faces`: Colour only the selected faces of a part.
- `capture_views`: Render several camera angles into one contact-sheet PNG.
- `batch`: Run many tool calls in one round trip.
- `sculpt_stroke`: Apply 3D sculpting brushes (`draw`, `inflate`, `smooth`, `flatten`, `grab`, `pinch`, `crease`) at a surface `point` or array of `points`, with optional `symmetry_x`/`y`/`z` and `dyntopo` auto-tessellation.
- `voxel_brush`: Apply volumetric SDF CSG operations (`add_sphere`, `subtract_sphere`, `add_box`, `subtract_box`, `smooth_field`, `paint_color`) and choose `smooth` (Dual Surface Nets) or `blocky` voxel meshing.
- `voxel_remesh`: Remesh any object into uniform quad topology at a specified voxel grid resolution.
- `set_camera_and_viewport`: Adjust camera orbit angles, target, zoom distance, first-person `eye`/`look_at` placement, view presets (`front`, `right`, `top`), projection (`orthographic`), shading (`lit`, `clay`, `wireframe`), and editor mode.
- `capture_viewport`: Render the 3D viewport to a PNG image file (`path`, `width`, `height`, and optional camera/shading overrides) so the AI agent can visually inspect its 3D models with `read_file`.
- `undo` / `redo`: Step backward or forward through the modeling history.
- `import_model` / `export_model`: Import or export `.obj`, `.stl`, and `.glb` (glTF 2.0) files.
