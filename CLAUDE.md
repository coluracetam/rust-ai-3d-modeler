# CLAUDE.md — Claude Code Guide for `rust_modeler`

Please read [`AGENTS.md`](AGENTS.md) for the complete architectural documentation, module map, and AI-Driven Spatial Comment workflow.

## Quick Commands for Claude Code
- **Build & Test**:
  ```bash
  cargo test
  cargo build
  ```
- **Inspect Open User Comments & Scene State**:
  Use the `rust-modeler` MCP tools (`wait_for_feedback`, `get_comments`, `get_scene_info`, `select_comment_region`, `query_geometry`, `select_faces`, `begin_revision`, `edit_mesh`, `add_mesh`, `add_extrude`, `add_loft`, `deform_object`, `align_object`, `scatter_object`, `paint_faces`, `set_texture`, `boolean`, `add_lathe`, `add_sweep`, `mirror_object`, `array_object`, `sculpt_stroke`, `voxel_brush`, `set_material`, `capture_viewport`, `capture_views`, `capture_studio`, `check_mesh`, `batch`, `finish_revision`, and the asset library tools `library_list`, `library_show`, `library_save_asset`, `library_insert_asset`, `library_create_section`, `library_move_asset`, `library_export`, `library_delete_asset`, `library_delete_section`) configured in [`.mcp.json`](.mcp.json), or invoke via CLI:
  ```bash
  ./target/debug/rust_modeler --cmd '{"tool":"get_comments","args":{}}'
  ./target/debug/rust_modeler --cmd '{"tool":"get_scene_info","args":{}}'
  ```
- **Launch the Interactive 3D GUI**:
  ```bash
  ./target/debug/rust_modeler --cmd '{"tool":"launch_gui_window","args":{}}'
  ```
