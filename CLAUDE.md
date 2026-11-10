# CLAUDE.md — Claude Code Guide for `rust_modeler`

Please read [`AGENTS.md`](AGENTS.md) for the complete architectural documentation, module map, and AI-Driven Spatial Comment workflow.

## Quick Commands for Claude Code
- **Build & Test**:
  ```bash
  cargo test
  cargo build
  ```
- **Inspect Open User Comments & Scene State**:
  Use the `rust-modeler` MCP tools (`get_comments`, `get_scene_info`, `select_comment_region`, `edit_mesh`, `sculpt_stroke`, `voxel_brush`, `set_material`, `capture_viewport`, `resolve_comment`) configured in [`.mcp.json`](.mcp.json), or invoke via CLI:
  ```bash
  ./target/debug/rust_modeler --cmd '{"tool":"get_comments","args":{}}'
  ./target/debug/rust_modeler --cmd '{"tool":"get_scene_info","args":{}}'
  ```
- **Launch the Interactive 3D GUI**:
  ```bash
  ./target/debug/rust_modeler --cmd '{"tool":"launch_gui_window","args":{}}'
  ```
