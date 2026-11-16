#![recursion_limit = "512"]

mod agent_tools;
mod autosave;
mod bake;
mod box_select;
pub mod codec;
pub mod camera;
pub mod gizmo;
pub mod gltf;
pub mod history;
pub mod io;
pub mod ipc;
pub mod library;
pub mod mcp;
pub mod mesh;
pub mod modeling;
mod modeling_ops;
mod navigation;
pub mod project;
pub mod renderer;
pub mod review;
#[cfg(test)]
mod review_tests;
pub mod scene;
pub mod ui;
mod viewport_input;
pub mod voxel;

use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::history::UndoHistory;
use crate::ipc::{IPC_SOCKET_PATH, IpcRequest, IpcResponse, execute_ipc_command};
use crate::mesh::EditElementMode;
use crate::renderer::{Renderer3D, ViewportShading};
use crate::scene::Scene;
use crate::ui::{SmartSelectMode, UiState, draw_ui, update_pending_anchor_from_selection};
use glam::{Vec2, Vec3};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::sync::{Arc, mpsc};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

#[derive(Debug)]
pub struct IpcEvent {
    pub request: IpcRequest,
    pub reply_tx: mpsc::Sender<IpcResponse>,
}

struct GpuContext {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pending_resize: Option<PhysicalSize<u32>>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    renderer_3d: Renderer3D,
}

struct BackgroundResult {
    response: IpcResponse,
    scene: Scene,
    history: UndoHistory,
}
struct BackgroundOperation {
    receiver: mpsc::Receiver<BackgroundResult>,
    reply: mpsc::Sender<IpcResponse>,
    tool: String,
    started: std::time::Instant,
    selected_id: Option<u64>,
}

impl BackgroundOperation {
    fn start(
        event: IpcEvent,
        scene: &Scene,
        camera: &OrbitCamera,
        history: &UndoHistory,
        ui_state: &mut UiState,
        gpu: Option<&GpuContext>,
    ) -> Self {
        let IpcEvent { request, reply_tx } = event;
        let selected_id = scene.active_object().map(|o| o.id);
        let (mut scene, mut camera, mut history) = (scene.clone(), camera.clone(), history.clone());
        let mut state = ui_state.clone();
        let mut shading = gpu.map_or(ViewportShading::StudioLit, |g| g.renderer_3d.shading);
        let mut wire = gpu.is_some_and(|g| g.renderer_3d.show_wireframe_overlay);
        let mut grid = gpu.is_none_or(|g| g.renderer_3d.show_grid);
        let (tx, receiver) = mpsc::channel();
        let started = std::time::Instant::now();
        let tool = request.tool.clone();
        ui_state.busy_operation = Some(ui::BusyOperation {
            name: tool.replace('_', " "),
            started,
            can_cancel: !ipc::is_export(&tool),
        });
        std::thread::spawn(move || {
            let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute_ipc_command(
                    &request,
                    &mut scene,
                    &mut camera,
                    &mut history,
                    &mut state,
                    &mut shading,
                    &mut wire,
                    &mut grid,
                )
            }))
            .unwrap_or_else(|_| {
                IpcResponse::error("Background operation failed; live geometry preserved")
            });
            let _ = tx.send(BackgroundResult {
                response,
                scene,
                history,
            });
        });
        Self {
            receiver,
            reply: reply_tx,
            tool,
            started,
            selected_id,
        }
    }
}

pub struct ModelerApp {
    gpu: Option<GpuContext>,
    background: Option<BackgroundOperation>,
    scene: Scene,
    camera: OrbitCamera,
    gizmo: GizmoState,
    history: UndoHistory,
    ui_state: UiState,

    // Input tracking
    modifiers: ModifiersState,
    cursor_pos: Vec2,
    last_cursor_pos: Vec2,
    left_mouse_down: bool,
    middle_mouse_down: bool,
    right_mouse_down: bool,
    last_sculpt_local_hit: Option<Vec3>,
    hover_brush_world: Option<(Vec3, Vec3)>,
    /// Window-pixel start of an in-progress box selection drag.
    box_drag_start: Option<Vec2>,
    fly: navigation::FlyController,
    /// Fly speed has been chosen for this session (otherwise derived from the view on entry).
    fly_speed_set: bool,
    /// The cursor moved since the last frame; hover/brush raycasts run once per frame.
    pointer_dirty: bool,
    /// Raw mouse-look motion gathered since the last frame while flying.
    fly_look_delta: Vec2,
    saver: autosave::SceneSaver,
}

impl Default for ModelerApp {
    fn default() -> Self {
        Self::with_scene(mcp::load_fallback_scene().unwrap_or_default())
    }
}

impl ModelerApp {
    fn with_scene(initial_scene: Scene) -> Self {
        let mut initial_camera = initial_scene.last_camera.clone().unwrap_or_default();
        if initial_scene.last_camera.is_none() {
            ui::frame_all(&initial_scene, &mut initial_camera);
        }
        Self {
            gpu: None,
            background: None,
            scene: initial_scene,
            camera: initial_camera,
            gizmo: GizmoState::default(),
            history: UndoHistory::default(),
            ui_state: UiState::default(),
            modifiers: ModifiersState::default(),
            cursor_pos: Vec2::ZERO,
            last_cursor_pos: Vec2::ZERO,
            left_mouse_down: false,
            middle_mouse_down: false,
            right_mouse_down: false,
            last_sculpt_local_hit: None,
            hover_brush_world: None,
            box_drag_start: None,
            fly: navigation::FlyController::default(),
            fly_speed_set: false,
            pointer_dirty: false,
            fly_look_delta: Vec2::ZERO,
            saver: autosave::SceneSaver::default(),
        }
    }
}

impl ModelerApp {
    fn cancel_operation(&mut self) -> IpcResponse {
        // An export writes files from the worker. Dropping its reply cannot undo those writes.
        if self.background.as_ref().is_some_and(|op| ipc::is_export(&op.tool)) {
            return IpcResponse::error("Wait for the export to finish");
        }
        if let Some(operation) = self.background.take() {
            let _ = operation
                .reply
                .send(IpcResponse::error("Operation cancelled; result discarded"));
            self.ui_state.busy_operation = None;
            self.ui_state.status_message = "Operation cancelled · live geometry preserved".into();
            IpcResponse::success("Operation cancelled", serde_json::json!({}))
        } else {
            IpcResponse::error("No operation is running")
        }
    }

    fn poll_operation(&mut self) {
        let result = self
            .background
            .as_ref()
            .and_then(|op| op.receiver.try_recv().ok());
        let Some(mut result) = result else {
            return;
        };
        let operation = self.background.take().unwrap();
        self.ui_state.busy_operation = None;
        let scene_changed = result.response.ok && !ipc::is_read_only(&operation.tool);
        if scene_changed {
            // Keep feedback and inspection changes made while geometry was computed.
            let current_id = self.scene.active_object().map(|o| o.id);
            let selected_id = if current_id == operation.selected_id {
                result.scene.active_object().map(|o| o.id)
            } else {
                current_id
            };
            for object in &mut result.scene.objects {
                if let Some(current) = self.scene.objects.iter().find(|o| o.id == object.id) {
                    object.visible = current.visible;
                    object.locked = current.locked;
                }
            }
            self.scene.objects = result.scene.objects;
            self.scene.next_id = self.scene.next_id.max(result.scene.next_id);
            self.scene.selected_object =
                selected_id.and_then(|id| self.scene.objects.iter().position(|o| o.id == id));
            self.history = result.history;
            self.ui_state.hover_preview_faces.clear();
            self.ui_state.hover_preview_obj_id = None;
            self.ui_state.hover_preview_seed_face = None;
            self.ui_state.target_sections = None;
            ui::update_pending_anchor_from_selection(&mut self.scene);
            self.saver.request();
        }
        self.ui_state.activity.push(format!(
            "{} · {} · {:.2}s",
            operation.tool,
            if result.response.ok { "Done" } else { "Failed" },
            operation.started.elapsed().as_secs_f32()
        ));
        if self.ui_state.activity.len() > 24 {
            self.ui_state.activity.remove(0);
        }
        self.ui_state.status_message = result.response.message.clone();
        if !result.response.ok {
            self.ui_state.last_error = Some(result.response.message.clone());
        }
        if scene_changed {
            self.scene.invalidate_mesh_caches();
        }
        if let Some(gpu) = &mut self.gpu {
            if scene_changed {
                gpu.renderer_3d.invalidate_cache();
            }
            gpu.window.request_redraw();
        }
        let _ = operation.reply.send(result.response);
    }

    fn viewport_size(&self) -> Vec2 {
        let rect = self.ui_state.viewport_rect;
        if rect.is_positive() {
            let scale = self
                .gpu
                .as_ref()
                .map(|g| g.egui_ctx.pixels_per_point())
                .unwrap_or(1.0);
            return Vec2::new(rect.width() * scale, rect.height() * scale);
        }
        Vec2::new(1280.0, 720.0)
    }

    fn viewport_cursor(&self) -> Vec2 {
        let scale = self
            .gpu
            .as_ref()
            .map(|g| g.egui_ctx.pixels_per_point())
            .unwrap_or(1.0);
        let rect = self.ui_state.viewport_rect;
        if rect.is_positive() {
            self.cursor_pos - Vec2::new(rect.left() * scale, rect.top() * scale)
        } else {
            self.cursor_pos
        }
    }

    /// Schedule a background save (see `autosave`); it never blocks input or rendering.
    fn persist(&mut self) {
        self.ui_state.scene_dirty_for_disk = false;
        self.saver.request();
    }

    /// Write everything before the window closes.
    fn flush_saves(&mut self) {
        if self.ui_state.scene_dirty_for_disk || self.ui_state.volume_drag_changed {
            self.saver.request();
        }
        if let Err(e) = self.saver.flush(&mut self.scene, &self.camera) {
            log::error!("Final save failed: {e}");
            eprintln!("Final save failed: {e}");
        }
    }

    fn handle_viewport_left_press(&mut self) {
        if self.scene.preview_revision.is_some() {
            return;
        }
        let vp_size = self.viewport_size();
        let ray = self.camera.screen_ray(self.viewport_cursor(), vp_size);

        if let Some(hit) = self.scene.raycast(&ray) {
            if self.ui_state.measure_active {
                if self.ui_state.measure_points.len() >= 3 {
                    self.ui_state.measure_points.clear();
                }
                self.ui_state.measure_points.push(hit.world_point);
                self.ui_state.status_message = "Measurement point placed".into();
                return;
            }
            if self.ui_state.select_mode == SmartSelectMode::Box {
                self.box_drag_start = Some(self.cursor_pos);
                return;
            }
            if self.ui_state.select_mode == SmartSelectMode::Volume {
                self.drop_selection_volume(&hit);
                return;
            }
            let multi_active = self.ui_state.multi_select
                || self.modifiers.shift_key()
                || self.modifiers.control_key();

            if !multi_active {
                self.scene.clear_all_selections();
                self.scene.selection_volume = None;
            }

            self.scene.selected_object = Some(hit.object_idx);

            let already_selected = self.scene.objects.get(hit.object_idx).map_or(false, |o| {
                o.mesh.selected_faces.contains(&hit.local_hit.face_idx)
            });

            let subtract = if self.modifiers.control_key() {
                true
            } else if self.ui_state.multi_select
                && !self.modifiers.shift_key()
                && already_selected
            {
                // In Multi-Select mode, clicking an already-selected sub-part or surface toggles it off!
                true
            } else {
                false
            };

            let additive = multi_active && !subtract;

            self.apply_section_selection_at_hit(&hit, additive, subtract);
            self.persist();
        } else if self.ui_state.select_mode == SmartSelectMode::Box
            && !self.ui_state.measure_active
        {
            self.box_drag_start = Some(self.cursor_pos);
        } else if self.ui_state.select_mode == SmartSelectMode::Volume {
            // Clicking empty space keeps the box; drag its handles or click a surface to move it.
        } else if !self.modifiers.shift_key()
            && !self.modifiers.control_key()
            && !self.ui_state.multi_select
        {
            self.scene.clear_all_selections();
            self.scene.selection_volume = None;
            self.scene.selected_object = None;
            self.persist();
        }
    }

    /// Place the 3D box selector on the clicked surface (keeping its size if it already exists)
    /// and select everything inside it.
    fn drop_selection_volume(&mut self, hit: &crate::scene::WorldRayHit) {
        let (half, label) = match &self.scene.selection_volume {
            Some(v) => (v.half_extents, v.label.clone()),
            None => {
                let size = self
                    .scene
                    .objects
                    .get(hit.object_idx)
                    .map(|obj| {
                        let (lo, hi) = obj.mesh.bounds();
                        ((hi - lo) * obj.transform.scale.abs()).max_element()
                    })
                    .unwrap_or(1.0);
                (Vec3::splat((size * 0.15).clamp(0.02, 50.0)), String::new())
            }
        };
        let mut volume = crate::scene::SelectionVolume::new(hit.world_point, half);
        volume.label = label;
        self.scene.selection_volume = Some(volume);
        let (faces, parts) = self.scene.apply_volume_selection();
        update_pending_anchor_from_selection(&mut self.scene);
        self.ui_state.status_message = format!(
            "[3D box] {faces} faces on {parts} part(s) inside · drag the handles to resize, the centre to move"
        );
        self.persist();
    }

    fn apply_section_selection_at_hit(
        &mut self,
        hit: &crate::scene::WorldRayHit,
        additive: bool,
        subtract: bool,
    ) {
        let select_mode = self.ui_state.select_mode;
        let crease_deg = self.ui_state.crease_angle_deg;
        let face_path = self.ui_state.face_path;
        let last_pick = self.ui_state.last_face_pick;
        let mut new_pick = last_pick;
        self.hover_brush_world = Some((hit.world_point, hit.world_normal));

        if let Some(obj) = self.scene.objects.get_mut(hit.object_idx) {
            let seed_face = hit.local_hit.face_idx;

            match select_mode {
                SmartSelectMode::SubPart => {
                    obj.mesh
                        .select_connected_component(seed_face, additive, subtract);
                }
                SmartSelectMode::SmartSurface => {
                    obj.mesh
                        .select_smart_surface(seed_face, crease_deg, additive, subtract);
                }
                SmartSelectMode::WholeLayer => {
                    if subtract {
                        obj.mesh.clear_selection();
                    } else {
                        obj.mesh.select_all(EditElementMode::Face);
                    }
                }
                SmartSelectMode::Box | SmartSelectMode::Volume => {}
                SmartSelectMode::Face => {
                    // With Path on, a click fills the shortest run of faces from the last pick.
                    let from = last_pick.filter(|(id, f)| {
                        face_path
                            && !subtract
                            && *id == obj.id
                            && *f != seed_face
                            && obj.mesh.selected_faces.contains(f)
                    });
                    let set: std::collections::HashSet<usize> =
                        match from.and_then(|(_, f)| obj.mesh.face_path(f, seed_face)) {
                            Some(path) => path.into_iter().collect(),
                            None => std::iter::once(seed_face).collect(),
                        };
                    let path_fill = from.is_some() && set.len() > 1;
                    obj.mesh.apply_face_set_selection(
                        &set,
                        additive || path_fill,
                        subtract && !path_fill,
                    );
                    new_pick = (!subtract || path_fill).then_some((obj.id, seed_face));
                }
            }

            let obj_id = obj.id;
            let obj_name = obj.name.clone();
            let sel_count = obj.mesh.selected_faces.len();
            let total_faces = obj.mesh.faces.len();

            if let Some((lc, ln, _min_b, _max_b)) = obj.mesh.selected_faces_bounds_and_normal() {
                let model = obj.transform.matrix();
                let wc = model.transform_point3(lc);
                let wn = model
                    .inverse()
                    .transpose()
                    .transform_vector3(ln)
                    .normalize_or_zero();
                self.scene.pending_anchor = Some((obj_id, wc, lc, wn));
            } else {
                self.scene.pending_anchor = Some((
                    obj_id,
                    hit.world_point,
                    hit.local_hit.point,
                    hit.world_normal,
                ));
            }

            self.ui_state.last_face_pick = new_pick;
            let mode_label = select_mode.label();
            self.ui_state.status_message = format!(
                "[{mode_label}] Selected {sel_count} / {total_faces} faces on '{obj_name}' (Shift+Click: Add, Ctrl+Click: Subtract, [ / ]: Shrink/Grow)"
            );
        }
    }

    fn update_hover_preview(&mut self, hit: Option<&crate::scene::WorldRayHit>) {
        let Some(hit) = hit else {
            self.ui_state.hover_preview_obj_id = None;
            self.ui_state.hover_preview_seed_face = None;
            self.ui_state.hover_preview_faces.clear();
            return;
        };

        let Some(obj) = self.scene.objects.get(hit.object_idx) else {
            return;
        };
        let obj_id = obj.id;
        let seed_face = hit.local_hit.face_idx;
        let select_mode = self.ui_state.select_mode;

        if select_mode.is_drag() {
            self.ui_state.hover_preview_obj_id = None;
            self.ui_state.hover_preview_seed_face = None;
            self.ui_state.hover_preview_faces.clear();
            return;
        }

        // Cache hit: avoid recomputing if still hovering the same face of the same object
        if self.ui_state.hover_preview_obj_id == Some(obj_id)
            && self.ui_state.hover_preview_seed_face == Some(seed_face)
        {
            return;
        }

        self.ui_state.hover_preview_obj_id = Some(obj_id);
        self.ui_state.hover_preview_seed_face = Some(seed_face);
        self.ui_state.hover_preview_faces = match select_mode {
            SmartSelectMode::SubPart => obj.mesh.compute_connected_component(seed_face),
            SmartSelectMode::SmartSurface => obj
                .mesh
                .compute_smart_surface(seed_face, self.ui_state.crease_angle_deg),
            SmartSelectMode::WholeLayer => (0..obj.mesh.faces.len()).collect(),
            SmartSelectMode::Face => std::iter::once(seed_face).collect(),
            SmartSelectMode::Box | SmartSelectMode::Volume => std::collections::HashSet::new(),
        };
    }

    fn handle_cursor_moved(&mut self, new_pos: Vec2) {
        let delta = new_pos - self.last_cursor_pos;
        self.last_cursor_pos = new_pos;
        self.cursor_pos = new_pos;

        // 1. Camera Orbit / Pan
        let alt = self.modifiers.alt_key();
        let shift = self.modifiers.shift_key();

        if self.middle_mouse_down || (alt && self.left_mouse_down) {
            if shift {
                self.camera.pan(delta.x, delta.y);
            } else {
                self.camera.orbit(delta.x, delta.y);
            }
            return;
        }
        if self.right_mouse_down {
            self.camera.pan(delta.x, delta.y);
            return;
        }

        if let Some(start) = self.box_drag_start {
            let scale = self
                .gpu
                .as_ref()
                .map(|g| g.egui_ctx.pixels_per_point())
                .unwrap_or(1.0);
            self.ui_state.box_rect = Some(egui::Rect::from_two_pos(
                egui::pos2(start.x / scale, start.y / scale),
                egui::pos2(new_pos.x / scale, new_pos.y / scale),
            ));
            self.hover_brush_world = None;
            self.update_hover_preview(None);
            return;
        }

        // Raycasting every event (hundreds per second) backs up input; do it once per frame.
        self.pointer_dirty = true;
    }

    /// Per-frame pointer work: hover preview and fly-mode look.
    fn process_pointer(&mut self) {
        if self.ui_state.fly_active {
            let look = std::mem::take(&mut self.fly_look_delta);
            if look != Vec2::ZERO {
                self.camera.fly_look(look.x, look.y);
                self.pointer_dirty = true;
            }
            if std::mem::take(&mut self.pointer_dirty) {
                self.update_fly_hover();
            }
            return;
        }
        self.fly_look_delta = Vec2::ZERO;
        if !std::mem::take(&mut self.pointer_dirty) || self.box_drag_start.is_some() {
            return;
        }
        let vp_size = self.viewport_size();
        let ray = self.camera.screen_ray(self.viewport_cursor(), vp_size);

        if self.scene.preview_revision.is_some() || self.ui_state.measure_active {
            self.hover_brush_world = None;
            self.update_hover_preview(None);
            return;
        }

        if let Some(hit) = self.scene.raycast(&ray) {
            self.hover_brush_world = Some((hit.world_point, hit.world_normal));
            self.update_hover_preview(Some(&hit));
        } else {
            self.hover_brush_world = None;
            self.update_hover_preview(None);
        }
    }

    fn handle_key_press(&mut self, code: KeyCode) {
        if code == KeyCode::F1 || (code == KeyCode::Slash && self.modifiers.shift_key()) {
            self.ui_state.show_shortcuts = !self.ui_state.show_shortcuts;
            return;
        }
        // Escape closes the open dialog/overlay (handled by the UI) instead of clearing the selection.
        if code == KeyCode::Escape && (self.ui_state.modal_open() || self.ui_state.show_shortcuts) {
            return;
        }
        if code == KeyCode::Backquote && !self.modifiers.control_key() {
            self.set_fly(true);
            return;
        }
        if code == KeyCode::Escape && self.box_drag_start.take().is_some() {
            self.ui_state.box_rect = None;
            self.ui_state.status_message = "Selection drag cancelled".into();
            return;
        }
        if self.scene.preview_revision.is_some() && code != KeyCode::KeyF {
            return;
        }
        if code == KeyCode::KeyC && self.modifiers.control_key() && !self.modifiers.shift_key() {
            self.toggle_box_select();
            return;
        }
        if self.modifiers.control_key() || self.modifiers.super_key() {
            return;
        }
        match code {
            KeyCode::Digit1 => {
                self.ui_state.select_mode = SmartSelectMode::SubPart;
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::Digit2 => {
                self.ui_state.select_mode = SmartSelectMode::SmartSurface;
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::Digit3 => {
                self.ui_state.select_mode = SmartSelectMode::WholeLayer;
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::Digit4 => {
                if self.ui_state.select_mode != SmartSelectMode::Box {
                    self.toggle_box_select();
                }
            }
            KeyCode::Digit5 => {
                self.ui_state.select_mode = SmartSelectMode::Face;
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::Digit6 => {
                self.ui_state.select_mode = SmartSelectMode::Volume;
                self.update_hover_preview(None);
                self.ui_state.status_message =
                    "3D box · click a surface to place the box, then drag its handles".into();
            }
            KeyCode::KeyI => {
                let mut any = false;
                for obj in &mut self.scene.objects {
                    if !obj.mesh.selected_faces.is_empty() {
                        obj.mesh.invert_selection();
                        any = true;
                    }
                }
                if any {
                    self.scene.selection_volume = None;
                    self.scene.region_labels.clear();
                    update_pending_anchor_from_selection(&mut self.scene);
                    self.ui_state.status_message = "Selection inverted".into();
                    self.persist();
                }
            }
            KeyCode::Tab => {
                self.ui_state.select_mode = match self.ui_state.select_mode {
                    SmartSelectMode::SubPart => SmartSelectMode::SmartSurface,
                    SmartSelectMode::SmartSurface => SmartSelectMode::WholeLayer,
                    SmartSelectMode::WholeLayer => SmartSelectMode::Box,
                    SmartSelectMode::Box => SmartSelectMode::Face,
                    SmartSelectMode::Face => SmartSelectMode::Volume,
                    SmartSelectMode::Volume => SmartSelectMode::SubPart,
                };
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::KeyF => {
                ui::frame_selection(&self.scene, &mut self.camera);
            }
            KeyCode::KeyA => {
                if let Some(obj) = self.scene.active_object_mut() {
                    obj.mesh.select_all(EditElementMode::Face);
                }
                update_pending_anchor_from_selection(&mut self.scene);
                self.persist();
            }
            KeyCode::KeyC | KeyCode::Escape => {
                self.scene.clear_all_selections();
                self.scene.selection_volume = None;
                self.ui_state.last_face_pick = None;
                self.scene.selected_object = None;
                self.scene.active_comment_id = None;
                self.persist();
            }
            KeyCode::BracketLeft => {
                let mut shrunk = false;
                for obj in &mut self.scene.objects {
                    if !obj.mesh.selected_faces.is_empty() {
                        shrunk |= obj.mesh.shrink_selection();
                    }
                }
                if shrunk {
                    update_pending_anchor_from_selection(&mut self.scene);
                    self.persist();
                }
            }
            KeyCode::BracketRight => {
                let mut grown = false;
                for obj in &mut self.scene.objects {
                    if !obj.mesh.selected_faces.is_empty() {
                        grown |= obj.mesh.grow_selection();
                    }
                }
                if grown {
                    update_pending_anchor_from_selection(&mut self.scene);
                    self.persist();
                }
            }
            _ => {}
        }
    }
}

impl ModelerApp {
    /// Ctrl+C: switch to the box selector, or back to the previous mode.
    fn toggle_box_select(&mut self) {
        if self.ui_state.select_mode == SmartSelectMode::Box {
            self.ui_state.select_mode = self.ui_state.prev_select_mode;
        } else {
            self.ui_state.prev_select_mode = self.ui_state.select_mode;
            self.ui_state.select_mode = SmartSelectMode::Box;
            self.ui_state.measure_active = false;
        }
        self.ui_state.hover_preview_seed_face = None;
        self.update_hover_preview(None);
        self.ui_state.status_message = if self.ui_state.select_mode == SmartSelectMode::Box {
            "Box select · drag over each place to change (Shift adds, Ctrl removes) · Ctrl+C to leave".into()
        } else {
            format!("{} select", self.ui_state.select_mode.label())
        };
    }

    fn finish_box_select(&mut self) {
        let Some(start) = self.box_drag_start.take() else {
            return;
        };
        self.ui_state.box_rect = None;
        let scale = self
            .gpu
            .as_ref()
            .map(|g| g.egui_ctx.pixels_per_point())
            .unwrap_or(1.0);
        let rect = self.ui_state.viewport_rect;
        let origin = if rect.is_positive() {
            Vec2::new(rect.left() * scale, rect.top() * scale)
        } else {
            Vec2::ZERO
        };
        let op = if self.modifiers.control_key() {
            box_select::BoxOp::Subtract
        } else if self.modifiers.shift_key() || self.ui_state.multi_select {
            box_select::BoxOp::Add
        } else {
            box_select::BoxOp::Replace
        };
        let viewport = self.viewport_size();
        let tool = "Box";
        let outcome = box_select::apply_box_selection(
            &mut self.scene,
            &self.camera,
            viewport,
            start - origin,
            self.cursor_pos - origin,
            self.ui_state.box_through,
            op,
        );
        if outcome.selected_faces > 0 {
            update_pending_anchor_from_selection(&mut self.scene);
        }
        let found: usize = outcome.hits.values().map(|f| f.len()).sum();
        self.ui_state.status_message = if outcome.selected_faces == 0 {
            format!("[{tool}] Nothing selected")
        } else {
            format!(
                "[{tool}] {} {found} faces · selection: {} faces on {} {} in {} {} (each region becomes its own AI target)",
                if op == box_select::BoxOp::Subtract {
                    "Removed"
                } else {
                    "Picked"
                },
                outcome.selected_faces,
                outcome.objects,
                if outcome.objects == 1 {
                    "part"
                } else {
                    "parts"
                },
                outcome.regions,
                if outcome.regions == 1 {
                    "region"
                } else {
                    "regions"
                },
            )
        };
        self.persist();
    }

    /// Window-pixel centre of the 3D viewport (the fly-mode crosshair).
    fn viewport_center_px(&self) -> Vec2 {
        let scale = self
            .gpu
            .as_ref()
            .map(|g| g.egui_ctx.pixels_per_point())
            .unwrap_or(1.0);
        let rect = self.ui_state.viewport_rect;
        if !rect.is_positive() {
            return self.viewport_size() * 0.5;
        }
        let c = rect.center();
        Vec2::new(c.x * scale, c.y * scale)
    }

    fn set_fly(&mut self, on: bool) {
        if on == self.ui_state.fly_active {
            return;
        }
        if on && self.ui_state.modal_open() {
            return;
        }
        self.fly.reset();
        self.box_drag_start = None;
        self.ui_state.box_rect = None;
        self.left_mouse_down = false;
        self.middle_mouse_down = false;
        self.right_mouse_down = false;
        self.ui_state.fly_active = on;
        if on {
            self.camera.orthographic = false;
            if !self.fly_speed_set {
                self.ui_state.fly_speed = navigation::default_speed_for(&self.camera);
            }
            self.ui_state.show_shortcuts = false;
            self.ui_state.status_message =
                "Fly mode · WASD move · mouse look · click selects at the crosshair · Esc exits"
                    .into();
        } else {
            // Orbit around whatever is under the crosshair after flying.
            let ray = self
                .camera
                .screen_ray(self.viewport_size() * 0.5, self.viewport_size());
            if let Some(hit) = self.scene.raycast(&ray) {
                self.camera.refocus(hit.world_distance);
            }
            self.ui_state.status_message = "Left fly mode".into();
            self.persist();
        }
        let center = self.viewport_center_px();
        if let Some(gpu) = &self.gpu {
            let window = &gpu.window;
            if on {
                if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                }
                window.set_cursor_visible(false);
                let _ = window.set_cursor_position(winit::dpi::PhysicalPosition::new(
                    center.x as f64,
                    center.y as f64,
                ));
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
            window.request_redraw();
        }
        if on {
            self.cursor_pos = center;
            self.last_cursor_pos = center;
            self.hover_brush_world = None;
            self.update_hover_preview(None);
        }
    }

    /// Point the hover preview at whatever is under the crosshair.
    fn update_fly_hover(&mut self) {
        let center = self.viewport_center_px();
        self.cursor_pos = center;
        self.last_cursor_pos = center;
        let ray = self
            .camera
            .screen_ray(self.viewport_cursor(), self.viewport_size());
        if self.scene.preview_revision.is_some() || self.ui_state.measure_active {
            self.hover_brush_world = None;
            self.update_hover_preview(None);
        } else if let Some(hit) = self.scene.raycast(&ray) {
            self.hover_brush_world = Some((hit.world_point, hit.world_normal));
            self.update_hover_preview(Some(&hit));
        } else {
            self.hover_brush_world = None;
            self.update_hover_preview(None);
        }
    }

    /// Input while flying. Returns true when the event was consumed.
    fn fly_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return true;
                };
                let pressed = event.state == ElementState::Pressed;
                if pressed && matches!(code, KeyCode::Escape | KeyCode::Backquote) {
                    self.set_fly(false);
                } else if pressed && code == KeyCode::KeyF && !event.repeat {
                    ui::frame_selection(&self.scene, &mut self.camera);
                    self.pointer_dirty = true;
                } else if pressed && code == KeyCode::F1 {
                    self.set_fly(false);
                    self.ui_state.show_shortcuts = true;
                } else if !event.repeat {
                    self.fly.key(code, pressed);
                }
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left if pressed => {
                        self.process_pointer();
                        self.update_fly_hover();
                        self.handle_viewport_left_press();
                        // A box press in fly mode selects the face under the crosshair.
                        self.finish_box_select();
                    }
                    MouseButton::Right if pressed => self.set_fly(false),
                    _ => {}
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if lines.is_finite() && lines != 0.0 {
                    self.ui_state.fly_speed =
                        navigation::adjust_speed(self.ui_state.fly_speed, lines);
                    self.fly_speed_set = true;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Confined (X11) grabs still move the pointer: keep it parked at the centre.
                let center = self.viewport_center_px();
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if pos.distance(center) > 80.0 {
                    if let Some(gpu) = &self.gpu {
                        let _ = gpu
                            .window
                            .set_cursor_position(winit::dpi::PhysicalPosition::new(
                                center.x as f64,
                                center.y as f64,
                            ));
                    }
                }
                true
            }
            WindowEvent::Focused(false) => {
                self.set_fly(false);
                false
            }
            _ => false,
        }
    }
}

impl ApplicationHandler<IpcEvent> for ModelerApp {
    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.ui_state.fly_active {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.fly_look_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: IpcEvent) {
        if event.request.tool == "quit_gui" {
            let _ = std::fs::remove_file(IPC_SOCKET_PATH);
            self.flush_saves();
            let _ = event.reply_tx.send(IpcResponse::success(
                "Closed GUI window",
                serde_json::json!({}),
            ));
            event_loop.exit();
            return;
        }
        if event.request.tool == "get_feedback_state" {
            // An agent is listening for feedback; not an edit worth logging.
            self.ui_state.agent_listening = Some(std::time::Instant::now());
        } else {
            self.ui_state.last_agent_activity = Some(std::time::Instant::now());
        }
        if event.request.tool == "cancel_operation" {
            let response = self.cancel_operation();
            let _ = event.reply_tx.send(response);
            return;
        }
        if self.background.is_some() && !ipc::allowed_during_operation(&event.request.tool) {
            let _ = event.reply_tx.send(IpcResponse::error(
                "Wait for the current operation to finish",
            ));
            return;
        }
        if (matches!(
            event.request.tool.as_str(),
            "sculpt_stroke"
                | "voxel_brush"
                | "voxel_remesh"
                | "set_material"
                | "edit_mesh"
                | "add_primitive"
                | "import_model"
                | "boolean"
                | "add_lathe"
                | "add_sweep"
                | "array_object"
                | "mirror_object"
                | "capture_viewport"
                | "capture_studio"
        ) || ipc::is_export(&event.request.tool)
            || agent_tools::HEAVY.contains(&event.request.tool.as_str())
            || library::HEAVY.contains(&event.request.tool.as_str()))
            && self.background.is_none()
        {
            self.background = Some(BackgroundOperation::start(
                event,
                &self.scene,
                &self.camera,
                &self.history,
                &mut self.ui_state,
                self.gpu.as_ref(),
            ));
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
            return;
        }
        let mut shading = self
            .gpu
            .as_ref()
            .map(|g| g.renderer_3d.shading)
            .unwrap_or(ViewportShading::StudioLit);
        let mut show_wire = self
            .gpu
            .as_ref()
            .map(|g| g.renderer_3d.show_wireframe_overlay)
            .unwrap_or(false);
        let mut show_grid = self
            .gpu
            .as_ref()
            .map(|g| g.renderer_3d.show_grid)
            .unwrap_or(true);

        let resp = execute_ipc_command(
            &event.request,
            &mut self.scene,
            &mut self.camera,
            &mut self.history,
            &mut self.ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );

        if resp.ok && !ipc::is_read_only(&event.request.tool) {
            self.persist();
        }

        if let Some(gpu) = &mut self.gpu {
            if !ipc::is_read_only(&event.request.tool) {
                self.scene.invalidate_mesh_caches();
                gpu.renderer_3d.invalidate_cache();
            }
            gpu.renderer_3d.shading = shading;
            gpu.renderer_3d.show_wireframe_overlay = show_wire;
            gpu.renderer_3d.show_grid = show_grid;
            gpu.window.request_redraw();
        }

        let _ = event.reply_tx.send(resp);
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let win_attrs = Window::default_attributes()
            .with_title("Spatial Studio · AI 3D Review")
            .with_inner_size(PhysicalSize::new(1440, 960))
            .with_min_inner_size(PhysicalSize::new(1100, 700));

        let window = Arc::new(
            event_loop
                .create_window(win_attrs)
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create wgpu surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create wgpu device");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        self.camera.aspect = (width as f32) / (height as f32);

        let egui_ctx = egui::Context::default();
        ui::apply_theme(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1, false);
        let renderer_3d = Renderer3D::new(&device, surface_format, width, height);

        self.gpu = Some(GpuContext {
            window,
            device,
            queue,
            surface,
            surface_config,
            pending_resize: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            renderer_3d,
        });
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        use winit::event_loop::ControlFlow;
        self.poll_operation();
        if self.ui_state.fly_toggle_requested {
            self.ui_state.fly_toggle_requested = false;
            self.set_fly(!self.ui_state.fly_active);
        }
        if self.ui_state.scene_dirty_for_disk {
            self.persist();
        }
        if let Some(error) = self
            .saver
            .poll(&mut self.scene, &self.camera)
            .into_iter()
            .last()
        {
            self.ui_state.last_error = Some(error);
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
        }
        let save_wake = self.saver.next_wake();
        let wait_until = |event_loop: &ActiveEventLoop, at: std::time::Instant| {
            let at = save_wake.map_or(at, |s| s.min(at));
            event_loop.set_control_flow(ControlFlow::WaitUntil(at));
        };
        if self.ui_state.fly_active && self.fly.moving() {
            let (fast, slow) = (self.modifiers.shift_key(), self.modifiers.control_key());
            if self
                .fly
                .tick(&mut self.camera, self.ui_state.fly_speed, fast, slow)
            {
                self.pointer_dirty = true;
            }
            wait_until(
                event_loop,
                std::time::Instant::now() + std::time::Duration::from_millis(8),
            );
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
            return;
        }
        if self.background.is_some() {
            wait_until(
                event_loop,
                std::time::Instant::now() + std::time::Duration::from_millis(33),
            );
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
        } else if let Some(at) = save_wake {
            event_loop.set_control_flow(ControlFlow::WaitUntil(at));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.gpu.is_none() {
            return;
        }
        // While flying the hidden cursor must not click or hover egui widgets.
        if self.ui_state.fly_active && self.fly_window_event(&event) {
            if let Some(gpu) = &self.gpu {
                gpu.window.request_redraw();
            }
            return;
        }
        let Some(gpu) = &mut self.gpu else {
            return;
        };

        let egui_resp = gpu.egui_state.on_window_event(&gpu.window, &event);
        if egui_resp.repaint {
            gpu.window.request_redraw();
        }
        let pointer_pos = match &event {
            WindowEvent::CursorMoved { position, .. } => {
                Vec2::new(position.x as f32, position.y as f32)
            }
            _ => self.cursor_pos,
        };
        let dialog_open = self.ui_state.modal_open();
        let viewport_accepts_pointer = viewport_input::accepts_pointer(
            &gpu.egui_ctx,
            self.ui_state.viewport_rect,
            pointer_pos,
            dialog_open,
        );
        let egui_wants_keyboard = gpu.egui_ctx.wants_keyboard_input();

        match event {
            WindowEvent::CloseRequested => {
                let _ = std::fs::remove_file(IPC_SOCKET_PATH);
                self.flush_saves();
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    gpu.pending_resize = Some(new_size);
                    self.camera.aspect = (new_size.width as f32) / (new_size.height as f32);
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(new_mods) => {
                self.modifiers = new_mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !egui_wants_keyboard && event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        self.handle_key_press(code);
                        if let Some(gpu) = &self.gpu {
                            gpu.window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.left_mouse_down = pressed && viewport_accepts_pointer;
                        if pressed && viewport_accepts_pointer && !self.modifiers.alt_key() {
                            self.handle_viewport_left_press();
                        } else if !pressed {
                            self.finish_box_select();
                            self.gizmo.end_drag();
                            self.last_sculpt_local_hit = None;
                        }
                    }
                    MouseButton::Middle => {
                        self.middle_mouse_down = pressed && viewport_accepts_pointer;
                    }
                    MouseButton::Right => {
                        self.right_mouse_down = pressed && viewport_accepts_pointer;
                    }
                    _ => {}
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                let camera_drag = self.middle_mouse_down
                    || self.right_mouse_down
                    || (self.modifiers.alt_key() && self.left_mouse_down);
                if viewport_accepts_pointer
                    || self.gizmo.dragging
                    || camera_drag
                    || self.box_drag_start.is_some()
                {
                    self.handle_cursor_moved(pos);
                } else {
                    self.last_cursor_pos = pos;
                    self.cursor_pos = pos;
                    self.pointer_dirty = false;
                    self.hover_brush_world = None;
                    self.update_hover_preview(None);
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if viewport_input::scroll_camera(
                    &gpu.egui_ctx,
                    &mut self.camera,
                    self.ui_state.viewport_rect,
                    self.cursor_pos,
                    delta,
                    dialog_open,
                ) {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                self.process_pointer();
                let Some(gpu) = &mut self.gpu else {
                    return;
                };

                if let Some(new_size) = gpu.pending_resize.take() {
                    if new_size.width > 0
                        && new_size.height > 0
                        && (gpu.surface_config.width != new_size.width
                            || gpu.surface_config.height != new_size.height)
                    {
                        gpu.surface_config.width = new_size.width;
                        gpu.surface_config.height = new_size.height;
                        gpu.surface.configure(&gpu.device, &gpu.surface_config);
                        gpu.renderer_3d
                            .resize(&gpu.device, new_size.width, new_size.height);
                    }
                }

                let frame = match gpu.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = gpu.window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            gpu.surface_config.width = size.width;
                            gpu.surface_config.height = size.height;
                            gpu.surface.configure(&gpu.device, &gpu.surface_config);
                            gpu.renderer_3d.resize(&gpu.device, size.width, size.height);
                        }
                        return;
                    }
                    Err(wgpu::SurfaceError::Timeout) => {
                        return;
                    }
                    Err(e) => {
                        log::error!("Surface error: {e:?}");
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let raw_input = gpu.egui_state.take_egui_input(&gpu.window);
                let hover_pt = self.hover_brush_world.map(|(p, _)| p);

                let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
                    draw_ui(
                        ctx,
                        &mut self.scene,
                        &mut self.camera,
                        &mut self.gizmo,
                        &mut gpu.renderer_3d,
                        &mut self.history,
                        &mut self.ui_state,
                        hover_pt,
                    );
                });

                for request in std::mem::take(&mut self.ui_state.actions) {
                    if request.tool == "cancel_operation" {
                        if self.background.as_ref().is_some_and(|op| !ipc::is_export(&op.tool)) {
                            let operation = self.background.take().unwrap();
                            let _ = operation.reply.send(IpcResponse::error("Operation cancelled; result discarded"));
                            self.ui_state.busy_operation = None;
                            self.ui_state.status_message = "Operation cancelled · live geometry preserved".into();
                        }
                        continue;
                    }
                    if self.background.is_some() && !ipc::allowed_during_operation(&request.tool) {
                        self.ui_state.last_error =
                            Some("Wait for the current operation to finish".into());
                        continue;
                    }
                    if ipc::is_export(&request.tool) {
                        let (reply_tx, _) = mpsc::channel();
                        self.background = Some(BackgroundOperation::start(
                            IpcEvent { request, reply_tx },
                            &self.scene,
                            &self.camera,
                            &self.history,
                            &mut self.ui_state,
                            Some(gpu),
                        ));
                        continue;
                    }
                    let response = execute_ipc_command(
                        &request,
                        &mut self.scene,
                        &mut self.camera,
                        &mut self.history,
                        &mut self.ui_state,
                        &mut gpu.renderer_3d.shading,
                        &mut gpu.renderer_3d.show_wireframe_overlay,
                        &mut gpu.renderer_3d.show_grid,
                    );
                    ui::after_action(&mut self.ui_state, &request, &response, &mut self.camera);
                    self.ui_state.status_message = response.message;
                    self.scene.invalidate_mesh_caches();
                    gpu.renderer_3d.invalidate_cache();
                }
                let viewport = self.ui_state.viewport_rect;
                let scale = full_output.pixels_per_point;
                if viewport.is_positive() {
                    self.camera.aspect = viewport.width() / viewport.height().max(1.0);
                    gpu.renderer_3d.viewport = Some([
                        viewport.left() * scale,
                        viewport.top() * scale,
                        viewport.width() * scale,
                        viewport.height() * scale,
                    ]);
                }

                gpu.egui_state
                    .handle_platform_output(&gpu.window, full_output.platform_output);

                let tris = gpu
                    .egui_ctx
                    .tessellate(full_output.shapes, full_output.pixels_per_point);

                for (id, image_delta) in &full_output.textures_delta.set {
                    gpu.egui_renderer
                        .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
                }

                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [gpu.surface_config.width, gpu.surface_config.height],
                    pixels_per_point: full_output.pixels_per_point,
                };

                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Main Frame Encoder"),
                        });

                let brush_cursor = self.hover_brush_world.map(|(p, n)| (p, n, 0.12));

                let hover_preview = self
                    .ui_state
                    .hover_preview_obj_id
                    .map(|oid| (oid, &self.ui_state.hover_preview_faces));

                gpu.renderer_3d.render_scene(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &view,
                    &self.camera,
                    &self.scene,
                    &self.gizmo,
                    brush_cursor,
                    hover_preview,
                );

                gpu.egui_renderer.update_buffers(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &tris,
                    &screen_descriptor,
                );

                {
                    let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Egui Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let mut rpass_static = rpass.forget_lifetime();
                    gpu.egui_renderer
                        .render(&mut rpass_static, &tris, &screen_descriptor);
                }

                for id in &full_output.textures_delta.free {
                    gpu.egui_renderer.free_texture(id);
                }

                gpu.queue.submit(std::iter::once(encoder.finish()));
                gpu.window.pre_present_notify();
                frame.present();
            }
            _ => {}
        }
    }
}

fn spawn_ipc_listener_thread(proxy: EventLoopProxy<IpcEvent>) {
    std::thread::spawn(move || {
        let _ = std::fs::remove_file(IPC_SOCKET_PATH);
        let listener = match UnixListener::bind(IPC_SOCKET_PATH) {
            Ok(l) => l,
            Err(e) => {
                log::warn!("Failed to bind IPC socket at {IPC_SOCKET_PATH}: {e}");
                return;
            }
        };

        for stream in listener.incoming().flatten() {
            let proxy = proxy.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(match stream.try_clone() {
                    Ok(s) => s,
                    Err(_) => return,
                });
                let mut line = String::new();
                if reader.read_line(&mut line).is_ok() && !line.trim().is_empty() {
                    if let Ok(req) = serde_json::from_str::<IpcRequest>(&line) {
                        let (tx, rx) = mpsc::channel();
                        if proxy
                            .send_event(IpcEvent {
                                request: req,
                                reply_tx: tx,
                            })
                            .is_ok()
                        {
                            if let Ok(resp) = rx.recv_timeout(std::time::Duration::from_secs(120)) {
                                if let Ok(mut out_bytes) = serde_json::to_vec(&resp) {
                                    out_bytes.push(b'\n');
                                    let mut write_stream = stream;
                                    let _ = write_stream.write_all(&out_bytes);
                                    let _ = write_stream.flush();
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--preview-studio") {
        let path = args
            .get(pos + 1)
            .map(String::as_str)
            .unwrap_or("/tmp/spatial_studio.png");
        let (scene, camera) = if let Some(source) = args
            .iter()
            .position(|a| a == "--project")
            .and_then(|i| args.get(i + 1))
        {
            match project::load(std::path::Path::new(source)) {
                Ok(p) => (p.scene, p.camera),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        } else {
            let scene = mcp::load_fallback_scene().unwrap_or_default();
            let mut camera = scene.last_camera.clone().unwrap_or_default();
            if scene.last_camera.is_none() {
                ui::frame_all(&scene, &mut camera);
            }
            (scene, camera)
        };
        let tab = usize::from(args.iter().any(|a| a == "--review-tab"));
        if let Err(e) = renderer::render_headless(
            &camera,
            &scene,
            ViewportShading::StudioLit,
            false,
            true,
            1440,
            960,
            std::path::Path::new(path),
            Some(tab),
            None,
        ) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        println!("Saved studio preview to {path}");
        return;
    }
    if args.iter().any(|arg| arg == "--mcp") {
        mcp::run_mcp_stdio_server();
        return;
    }

    if let Some(pos) = args.iter().position(|arg| arg == "--cmd") {
        if let Some(json_str) = args.get(pos + 1) {
            match serde_json::from_str::<IpcRequest>(json_str) {
                Ok(req) => {
                    let mut scene = Scene::empty();
                    let mut camera = OrbitCamera::default();
                    let mut history = UndoHistory::default();
                    let mut ui_state = UiState::default();
                    let mut route = |req: &IpcRequest| {
                        mcp::send_to_live_gui(req).unwrap_or_else(|| {
                            mcp::execute_headless(
                                req,
                                &mut scene,
                                &mut camera,
                                &mut history,
                                &mut ui_state,
                                &mut ViewportShading::StudioLit,
                                &mut false,
                                &mut true,
                            )
                        })
                    };
                    let resp = if req.tool == "wait_for_feedback" {
                        mcp::wait_for_feedback(&req.args, &mut route)
                    } else {
                        route(&req)
                    };
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&resp).unwrap_or_default()
                    );
                }
                Err(e) => {
                    eprintln!("Invalid --cmd JSON: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
    }

    env_logger::init();
    let event_loop = EventLoop::<IpcEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");
    spawn_ipc_listener_thread(event_loop.create_proxy());

    let mut app = ModelerApp::default();
    let res = event_loop.run_app(&mut app);
    let _ = std::fs::remove_file(IPC_SOCKET_PATH);
    res.expect("Event loop terminated with error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::voxel::{
        SculptBrush, SculptSettings, VoxelBrushOp, VoxelGrid, apply_sculpt_stroke, voxel_remesh,
    };

    #[test]
    fn closing_during_a_volume_drag_saves_the_latest_box() {
        let mut app = ModelerApp::with_scene(Scene::empty());
        app.saver = autosave::SceneSaver::new(|scene, _| {
            assert_eq!(scene.selection_volume.as_ref().unwrap().center, Vec3::X);
            scene.project_name = "Saved".into();
            Ok(())
        });
        app.scene.selection_volume = Some(crate::scene::SelectionVolume::new(Vec3::X, Vec3::ONE));
        app.ui_state.volume_dragging = true;
        app.ui_state.volume_drag_changed = true;
        assert!(!app.saver.pending());

        app.flush_saves();

        assert_eq!(app.scene.project_name, "Saved");
    }

    #[test]
    fn test_primitives_and_mesh_editing() {
        let mut cube = Mesh::cube(2.0);
        assert_eq!(cube.positions.len(), 8);
        assert_eq!(cube.faces.len(), 6);

        // Select top face and extrude
        cube.selected_faces.insert(4);
        assert!(cube.extrude_selected_faces(0.5));
        assert_eq!(cube.positions.len(), 12);
        assert_eq!(cube.faces.len(), 10); // 6 original + 4 side walls

        // Inset top face
        assert!(cube.inset_selected_faces(0.25, 0.1));
        assert_eq!(cube.faces.len(), 14);

        // Loop cut / Subdivide
        assert!(cube.subdivide_or_loop_cut());
        assert!(cube.faces.len() > 14);
    }

    #[test]
    fn test_voxel_grid_and_sculpting() {
        let mut grid = VoxelGrid::new_sphere(20, 3.0, 1.0);
        let initial_mesh = grid.extract_mesh();
        assert!(!initial_mesh.positions.is_empty());
        assert!(!initial_mesh.faces.is_empty());

        // Apply Voxel CSG addition
        grid.apply_brush(
            VoxelBrushOp::AddSphere,
            Vec3::new(0.8, 0.0, 0.0),
            0.5,
            [1.0, 0.5, 0.2],
            true,
        );
        let mut sculpted_mesh = grid.extract_mesh();
        assert!(sculpted_mesh.positions.len() >= initial_mesh.positions.len());

        // Apply Sculpt stroke with Dyntopo enabled
        let settings = SculptSettings {
            brush: SculptBrush::Draw,
            radius: 0.6,
            strength: 0.5,
            invert: false,
            symmetry_x: true,
            symmetry_y: false,
            symmetry_z: false,
            dyntopo_enabled: true,
            dyntopo_detail_size: 0.15,
            voxel_remesh_resolution: 20,
        };
        apply_sculpt_stroke(
            &mut sculpted_mesh,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            Vec3::ZERO,
            &settings,
        );
        assert!(!sculpted_mesh.faces.is_empty());

        // Test Voxel Remesher
        let remeshed = voxel_remesh(&sculpted_mesh, 18);
        assert!(!remeshed.positions.is_empty());
        assert!(!remeshed.faces.is_empty());
    }

    #[test]
    fn test_obj_roundtrip_and_undo_history() {
        let mut scene = Scene::default();
        let mut history = UndoHistory::default();

        history.record("Add Torus", &scene);
        scene.add_mesh_object("Torus", Mesh::torus(1.0, 0.3, 16, 12), [0.8, 0.7, 0.9]);
        assert_eq!(scene.objects.len(), 2);

        let obj_text = io::export_obj_string(&scene);
        let imported = io::import_obj_str(&obj_text).expect("OBJ import should succeed");
        assert!(!imported.positions.is_empty());
        assert!(!imported.faces.is_empty());

        history.undo(&mut scene);
        assert_eq!(scene.objects.len(), 1);
        history.redo(&mut scene);
        assert_eq!(scene.objects.len(), 2);

        // Test STL & GLB export/import roundtrips
        let stl_path = std::env::temp_dir().join("test_modeler_roundtrip.stl");
        io::export_stl_file(&scene, &stl_path).expect("STL export should succeed");
        let stl_mesh = io::import_stl_file(&stl_path).expect("STL import should succeed");
        assert!(!stl_mesh.positions.is_empty());
        assert!(!stl_mesh.faces.is_empty());
        let _ = std::fs::remove_file(&stl_path);

        let glb_path = std::env::temp_dir().join("test_modeler_roundtrip.glb");
        io::export_glb_file(&scene, &glb_path).expect("GLB export should succeed");
        let glb_mesh = io::import_glb_file(&glb_path).expect("GLB import should succeed");
        assert!(!glb_mesh.positions.is_empty());
        assert!(!glb_mesh.faces.is_empty());
        let _ = std::fs::remove_file(&glb_path);
    }

    #[test]
    fn test_mcp_ipc_tool_commands() {
        let mut scene = Scene::default();
        let mut camera = OrbitCamera::default();
        let mut history = UndoHistory::default();
        let mut ui_state = UiState::default();
        let mut shading = ViewportShading::StudioLit;
        let mut show_wire = false;
        let mut show_grid = true;

        // 1. Add a cube via MCP tool
        let add_req = IpcRequest {
            tool: "add_primitive".to_string(),
            args: serde_json::json!({
                "shape": "cube",
                "name": "TowerBase",
                "size": 2.0,
                "position": [0.0, 1.0, 0.0],
                "color": [0.8, 0.3, 0.2]
            }),
        };
        let resp = execute_ipc_command(
            &add_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(resp.ok);
        let obj_id = resp.data["object_id"].as_u64().unwrap();

        // 2. Extrude top face via MCP edit_mesh tool
        let edit_req = IpcRequest {
            tool: "edit_mesh".to_string(),
            args: serde_json::json!({
                "object_id": obj_id,
                "operation": "extrude",
                "select_faces_by_normal": [0.0, 1.0, 0.0],
                "distance": 0.8
            }),
        };
        let edit_resp = execute_ipc_command(
            &edit_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(edit_resp.ok);
        assert_eq!(edit_resp.data["faces"].as_u64().unwrap(), 10);

        // 3. Apply sculpt stroke via MCP
        let sculpt_req = IpcRequest {
            tool: "sculpt_stroke".to_string(),
            args: serde_json::json!({
                "object_id": obj_id,
                "brush": "inflate",
                "point": [0.0, 1.8, 0.0],
                "radius": 0.8,
                "strength": 0.5,
                "dyntopo": true
            }),
        };
        let sculpt_resp = execute_ipc_command(
            &sculpt_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(sculpt_resp.ok);

        // 4. Apply procedural texture, emission glow, AO baking & top-coat painting via set_material
        let mat_req = IpcRequest {
            tool: "set_material".to_string(),
            args: serde_json::json!({
                "object_id": obj_id,
                "pattern": "stone_blocks",
                "secondary_color": [0.22, 0.20, 0.18],
                "texture_scale": 4.5,
                "emission_color": [1.0, 0.6, 0.2],
                "emission_strength": 0.35,
                "bake_ao": true,
                "ao_strength": 0.7,
                "wear_strength": 0.5,
                "top_coat_color": [0.25, 0.55, 0.28],
                "top_coat_strength": 0.6
            }),
        };
        let mat_resp = execute_ipc_command(
            &mat_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(mat_resp.ok);
        assert_eq!(
            mat_resp.data["pattern"].as_str().unwrap(),
            "Ashlar Stone Blocks"
        );

        // 5. Add a spatial comment on a selected section, retrieve it, select its region, and resolve it
        let comment_req = IpcRequest {
            tool: "add_comment".to_string(),
            args: serde_json::json!({
                "object_id": obj_id,
                "local_point": [0.0, 1.8, 0.0],
                "radius": 0.9,
                "comment": "Extrude this upper battlement section and make it glow gold."
            }),
        };
        let comment_resp = execute_ipc_command(
            &comment_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(comment_resp.ok);
        let cid = comment_resp.data["comment_id"].as_u64().unwrap();
        assert!(comment_resp.data["selected_faces_count"].as_u64().unwrap() > 0);

        let list_req = IpcRequest {
            tool: "get_comments".to_string(),
            args: serde_json::json!({}),
        };
        let list_resp = execute_ipc_command(
            &list_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(list_resp.ok);
        assert_eq!(list_resp.data["count"].as_u64().unwrap(), 1);

        let resolve_req = IpcRequest {
            tool: "resolve_comment".to_string(),
            args: serde_json::json!({
                "comment_id": cid,
                "resolved": true,
                "reply": "Extruded the upper battlement section and applied golden emission."
            }),
        };
        let resolve_resp = execute_ipc_command(
            &resolve_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(resolve_resp.ok);
        assert!(resolve_resp.data["resolved"].as_bool().unwrap());

        // 5b. Add a General Scene Comment (not tied to a specific 3D section)
        let gen_comment_req = IpcRequest {
            tool: "add_comment".to_string(),
            args: serde_json::json!({
                "is_general": true,
                "comment": "Add a floating dragon above the cathedral spire."
            }),
        };
        let gen_resp = execute_ipc_command(
            &gen_comment_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(gen_resp.ok);
        assert!(gen_resp.data["is_general"].as_bool().unwrap());

        // 6. Query scene info
        let info_req = IpcRequest {
            tool: "get_scene_info".to_string(),
            args: serde_json::json!({}),
        };
        let info_resp = execute_ipc_command(
            &info_req,
            &mut scene,
            &mut camera,
            &mut history,
            &mut ui_state,
            &mut shading,
            &mut show_wire,
            &mut show_grid,
        );
        assert!(info_resp.ok);
        assert_eq!(info_resp.data["object_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_smart_subpart_and_surface_selection() {
        // Build a mesh containing two disconnected cubes (representing 2 sub-parts in one layer)
        let mut mesh = Mesh::cube(2.0);
        let second_cube = Mesh::cube(2.0);
        let vert_offset = mesh.positions.len();
        for p in &second_cube.positions {
            mesh.positions.push(*p + Vec3::new(6.0, 0.0, 0.0));
        }
        for c in &second_cube.colors {
            mesh.colors.push(*c);
        }
        for f in &second_cube.faces {
            mesh.faces
                .push(f.iter().map(|&vi| vi + vert_offset).collect());
        }
        mesh.recalculate_normals();
        assert_eq!(mesh.faces.len(), 12);

        // 1. Sub-Part Selection: clicking face 0 selects only the first cube (6 faces)
        mesh.select_connected_component(0, false, false);
        assert_eq!(mesh.selected_faces.len(), 6);

        // Additive Shift-click on face 6 adds the second cube (12 faces total)
        mesh.select_connected_component(6, true, false);
        assert_eq!(mesh.selected_faces.len(), 12);

        // Subtractive Ctrl-click on face 6 removes the second cube (back to 6 faces)
        mesh.select_connected_component(6, false, true);
        assert_eq!(mesh.selected_faces.len(), 6);

        // 2. Smart Surface Selection: subdivide first so each cube face has 4 coplanar quads
        mesh.clear_selection();
        mesh.subdivide_or_loop_cut();
        // Clicking face 0 with 30° crease angle selects only the 4 coplanar quads of that cube wall!
        mesh.select_smart_surface(0, 30.0, false, false);
        assert_eq!(mesh.selected_faces.len(), 4);

        // 3. Topological Grow & Shrink
        assert!(mesh.grow_selection());
        assert!(mesh.selected_faces.len() > 4);
        assert!(mesh.shrink_selection());
        assert_eq!(mesh.selected_faces.len(), 4);

        // 4. Multi-Selection across multiple scene objects for a single SpatialComment
        let mut scene = Scene::default();
        let idx_a = scene.add_mesh_object("North Pillar", Mesh::cube(2.0), [0.8, 0.7, 0.6]);
        let idx_b = scene.add_mesh_object("South Arch", Mesh::cube(2.0), [0.6, 0.7, 0.8]);
        scene.objects[idx_a]
            .mesh
            .select_connected_component(0, false, false);
        scene.objects[idx_b]
            .mesh
            .select_smart_surface(0, 30.0, false, false);

        let comment = scene
            .add_comment_for_selection("Make both the North Pillar and South Arch taller", 0.45)
            .expect("Should create multi-target comment");
        assert_eq!(comment.targets.len(), 2);
        assert_eq!(comment.targets[0].selected_faces.len(), 6);
        assert_eq!(comment.targets[1].selected_faces.len(), 1);

        // Clear all selections and verify focus_comment restores both selections
        scene.clear_all_selections();
        assert!(scene.objects[idx_a].mesh.selected_faces.is_empty());
        assert!(scene.objects[idx_b].mesh.selected_faces.is_empty());

        scene.focus_comment(comment.id);
        assert_eq!(scene.objects[idx_a].mesh.selected_faces.len(), 6);
        assert_eq!(scene.objects[idx_b].mesh.selected_faces.len(), 1);
    }
}
