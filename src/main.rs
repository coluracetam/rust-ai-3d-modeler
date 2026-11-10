#![recursion_limit = "512"]

pub mod camera;
pub mod gizmo;
pub mod history;
pub mod io;
pub mod ipc;
pub mod mcp;
pub mod mesh;
pub mod renderer;
pub mod scene;
pub mod ui;
pub mod voxel;

use crate::camera::OrbitCamera;
use crate::gizmo::GizmoState;
use crate::history::UndoHistory;
use crate::ipc::{execute_ipc_command, IpcRequest, IpcResponse, IPC_SOCKET_PATH};
use crate::mesh::EditElementMode;
use crate::renderer::{Renderer3D, ViewportShading};
use crate::scene::Scene;
use crate::ui::{draw_ui, update_pending_anchor_from_selection, SmartSelectMode, UiState};
use glam::{Vec2, Vec3};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::sync::{mpsc, Arc};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

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

pub struct ModelerApp {
    gpu: Option<GpuContext>,
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
    stroke_active: bool,
    last_sculpt_local_hit: Option<Vec3>,
    hover_brush_world: Option<(Vec3, Vec3)>,
}

impl Default for ModelerApp {
    fn default() -> Self {
        let initial_scene = mcp::load_fallback_scene().unwrap_or_default();
        Self {
            gpu: None,
            scene: initial_scene,
            camera: OrbitCamera::default(),
            gizmo: GizmoState::default(),
            history: UndoHistory::default(),
            ui_state: UiState::default(),
            modifiers: ModifiersState::default(),
            cursor_pos: Vec2::ZERO,
            last_cursor_pos: Vec2::ZERO,
            left_mouse_down: false,
            middle_mouse_down: false,
            right_mouse_down: false,
            stroke_active: false,
            last_sculpt_local_hit: None,
            hover_brush_world: None,
        }
    }
}

impl ModelerApp {
    fn viewport_size(&self) -> Vec2 {
        if let Some(gpu) = &self.gpu {
            Vec2::new(
                gpu.surface_config.width.max(1) as f32,
                gpu.surface_config.height.max(1) as f32,
            )
        } else {
            Vec2::new(1280.0, 720.0)
        }
    }

    fn handle_viewport_left_press(&mut self) {
        let vp_size = self.viewport_size();
        let ray = self.camera.screen_ray(self.cursor_pos, vp_size);

        if let Some(hit) = self.scene.raycast(&ray) {
            let multi_active = self.ui_state.multi_select
                || self.modifiers.shift_key()
                || self.modifiers.control_key();

            if !multi_active {
                self.scene.clear_all_selections();
            }

            self.scene.selected_object = Some(hit.object_idx);
            self.stroke_active = true;

            let already_selected = self
                .scene
                .objects
                .get(hit.object_idx)
                .map_or(false, |o| {
                    o.mesh.selected_faces.contains(&hit.local_hit.face_idx)
                });

            let subtract = if self.modifiers.control_key() {
                true
            } else if self.ui_state.multi_select
                && !self.modifiers.shift_key()
                && self.ui_state.select_mode != SmartSelectMode::Brush
                && already_selected
            {
                // In Multi-Select mode, clicking an already-selected sub-part or surface toggles it off!
                true
            } else {
                false
            };

            let additive = multi_active && !subtract;

            self.apply_section_selection_at_hit(&hit, additive, subtract);
            mcp::save_fallback_scene(&self.scene);
        } else if !self.modifiers.shift_key()
            && !self.modifiers.control_key()
            && !self.ui_state.multi_select
        {
            self.scene.clear_all_selections();
            self.scene.selected_object = None;
            mcp::save_fallback_scene(&self.scene);
        }
    }

    fn apply_section_selection_at_hit(
        &mut self,
        hit: &crate::scene::WorldRayHit,
        additive: bool,
        subtract: bool,
    ) {
        let select_mode = self.ui_state.select_mode;
        let crease_deg = self.ui_state.crease_angle_deg;
        let radius = self.ui_state.selection_radius;
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
                SmartSelectMode::Brush => {
                    let avg_scale = ((obj.transform.scale.x.abs()
                        + obj.transform.scale.y.abs()
                        + obj.transform.scale.z.abs())
                        / 3.0)
                        .max(0.01);
                    let local_radius = radius / avg_scale;
                    obj.mesh.select_faces_in_sphere(
                        hit.local_hit.point,
                        local_radius,
                        Some(seed_face),
                        additive,
                        subtract,
                    );
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

            let mode_label = match select_mode {
                SmartSelectMode::SubPart => "Sub-Part",
                SmartSelectMode::SmartSurface => "Smart Surface",
                SmartSelectMode::WholeLayer => "Whole Layer",
                SmartSelectMode::Brush => "Brush",
            };
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

        if select_mode == SmartSelectMode::Brush {
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
            SmartSelectMode::Brush => std::collections::HashSet::new(),
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

        let vp_size = self.viewport_size();
        let ray = self.camera.screen_ray(self.cursor_pos, vp_size);

        // 2. Continuous Section Paint Selection while dragging in Brush mode (or Shift/Ctrl drag)
        if self.left_mouse_down
            && self.stroke_active
            && self.ui_state.select_mode == SmartSelectMode::Brush
        {
            if let Some(hit) = self.scene.raycast_active(&ray) {
                let subtract = self.modifiers.control_key();
                self.apply_section_selection_at_hit(&hit, true, subtract);
            }
        } else if let Some(hit) = self.scene.raycast(&ray) {
            self.hover_brush_world = Some((hit.world_point, hit.world_normal));
            self.update_hover_preview(Some(&hit));
        } else {
            self.hover_brush_world = None;
            self.update_hover_preview(None);
        }
    }

    fn handle_key_press(&mut self, code: KeyCode) {
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
                self.ui_state.select_mode = SmartSelectMode::Brush;
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::Tab => {
                self.ui_state.select_mode = match self.ui_state.select_mode {
                    SmartSelectMode::SubPart => SmartSelectMode::SmartSurface,
                    SmartSelectMode::SmartSurface => SmartSelectMode::WholeLayer,
                    SmartSelectMode::WholeLayer => SmartSelectMode::Brush,
                    SmartSelectMode::Brush => SmartSelectMode::SubPart,
                };
                self.ui_state.hover_preview_seed_face = None;
            }
            KeyCode::KeyF => {
                if let Some((_id, wp, _lp, _wn)) = self.scene.pending_anchor {
                    self.camera.target = wp;
                } else if let Some(obj) = self.scene.active_object() {
                    self.camera.target = obj.transform.translation;
                }
            }
            KeyCode::KeyA => {
                if let Some(obj) = self.scene.active_object_mut() {
                    obj.mesh.select_all(EditElementMode::Face);
                }
                update_pending_anchor_from_selection(&mut self.scene);
                mcp::save_fallback_scene(&self.scene);
            }
            KeyCode::KeyC | KeyCode::Escape => {
                self.scene.clear_all_selections();
                self.scene.selected_object = None;
                self.scene.active_comment_id = None;
                mcp::save_fallback_scene(&self.scene);
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
                    mcp::save_fallback_scene(&self.scene);
                } else if self.ui_state.select_mode == SmartSelectMode::Brush {
                    self.ui_state.selection_radius =
                        (self.ui_state.selection_radius - 0.05).max(0.05);
                    self.ui_state.sculpt.radius = self.ui_state.selection_radius;
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
                    mcp::save_fallback_scene(&self.scene);
                } else if self.ui_state.select_mode == SmartSelectMode::Brush {
                    self.ui_state.selection_radius =
                        (self.ui_state.selection_radius + 0.05).min(2.50);
                    self.ui_state.sculpt.radius = self.ui_state.selection_radius;
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler<IpcEvent> for ModelerApp {
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: IpcEvent) {
        if event.request.tool == "quit_gui" {
            let _ = std::fs::remove_file(IPC_SOCKET_PATH);
            mcp::save_fallback_scene(&self.scene);
            let _ = event
                .reply_tx
                .send(IpcResponse::success("Closed GUI window", serde_json::json!({})));
            event_loop.exit();
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

        if resp.ok
            && event.request.tool != "get_scene_info"
            && event.request.tool != "capture_viewport"
        {
            mcp::save_fallback_scene(&self.scene);
        }

        if let Some(gpu) = &mut self.gpu {
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
            .with_title("Rust 3D Modeler & Voxel Sculpt Studio")
            .with_inner_size(PhysicalSize::new(1400, 860));

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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = &mut self.gpu else {
            return;
        };

        let egui_resp = gpu.egui_state.on_window_event(&gpu.window, &event);
        if egui_resp.repaint {
            gpu.window.request_redraw();
        }
        let egui_wants_pointer = egui_resp.consumed || gpu.egui_ctx.wants_pointer_input();
        let egui_wants_keyboard = gpu.egui_ctx.wants_keyboard_input();

        match event {
            WindowEvent::CloseRequested => {
                let _ = std::fs::remove_file(IPC_SOCKET_PATH);
                mcp::save_fallback_scene(&self.scene);
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
                        self.left_mouse_down = pressed;
                        if pressed && !egui_wants_pointer && !self.modifiers.alt_key() {
                            self.handle_viewport_left_press();
                        } else if !pressed {
                            if self.stroke_active {
                                mcp::save_fallback_scene(&self.scene);
                            }
                            self.gizmo.end_drag();
                            self.stroke_active = false;
                            self.last_sculpt_local_hit = None;
                        }
                    }
                    MouseButton::Middle => {
                        self.middle_mouse_down = pressed;
                    }
                    MouseButton::Right => {
                        self.right_mouse_down = pressed;
                    }
                    _ => {}
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if !egui_wants_pointer || self.gizmo.dragging || self.stroke_active {
                    self.handle_cursor_moved(pos);
                } else {
                    self.last_cursor_pos = pos;
                    self.cursor_pos = pos;
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_wants_pointer {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.05,
                    };
                    self.camera.zoom(scroll);
                    if let Some(gpu) = &self.gpu {
                        gpu.window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
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
                            gpu.renderer_3d
                                .resize(&gpu.device, size.width, size.height);
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

                if self.ui_state.scene_dirty_for_disk {
                    self.ui_state.scene_dirty_for_disk = false;
                    mcp::save_fallback_scene(&self.scene);
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
                    pixels_per_point: gpu.window.scale_factor() as f32,
                };

                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Main Frame Encoder"),
                        });

                let brush_cursor = if self.ui_state.select_mode == SmartSelectMode::Brush {
                    self.hover_brush_world
                        .map(|(p, n)| (p, n, self.ui_state.selection_radius))
                } else {
                    self.hover_brush_world.map(|(p, n)| (p, n, 0.12))
                };

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
            let mut reader = BufReader::new( match stream.try_clone() {
                Ok(s) => s,
                Err(_) => continue,
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
                        if let Ok(resp) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
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
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--mcp") {
        mcp::run_mcp_stdio_server();
        return;
    }

    if let Some(pos) = args.iter().position(|arg| arg == "--cmd") {
        if let Some(json_str) = args.get(pos + 1) {
            match serde_json::from_str::<IpcRequest>(json_str) {
                Ok(req) => {
                    let pre_disk_scene = mcp::load_fallback_scene();
                    let resp = if let Some(live) = mcp::send_to_live_gui(&req) {
                        if (!live.ok && live.message.starts_with("Unknown MCP tool"))
                            || (req.tool == "set_material" && live.data.get("pattern").is_none())
                        {
                            let mut scene = pre_disk_scene
                                .or_else(mcp::load_fallback_scene)
                                .unwrap_or_default();
                            let mut camera = OrbitCamera::default();
                            let mut history = UndoHistory::default();
                            let mut ui_state = UiState::default();
                            let mut shading = ViewportShading::StudioLit;
                            let mut wire = false;
                            let mut grid = true;
                            let r = execute_ipc_command(
                                &req,
                                &mut scene,
                                &mut camera,
                                &mut history,
                                &mut ui_state,
                                &mut shading,
                                &mut wire,
                                &mut grid,
                            );
                            mcp::save_fallback_scene(&scene);
                            r
                        } else {
                            live
                        }
                    } else {
                        let mut scene = mcp::load_fallback_scene().unwrap_or_default();
                        let mut camera = OrbitCamera::default();
                        let mut history = UndoHistory::default();
                        let mut ui_state = UiState::default();
                        let mut shading = ViewportShading::StudioLit;
                        let mut wire = false;
                        let mut grid = true;
                        let r = execute_ipc_command(
                            &req,
                            &mut scene,
                            &mut camera,
                            &mut history,
                            &mut ui_state,
                            &mut shading,
                            &mut wire,
                            &mut grid,
                        );
                        mcp::save_fallback_scene(&scene);
                        r
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
        apply_sculpt_stroke, voxel_remesh, SculptBrush, SculptSettings, VoxelBrushOp, VoxelGrid,
    };

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
        scene.objects[idx_a].mesh.select_connected_component(0, false, false);
        scene.objects[idx_b].mesh.select_smart_surface(0, 30.0, false, false);

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
