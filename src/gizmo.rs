use crate::camera::{OrbitCamera, Ray};
use crate::mesh::GpuVertex;
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    pub fn label(&self) -> &'static str {
        match self {
            GizmoMode::Translate => "Translate (G)",
            GizmoMode::Rotate => "Rotate (R)",
            GizmoMode::Scale => "Scale (S)",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GizmoAxis {
    None,
    X,
    Y,
    Z,
    All,
}

impl GizmoAxis {
    pub fn unit_vector(&self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            GizmoAxis::None | GizmoAxis::All => Vec3::ONE,
        }
    }

    pub fn color(&self, hovered: bool) -> [f32; 3] {
        if hovered {
            return [1.0, 0.92, 0.25];
        }
        match self {
            GizmoAxis::X => [0.95, 0.26, 0.26],
            GizmoAxis::Y => [0.28, 0.88, 0.36],
            GizmoAxis::Z => [0.26, 0.56, 0.98],
            GizmoAxis::All => [0.92, 0.92, 0.92],
            GizmoAxis::None => [0.6, 0.6, 0.6],
        }
    }
}

#[derive(Clone, Debug)]
pub struct GizmoState {
    pub mode: GizmoMode,
    pub active_axis: GizmoAxis,
    pub hovered_axis: GizmoAxis,
    pub dragging: bool,
    pub snap_enabled: bool,
    pub snap_translate: f32,
    pub snap_rotate_deg: f32,
    drag_start_origin: Vec3,
    drag_last_param: f32,
    drag_last_screen: Vec2,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            active_axis: GizmoAxis::None,
            hovered_axis: GizmoAxis::None,
            dragging: false,
            snap_enabled: false,
            snap_translate: 0.25,
            snap_rotate_deg: 15.0,
            drag_start_origin: Vec3::ZERO,
            drag_last_param: 0.0,
            drag_last_screen: Vec2::ZERO,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GizmoDelta {
    pub translation: Vec3,
    pub rotation_deg: Vec3,
    pub rotation_quat: Quat,
    pub scale_factor: Vec3,
}

impl GizmoState {
    pub fn gizmo_world_scale(origin: Vec3, camera: &OrbitCamera) -> f32 {
        let dist = (origin - camera.eye()).length().max(0.5);
        dist * 0.18
    }

    /// Test if the cursor ray hovers over one of the gizmo axes at `origin`.
    pub fn pick_axis(&self, origin: Vec3, ray: &Ray, camera: &OrbitCamera) -> GizmoAxis {
        let scale = Self::gizmo_world_scale(origin, camera);
        let threshold = scale * 0.14;
        let mut best_axis = GizmoAxis::None;
        let mut best_dist = threshold;

        for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
            let dir = axis.unit_vector();
            match self.mode {
                GizmoMode::Translate | GizmoMode::Scale => {
                    let t = ray.closest_t_on_axis(origin, dir).clamp(0.0, scale);
                    let pt_axis = origin + dir * t;
                    let t_ray = (pt_axis - ray.origin).dot(ray.direction);
                    if t_ray > 0.0 {
                        let dist = (pt_axis - ray.at(t_ray)).length();
                        if dist < best_dist {
                            best_dist = dist;
                            best_axis = axis;
                        }
                    }
                }
                GizmoMode::Rotate => {
                    // Intersect ray with the rotation plane perpendicular to `dir`
                    if let Some(t_ray) = ray.intersect_plane(origin, dir) {
                        let hit = ray.at(t_ray);
                        let radial_dist = ((hit - origin).length() - scale * 0.85).abs();
                        if radial_dist < best_dist {
                            best_dist = radial_dist;
                            best_axis = axis;
                        }
                    }
                }
            }
        }

        best_axis
    }

    pub fn begin_drag(
        &mut self,
        axis: GizmoAxis,
        origin: Vec3,
        ray: &Ray,
        screen_pos: Vec2,
    ) {
        self.active_axis = axis;
        self.dragging = true;
        self.drag_start_origin = origin;
        self.drag_last_screen = screen_pos;
        if axis != GizmoAxis::None && axis != GizmoAxis::All {
            self.drag_last_param = ray.closest_t_on_axis(origin, axis.unit_vector());
        } else {
            self.drag_last_param = 0.0;
        }
    }

    pub fn update_drag(
        &mut self,
        origin: Vec3,
        ray: &Ray,
        screen_pos: Vec2,
        camera: &OrbitCamera,
    ) -> GizmoDelta {
        let mut out = GizmoDelta {
            translation: Vec3::ZERO,
            rotation_deg: Vec3::ZERO,
            rotation_quat: Quat::IDENTITY,
            scale_factor: Vec3::ONE,
        };

        if !self.dragging || self.active_axis == GizmoAxis::None {
            return out;
        }

        let screen_delta = screen_pos - self.drag_last_screen;
        self.drag_last_screen = screen_pos;

        match self.mode {
            GizmoMode::Translate => {
                if self.active_axis == GizmoAxis::All {
                    let factor = camera.distance * 0.002;
                    out.translation =
                        camera.right() * (screen_delta.x * factor) - camera.up() * (screen_delta.y * factor);
                } else {
                    let dir = self.active_axis.unit_vector();
                    let new_t = ray.closest_t_on_axis(self.drag_start_origin, dir);
                    let mut dt = new_t - self.drag_last_param;
                    if self.snap_enabled && self.snap_translate > 0.001 {
                        if dt.abs() >= self.snap_translate {
                            dt = (dt / self.snap_translate).round() * self.snap_translate;
                            self.drag_last_param += dt;
                        } else {
                            dt = 0.0;
                        }
                    } else {
                        self.drag_last_param = new_t;
                    }
                    out.translation = dir * dt;
                }
            }
            GizmoMode::Scale => {
                let delta_s = 1.0 + (screen_delta.x - screen_delta.y) * 0.008;
                let clamped = delta_s.clamp(0.8, 1.25);
                out.scale_factor = match self.active_axis {
                    GizmoAxis::X => Vec3::new(clamped, 1.0, 1.0),
                    GizmoAxis::Y => Vec3::new(1.0, clamped, 1.0),
                    GizmoAxis::Z => Vec3::new(1.0, 1.0, clamped),
                    GizmoAxis::All | GizmoAxis::None => Vec3::splat(clamped),
                };
            }
            GizmoMode::Rotate => {
                let deg = (screen_delta.x - screen_delta.y) * 0.6;
                let dir = self.active_axis.unit_vector();
                out.rotation_deg = dir * deg;
                out.rotation_quat = Quat::from_axis_angle(dir.normalize_or_zero(), deg.to_radians());
            }
        }

        let _ = origin;
        out
    }

    pub fn end_drag(&mut self) {
        self.dragging = false;
        self.active_axis = GizmoAxis::None;
    }

    /// Generate 3D line segments (`GpuVertex` pairs) for rendering the transform gizmo at `origin`.
    pub fn build_gizmo_lines(&self, origin: Vec3, camera: &OrbitCamera) -> Vec<GpuVertex> {
        let scale = Self::gizmo_world_scale(origin, camera);
        let mut lines = Vec::new();

        let mut push_seg = |a: Vec3, b: Vec3, color: [f32; 3]| {
            lines.push(GpuVertex {
                position: a.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
            lines.push(GpuVertex {
                position: b.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
        };

        for (axis, dir, perp1, perp2) in [
            (GizmoAxis::X, Vec3::X, Vec3::Y, Vec3::Z),
            (GizmoAxis::Y, Vec3::Y, Vec3::Z, Vec3::X),
            (GizmoAxis::Z, Vec3::Z, Vec3::X, Vec3::Y),
        ] {
            let highlighted = self.hovered_axis == axis || self.active_axis == axis;
            let col = axis.color(highlighted);

            match self.mode {
                GizmoMode::Translate => {
                    let end = origin + dir * scale;
                    push_seg(origin, end, col);
                    // Arrowhead cone lines
                    let tip_base = origin + dir * (scale * 0.82);
                    let r = scale * 0.07;
                    for &p in &[perp1 * r, -perp1 * r, perp2 * r, -perp2 * r] {
                        push_seg(end, tip_base + p, col);
                    }
                }
                GizmoMode::Scale => {
                    let end = origin + dir * scale;
                    push_seg(origin, end, col);
                    // Small box tip at end
                    let r = scale * 0.06;
                    push_seg(end - perp1 * r, end + perp1 * r, col);
                    push_seg(end - perp2 * r, end + perp2 * r, col);
                }
                GizmoMode::Rotate => {
                    let ring_r = scale * 0.85;
                    let segs = 36;
                    for i in 0..segs {
                        let a0 = 2.0 * std::f32::consts::PI * (i as f32) / (segs as f32);
                        let a1 = 2.0 * std::f32::consts::PI * ((i + 1) as f32) / (segs as f32);
                        let p0 = origin + (perp1 * a0.cos() + perp2 * a0.sin()) * ring_r;
                        let p1 = origin + (perp1 * a1.cos() + perp2 * a1.sin()) * ring_r;
                        push_seg(p0, p1, col);
                    }
                }
            }
        }

        lines
    }
}
