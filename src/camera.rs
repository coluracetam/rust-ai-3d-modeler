use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Intersect ray with a plane defined by `plane_point` and `plane_normal`.
    pub fn intersect_plane(&self, plane_point: Vec3, plane_normal: Vec3) -> Option<f32> {
        let denom = plane_normal.dot(self.direction);
        if denom.abs() > 1e-6 {
            let t = (plane_point - self.origin).dot(plane_normal) / denom;
            if t >= 0.0 {
                return Some(t);
            }
        }
        None
    }

    /// Closest point parameter `t` along an axis line `(axis_origin, axis_dir)` to this ray.
    pub fn closest_t_on_axis(&self, axis_origin: Vec3, axis_dir: Vec3) -> f32 {
        let a = axis_dir.normalize_or_zero();
        let r = self.direction;
        let w0 = axis_origin - self.origin;
        let b = a.dot(r);
        let d = a.dot(w0);
        let e = r.dot(w0);
        let denom = 1.0 - b * b;
        if denom.abs() < 1e-5 {
            0.0
        } else {
            (b * e - d) / denom
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub orthographic: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 6.0,
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.45,
            fov_y_radians: 45.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            z_near: 0.05,
            z_far: 500.0,
            orthographic: false,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or_zero()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        if self.orthographic {
            let half_h = self.distance * (self.fov_y_radians * 0.5).tan();
            let half_w = half_h * self.aspect.max(0.01);
            Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.z_near, self.z_far)
        } else {
            Mat4::perspective_rh(
                self.fov_y_radians,
                self.aspect.max(0.01),
                self.z_near,
                self.z_far,
            )
        }
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.007;
        self.yaw -= delta_x * sensitivity;
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.02;
        self.pitch = (self.pitch + delta_y * sensitivity).clamp(-max_pitch, max_pitch);
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let factor = self.distance * 0.0015;
        let right = self.right();
        let up = self.up();
        self.target += -right * (delta_x * factor) + up * (delta_y * factor);
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let zoom_factor = (1.0 - scroll_delta * 0.1).clamp(0.5, 2.0);
        self.distance = (self.distance * zoom_factor).clamp(0.3, 200.0);
    }

    pub fn set_view_front(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
    }

    pub fn set_view_right(&mut self) {
        self.yaw = std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
    }

    pub fn set_view_top(&mut self) {
        self.yaw = 0.0;
        self.pitch = std::f32::consts::FRAC_PI_2 - 0.021;
    }

    /// Generate a world-space ray from normalized viewport coordinates `pos` in `[0..width, 0..height]`.
    pub fn screen_ray(&self, pos: Vec2, viewport_size: Vec2) -> Ray {
        let ndc_x = (2.0 * pos.x / viewport_size.x.max(1.0)) - 1.0;
        let ndc_y = 1.0 - (2.0 * pos.y / viewport_size.y.max(1.0));
        let inv_vp = self.view_proj_matrix().inverse();
        let near_pt = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far_pt = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        Ray::new(near_pt, far_pt - near_pt)
    }

    /// Project a 3D world point to 2D viewport pixel coordinates `[0..width, 0..height]`.
    pub fn world_to_screen(&self, world_pos: Vec3, viewport_size: Vec2) -> Option<Vec2> {
        let clip = self.view_proj_matrix() * world_pos.extend(1.0);
        if clip.w <= 0.0001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }
        let x = (ndc.x + 1.0) * 0.5 * viewport_size.x;
        let y = (1.0 - ndc.y) * 0.5 * viewport_size.y;
        Some(Vec2::new(x, y))
    }
}
