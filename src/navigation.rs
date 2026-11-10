//! Game-style free-flight navigation: WASD to move, mouse to look, Shift/Ctrl to change speed.
use crate::camera::OrbitCamera;
use glam::Vec3;
use std::collections::HashSet;
use std::time::Instant;
use winit::keyboard::KeyCode;

pub const MIN_FLY_SPEED: f32 = 0.1;
pub const MAX_FLY_SPEED: f32 = 200.0;

#[derive(Default)]
pub struct FlyController {
    held: HashSet<KeyCode>,
    last_tick: Option<Instant>,
}

impl FlyController {
    pub fn reset(&mut self) {
        self.held.clear();
        self.last_tick = None;
    }

    pub fn key(&mut self, code: KeyCode, pressed: bool) {
        if pressed {
            self.held.insert(code);
        } else {
            self.held.remove(&code);
        }
        self.last_tick.get_or_insert_with(Instant::now);
    }

    /// Movement direction as (right, up, forward), each in -1..=1.
    pub fn direction(&self) -> Vec3 {
        let axis = |pos: &[KeyCode], neg: &[KeyCode]| {
            let p = pos.iter().any(|k| self.held.contains(k)) as i32;
            let n = neg.iter().any(|k| self.held.contains(k)) as i32;
            (p - n) as f32
        };
        Vec3::new(
            axis(
                &[KeyCode::KeyD, KeyCode::ArrowRight],
                &[KeyCode::KeyA, KeyCode::ArrowLeft],
            ),
            axis(&[KeyCode::KeyE, KeyCode::Space], &[KeyCode::KeyQ]),
            axis(
                &[KeyCode::KeyW, KeyCode::ArrowUp],
                &[KeyCode::KeyS, KeyCode::ArrowDown],
            ),
        )
    }

    pub fn moving(&self) -> bool {
        self.direction() != Vec3::ZERO
    }

    /// Advance the camera by the time since the last tick. Returns true when it moved.
    pub fn tick(&mut self, camera: &mut OrbitCamera, speed: f32, fast: bool, slow: bool) -> bool {
        let now = Instant::now();
        let dt = self
            .last_tick
            .replace(now)
            .map_or(0.0, |t| (now - t).as_secs_f32().min(0.1));
        let dir = self.direction();
        if dir == Vec3::ZERO || dt <= 0.0 {
            return false;
        }
        let multiplier = if fast {
            4.0
        } else if slow {
            0.25
        } else {
            1.0
        };
        camera.fly_move(dir, speed * multiplier * dt);
        true
    }
}

/// Scroll-wheel speed change while flying.
pub fn adjust_speed(speed: f32, lines: f32) -> f32 {
    (speed * 1.2_f32.powf(lines)).clamp(MIN_FLY_SPEED, MAX_FLY_SPEED)
}

/// A comfortable starting speed: cross the framed view in about three seconds.
pub fn default_speed_for(camera: &OrbitCamera) -> f32 {
    (camera.distance / 3.0).clamp(0.5, 40.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_look_keeps_the_eye_fixed() {
        let mut camera = OrbitCamera::default();
        let eye = camera.eye();
        camera.fly_look(120.0, -40.0);
        assert!(camera.eye().distance(eye) < 1e-4);
        assert!(camera.forward().dot(OrbitCamera::default().forward()) < 0.999);
    }

    #[test]
    fn mouse_right_turns_right_and_mouse_down_looks_down() {
        let mut camera = OrbitCamera {
            yaw: 0.0,
            pitch: 0.0,
            ..Default::default()
        };
        let right = camera.right();
        camera.fly_look(50.0, 0.0);
        assert!(camera.forward().dot(right) > 0.0);
        let mut camera = OrbitCamera {
            yaw: 0.0,
            pitch: 0.0,
            ..Default::default()
        };
        camera.fly_look(0.0, 50.0);
        assert!(camera.forward().y < 0.0);
    }

    #[test]
    fn wasd_moves_relative_to_view() {
        let mut camera = OrbitCamera::default();
        let start = camera.eye();
        let forward = camera.forward();
        camera.fly_move(Vec3::Z, 2.0);
        assert!((camera.eye() - start - forward * 2.0).length() < 1e-4);
        let start = camera.eye();
        camera.fly_move(Vec3::Y, 1.0);
        assert!((camera.eye() - start - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn held_keys_map_to_axes() {
        let mut fly = FlyController::default();
        fly.key(KeyCode::KeyW, true);
        fly.key(KeyCode::KeyA, true);
        fly.key(KeyCode::KeyQ, true);
        assert_eq!(fly.direction(), Vec3::new(-1.0, -1.0, 1.0));
        fly.key(KeyCode::KeyA, false);
        fly.key(KeyCode::KeyD, true);
        assert_eq!(fly.direction(), Vec3::new(1.0, -1.0, 1.0));
        fly.reset();
        assert!(!fly.moving());
    }

    #[test]
    fn look_from_and_refocus_preserve_view() {
        let mut camera = OrbitCamera::default();
        camera.look_from(Vec3::new(3.0, 2.0, 5.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(camera.eye().distance(Vec3::new(3.0, 2.0, 5.0)) < 1e-3);
        assert!(camera.target.distance(Vec3::new(0.0, 1.0, 0.0)) < 1e-3);
        let (eye, forward) = (camera.eye(), camera.forward());
        camera.refocus(1.5);
        assert!(camera.eye().distance(eye) < 1e-3);
        assert!(camera.forward().dot(forward) > 0.9999);
        assert!((camera.distance - 1.5).abs() < 1e-5);
    }

    #[test]
    fn scroll_speed_is_clamped() {
        assert!(adjust_speed(1.0, 1.0) > 1.0);
        assert!(adjust_speed(1.0, -1.0) < 1.0);
        assert_eq!(adjust_speed(MAX_FLY_SPEED, 10.0), MAX_FLY_SPEED);
        assert_eq!(adjust_speed(MIN_FLY_SPEED, -10.0), MIN_FLY_SPEED);
    }
}
