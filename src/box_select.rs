//! Screen-space box selection.
//!
//! Faces are rasterized into a small depth/face-id buffer covering only the dragged rectangle, so
//! the box picks what the user can actually see (respecting occlusion and the section plane).
//! With `through` enabled every face that overlaps the rectangle is picked, hidden or not.
use crate::camera::OrbitCamera;
use crate::scene::Scene;
use glam::{Vec2, Vec3, Vec4};
use std::collections::{HashMap, HashSet};

/// Rasterization budget for the box; larger boxes are sampled at a coarser resolution.
const MAX_BUFFER_PIXELS: f32 = 640.0 * 640.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxOp {
    Replace,
    Add,
    Subtract,
}

#[derive(Debug, Default)]
pub struct BoxSelectOutcome {
    /// Faces found inside the box, per object index.
    pub hits: HashMap<usize, HashSet<usize>>,
    /// Total faces selected across the scene afterwards.
    pub selected_faces: usize,
    /// Separate regions (one AI comment target each) across the scene afterwards.
    pub regions: usize,
    pub objects: usize,
}

struct Buffer {
    width: usize,
    height: usize,
    origin: Vec2,
    scale: f32,
    depth: Vec<f32>,
    id: Vec<Option<(usize, usize)>>,
}

impl Buffer {
    fn new(min: Vec2, max: Vec2) -> Self {
        let size = (max - min).max(Vec2::ONE);
        let scale = (MAX_BUFFER_PIXELS / (size.x * size.y)).sqrt().min(1.0);
        let width = ((size.x * scale).ceil() as usize).max(1);
        let height = ((size.y * scale).ceil() as usize).max(1);
        Self {
            width,
            height,
            origin: min,
            scale,
            depth: vec![f32::INFINITY; width * height],
            id: vec![None; width * height],
        }
    }

    fn to_buffer(&self, screen: Vec2) -> Vec2 {
        (screen - self.origin) * self.scale
    }

    fn sample(&self, screen: Vec2) -> Option<(f32, Option<(usize, usize)>)> {
        let b = self.to_buffer(screen);
        if b.x < 0.0 || b.y < 0.0 {
            return None;
        }
        let (x, y) = (b.x as usize, b.y as usize);
        (x < self.width && y < self.height).then(|| {
            let i = y * self.width + x;
            (self.depth[i], self.id[i])
        })
    }

    /// Rasterize a screen-space triangle (x, y in pixels, z = NDC depth). Returns true if it
    /// covered at least one sample.
    fn raster(&mut self, tri: [Vec3; 3], id: (usize, usize), depth_test: bool) -> bool {
        let p: [Vec3; 3] = tri.map(|v| {
            let b = self.to_buffer(v.truncate());
            Vec3::new(b.x, b.y, v.z)
        });
        let area = edge(p[0], p[1], p[2]);
        if !area.is_finite() || area.abs() < 1e-12 {
            return false;
        }
        let min_x = p.iter().map(|v| v.x).fold(f32::INFINITY, f32::min).floor().max(0.0);
        let min_y = p.iter().map(|v| v.y).fold(f32::INFINITY, f32::min).floor().max(0.0);
        let max_x = p
            .iter()
            .map(|v| v.x)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min(self.width as f32);
        let max_y = p
            .iter()
            .map(|v| v.y)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min(self.height as f32);
        if min_x >= max_x || min_y >= max_y {
            return false;
        }
        let mut covered = false;
        for y in min_y as usize..max_y as usize {
            for x in min_x as usize..max_x as usize {
                let s = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let w0 = edge(p[1], p[2], s) / area;
                let w1 = edge(p[2], p[0], s) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                covered = true;
                if depth_test {
                    let z = w0 * p[0].z + w1 * p[1].z + w2 * p[2].z;
                    let i = y * self.width + x;
                    if z < self.depth[i] {
                        self.depth[i] = z;
                        self.id[i] = Some(id);
                    }
                }
            }
        }
        covered
    }
}

fn edge(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Clip a clip-space polygon against the near plane (z >= 0 in wgpu's depth range).
fn clip_near(poly: &[Vec4]) -> Vec<Vec4> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let (da, db) = (a.z, b.z);
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push(a + (b - a) * t);
        }
    }
    out
}

fn to_screen(clip: Vec4, viewport: Vec2) -> Vec3 {
    let ndc = clip.truncate() / clip.w.max(1e-6);
    Vec3::new(
        (ndc.x + 1.0) * 0.5 * viewport.x,
        (1.0 - ndc.y) * 0.5 * viewport.y,
        ndc.z,
    )
}

/// Find the faces inside the viewport-pixel rectangle `a`..`b`.
pub fn faces_in_rect(
    scene: &Scene,
    camera: &OrbitCamera,
    viewport: Vec2,
    a: Vec2,
    b: Vec2,
    through: bool,
) -> HashMap<usize, HashSet<usize>> {
    let mut min = a.min(b).max(Vec2::ZERO);
    let mut max = a.max(b).min(viewport);
    // A click (or a sliver) still picks the face under the cursor.
    if max.x - min.x < 1.0 {
        min.x = (min.x - 0.5).max(0.0);
        max.x = min.x + 1.0;
    }
    if max.y - min.y < 1.0 {
        min.y = (min.y - 0.5).max(0.0);
        max.y = min.y + 1.0;
    }
    let mut buffer = Buffer::new(min, max);
    let view_proj = camera.view_proj_matrix();
    let plane = scene.clip_plane.map(|p| (Vec3::new(p[0], p[1], p[2]), p[3]));
    let mut hits: HashMap<usize, HashSet<usize>> = HashMap::new();
    // Candidate faces whose centre projects inside the box: (object, face, screen centre).
    let mut centres = Vec::new();

    for (oi, obj) in scene.objects.iter().enumerate() {
        if !obj.visible {
            continue;
        }
        let model = obj.transform.matrix();
        let mvp = view_proj * model;
        let clip: Vec<Vec4> = obj
            .mesh
            .positions
            .iter()
            .map(|p| mvp * p.extend(1.0))
            .collect();
        for (fi, face) in obj.mesh.faces.iter().enumerate() {
            if face.len() < 3 || face.iter().any(|&v| v >= clip.len()) {
                continue;
            }
            if let Some((n, d)) = plane {
                let centre = model.transform_point3(obj.mesh.face_center(fi));
                if n.dot(centre) + d < 0.0 {
                    continue;
                }
            }
            let poly = clip_near(&face.iter().map(|&v| clip[v]).collect::<Vec<_>>());
            if poly.len() < 3 {
                continue;
            }
            let screen: Vec<Vec3> = poly.iter().map(|&c| to_screen(c, viewport)).collect();
            // Cheap reject before rasterizing.
            let lo = screen.iter().fold(Vec2::splat(f32::INFINITY), |m, s| m.min(s.truncate()));
            let hi = screen
                .iter()
                .fold(Vec2::splat(f32::NEG_INFINITY), |m, s| m.max(s.truncate()));
            if hi.x < min.x || hi.y < min.y || lo.x > max.x || lo.y > max.y {
                continue;
            }
            let mut covered = false;
            for i in 1..screen.len() - 1 {
                covered |= buffer.raster([screen[0], screen[i], screen[i + 1]], (oi, fi), !through);
            }
            if through && covered {
                hits.entry(oi).or_default().insert(fi);
            }
            let centre_clip = mvp * obj.mesh.face_center(fi).extend(1.0);
            if centre_clip.z >= 0.0 && centre_clip.w > 0.0 {
                let c = to_screen(centre_clip, viewport);
                if c.x >= min.x && c.x <= max.x && c.y >= min.y && c.y <= max.y && c.z <= 1.0 {
                    centres.push((oi, fi, c));
                }
            }
        }
    }

    if through {
        for (oi, fi, _) in centres {
            hits.entry(oi).or_default().insert(fi);
        }
        return hits;
    }
    for &(oi, fi) in buffer.id.iter().flatten() {
        hits.entry(oi).or_default().insert(fi);
    }
    // Faces smaller than a sample: keep them when their centre is not hidden behind something.
    for (oi, fi, c) in centres {
        let Some((depth, _)) = buffer.sample(c.truncate()) else {
            continue;
        };
        let visible = !depth.is_finite() || {
            let (mine, front) = (camera.linear_depth(c.z), camera.linear_depth(depth));
            mine <= front * 1.01 + 1e-3
        };
        if visible {
            hits.entry(oi).or_default().insert(fi);
        }
    }
    hits
}

/// Apply a box selection to the scene and report what is selected afterwards.
pub fn apply_box_selection(
    scene: &mut Scene,
    camera: &OrbitCamera,
    viewport: Vec2,
    a: Vec2,
    b: Vec2,
    through: bool,
    op: BoxOp,
) -> BoxSelectOutcome {
    let hits = faces_in_rect(scene, camera, viewport, a, b, through);
    if op == BoxOp::Replace {
        scene.clear_all_selections();
    }
    for (&oi, faces) in &hits {
        if let Some(obj) = scene.objects.get_mut(oi) {
            obj.mesh
                .apply_face_set_selection(faces, true, op == BoxOp::Subtract);
        }
    }
    // Make the part with the largest share of the box active.
    let largest = hits
        .iter()
        .filter(|(oi, _)| {
            scene
                .objects
                .get(**oi)
                .is_some_and(|o| !o.mesh.selected_faces.is_empty())
        })
        .max_by_key(|(oi, f)| (f.len(), std::cmp::Reverse(**oi)))
        .map(|(&oi, _)| oi);
    let current_has_faces = scene
        .active_object()
        .is_some_and(|o| !o.mesh.selected_faces.is_empty());
    if let Some(oi) = largest.filter(|_| op != BoxOp::Subtract || !current_has_faces) {
        scene.selected_object = Some(oi);
    } else if !current_has_faces {
        scene.selected_object = scene
            .objects
            .iter()
            .position(|o| !o.mesh.selected_faces.is_empty());
    }
    let mut outcome = BoxSelectOutcome {
        hits,
        ..Default::default()
    };
    for obj in &scene.objects {
        if !obj.mesh.selected_faces.is_empty() {
            outcome.objects += 1;
            outcome.selected_faces += obj.mesh.selected_faces.len();
            outcome.regions += obj.mesh.selected_components().len();
        }
    }
    if outcome.selected_faces == 0 {
        scene.pending_anchor = None;
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    fn scene_with(objects: &[(Vec3, f32)]) -> Scene {
        let mut scene = Scene::empty();
        for (i, &(pos, size)) in objects.iter().enumerate() {
            let idx = scene.add_mesh_object(format!("Cube {i}"), Mesh::cube(size), [0.7; 3]);
            scene.objects[idx].transform.translation = pos;
        }
        scene
    }

    fn front_camera() -> OrbitCamera {
        let mut camera = OrbitCamera {
            aspect: 1.0,
            ..Default::default()
        };
        camera.look_from(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
        camera
    }

    const VP: Vec2 = Vec2::new(800.0, 800.0);

    #[test]
    fn box_picks_only_visible_faces() {
        let mut scene = scene_with(&[(Vec3::ZERO, 2.0)]);
        let camera = front_camera();
        let hits = faces_in_rect(&scene, &camera, VP, Vec2::ZERO, VP, false);
        let faces = &hits[&0];
        // Looking straight down -Z only the +Z face is visible.
        assert_eq!(faces.len(), 1);
        let fi = *faces.iter().next().unwrap();
        assert!(scene.objects[0].mesh.face_normal(fi).z > 0.9);

        let through = faces_in_rect(&scene, &camera, VP, Vec2::ZERO, VP, true);
        assert_eq!(through[&0].len(), 6);

        let out = apply_box_selection(&mut scene, &camera, VP, Vec2::ZERO, VP, false, BoxOp::Replace);
        assert_eq!(out.selected_faces, 1);
        assert_eq!(scene.selected_object, Some(0));
    }

    #[test]
    fn occluded_objects_are_not_picked_unless_selecting_through() {
        // A small cube hidden directly behind a big one.
        let scene = scene_with(&[(Vec3::ZERO, 3.0), (Vec3::new(0.0, 0.0, -4.0), 1.0)]);
        let camera = front_camera();
        let hits = faces_in_rect(&scene, &camera, VP, Vec2::splat(300.0), Vec2::splat(500.0), false);
        assert!(hits.contains_key(&0));
        assert!(!hits.contains_key(&1));
        let through = faces_in_rect(&scene, &camera, VP, Vec2::splat(300.0), Vec2::splat(500.0), true);
        assert!(through.contains_key(&1));
    }

    #[test]
    fn separate_boxes_build_separate_regions_and_subtract_removes() {
        let mut scene = scene_with(&[(Vec3::new(-2.5, 0.0, 0.0), 1.0), (Vec3::new(2.5, 0.0, 0.0), 1.0)]);
        let camera = front_camera();
        let left = camera.world_to_screen(Vec3::new(-2.5, 0.0, 0.5), VP).unwrap();
        let right = camera.world_to_screen(Vec3::new(2.5, 0.0, 0.5), VP).unwrap();
        let d = Vec2::splat(15.0);
        let out = apply_box_selection(&mut scene, &camera, VP, left - d, left + d, false, BoxOp::Replace);
        assert_eq!((out.objects, out.regions), (1, 1));
        let out = apply_box_selection(&mut scene, &camera, VP, right - d, right + d, false, BoxOp::Add);
        assert_eq!((out.objects, out.regions), (2, 2));
        assert_eq!(scene.collect_current_target_sections(0.4).len(), 2);
        let out = apply_box_selection(&mut scene, &camera, VP, left - d, left + d, false, BoxOp::Subtract);
        assert_eq!((out.objects, out.regions), (1, 1));
        assert!(scene.objects[0].mesh.selected_faces.is_empty());
        assert_eq!(scene.selected_object, Some(1));
        // Replacing with an empty box clears everything.
        let out = apply_box_selection(&mut scene, &camera, VP, Vec2::splat(1.0), Vec2::splat(3.0), false, BoxOp::Replace);
        assert_eq!(out.selected_faces, 0);
        assert!(!scene.has_any_selection());
    }

    #[test]
    fn click_sized_box_picks_face_under_cursor_and_respects_hidden_and_section() {
        let mut scene = scene_with(&[(Vec3::ZERO, 2.0)]);
        let camera = front_camera();
        let c = VP * 0.5;
        assert_eq!(faces_in_rect(&scene, &camera, VP, c, c, false)[&0].len(), 1);
        scene.clip_plane = Some([0.0, 0.0, -1.0, 0.0]);
        let hits = faces_in_rect(&scene, &camera, VP, c, c, false);
        // The +Z face is cut away, so the inside of the back face shows through.
        let fi = *hits[&0].iter().next().unwrap();
        assert!(scene.objects[0].mesh.face_normal(fi).z < -0.9);
        scene.objects[0].visible = false;
        assert!(faces_in_rect(&scene, &camera, VP, Vec2::ZERO, VP, true).is_empty());
    }

    #[test]
    fn geometry_crossing_the_near_plane_is_clipped_not_dropped() {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Floor", Mesh::plane(40.0, 1), [0.5; 3]);
        let mut camera = front_camera();
        camera.look_from(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -5.0));
        let hits = faces_in_rect(&scene, &camera, VP, Vec2::new(0.0, 600.0), VP, false);
        assert!(!hits.is_empty());
    }
}
