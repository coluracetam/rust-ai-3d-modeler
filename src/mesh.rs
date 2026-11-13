use crate::camera::Ray;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditElementMode {
    Vertex,
    Edge,
    Face,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RayHit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub face_idx: usize,
}

type QPt = (i32, i32, i32);

/// Faces per bounding box in the raycast acceleration list.
const RAY_CHUNK: usize = 64;

/// Derived lookup structures, built lazily and reused until the mesh changes.
struct MeshDerived {
    key: (u64, usize, usize),
    vert_faces: OnceLock<HashMap<QPt, Vec<usize>>>,
    edge_faces: OnceLock<HashMap<(QPt, QPt), Vec<usize>>>,
    bounds: OnceLock<(Vec3, Vec3)>,
    /// Bounding box of every `RAY_CHUNK` consecutive faces.
    chunks: OnceLock<Vec<(Vec3, Vec3)>>,
}

/// Per-mesh cache. Cloning yields an empty cache, so a copy edited elsewhere never inherits
/// stale data; call `Mesh::invalidate_caches` after editing geometry in place.
#[derive(Default)]
pub struct MeshCache(Mutex<Option<Arc<MeshDerived>>>);

impl Clone for MeshCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for MeshCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MeshCache")
    }
}

/// Entry distance of `ray` into an axis-aligned box, if it hits.
pub fn ray_box_entry(ray: &Ray, min: Vec3, max: Vec3) -> Option<f32> {
    let inv = ray.direction.recip();
    let t0 = (min - ray.origin) * inv;
    let t1 = (max - ray.origin) * inv;
    let near = t0.min(t1).max_element();
    let far = t0.max(t1).min_element();
    // NaN (ray parallel to and on a slab) compares false and falls through to "hit".
    if far < near.max(0.0) {
        None
    } else {
        Some(near.max(0.0))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mesh {
    #[serde(with = "crate::codec::vec3s")]
    pub positions: Vec<Vec3>,
    #[serde(with = "crate::codec::vec3s")]
    pub normals: Vec<Vec3>,
    #[serde(with = "crate::codec::rgbs")]
    pub colors: Vec<[f32; 3]>,
    /// Each face is a polygon (typically 3 or 4 vertex indices in counter-clockwise order).
    #[serde(with = "crate::codec::faces")]
    pub faces: Vec<Vec<usize>>,

    #[serde(skip)]
    pub selected_vertices: HashSet<usize>,
    #[serde(skip)]
    pub selected_edges: HashSet<(usize, usize)>,
    #[serde(default)]
    pub selected_faces: HashSet<usize>,
    #[serde(skip)]
    pub version: u64,
    #[serde(skip)]
    cache: MeshCache,
}

impl Mesh {
    fn derived(&self) -> Arc<MeshDerived> {
        let key = (self.version, self.positions.len(), self.faces.len());
        let mut slot = self.cache.0.lock().unwrap_or_else(|e| e.into_inner());
        match &*slot {
            Some(d) if d.key == key => d.clone(),
            _ => {
                let d = Arc::new(MeshDerived {
                    key,
                    vert_faces: OnceLock::new(),
                    edge_faces: OnceLock::new(),
                    bounds: OnceLock::new(),
                    chunks: OnceLock::new(),
                });
                *slot = Some(d.clone());
                d
            }
        }
    }

    /// Drop cached adjacency/bounds after editing geometry without `bump_version`.
    pub fn invalidate_caches(&self) {
        *self.cache.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn with_vert_faces<R>(&self, f: impl FnOnce(&HashMap<QPt, Vec<usize>>) -> R) -> R {
        let d = self.derived();
        f(d.vert_faces.get_or_init(|| {
            let mut map: HashMap<QPt, Vec<usize>> = HashMap::new();
            for (fi, face) in self.faces.iter().enumerate() {
                for &vi in face {
                    if let Some(&p) = self.positions.get(vi) {
                        map.entry(Self::quantize_pos(p)).or_default().push(fi);
                    }
                }
            }
            map
        }))
    }

    fn with_edge_faces<R>(&self, f: impl FnOnce(&HashMap<(QPt, QPt), Vec<usize>>) -> R) -> R {
        let d = self.derived();
        f(d.edge_faces.get_or_init(|| {
            let mut map: HashMap<(QPt, QPt), Vec<usize>> = HashMap::new();
            for (fi, face) in self.faces.iter().enumerate() {
                let n = face.len();
                if n < 3 {
                    continue;
                }
                for i in 0..n {
                    if let (Some(&pa), Some(&pb)) = (
                        self.positions.get(face[i]),
                        self.positions.get(face[(i + 1) % n]),
                    ) {
                        map.entry(Self::edge_key(pa, pb)).or_default().push(fi);
                    }
                }
            }
            map
        }))
    }

    fn edge_key(a: Vec3, b: Vec3) -> (QPt, QPt) {
        let (qa, qb) = (Self::quantize_pos(a), Self::quantize_pos(b));
        if qa <= qb { (qa, qb) } else { (qb, qa) }
    }

    fn chunk_bounds(&self) -> Arc<MeshDerived> {
        let d = self.derived();
        d.chunks.get_or_init(|| {
            self.faces
                .chunks(RAY_CHUNK)
                .map(|chunk| {
                    let mut lo = Vec3::splat(f32::INFINITY);
                    let mut hi = Vec3::splat(f32::NEG_INFINITY);
                    for &vi in chunk.iter().flatten() {
                        if let Some(&p) = self.positions.get(vi) {
                            lo = lo.min(p);
                            hi = hi.max(p);
                        }
                    }
                    (lo, hi)
                })
                .collect()
        });
        d
    }

    pub fn new(positions: Vec<Vec3>, faces: Vec<Vec<usize>>) -> Self {
        let count = positions.len();
        let mut mesh = Self {
            positions,
            normals: vec![Vec3::Y; count],
            colors: vec![[1.0, 1.0, 1.0]; count],
            faces,
            selected_vertices: HashSet::new(),
            selected_edges: HashSet::new(),
            selected_faces: HashSet::new(),
            version: 1,
            cache: MeshCache::default(),
        };
        mesh.recalculate_normals();
        mesh
    }

    #[inline]
    pub fn bump_version(&mut self) {
        static VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(10);
        self.version = VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stable topology identity, independent of selection and rendering caches.
    pub fn topology_hash(&self) -> u64 {
        let mut hash = 14695981039346656037u64;
        for n in std::iter::once(self.positions.len()).chain(
            self.faces
                .iter()
                .flat_map(|f| std::iter::once(f.len()).chain(f.iter().copied())),
        ) {
            for byte in (n as u64).to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(1099511628211);
            }
        }
        hash
    }

    pub fn face_signature(&self, index: usize) -> u64 {
        let mut points: Vec<_> = self.faces[index]
            .iter()
            .map(|&i| {
                let p = self.positions[i] * 100000.0;
                (p.x.round() as i64, p.y.round() as i64, p.z.round() as i64)
            })
            .collect();
        points.sort_unstable();
        let mut hash = 14695981039346656037u64;
        for (x, y, z) in points {
            for v in [x, y, z] {
                for b in v.to_le_bytes() {
                    hash = (hash ^ b as u64).wrapping_mul(1099511628211);
                }
            }
        }
        hash
    }

    /// Connected islands within the selected subset, including split imported vertices.
    pub fn selected_components(&self) -> Vec<Vec<usize>> {
        let mut adjacency: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        let mut remaining: HashSet<_> = self
            .selected_faces
            .iter()
            .copied()
            .filter(|&i| i < self.faces.len())
            .collect();
        for &fi in &remaining {
            for &vi in &self.faces[fi] {
                adjacency
                    .entry(Self::quantize_pos(self.positions[vi]))
                    .or_default()
                    .push(fi);
            }
        }
        let mut result = Vec::new();
        while let Some(&seed) = remaining.iter().min() {
            remaining.remove(&seed);
            let mut group = vec![seed];
            let mut cursor = 0;
            while cursor < group.len() {
                for &vi in &self.faces[group[cursor]] {
                    for &fi in &adjacency[&Self::quantize_pos(self.positions[vi])] {
                        if remaining.remove(&fi) {
                            group.push(fi);
                        }
                    }
                }
                cursor += 1;
            }
            group.sort_unstable();
            result.push(group);
        }
        result
    }

    pub fn canonical_edge(a: usize, b: usize) -> (usize, usize) {
        if a <= b { (a, b) } else { (b, a) }
    }

    pub fn unique_edges(&self) -> Vec<(usize, usize)> {
        let mut set = HashSet::new();
        for face in &self.faces {
            let n = face.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                set.insert(Self::canonical_edge(a, b));
            }
        }
        let mut edges: Vec<_> = set.into_iter().collect();
        edges.sort_unstable();
        edges
    }

    pub fn face_normal(&self, face_idx: usize) -> Vec3 {
        let face = &self.faces[face_idx];
        if face.len() < 3 {
            return Vec3::Y;
        }
        let p0 = self.positions[face[0]];
        let p1 = self.positions[face[1]];
        let p2 = self.positions[face[2]];
        (p1 - p0).cross(p2 - p0).normalize_or_zero()
    }

    pub fn face_center(&self, face_idx: usize) -> Vec3 {
        let face = &self.faces[face_idx];
        if face.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = face.iter().map(|&idx| self.positions[idx]).sum();
        sum / (face.len() as f32)
    }

    pub fn recalculate_normals(&mut self) {
        if self.colors.len() < self.positions.len() {
            self.colors.resize(self.positions.len(), [1.0, 1.0, 1.0]);
        }
        let mut accum = vec![Vec3::ZERO; self.positions.len()];
        for (fi, face) in self.faces.iter().enumerate() {
            let n = self.face_normal(fi);
            for &vi in face {
                if vi < accum.len() {
                    accum[vi] += n;
                }
            }
        }
        self.normals = accum
            .into_iter()
            .map(|n| {
                if n.length_squared() > 1e-8 {
                    n.normalize()
                } else {
                    Vec3::Y
                }
            })
            .collect();
        self.bump_version();
    }

    // =========================================================================
    // PRIMITIVE GENERATORS
    // =========================================================================

    pub fn cube(size: f32) -> Self {
        let h = size * 0.5;
        let positions = vec![
            Vec3::new(-h, -h, h),  // 0
            Vec3::new(h, -h, h),   // 1
            Vec3::new(h, h, h),    // 2
            Vec3::new(-h, h, h),   // 3
            Vec3::new(-h, -h, -h), // 4
            Vec3::new(h, -h, -h),  // 5
            Vec3::new(h, h, -h),   // 6
            Vec3::new(-h, h, -h),  // 7
        ];
        let faces = vec![
            vec![0, 1, 2, 3], // Front (+Z)
            vec![5, 4, 7, 6], // Back (-Z)
            vec![4, 0, 3, 7], // Left (-X)
            vec![1, 5, 6, 2], // Right (+X)
            vec![3, 2, 6, 7], // Top (+Y)
            vec![4, 5, 1, 0], // Bottom (-Y)
        ];
        Self::new(positions, faces)
    }

    pub fn plane(size: f32, subdivisions: usize) -> Self {
        let subs = subdivisions.max(1);
        let half = size * 0.5;
        let step = size / (subs as f32);
        let mut positions = Vec::with_capacity((subs + 1) * (subs + 1));
        for z in 0..=subs {
            for x in 0..=subs {
                let px = -half + (x as f32) * step;
                let pz = -half + (z as f32) * step;
                positions.push(Vec3::new(px, 0.0, pz));
            }
        }
        let mut faces = Vec::with_capacity(subs * subs);
        let row = subs + 1;
        for z in 0..subs {
            for x in 0..subs {
                let i0 = z * row + x;
                let i1 = i0 + 1;
                let i2 = (z + 1) * row + x + 1;
                let i3 = (z + 1) * row + x;
                faces.push(vec![i0, i3, i2, i1]);
            }
        }
        Self::new(positions, faces)
    }

    pub fn sphere(radius: f32, rings: usize, sectors: usize) -> Self {
        let rings = rings.max(4);
        let sectors = sectors.max(6);
        let mut positions = Vec::new();
        let mut faces = Vec::new();

        // Top pole (0)
        positions.push(Vec3::new(0.0, radius, 0.0));

        // Intermediate rings
        for r in 1..rings {
            let phi = std::f32::consts::PI * (r as f32) / (rings as f32);
            let y = radius * phi.cos();
            let ring_r = radius * phi.sin();
            for s in 0..sectors {
                let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sectors as f32);
                let x = ring_r * theta.sin();
                let z = ring_r * theta.cos();
                positions.push(Vec3::new(x, y, z));
            }
        }

        // Bottom pole
        let bottom_idx = positions.len();
        positions.push(Vec3::new(0.0, -radius, 0.0));

        // Top cap triangles
        for s in 0..sectors {
            let next_s = (s + 1) % sectors;
            faces.push(vec![0, 1 + s, 1 + next_s]);
        }

        // Quad rings
        for r in 0..(rings - 2) {
            let ring_start = 1 + r * sectors;
            let next_ring_start = ring_start + sectors;
            for s in 0..sectors {
                let next_s = (s + 1) % sectors;
                faces.push(vec![
                    ring_start + s,
                    next_ring_start + s,
                    next_ring_start + next_s,
                    ring_start + next_s,
                ]);
            }
        }

        // Bottom cap triangles
        let last_ring_start = 1 + (rings - 2) * sectors;
        for s in 0..sectors {
            let next_s = (s + 1) % sectors;
            faces.push(vec![
                last_ring_start + s,
                bottom_idx,
                last_ring_start + next_s,
            ]);
        }

        Self::new(positions, faces)
    }

    pub fn cylinder(radius: f32, height: f32, segments: usize) -> Self {
        let segs = segments.max(6);
        let half_h = height * 0.5;
        let mut positions = Vec::new();
        let mut faces = Vec::new();

        // Top ring: 0..segs
        for s in 0..segs {
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (segs as f32);
            positions.push(Vec3::new(
                radius * theta.sin(),
                half_h,
                radius * theta.cos(),
            ));
        }
        // Bottom ring: segs..2*segs
        for s in 0..segs {
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (segs as f32);
            positions.push(Vec3::new(
                radius * theta.sin(),
                -half_h,
                radius * theta.cos(),
            ));
        }

        // Top center & bottom center for clean caps
        let top_center = positions.len();
        positions.push(Vec3::new(0.0, half_h, 0.0));
        let bottom_center = positions.len();
        positions.push(Vec3::new(0.0, -half_h, 0.0));

        for s in 0..segs {
            let next = (s + 1) % segs;
            // Side quad
            faces.push(vec![s, segs + s, segs + next, next]);
            // Top cap triangle
            faces.push(vec![top_center, s, next]);
            // Bottom cap triangle
            faces.push(vec![bottom_center, segs + next, segs + s]);
        }

        Self::new(positions, faces)
    }

    pub fn torus(
        major_radius: f32,
        minor_radius: f32,
        major_segments: usize,
        minor_segments: usize,
    ) -> Self {
        let maj = major_segments.max(8);
        let min = minor_segments.max(6);
        let mut positions = Vec::with_capacity(maj * min);
        let mut faces = Vec::with_capacity(maj * min);

        for i in 0..maj {
            let u = 2.0 * std::f32::consts::PI * (i as f32) / (maj as f32);
            let cos_u = u.cos();
            let sin_u = u.sin();
            for j in 0..min {
                let v = 2.0 * std::f32::consts::PI * (j as f32) / (min as f32);
                let cos_v = v.cos();
                let sin_v = v.sin();
                let r = major_radius + minor_radius * cos_v;
                let x = r * sin_u;
                let y = minor_radius * sin_v;
                let z = r * cos_u;
                positions.push(Vec3::new(x, y, z));
            }
        }

        for i in 0..maj {
            let next_i = (i + 1) % maj;
            for j in 0..min {
                let next_j = (j + 1) % min;
                let i0 = i * min + j;
                let i1 = next_i * min + j;
                let i2 = next_i * min + next_j;
                let i3 = i * min + next_j;
                faces.push(vec![i0, i1, i2, i3]);
            }
        }

        Self::new(positions, faces)
    }

    // =========================================================================
    // SELECTION & EDIT MODE OPERATIONS
    // =========================================================================

    pub fn clear_selection(&mut self) {
        self.selected_vertices.clear();
        self.selected_edges.clear();
        self.selected_faces.clear();
        self.bump_version();
    }

    pub fn select_all(&mut self, mode: EditElementMode) {
        match mode {
            EditElementMode::Vertex => {
                self.selected_vertices = (0..self.positions.len()).collect();
            }
            EditElementMode::Edge => {
                self.selected_edges = self.unique_edges().into_iter().collect();
            }
            EditElementMode::Face => {
                self.selected_faces = (0..self.faces.len()).collect();
            }
        }
        self.bump_version();
    }

    /// Returns the set of vertex indices affected by the current selection in `mode`.
    pub fn active_vertex_indices(&self, mode: EditElementMode) -> HashSet<usize> {
        let mut verts = HashSet::new();
        match mode {
            EditElementMode::Vertex => {
                verts.extend(&self.selected_vertices);
            }
            EditElementMode::Edge => {
                for &(a, b) in &self.selected_edges {
                    verts.insert(a);
                    verts.insert(b);
                }
            }
            EditElementMode::Face => {
                for &fi in &self.selected_faces {
                    if let Some(face) = self.faces.get(fi) {
                        verts.extend(face.iter().copied());
                    }
                }
            }
        }
        verts
    }

    pub fn selection_centroid(&self, mode: EditElementMode) -> Option<Vec3> {
        let verts = self.active_vertex_indices(mode);
        if verts.is_empty() {
            return None;
        }
        let sum: Vec3 = verts.iter().map(|&vi| self.positions[vi]).sum();
        Some(sum / (verts.len() as f32))
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        if self.positions.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        *self.derived().bounds.get_or_init(|| {
            let mut min_p = Vec3::splat(f32::MAX);
            let mut max_p = Vec3::splat(f32::MIN);
            for &p in &self.positions {
                min_p = min_p.min(p);
                max_p = max_p.max(p);
            }
            (min_p, max_p)
        })
    }

    /// Selects or unselects faces within `radius` of `local_center` (for 3D section brush/pin selection).
    pub fn select_faces_in_sphere(
        &mut self,
        local_center: Vec3,
        radius: f32,
        hit_face_idx: Option<usize>,
        additive: bool,
        subtract: bool,
    ) {
        if !additive && !subtract {
            self.selected_faces.clear();
            self.selected_vertices.clear();
            self.selected_edges.clear();
        }
        let r2 = (radius.max(0.01)) * (radius.max(0.01));
        for fi in 0..self.faces.len() {
            let fc = self.face_center(fi);
            let mut inside = fc.distance_squared(local_center) <= r2;
            if !inside {
                for &vi in &self.faces[fi] {
                    if let Some(&p) = self.positions.get(vi) {
                        if p.distance_squared(local_center) <= r2 {
                            inside = true;
                            break;
                        }
                    }
                }
            }
            if inside {
                if subtract {
                    self.selected_faces.remove(&fi);
                } else {
                    self.selected_faces.insert(fi);
                }
            }
        }
        if let Some(fi) = hit_face_idx {
            if fi < self.faces.len() {
                if subtract {
                    self.selected_faces.remove(&fi);
                } else {
                    self.selected_faces.insert(fi);
                }
            }
        }
        self.bump_version();
    }

    /// Computes `(centroid, avg_normal, bounds_min, bounds_max)` in local space for `selected_faces`.
    pub fn selected_faces_bounds_and_normal(&self) -> Option<(Vec3, Vec3, Vec3, Vec3)> {
        if self.selected_faces.is_empty() {
            return None;
        }
        let mut min_pt = Vec3::splat(f32::MAX);
        let mut max_pt = Vec3::splat(f32::MIN);
        let mut sum_pt = Vec3::ZERO;
        let mut v_count = 0usize;
        let mut sum_norm = Vec3::ZERO;

        for &fi in &self.selected_faces {
            if fi < self.faces.len() {
                sum_norm += self.face_normal(fi);
                for &vi in &self.faces[fi] {
                    if let Some(&p) = self.positions.get(vi) {
                        min_pt = min_pt.min(p);
                        max_pt = max_pt.max(p);
                        sum_pt += p;
                        v_count += 1;
                    }
                }
            }
        }
        if v_count == 0 {
            return None;
        }
        let centroid = sum_pt / (v_count as f32);
        let avg_norm = if sum_norm.length_squared() > 1e-6 {
            sum_norm.normalize()
        } else {
            Vec3::Y
        };
        Some((centroid, avg_norm, min_pt, max_pt))
    }

    #[inline]
    fn quantize_pos(p: Vec3) -> (i32, i32, i32) {
        (
            (p.x * 500.0).round() as i32,
            (p.y * 500.0).round() as i32,
            (p.z * 500.0).round() as i32,
        )
    }

    /// Finds all faces belonging to the connected 3D sub-part (island/component) containing `seed_face_idx`.
    /// Uses spatially welded vertex positions (2mm tolerance) so faceted pieces with split vertices still group as one sub-part.
    pub fn compute_connected_component(&self, seed_face_idx: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        if seed_face_idx >= self.faces.len() {
            return result;
        }

        self.with_vert_faces(|vert_to_faces| {
            let mut queue = std::collections::VecDeque::new();
            result.insert(seed_face_idx);
            queue.push_back(seed_face_idx);

            while let Some(curr) = queue.pop_front() {
                for &vi in &self.faces[curr] {
                    if let Some(&p) = self.positions.get(vi) {
                        if let Some(neighbors) = vert_to_faces.get(&Self::quantize_pos(p)) {
                            for &nb in neighbors {
                                if result.insert(nb) {
                                    queue.push_back(nb);
                                }
                            }
                        }
                    }
                }
            }
        });

        result
    }

    /// Selects or unselects the connected 3D sub-part containing `seed_face_idx`.
    pub fn select_connected_component(
        &mut self,
        seed_face_idx: usize,
        additive: bool,
        subtract: bool,
    ) {
        let comp = self.compute_connected_component(seed_face_idx);
        self.apply_face_set_selection(&comp, additive, subtract);
    }

    /// Finds a contiguous surface patch starting at `seed_face_idx` bounded by sharp creases (`crease_angle_deg`).
    pub fn compute_smart_surface(
        &self,
        seed_face_idx: usize,
        crease_angle_deg: f32,
    ) -> HashSet<usize> {
        let mut result = HashSet::new();
        if seed_face_idx >= self.faces.len() {
            return result;
        }

        let cos_step = crease_angle_deg.clamp(2.0, 90.0).to_radians().cos();
        let cos_seed = (crease_angle_deg * 1.75)
            .clamp(5.0, 88.0)
            .to_radians()
            .cos();
        let seed_normal = self.face_normal(seed_face_idx);

        self.with_edge_faces(|edge_to_faces| {
            let mut queue = std::collections::VecDeque::new();
            result.insert(seed_face_idx);
            queue.push_back(seed_face_idx);

            while let Some(curr) = queue.pop_front() {
                let curr_normal = self.face_normal(curr);
                let face = &self.faces[curr];
                let n = face.len();
                for i in 0..n {
                    let va = face[i];
                    let vb = face[(i + 1) % n];
                    if let (Some(&pa), Some(&pb)) = (self.positions.get(va), self.positions.get(vb))
                    {
                        let key = Self::edge_key(pa, pb);
                        if let Some(neighbors) = edge_to_faces.get(&key) {
                            for &nb in neighbors {
                                if !result.contains(&nb) {
                                    let nb_normal = self.face_normal(nb);
                                    if curr_normal.dot(nb_normal) >= cos_step
                                        && seed_normal.dot(nb_normal) >= cos_seed
                                    {
                                        result.insert(nb);
                                        queue.push_back(nb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        result
    }

    /// Selects or unselects a contiguous crease-bounded surface patch starting at `seed_face_idx`.
    pub fn select_smart_surface(
        &mut self,
        seed_face_idx: usize,
        crease_angle_deg: f32,
        additive: bool,
        subtract: bool,
    ) {
        let patch = self.compute_smart_surface(seed_face_idx, crease_angle_deg);
        self.apply_face_set_selection(&patch, additive, subtract);
    }

    pub fn apply_face_set_selection(
        &mut self,
        faces_set: &HashSet<usize>,
        additive: bool,
        subtract: bool,
    ) {
        if !additive && !subtract {
            self.selected_faces.clear();
            self.selected_vertices.clear();
            self.selected_edges.clear();
        }
        for &fi in faces_set {
            if fi < self.faces.len() {
                if subtract {
                    self.selected_faces.remove(&fi);
                } else {
                    self.selected_faces.insert(fi);
                }
            }
        }
        self.bump_version();
    }

    /// Expands `selected_faces` by 1 ring of adjacent faces across shared vertices.
    pub fn grow_selection(&mut self) -> bool {
        if self.selected_faces.is_empty() {
            return false;
        }
        let before_len = self.selected_faces.len();
        let mut active_qverts = HashSet::new();
        for &fi in &self.selected_faces {
            if let Some(face) = self.faces.get(fi) {
                for &vi in face {
                    if let Some(&p) = self.positions.get(vi) {
                        active_qverts.insert(Self::quantize_pos(p));
                    }
                }
            }
        }
        for (fi, face) in self.faces.iter().enumerate() {
            if self.selected_faces.contains(&fi) {
                continue;
            }
            for &vi in face {
                if let Some(&p) = self.positions.get(vi) {
                    if active_qverts.contains(&Self::quantize_pos(p)) {
                        self.selected_faces.insert(fi);
                        break;
                    }
                }
            }
        }
        self.bump_version();
        self.selected_faces.len() != before_len
    }

    /// Contracts `selected_faces` by removing faces that touch unselected faces.
    pub fn shrink_selection(&mut self) -> bool {
        if self.selected_faces.is_empty() {
            return false;
        }
        let mut unselected_qverts = HashSet::new();
        for (fi, face) in self.faces.iter().enumerate() {
            if !self.selected_faces.contains(&fi) {
                for &vi in face {
                    if let Some(&p) = self.positions.get(vi) {
                        unselected_qverts.insert(Self::quantize_pos(p));
                    }
                }
            }
        }
        let mut to_remove = Vec::new();
        for &fi in &self.selected_faces {
            if let Some(face) = self.faces.get(fi) {
                for &vi in face {
                    if let Some(&p) = self.positions.get(vi) {
                        if unselected_qverts.contains(&Self::quantize_pos(p)) {
                            to_remove.push(fi);
                            break;
                        }
                    }
                }
            }
        }
        let changed = !to_remove.is_empty();
        for fi in to_remove {
            self.selected_faces.remove(&fi);
        }
        self.bump_version();
        changed
    }

    /// Builds 3D highlight wireframe lines for a subset of faces (used for live hover preview of sub-parts/surfaces).
    pub fn build_face_subset_lines(
        &self,
        face_indices: &HashSet<usize>,
        model_matrix: Mat4,
        color: [f32; 3],
    ) -> Vec<GpuVertex> {
        let mut seen_edges = HashSet::new();
        let mut lines = Vec::with_capacity(face_indices.len() * 6);
        for &fi in face_indices {
            let Some(face) = self.faces.get(fi) else {
                continue;
            };
            let n = face.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                let edge = Self::canonical_edge(a, b);
                if seen_edges.insert(edge) {
                    let pa = model_matrix.transform_point3(self.positions[a]);
                    let pb = model_matrix.transform_point3(self.positions[b]);
                    lines.push(GpuVertex {
                        position: pa.to_array(),
                        normal: [0.0, 1.0, 0.0],
                        color,
                    });
                    lines.push(GpuVertex {
                        position: pb.to_array(),
                        normal: [0.0, 1.0, 0.0],
                        color,
                    });
                }
            }
        }
        lines
    }

    pub fn translate_selected(&mut self, delta: Vec3, mode: EditElementMode) {
        let verts = self.active_vertex_indices(mode);
        for vi in verts {
            if let Some(p) = self.positions.get_mut(vi) {
                *p += delta;
            }
        }
        self.recalculate_normals();
    }

    pub fn scale_selected(&mut self, scale: Vec3, mode: EditElementMode) {
        let Some(center) = self.selection_centroid(mode) else {
            return;
        };
        let verts = self.active_vertex_indices(mode);
        for vi in verts {
            if let Some(p) = self.positions.get_mut(vi) {
                *p = center + (*p - center) * scale;
            }
        }
        self.recalculate_normals();
    }

    pub fn rotate_selected(&mut self, rotation: Quat, mode: EditElementMode) {
        let Some(center) = self.selection_centroid(mode) else {
            return;
        };
        let verts = self.active_vertex_indices(mode);
        for vi in verts {
            if let Some(p) = self.positions.get_mut(vi) {
                *p = center + rotation * (*p - center);
            }
        }
        self.recalculate_normals();
    }

    /// Extrude all selected faces along their average normal (or individual face normals) by `distance`.
    /// Creates connecting side-wall quads along the boundary edges of the selected face region.
    pub fn extrude_selected_faces(&mut self, distance: f32) -> bool {
        if self.selected_faces.is_empty() {
            return false;
        }

        let selected: Vec<usize> = self.selected_faces.iter().copied().collect();

        // Compute average normal of selected faces
        let mut avg_normal = Vec3::ZERO;
        for &fi in &selected {
            if fi < self.faces.len() {
                avg_normal += self.face_normal(fi);
            }
        }
        avg_normal = if avg_normal.length_squared() > 1e-6 {
            avg_normal.normalize()
        } else {
            Vec3::Y
        };

        // Count directed edges `(u, v)` in the selected faces.
        // An edge `(u, v)` is on the region boundary if its reverse `(v, u)` is NOT in the selected faces.
        let mut directed_edges = HashSet::new();
        for &fi in &selected {
            let face = &self.faces[fi];
            let n = face.len();
            for i in 0..n {
                directed_edges.insert((face[i], face[(i + 1) % n]));
            }
        }

        let mut boundary_edges = Vec::new();
        for &(u, v) in &directed_edges {
            if !directed_edges.contains(&(v, u)) {
                boundary_edges.push((u, v));
            }
        }

        // Duplicate all vertices belonging to the selected faces
        let mut vert_map: HashMap<usize, usize> = HashMap::new();
        for &fi in &selected {
            let face = &self.faces[fi];
            for &vi in face {
                vert_map.entry(vi).or_insert_with(|| {
                    let new_idx = self.positions.len();
                    let new_pos = self.positions[vi] + avg_normal * distance;
                    let color = self.colors.get(vi).copied().unwrap_or([0.78, 0.80, 0.84]);
                    self.positions.push(new_pos);
                    self.normals.push(avg_normal);
                    self.colors.push(color);
                    new_idx
                });
            }
        }

        // Update selected faces to use the new extruded vertices
        for &fi in &selected {
            for vi in &mut self.faces[fi] {
                if let Some(&new_vi) = vert_map.get(vi) {
                    *vi = new_vi;
                }
            }
        }

        // Create side walls for each boundary edge `(u, v)`
        for (u, v) in boundary_edges {
            let u_new = vert_map[&u];
            let v_new = vert_map[&v];
            self.faces.push(vec![u, v, v_new, u_new]);
        }

        self.recalculate_normals();
        true
    }

    /// Inset each selected face by `inset_factor` (in `0.05..0.95`) and optional `depth` offset along normal.
    pub fn inset_selected_faces(&mut self, inset_factor: f32, depth: f32) -> bool {
        if self.selected_faces.is_empty() {
            return false;
        }
        let factor = inset_factor.clamp(0.05, 0.95);
        let selected: Vec<usize> = self.selected_faces.iter().copied().collect();

        for fi in selected {
            if fi >= self.faces.len() {
                continue;
            }
            let old_face = self.faces[fi].clone();
            if old_face.len() < 3 {
                continue;
            }
            let center = self.face_center(fi);
            let normal = self.face_normal(fi);

            let mut new_ring = Vec::with_capacity(old_face.len());
            for &vi in &old_face {
                let p = self.positions[vi];
                let inset_pos = p.lerp(center, factor) + normal * depth;
                let color = self.colors.get(vi).copied().unwrap_or([0.78, 0.80, 0.84]);
                let new_vi = self.positions.len();
                self.positions.push(inset_pos);
                self.normals.push(normal);
                self.colors.push(color);
                new_ring.push(new_vi);
            }

            // Replace original face with the inner inset face
            self.faces[fi] = new_ring.clone();

            // Connect outer ring to inner ring with quad border faces
            let n = old_face.len();
            for i in 0..n {
                let next = (i + 1) % n;
                self.faces.push(vec![
                    old_face[i],
                    old_face[next],
                    new_ring[next],
                    new_ring[i],
                ]);
            }
        }

        self.recalculate_normals();
        true
    }

    /// Loop cut / Subdivide:
    /// If faces are selected (or all faces if none selected), subdivides quads into 4 quads and triangles into 3 quads / 4 triangles,
    /// inserting edge midpoints (loop cuts across quad strips).
    pub fn subdivide_or_loop_cut(&mut self) -> bool {
        if self.faces.is_empty() {
            return false;
        }

        let target_faces: HashSet<usize> = if self.selected_faces.is_empty() {
            (0..self.faces.len()).collect()
        } else {
            self.selected_faces.clone()
        };

        let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();
        let mut get_midpoint = |mesh: &mut Self, a: usize, b: usize| -> usize {
            let key = Self::canonical_edge(a, b);
            if let Some(&idx) = edge_midpoints.get(&key) {
                return idx;
            }
            let mid_pos = (mesh.positions[a] + mesh.positions[b]) * 0.5;
            let ca = mesh.colors.get(a).copied().unwrap_or([0.78, 0.80, 0.84]);
            let cb = mesh.colors.get(b).copied().unwrap_or([0.78, 0.80, 0.84]);
            let mid_col = [
                (ca[0] + cb[0]) * 0.5,
                (ca[1] + cb[1]) * 0.5,
                (ca[2] + cb[2]) * 0.5,
            ];
            let new_idx = mesh.positions.len();
            mesh.positions.push(mid_pos);
            mesh.normals.push(Vec3::Y);
            mesh.colors.push(mid_col);
            edge_midpoints.insert(key, new_idx);
            new_idx
        };

        let mut new_faces = Vec::with_capacity(self.faces.len() * 4);
        let mut new_selected_faces = HashSet::new();

        for fi in 0..self.faces.len() {
            let face = self.faces[fi].clone();
            if !target_faces.contains(&fi) || face.len() < 3 {
                new_faces.push(face);
                continue;
            }

            let n = face.len();
            let center_pos = self.face_center(fi);
            let center_idx = self.positions.len();
            let c0 = self
                .colors
                .get(face[0])
                .copied()
                .unwrap_or([0.78, 0.80, 0.84]);
            self.positions.push(center_pos);
            self.normals.push(Vec3::Y);
            self.colors.push(c0);

            let mut mids = Vec::with_capacity(n);
            for i in 0..n {
                let a = face[i];
                let b = face[(i + 1) % n];
                mids.push(get_midpoint(self, a, b));
            }

            for i in 0..n {
                let prev = (i + n - 1) % n;
                let sub_face_idx = new_faces.len();
                new_faces.push(vec![face[i], mids[i], center_idx, mids[prev]]);
                if !self.selected_faces.is_empty() {
                    new_selected_faces.insert(sub_face_idx);
                }
            }
        }

        self.faces = new_faces;
        self.selected_faces = new_selected_faces;
        self.recalculate_normals();
        true
    }

    /// Delete currently selected faces (or faces incident to selected vertices/edges).
    pub fn delete_selected(&mut self, mode: EditElementMode) -> bool {
        let before = self.faces.len();
        match mode {
            EditElementMode::Face => {
                if self.selected_faces.is_empty() {
                    return false;
                }
                let selected = self.selected_faces.clone();
                self.faces = self
                    .faces
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        if !selected.contains(&i) {
                            Some(f.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                self.selected_faces.clear();
            }
            EditElementMode::Edge => {
                if self.selected_edges.is_empty() {
                    return false;
                }
                let sel_edges = self.selected_edges.clone();
                self.faces.retain(|f| {
                    let n = f.len();
                    for i in 0..n {
                        let e = Self::canonical_edge(f[i], f[(i + 1) % n]);
                        if sel_edges.contains(&e) {
                            return false;
                        }
                    }
                    true
                });
                self.selected_edges.clear();
            }
            EditElementMode::Vertex => {
                if self.selected_vertices.is_empty() {
                    return false;
                }
                let sel_verts = self.selected_vertices.clone();
                self.faces
                    .retain(|f| !f.iter().any(|vi| sel_verts.contains(vi)));
                self.selected_vertices.clear();
            }
        }
        self.recalculate_normals();
        self.faces.len() != before
    }

    // =========================================================================
    // RAYCASTING & PICKING
    // =========================================================================

    /// Möller–Trumbore ray-triangle intersection
    fn ray_triangle_intersect(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, Vec3)> {
        let eps = 1e-6;
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = ray.direction.cross(edge2);
        let a = edge1.dot(h);
        if a.abs() < eps {
            return None;
        }
        let f = 1.0 / a;
        let s = ray.origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(edge1);
        let v = f * ray.direction.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * edge2.dot(q);
        if t > eps {
            let normal = edge1.cross(edge2).normalize_or_zero();
            Some((t, normal))
        } else {
            None
        }
    }

    /// Intersect a local-space ray with the mesh. Returns the closest `RayHit`.
    pub fn ray_intersect(&self, local_ray: &Ray) -> Option<RayHit> {
        self.ray_intersect_filtered(local_ray, |_| true)
    }

    pub fn ray_intersect_filtered(
        &self,
        local_ray: &Ray,
        visible: impl Fn(Vec3) -> bool,
    ) -> Option<RayHit> {
        if self.faces.is_empty() {
            return None;
        }
        let (lo, hi) = self.bounds();
        let pad = (hi - lo).abs() * 1e-4 + Vec3::splat(1e-5);
        ray_box_entry(local_ray, lo - pad, hi + pad)?;
        let mut closest: Option<RayHit> = None;
        let derived = self.chunk_bounds();
        let chunks = derived.chunks.get().expect("chunk bounds built");

        for (ci, &(lo, hi)) in chunks.iter().enumerate() {
            let pad = (hi - lo).abs() * 1e-4 + Vec3::splat(1e-5);
            match ray_box_entry(local_ray, lo - pad, hi + pad) {
                Some(t) if closest.as_ref().is_none_or(|c| t <= c.distance) => {}
                _ => continue,
            }
            let start = ci * RAY_CHUNK;
            for (offset, face) in self.faces[start..(start + RAY_CHUNK).min(self.faces.len())]
                .iter()
                .enumerate()
            {
                let fi = start + offset;
                if face.len() < 3 {
                    continue;
                }
                let v0 = self.positions[face[0]];
                for i in 1..(face.len() - 1) {
                    let v1 = self.positions[face[i]];
                    let v2 = self.positions[face[i + 1]];
                    if let Some((t, normal)) = Self::ray_triangle_intersect(local_ray, v0, v1, v2) {
                        if visible(local_ray.at(t))
                            && closest.as_ref().map_or(true, |c| t < c.distance)
                        {
                            closest = Some(RayHit {
                                distance: t,
                                point: local_ray.at(t),
                                normal,
                                face_idx: fi,
                            });
                        }
                    }
                }
            }
        }

        closest
    }

    /// Pick the closest vertex to `local_ray` within `max_radius`.
    pub fn pick_vertex(&self, local_ray: &Ray, max_radius: f32) -> Option<usize> {
        let mut best_vi = None;
        let mut best_dist = max_radius;
        for (vi, &pos) in self.positions.iter().enumerate() {
            let to_pt = pos - local_ray.origin;
            let t = to_pt.dot(local_ray.direction);
            if t < 0.0 {
                continue;
            }
            let closest_pt = local_ray.at(t);
            let perp_dist = (pos - closest_pt).length();
            if perp_dist < best_dist {
                best_dist = perp_dist;
                best_vi = Some(vi);
            }
        }
        best_vi
    }

    /// Pick the closest edge to `local_ray` within `max_radius`.
    pub fn pick_edge(&self, local_ray: &Ray, max_radius: f32) -> Option<(usize, usize)> {
        let mut best_edge = None;
        let mut best_dist = max_radius;

        for (a, b) in self.unique_edges() {
            let pa = self.positions[a];
            let pb = self.positions[b];
            let seg = pb - pa;
            let seg_len = seg.length();
            if seg_len < 1e-6 {
                continue;
            }
            let seg_dir = seg / seg_len;
            let t_seg = local_ray.closest_t_on_axis(pa, seg_dir).clamp(0.0, seg_len);
            let pt_on_seg = pa + seg_dir * t_seg;
            let t_ray = (pt_on_seg - local_ray.origin).dot(local_ray.direction);
            if t_ray < 0.0 {
                continue;
            }
            let pt_on_ray = local_ray.at(t_ray);
            let dist = (pt_on_seg - pt_on_ray).length();
            if dist < best_dist {
                best_dist = dist;
                best_edge = Some((a, b));
            }
        }
        best_edge
    }

    // =========================================================================
    // AI VERTEX COLOR BAKING (AMBIENT OCCLUSION, CAVITY WEAR & GRADIENTS)
    // =========================================================================

    pub fn reset_vertex_colors(&mut self, color: [f32; 3]) {
        self.colors = vec![color; self.positions.len()];
        self.bump_version();
    }

    /// Bakes Ambient Occlusion (crevice darkening + ground shadowing) and Curvature Edge Wear
    /// into the mesh's vertex colors so surface details pop immediately.
    pub fn bake_ao_and_curvature(
        &mut self,
        ao_strength: f32,
        wear_strength: f32,
        crevice_color: [f32; 3],
        wear_color: [f32; 3],
    ) {
        let n_verts = self.positions.len();
        if n_verts == 0 {
            return;
        }
        if self.colors.len() < n_verts {
            self.colors.resize(n_verts, [1.0, 1.0, 1.0]);
        }

        // Build 1-ring neighbor adjacency
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
        for face in &self.faces {
            let fn_len = face.len();
            for i in 0..fn_len {
                let a = face[i];
                let b = face[(i + 1) % fn_len];
                if a < n_verts && b < n_verts {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }
        }

        // Bounding box for vertical ground-proximity AO
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for &p in &self.positions {
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        let span_y = (max_y - min_y).max(0.001);

        for vi in 0..n_verts {
            let p = self.positions[vi];
            let n = self.normals.get(vi).copied().unwrap_or(Vec3::Y);
            let neighbors = &adj[vi];

            let mut curvature = 0.0f32;
            if !neighbors.is_empty() {
                let mut avg = Vec3::ZERO;
                let mut avg_edge_len = 0.0f32;
                for &nb in neighbors {
                    avg += self.positions[nb];
                    avg_edge_len += p.distance(self.positions[nb]);
                }
                avg /= neighbors.len() as f32;
                avg_edge_len = (avg_edge_len / (neighbors.len() as f32)).max(1e-4);
                // Positive when vertex sticks out (convex ridge), negative in concave valley
                curvature = ((p - avg).dot(n) / avg_edge_len).clamp(-1.0, 1.0);
            }

            // Ground/downward normal occlusion factor
            let height_t = ((p.y - min_y) / span_y).clamp(0.0, 1.0);
            let down_occ = ((-n.y).max(0.0) * 0.45 + (1.0 - height_t) * 0.25).clamp(0.0, 1.0);
            let cavity_occ = ((-curvature).max(0.0) * 1.6 + down_occ).clamp(0.0, 1.0);
            let ao_w = (cavity_occ * ao_strength).clamp(0.0, 0.92);

            let ridge_w = (curvature.max(0.0) * 1.8 * wear_strength).clamp(0.0, 0.90);

            let mut c = self.colors[vi];
            // Blend toward dark crevice AO color
            c[0] = c[0] * (1.0 - ao_w) + crevice_color[0] * ao_w;
            c[1] = c[1] * (1.0 - ao_w) + crevice_color[1] * ao_w;
            c[2] = c[2] * (1.0 - ao_w) + crevice_color[2] * ao_w;

            // Blend convex ridges toward edge-wear highlight color
            c[0] = c[0] * (1.0 - ridge_w) + wear_color[0] * ridge_w;
            c[1] = c[1] * (1.0 - ridge_w) + wear_color[1] * ridge_w;
            c[2] = c[2] * (1.0 - ridge_w) + wear_color[2] * ridge_w;

            self.colors[vi] = c;
        }

        self.bump_version();
    }

    /// Paints a vertical height gradient (`bottom_color` -> `top_color`) and optional directional normal coat
    /// (such as moss, snow, or dust on upward-facing surfaces).
    pub fn paint_gradient_and_normal_coat(
        &mut self,
        bottom_color: [f32; 3],
        top_color: [f32; 3],
        gradient_blend: f32,
        top_coat_color: Option<[f32; 3]>,
        top_coat_threshold: f32,
        top_coat_strength: f32,
    ) {
        let n_verts = self.positions.len();
        if n_verts == 0 {
            return;
        }
        if self.colors.len() < n_verts {
            self.colors.resize(n_verts, [1.0, 1.0, 1.0]);
        }

        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for &p in &self.positions {
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        let span_y = (max_y - min_y).max(0.001);
        let g_blend = gradient_blend.clamp(0.0, 1.0);

        for vi in 0..n_verts {
            let p = self.positions[vi];
            let n = self.normals.get(vi).copied().unwrap_or(Vec3::Y);
            let t = ((p.y - min_y) / span_y).clamp(0.0, 1.0);

            let grad = [
                bottom_color[0] + (top_color[0] - bottom_color[0]) * t,
                bottom_color[1] + (top_color[1] - bottom_color[1]) * t,
                bottom_color[2] + (top_color[2] - bottom_color[2]) * t,
            ];

            let mut c = self.colors[vi];
            c[0] = c[0] * (1.0 - g_blend) + grad[0] * g_blend;
            c[1] = c[1] * (1.0 - g_blend) + grad[1] * g_blend;
            c[2] = c[2] * (1.0 - g_blend) + grad[2] * g_blend;

            if let Some(coat) = top_coat_color {
                if n.y > top_coat_threshold {
                    let ny_w = ((n.y - top_coat_threshold) / (1.0 - top_coat_threshold).max(0.05))
                        .clamp(0.0, 1.0)
                        * top_coat_strength.clamp(0.0, 1.0);
                    c[0] = c[0] * (1.0 - ny_w) + coat[0] * ny_w;
                    c[1] = c[1] * (1.0 - ny_w) + coat[1] * ny_w;
                    c[2] = c[2] * (1.0 - ny_w) + coat[2] * ny_w;
                }
            }

            self.colors[vi] = c;
        }

        self.bump_version();
    }

    // =========================================================================
    // GPU BUFFER GENERATION
    // =========================================================================

    pub fn build_gpu_triangles(&self, base_color: [f32; 3], flat_shading: bool) -> Vec<GpuVertex> {
        let mut out = Vec::with_capacity(self.faces.len() * 6);
        for (fi, face) in self.faces.iter().enumerate() {
            if face.len() < 3 {
                continue;
            }
            let is_face_selected = self.selected_faces.contains(&fi);
            let face_n = self.face_normal(fi);

            let v0_idx = face[0];
            for i in 1..(face.len() - 1) {
                let v1_idx = face[i];
                let v2_idx = face[i + 1];
                for &vi in &[v0_idx, v1_idx, v2_idx] {
                    let pos = self.positions[vi].to_array();
                    let norm = if flat_shading {
                        face_n.to_array()
                    } else {
                        self.normals.get(vi).copied().unwrap_or(face_n).to_array()
                    };
                    let vc = self.colors.get(vi).copied().unwrap_or([1.0, 1.0, 1.0]);
                    let mut col = [
                        vc[0] * base_color[0],
                        vc[1] * base_color[1],
                        vc[2] * base_color[2],
                    ];
                    if is_face_selected {
                        // Studio accent (teal), strong enough to read on any material.
                        col = [
                            col[0] * 0.52 + 0.48 * 0.30,
                            col[1] * 0.52 + 0.48 * 0.92,
                            col[2] * 0.52 + 0.48 * 0.78,
                        ];
                    }
                    out.push(GpuVertex {
                        position: pos,
                        normal: norm,
                        color: col,
                    });
                }
            }
        }
        out
    }

    /// Border edges of the selected face region (welded by position), in local space.
    pub fn build_selection_outline(&self, color: [f32; 3]) -> Vec<GpuVertex> {
        let mut directed: HashMap<((i32, i32, i32), (i32, i32, i32)), (usize, usize)> =
            HashMap::new();
        for &fi in &self.selected_faces {
            let Some(face) = self.faces.get(fi) else { continue };
            for i in 0..face.len() {
                let (a, b) = (face[i], face[(i + 1) % face.len()]);
                let (qa, qb) = (
                    Self::quantize_pos(self.positions[a]),
                    Self::quantize_pos(self.positions[b]),
                );
                if qa != qb {
                    directed.insert((qa, qb), (a, b));
                }
            }
        }
        let mut lines = Vec::new();
        for (&(qa, qb), &(a, b)) in &directed {
            if directed.contains_key(&(qb, qa)) {
                continue;
            }
            for v in [a, b] {
                lines.push(GpuVertex {
                    position: self.positions[v].to_array(),
                    normal: [0.0, 1.0, 0.0],
                    color,
                });
            }
        }
        lines
    }

    /// Builds line segments (`GpuVertex` pairs) for wireframe and selection highlights in world space.
    pub fn build_wireframe_lines(
        &self,
        model_matrix: Mat4,
        default_color: [f32; 3],
        highlight_selection: bool,
    ) -> Vec<GpuVertex> {
        let edges = self.unique_edges();
        let mut lines = Vec::with_capacity(edges.len() * 2);
        for (a, b) in edges {
            let pa = model_matrix.transform_point3(self.positions[a]);
            let pb = model_matrix.transform_point3(self.positions[b]);
            let is_sel = highlight_selection
                && (self.selected_edges.contains(&(a, b))
                    || (self.selected_vertices.contains(&a)
                        && self.selected_vertices.contains(&b)));
            let color = if is_sel {
                [1.0, 0.65, 0.1]
            } else {
                default_color
            };
            lines.push(GpuVertex {
                position: pa.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
            lines.push(GpuVertex {
                position: pb.to_array(),
                normal: [0.0, 1.0, 0.0],
                color,
            });
        }
        lines
    }
}

/// What `Mesh::select_similar` compares against the faces already selected.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SimilarBy {
    /// Faces pointing the same way (tolerance in degrees).
    Normal,
    /// Faces of about the same size (tolerance as a fraction, 0.15 = within 15%).
    Area,
    /// Faces with about the same vertex colour (tolerance as RGB distance).
    Color,
}

impl SimilarBy {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "normal" | "direction" => Ok(Self::Normal),
            "area" | "size" => Ok(Self::Area),
            "color" | "colour" => Ok(Self::Color),
            other => Err(format!("Unknown similarity '{other}'. Use normal, area or color.")),
        }
    }

    pub fn default_tolerance(self) -> f32 {
        match self {
            Self::Normal => 8.0,
            Self::Area => 0.15,
            Self::Color => 0.08,
        }
    }
}

// =========================================================================
// SELECTION QUERIES (face picking, similarity, paths, volumes)
// =========================================================================
impl Mesh {
    pub fn face_area(&self, fi: usize) -> f32 {
        let Some(face) = self.faces.get(fi) else { return 0.0 };
        if face.len() < 3 {
            return 0.0;
        }
        let p0 = self.positions[face[0]];
        let mut sum = Vec3::ZERO;
        for i in 1..face.len() - 1 {
            sum += (self.positions[face[i]] - p0).cross(self.positions[face[i + 1]] - p0);
        }
        sum.length() * 0.5
    }

    /// Average vertex colour of a face (white when the mesh has no vertex colours).
    pub fn face_color(&self, fi: usize) -> [f32; 3] {
        let Some(face) = self.faces.get(fi) else { return [1.0; 3] };
        let mut sum = Vec3::ZERO;
        for &vi in face {
            sum += Vec3::from_array(self.colors.get(vi).copied().unwrap_or([1.0; 3]));
        }
        (sum / face.len().max(1) as f32).to_array()
    }

    /// Select every unselected face and deselect the rest.
    pub fn invert_selection(&mut self) {
        let inverted: HashSet<usize> = (0..self.faces.len())
            .filter(|i| !self.selected_faces.contains(i))
            .collect();
        self.selected_faces = inverted;
        self.selected_vertices.clear();
        self.selected_edges.clear();
        self.bump_version();
    }

    /// Faces that resemble any currently selected face. Does not change the selection.
    pub fn compute_similar(&self, by: SimilarBy, tolerance: f32) -> HashSet<usize> {
        const MAX_REFERENCES: usize = 48;
        let mut out = HashSet::new();
        let selected: Vec<usize> = {
            let mut v: Vec<usize> = self
                .selected_faces
                .iter()
                .copied()
                .filter(|&i| i < self.faces.len())
                .collect();
            v.sort_unstable();
            v
        };
        if selected.is_empty() {
            return out;
        }
        match by {
            SimilarBy::Normal => {
                let min_dot = tolerance.clamp(0.0, 180.0).to_radians().cos();
                let mut refs: Vec<Vec3> = Vec::new();
                for &fi in &selected {
                    let n = self.face_normal(fi);
                    if n.length_squared() > 0.5 && !refs.iter().any(|r| r.dot(n) > 0.9995) {
                        refs.push(n);
                        if refs.len() >= MAX_REFERENCES {
                            break;
                        }
                    }
                }
                for fi in 0..self.faces.len() {
                    let n = self.face_normal(fi);
                    if refs.iter().any(|r| r.dot(n) >= min_dot) {
                        out.insert(fi);
                    }
                }
            }
            SimilarBy::Area => {
                let tol = tolerance.max(0.0);
                let mut refs: Vec<f32> = Vec::new();
                for &fi in &selected {
                    let a = self.face_area(fi);
                    if a > 1e-10 && !refs.iter().any(|r| (r - a).abs() <= r * 0.01) {
                        refs.push(a);
                        if refs.len() >= MAX_REFERENCES {
                            break;
                        }
                    }
                }
                for fi in 0..self.faces.len() {
                    let a = self.face_area(fi);
                    if refs.iter().any(|r| (r - a).abs() <= r * tol) {
                        out.insert(fi);
                    }
                }
            }
            SimilarBy::Color => {
                let mut refs: Vec<Vec3> = Vec::new();
                for &fi in &selected {
                    let c = Vec3::from_array(self.face_color(fi));
                    if !refs.iter().any(|r| r.distance(c) < 0.01) {
                        refs.push(c);
                        if refs.len() >= MAX_REFERENCES {
                            break;
                        }
                    }
                }
                for fi in 0..self.faces.len() {
                    let c = Vec3::from_array(self.face_color(fi));
                    if refs.iter().any(|r| r.distance(c) <= tolerance) {
                        out.insert(fi);
                    }
                }
            }
        }
        out
    }

    /// Add every face that resembles the current selection. Returns how many were added.
    pub fn select_similar(&mut self, by: SimilarBy, tolerance: f32) -> usize {
        let similar = self.compute_similar(by, tolerance);
        let before = self.selected_faces.len();
        self.selected_faces.extend(similar);
        self.bump_version();
        self.selected_faces.len() - before
    }

    /// Faces sharing an edge with `fi` (welded by position).
    pub fn face_neighbors(&self, fi: usize) -> Vec<usize> {
        let Some(face) = self.faces.get(fi) else { return Vec::new() };
        let n = face.len();
        self.with_edge_faces(|edges| {
            let mut out = Vec::new();
            for i in 0..n {
                let key = Self::edge_key(self.positions[face[i]], self.positions[face[(i + 1) % n]]);
                if let Some(list) = edges.get(&key) {
                    for &other in list {
                        if other != fi && !out.contains(&other) {
                            out.push(other);
                        }
                    }
                }
            }
            out
        })
    }

    /// Shortest chain of edge-adjacent faces from `from` to `to`, both included.
    pub fn face_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        if from >= self.faces.len() || to >= self.faces.len() {
            return None;
        }
        if from == to {
            return Some(vec![from]);
        }
        let mut dist: HashMap<usize, f32> = HashMap::new();
        let mut prev: HashMap<usize, usize> = HashMap::new();
        // Distances are non-negative, so their bit patterns order the same way.
        let mut heap = BinaryHeap::new();
        dist.insert(from, 0.0);
        heap.push(Reverse((0u32, from)));
        while let Some(Reverse((bits, fi))) = heap.pop() {
            let d = f32::from_bits(bits);
            if fi == to {
                let mut path = vec![to];
                let mut cur = to;
                while let Some(&p) = prev.get(&cur) {
                    path.push(p);
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            if d > dist.get(&fi).copied().unwrap_or(f32::INFINITY) {
                continue;
            }
            let centre = self.face_center(fi);
            for next in self.face_neighbors(fi) {
                let nd = d + centre.distance(self.face_center(next));
                if nd < dist.get(&next).copied().unwrap_or(f32::INFINITY) {
                    dist.insert(next, nd);
                    prev.insert(next, fi);
                    heap.push(Reverse((nd.to_bits(), next)));
                }
            }
        }
        None
    }

    /// Faces whose centre, moved to world space by `model`, lies inside the world box.
    pub fn faces_in_world_box(&self, model: Mat4, min: Vec3, max: Vec3) -> HashSet<usize> {
        (0..self.faces.len())
            .filter(|&fi| {
                if self.faces[fi].len() < 3 {
                    return false;
                }
                let c = model.transform_point3(self.face_center(fi));
                c.cmpge(min).all() && c.cmple(max).all()
            })
            .collect()
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn brute_force(mesh: &Mesh, ray: &Ray) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (fi, face) in mesh.faces.iter().enumerate() {
            for i in 1..face.len() - 1 {
                let (a, b, c) = (
                    mesh.positions[face[0]],
                    mesh.positions[face[i]],
                    mesh.positions[face[i + 1]],
                );
                if let Some((t, _)) = Mesh::ray_triangle_intersect(ray, a, b, c) {
                    if best.is_none_or(|(_, bt)| t < bt) {
                        best = Some((fi, t));
                    }
                }
            }
        }
        best
    }

    #[test]
    fn chunked_raycast_matches_brute_force() {
        let mesh = Mesh::sphere(1.0, 48, 64);
        assert!(mesh.faces.len() > RAY_CHUNK * 10);
        for i in 0..400 {
            let a = i as f32 * 0.37;
            let origin = Vec3::new(a.cos() * 4.0, (a * 0.7).sin() * 3.0, a.sin() * 4.0);
            let target = Vec3::new((a * 1.3).sin() * 0.9, (a * 2.1).cos() * 0.9, 0.0);
            let ray = Ray::new(origin, target - origin);
            let fast = mesh.ray_intersect(&ray).map(|h| (h.face_idx, h.distance));
            let slow = brute_force(&mesh, &ray);
            match (fast, slow) {
                (Some((_, t1)), Some((_, t2))) => assert!((t1 - t2).abs() < 1e-4),
                (None, None) => {}
                other => panic!("mismatch for ray {i}: {other:?}"),
            }
        }
        let miss = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::X);
        assert!(mesh.ray_intersect(&miss).is_none());
    }

    #[test]
    fn caches_follow_geometry_edits() {
        let mut mesh = Mesh::cube(2.0);
        let down = Ray::new(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y);
        assert!((mesh.ray_intersect(&down).unwrap().distance - 4.0).abs() < 1e-4);
        assert_eq!(mesh.bounds().1, Vec3::ONE);
        assert_eq!(mesh.compute_connected_component(0).len(), 6);

        // In-place edit without a version bump: explicit invalidation refreshes everything.
        for p in &mut mesh.positions {
            *p *= 2.0;
        }
        mesh.invalidate_caches();
        assert_eq!(mesh.bounds().1, Vec3::splat(2.0));
        assert!((mesh.ray_intersect(&down).unwrap().distance - 3.0).abs() < 1e-4);

        // Topology changes change the cache key on their own.
        let base = mesh.positions.len();
        mesh.positions.extend(
            Mesh::cube(1.0)
                .positions
                .iter()
                .map(|p| *p + Vec3::X * 10.0),
        );
        mesh.faces.extend(
            Mesh::cube(1.0)
                .faces
                .iter()
                .map(|f| f.iter().map(|v| v + base).collect()),
        );
        assert_eq!(mesh.compute_connected_component(0).len(), 6);
        assert_eq!(mesh.compute_connected_component(6).len(), 6);
        assert!(mesh.bounds().1.x > 10.0);

        // Clones never share a cache.
        let clone = mesh.clone();
        mesh.positions[0] = Vec3::splat(-50.0);
        mesh.invalidate_caches();
        assert!(clone.bounds().0.x > -10.0);
        assert!(mesh.bounds().0.x <= -50.0);
    }
}
