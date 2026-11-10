use crate::mesh::Mesh;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SculptBrush {
    Draw,
    Inflate,
    Smooth,
    Flatten,
    Grab,
    Pinch,
    Crease,
}

impl SculptBrush {
    pub fn label(&self) -> &'static str {
        match self {
            SculptBrush::Draw => "Draw / Clay",
            SculptBrush::Inflate => "Inflate",
            SculptBrush::Smooth => "Smooth",
            SculptBrush::Flatten => "Flatten",
            SculptBrush::Grab => "Grab",
            SculptBrush::Pinch => "Pinch",
            SculptBrush::Crease => "Crease",
        }
    }

    pub fn all() -> &'static [SculptBrush] {
        &[
            SculptBrush::Draw,
            SculptBrush::Inflate,
            SculptBrush::Smooth,
            SculptBrush::Flatten,
            SculptBrush::Grab,
            SculptBrush::Pinch,
            SculptBrush::Crease,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SculptSettings {
    pub brush: SculptBrush,
    pub radius: f32,
    pub strength: f32,
    pub invert: bool,
    pub symmetry_x: bool,
    pub symmetry_y: bool,
    pub symmetry_z: bool,
    pub dyntopo_enabled: bool,
    pub dyntopo_detail_size: f32,
    pub voxel_remesh_resolution: usize,
}

impl Default for SculptSettings {
    fn default() -> Self {
        Self {
            brush: SculptBrush::Draw,
            radius: 0.45,
            strength: 0.35,
            invert: false,
            symmetry_x: false,
            symmetry_y: false,
            symmetry_z: false,
            dyntopo_enabled: false,
            dyntopo_detail_size: 0.18,
            voxel_remesh_resolution: 32,
        }
    }
}

/// Smooth polynomial falloff in `[0..1]` for normalized distance `t = dist / radius`.
#[inline]
pub fn brush_falloff(t: f32) -> f32 {
    if t >= 1.0 {
        0.0
    } else {
        let u = 1.0 - t * t;
        u * u
    }
}

/// Apply a sculpt stroke (including any active mirror symmetry axes) to `mesh` in local space.
pub fn apply_sculpt_stroke(
    mesh: &mut Mesh,
    local_hit: Vec3,
    local_normal: Vec3,
    grab_delta: Vec3,
    settings: &SculptSettings,
) {
    let mut mirror_signs = vec![Vec3::ONE];
    if settings.symmetry_x {
        let current = mirror_signs.clone();
        for s in current {
            mirror_signs.push(Vec3::new(-s.x, s.y, s.z));
        }
    }
    if settings.symmetry_y {
        let current = mirror_signs.clone();
        for s in current {
            mirror_signs.push(Vec3::new(s.x, -s.y, s.z));
        }
    }
    if settings.symmetry_z {
        let current = mirror_signs.clone();
        for s in current {
            mirror_signs.push(Vec3::new(s.x, s.y, -s.z));
        }
    }

    for sign in mirror_signs {
        let center = local_hit * sign;
        let normal = (local_normal * sign).normalize_or_zero();
        let delta = grab_delta * sign;

        if settings.dyntopo_enabled && settings.brush != SculptBrush::Grab {
            subdivide_edges_in_sphere(mesh, center, settings.radius, settings.dyntopo_detail_size);
        }

        apply_single_sculpt_dab(mesh, center, normal, delta, settings);
    }

    mesh.recalculate_normals();
}

fn apply_single_sculpt_dab(
    mesh: &mut Mesh,
    center: Vec3,
    normal: Vec3,
    grab_delta: Vec3,
    settings: &SculptSettings,
) {
    let radius = settings.radius.max(0.01);
    let r2 = radius * radius;
    let dir_sign = if settings.invert { -1.0 } else { 1.0 };
    let strength = settings.strength * 0.15;

    match settings.brush {
        SculptBrush::Draw => {
            for pos in &mut mesh.positions {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let w = brush_falloff(d2.sqrt() / radius);
                    *pos += normal * (dir_sign * strength * radius * w);
                }
            }
        }
        SculptBrush::Inflate => {
            for (i, pos) in mesh.positions.iter_mut().enumerate() {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let w = brush_falloff(d2.sqrt() / radius);
                    let vn = mesh.normals.get(i).copied().unwrap_or(normal);
                    *pos += vn * (dir_sign * strength * radius * w);
                }
            }
        }
        SculptBrush::Smooth => {
            // Build vertex neighbor adjacency for vertices inside the brush sphere
            let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
            for face in &mesh.faces {
                let n = face.len();
                for i in 0..n {
                    let a = face[i];
                    let b = face[(i + 1) % n];
                    if mesh.positions[a].distance_squared(center) < r2 {
                        neighbors.entry(a).or_default().push(b);
                    }
                    if mesh.positions[b].distance_squared(center) < r2 {
                        neighbors.entry(b).or_default().push(a);
                    }
                }
            }
            let old_positions = mesh.positions.clone();
            let smooth_factor = (settings.strength * 0.65).clamp(0.05, 0.95);
            for (vi, adj) in neighbors {
                if adj.is_empty() {
                    continue;
                }
                let d = old_positions[vi].distance(center);
                let w = brush_falloff(d / radius) * smooth_factor;
                let avg: Vec3 = adj.iter().map(|&ni| old_positions[ni]).sum::<Vec3>() / (adj.len() as f32);
                mesh.positions[vi] = old_positions[vi].lerp(avg, w);
            }
        }
        SculptBrush::Flatten => {
            // Compute average plane of vertices within radius
            let mut sum_pos = Vec3::ZERO;
            let mut sum_norm = Vec3::ZERO;
            let mut count = 0usize;
            for (i, &pos) in mesh.positions.iter().enumerate() {
                if pos.distance_squared(center) < r2 {
                    sum_pos += pos;
                    sum_norm += mesh.normals.get(i).copied().unwrap_or(normal);
                    count += 1;
                }
            }
            if count == 0 {
                return;
            }
            let plane_pt = sum_pos / (count as f32);
            let plane_n = if sum_norm.length_squared() > 1e-6 {
                sum_norm.normalize()
            } else {
                normal
            };
            let flat_strength = (settings.strength * 0.5).clamp(0.05, 1.0);
            for pos in &mut mesh.positions {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let w = brush_falloff(d2.sqrt() / radius) * flat_strength;
                    let signed_dist = (*pos - plane_pt).dot(plane_n);
                    let projected = *pos - plane_n * signed_dist;
                    *pos = pos.lerp(projected, w);
                }
            }
        }
        SculptBrush::Grab => {
            for pos in &mut mesh.positions {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let w = brush_falloff(d2.sqrt() / radius);
                    *pos += grab_delta * w;
                }
            }
        }
        SculptBrush::Pinch => {
            let pinch_strength = (settings.strength * 0.35 * dir_sign).clamp(-0.8, 0.8);
            for pos in &mut mesh.positions {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let w = brush_falloff(d2.sqrt() / radius) * pinch_strength;
                    *pos = pos.lerp(center, w);
                }
            }
        }
        SculptBrush::Crease => {
            for pos in &mut mesh.positions {
                let d2 = pos.distance_squared(center);
                if d2 < r2 {
                    let t = d2.sqrt() / radius;
                    let w = brush_falloff(t);
                    let sharp = w * w;
                    // Pull toward center and push/carve along normal
                    let to_center = center - *pos;
                    *pos += to_center * (sharp * settings.strength * 0.25)
                        - normal * (dir_sign * sharp * strength * radius * 0.9);
                }
            }
        }
    }
}

/// Dynamic Topology (Dyntopo): Splits triangles/quads inside the brush sphere whose edges exceed `max_edge_len`.
pub fn subdivide_edges_in_sphere(
    mesh: &mut Mesh,
    center: Vec3,
    radius: f32,
    max_edge_len: f32,
) {
    let r2 = radius * radius;
    let max_l2 = max_edge_len.max(0.03).powi(2);

    let mut midpoint_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_faces = Vec::with_capacity(mesh.faces.len() + 32);

    let mut get_mid = |positions: &mut Vec<Vec3>,
                       normals: &mut Vec<Vec3>,
                       colors: &mut Vec<[f32; 3]>,
                       a: usize,
                       b: usize|
     -> usize {
        let key = Mesh::canonical_edge(a, b);
        if let Some(&idx) = midpoint_cache.get(&key) {
            return idx;
        }
        let mid_pos = (positions[a] + positions[b]) * 0.5;
        let mid_norm = (normals[a] + normals[b]).normalize_or_zero();
        let ca = colors.get(a).copied().unwrap_or([0.78, 0.80, 0.84]);
        let cb = colors.get(b).copied().unwrap_or([0.78, 0.80, 0.84]);
        let mid_col = [
            (ca[0] + cb[0]) * 0.5,
            (ca[1] + cb[1]) * 0.5,
            (ca[2] + cb[2]) * 0.5,
        ];
        let idx = positions.len();
        positions.push(mid_pos);
        normals.push(mid_norm);
        colors.push(mid_col);
        midpoint_cache.insert(key, idx);
        idx
    };

    // Triangulate and split faces within the sphere that have long edges
    let old_faces = std::mem::take(&mut mesh.faces);
    for face in old_faces {
        if face.len() < 3 {
            continue;
        }
        let f_center: Vec3 =
            face.iter().map(|&vi| mesh.positions[vi]).sum::<Vec3>() / (face.len() as f32);
        let in_sphere = f_center.distance_squared(center) < r2
            || face
                .iter()
                .any(|&vi| mesh.positions[vi].distance_squared(center) < r2);

        if !in_sphere {
            new_faces.push(face);
            continue;
        }

        // Fan-triangulate the polygon first, then split long triangle edges
        let v0 = face[0];
        for i in 1..(face.len() - 1) {
            let a = v0;
            let b = face[i];
            let c = face[i + 1];

            let ab_long = mesh.positions[a].distance_squared(mesh.positions[b]) > max_l2;
            let bc_long = mesh.positions[b].distance_squared(mesh.positions[c]) > max_l2;
            let ca_long = mesh.positions[c].distance_squared(mesh.positions[a]) > max_l2;

            if ab_long || bc_long || ca_long {
                let mab = get_mid(&mut mesh.positions, &mut mesh.normals, &mut mesh.colors, a, b);
                let mbc = get_mid(&mut mesh.positions, &mut mesh.normals, &mut mesh.colors, b, c);
                let mca = get_mid(&mut mesh.positions, &mut mesh.normals, &mut mesh.colors, c, a);
                new_faces.push(vec![a, mab, mca]);
                new_faces.push(vec![mab, b, mbc]);
                new_faces.push(vec![mbc, c, mca]);
                new_faces.push(vec![mab, mbc, mca]);
            } else {
                new_faces.push(vec![a, b, c]);
            }
        }
    }

    mesh.faces = new_faces;
}

// =============================================================================
// VOXEL GRID & SURFACE NETS / BLOCKY MESHER
// =============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoxelMeshingMode {
    SmoothSurfaceNets,
    BlockyVoxels,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoxelBrushOp {
    AddSphere,
    SubtractSphere,
    AddBox,
    SubtractBox,
    SmoothField,
    PaintColor,
}

impl VoxelBrushOp {
    pub fn label(&self) -> &'static str {
        match self {
            VoxelBrushOp::AddSphere => "Add Voxel Sphere",
            VoxelBrushOp::SubtractSphere => "Carve Voxel Sphere",
            VoxelBrushOp::AddBox => "Add Voxel Box",
            VoxelBrushOp::SubtractBox => "Carve Voxel Box",
            VoxelBrushOp::SmoothField => "Smooth Voxel Field",
            VoxelBrushOp::PaintColor => "Paint Voxel Color",
        }
    }

    pub fn all() -> &'static [VoxelBrushOp] {
        &[
            VoxelBrushOp::AddSphere,
            VoxelBrushOp::SubtractSphere,
            VoxelBrushOp::AddBox,
            VoxelBrushOp::SubtractBox,
            VoxelBrushOp::SmoothField,
            VoxelBrushOp::PaintColor,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGrid {
    pub dims: [usize; 3],
    pub voxel_size: f32,
    /// Signed distance / density values (`< 0.0` is inside/solid, `> 0.0` is outside/empty).
    pub sdf: Vec<f32>,
    pub colors: Vec<[f32; 3]>,
    pub meshing_mode: VoxelMeshingMode,
}

impl VoxelGrid {
    pub fn new_sphere(resolution: usize, world_size: f32, radius: f32) -> Self {
        let res = resolution.clamp(8, 96);
        let voxel_size = world_size / (res as f32);
        let total = res * res * res;
        let sdf = vec![1.0; total];
        let colors = vec![[0.82, 0.76, 0.70]; total];

        let mut grid = Self {
            dims: [res, res, res],
            voxel_size,
            sdf,
            colors,
            meshing_mode: VoxelMeshingMode::SmoothSurfaceNets,
        };

        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let p = grid.cell_pos(x, y, z);
                    let idx = grid.idx(x, y, z);
                    grid.sdf[idx] = p.length() - radius;
                }
            }
        }
        grid
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    #[inline]
    pub fn cell_pos(&self, x: usize, y: usize, z: usize) -> Vec3 {
        let half = Vec3::new(
            self.dims[0] as f32,
            self.dims[1] as f32,
            self.dims[2] as f32,
        ) * 0.5;
        (Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) - half) * self.voxel_size
    }

    #[inline]
    pub fn get_sdf(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 1.0;
        }
        self.sdf[self.idx(x, y, z)]
    }

    /// Apply a volumetric CSG / brush operation at `local_center` with `radius`.
    pub fn apply_brush(
        &mut self,
        op: VoxelBrushOp,
        local_center: Vec3,
        radius: f32,
        brush_color: [f32; 3],
        symmetry_x: bool,
    ) {
        let centers = if symmetry_x && local_center.x.abs() > 1e-4 {
            vec![
                local_center,
                Vec3::new(-local_center.x, local_center.y, local_center.z),
            ]
        } else {
            vec![local_center]
        };

        for center in centers {
            self.apply_single_brush(op, center, radius, brush_color);
        }
    }

    fn apply_single_brush(
        &mut self,
        op: VoxelBrushOp,
        center: Vec3,
        radius: f32,
        brush_color: [f32; 3],
    ) {
        let r = radius.max(self.voxel_size * 1.2);
        let [nx, ny, nz] = self.dims;

        if op == VoxelBrushOp::SmoothField {
            let old = self.sdf.clone();
            let r2 = (r * 1.25).powi(2);
            for z in 1..(nz.saturating_sub(1)) {
                for y in 1..(ny.saturating_sub(1)) {
                    for x in 1..(nx.saturating_sub(1)) {
                        let p = self.cell_pos(x, y, z);
                        if p.distance_squared(center) <= r2 {
                            let avg = (old[self.idx(x - 1, y, z)]
                                + old[self.idx(x + 1, y, z)]
                                + old[self.idx(x, y - 1, z)]
                                + old[self.idx(x, y + 1, z)]
                                + old[self.idx(x, y, z - 1)]
                                + old[self.idx(x, y, z + 1)])
                                / 6.0;
                            let id = self.idx(x, y, z);
                            self.sdf[id] = old[id] * 0.4 + avg * 0.6;
                        }
                    }
                }
            }
            return;
        }

        // Keep a 1-voxel border empty so surfaces remain closed
        for z in 1..(nz.saturating_sub(1)) {
            for y in 1..(ny.saturating_sub(1)) {
                for x in 1..(nx.saturating_sub(1)) {
                    let p = self.cell_pos(x, y, z);
                    let id = self.idx(x, y, z);
                    match op {
                        VoxelBrushOp::AddSphere => {
                            let d = p.distance(center) - r;
                            if d < self.sdf[id] {
                                self.sdf[id] = d;
                                self.colors[id] = brush_color;
                            }
                        }
                        VoxelBrushOp::SubtractSphere => {
                            let d = p.distance(center) - r;
                            self.sdf[id] = self.sdf[id].max(-d);
                        }
                        VoxelBrushOp::AddBox => {
                            let q = (p - center).abs() - Vec3::splat(r * 0.85);
                            let d = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);
                            if d < self.sdf[id] {
                                self.sdf[id] = d;
                                self.colors[id] = brush_color;
                            }
                        }
                        VoxelBrushOp::SubtractBox => {
                            let q = (p - center).abs() - Vec3::splat(r * 0.85);
                            let d = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);
                            self.sdf[id] = self.sdf[id].max(-d);
                        }
                        VoxelBrushOp::PaintColor => {
                            let d = p.distance(center);
                            if d <= r {
                                let w = brush_falloff(d / r) * 0.5;
                                let c = &mut self.colors[id];
                                c[0] = c[0] * (1.0 - w) + brush_color[0] * w;
                                c[1] = c[1] * (1.0 - w) + brush_color[1] * w;
                                c[2] = c[2] * (1.0 - w) + brush_color[2] * w;
                            }
                        }
                        VoxelBrushOp::SmoothField => {}
                    }
                }
            }
        }
    }

    pub fn extract_mesh(&self) -> Mesh {
        match self.meshing_mode {
            VoxelMeshingMode::SmoothSurfaceNets => self.extract_surface_nets_mesh(),
            VoxelMeshingMode::BlockyVoxels => self.extract_blocky_mesh(),
        }
    }

    /// Dual Surface Nets iso-surface extraction (`sdf == 0.0`):
    /// Produces a clean, smooth quad mesh with adaptive vertex placement inside active dual cells.
    pub fn extract_surface_nets_mesh(&self) -> Mesh {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return Mesh::default();
        }

        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let mut cell_to_vert: HashMap<(usize, usize, usize), usize> = HashMap::new();

        let corner_offsets = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ];

        let cell_edges = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        // 1. Place dual vertex in every `(x, y, z)` cell that straddles the 0-isosurface
        for z in 0..(nz - 1) {
            for y in 0..(ny - 1) {
                for x in 0..(nx - 1) {
                    let mut vals = [0.0f32; 8];
                    let mut pos8 = [Vec3::ZERO; 8];
                    let mut mask = 0u8;

                    for (i, &(dx, dy, dz)) in corner_offsets.iter().enumerate() {
                        let v = self.get_sdf(x + dx, y + dy, z + dz);
                        vals[i] = v;
                        pos8[i] = self.cell_pos(x + dx, y + dy, z + dz);
                        if v < 0.0 {
                            mask |= 1 << i;
                        }
                    }

                    if mask == 0 || mask == 0xFF {
                        continue;
                    }

                    // Average zero-crossing points across the 12 cell edges
                    let mut avg_pt = Vec3::ZERO;
                    let mut crossings = 0usize;
                    for &(a, b) in &cell_edges {
                        let va = vals[a];
                        let vb = vals[b];
                        if (va < 0.0) != (vb < 0.0) {
                            let denom = (va - vb).abs().max(1e-6);
                            let t = (va.abs() / denom).clamp(0.0, 1.0);
                            avg_pt += pos8[a].lerp(pos8[b], t);
                            crossings += 1;
                        }
                    }

                    let dual_pt = if crossings > 0 {
                        avg_pt / (crossings as f32)
                    } else {
                        (pos8[0] + pos8[7]) * 0.5
                    };

                    let v_idx = positions.len();
                    positions.push(dual_pt);
                    colors.push(self.colors[self.idx(x, y, z)]);
                    cell_to_vert.insert((x, y, z), v_idx);
                }
            }
        }

        // 2. Emit quads for every interior grid edge that changes sign
        let mut faces = Vec::new();

        for z in 1..(nz - 1) {
            for y in 1..(ny - 1) {
                for x in 1..(nx - 1) {
                    let v0 = self.get_sdf(x, y, z);

                    // X-axis edge from (x, y, z) to (x + 1, y, z)
                    let vx = self.get_sdf(x + 1, y, z);
                    if (v0 < 0.0) != (vx < 0.0) {
                        if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                            cell_to_vert.get(&(x, y - 1, z - 1)),
                            cell_to_vert.get(&(x, y, z - 1)),
                            cell_to_vert.get(&(x, y, z)),
                            cell_to_vert.get(&(x, y - 1, z)),
                        ) {
                            if v0 < 0.0 {
                                faces.push(vec![c0, c1, c2, c3]);
                            } else {
                                faces.push(vec![c0, c3, c2, c1]);
                            }
                        }
                    }

                    // Y-axis edge from (x, y, z) to (x, y + 1, z)
                    let vy = self.get_sdf(x, y + 1, z);
                    if (v0 < 0.0) != (vy < 0.0) {
                        if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                            cell_to_vert.get(&(x - 1, y, z - 1)),
                            cell_to_vert.get(&(x - 1, y, z)),
                            cell_to_vert.get(&(x, y, z)),
                            cell_to_vert.get(&(x, y, z - 1)),
                        ) {
                            if v0 < 0.0 {
                                faces.push(vec![c0, c1, c2, c3]);
                            } else {
                                faces.push(vec![c0, c3, c2, c1]);
                            }
                        }
                    }

                    // Z-axis edge from (x, y, z) to (x, y, z + 1)
                    let vz = self.get_sdf(x, y, z + 1);
                    if (v0 < 0.0) != (vz < 0.0) {
                        if let (Some(&c0), Some(&c1), Some(&c2), Some(&c3)) = (
                            cell_to_vert.get(&(x - 1, y - 1, z)),
                            cell_to_vert.get(&(x, y - 1, z)),
                            cell_to_vert.get(&(x, y, z)),
                            cell_to_vert.get(&(x - 1, y, z)),
                        ) {
                            if v0 < 0.0 {
                                faces.push(vec![c0, c1, c2, c3]);
                            } else {
                                faces.push(vec![c0, c3, c2, c1]);
                            }
                        }
                    }
                }
            }
        }

        let mut mesh = Mesh::new(positions, faces);
        mesh.colors = colors;
        mesh.recalculate_normals();
        mesh
    }

    /// Blocky cubic voxel mesh generator with hidden-face culling.
    pub fn extract_blocky_mesh(&self) -> Mesh {
        let [nx, ny, nz] = self.dims;
        let h = self.voxel_size * 0.5;
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let mut faces = Vec::new();

        let dirs: [([isize; 3], [Vec3; 4]); 6] = [
            (
                [1, 0, 0],
                [
                    Vec3::new(h, -h, h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, h, -h),
                    Vec3::new(h, h, h),
                ],
            ),
            (
                [-1, 0, 0],
                [
                    Vec3::new(-h, -h, -h),
                    Vec3::new(-h, -h, h),
                    Vec3::new(-h, h, h),
                    Vec3::new(-h, h, -h),
                ],
            ),
            (
                [0, 1, 0],
                [
                    Vec3::new(-h, h, h),
                    Vec3::new(h, h, h),
                    Vec3::new(h, h, -h),
                    Vec3::new(-h, h, -h),
                ],
            ),
            (
                [0, -1, 0],
                [
                    Vec3::new(-h, -h, -h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, -h, h),
                    Vec3::new(-h, -h, h),
                ],
            ),
            (
                [0, 0, 1],
                [
                    Vec3::new(-h, -h, h),
                    Vec3::new(h, -h, h),
                    Vec3::new(h, h, h),
                    Vec3::new(-h, h, h),
                ],
            ),
            (
                [0, 0, -1],
                [
                    Vec3::new(h, -h, -h),
                    Vec3::new(-h, -h, -h),
                    Vec3::new(-h, h, -h),
                    Vec3::new(h, h, -h),
                ],
            ),
        ];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if self.get_sdf(x, y, z) >= 0.0 {
                        continue;
                    }
                    let center = self.cell_pos(x, y, z);
                    let col = self.colors[self.idx(x, y, z)];

                    for (d, quad_offsets) in &dirs {
                        let nx_i = x as isize + d[0];
                        let ny_i = y as isize + d[1];
                        let nz_i = z as isize + d[2];
                        let neighbor_solid = nx_i >= 0
                            && ny_i >= 0
                            && nz_i >= 0
                            && (nx_i as usize) < nx
                            && (ny_i as usize) < ny
                            && (nz_i as usize) < nz
                            && self.get_sdf(nx_i as usize, ny_i as usize, nz_i as usize) < 0.0;

                        if !neighbor_solid {
                            let base = positions.len();
                            for &off in quad_offsets {
                                positions.push(center + off);
                                colors.push(col);
                            }
                            faces.push(vec![base, base + 1, base + 2, base + 3]);
                        }
                    }
                }
            }
        }

        let mut mesh = Mesh::new(positions, faces);
        mesh.colors = colors;
        mesh.recalculate_normals();
        mesh
    }
}

/// Voxel Remesh: Converts an arbitrary polygon `Mesh` into a signed distance field at `resolution`
/// and extracts a clean, uniform quad mesh via Dual Surface Nets.
pub fn voxel_remesh(mesh: &Mesh, resolution: usize) -> Mesh {
    if mesh.positions.is_empty() || mesh.faces.is_empty() {
        return mesh.clone();
    }

    let res = resolution.clamp(12, 64);

    // Compute bounding box
    let mut min_pt = Vec3::splat(f32::MAX);
    let mut max_pt = Vec3::splat(f32::MIN);
    for &p in &mesh.positions {
        min_pt = min_pt.min(p);
        max_pt = max_pt.max(p);
    }
    let extent = (max_pt - min_pt).max_element().max(0.2);
    let center = (min_pt + max_pt) * 0.5;
    let padded_size = extent * 1.35;
    let voxel_size = padded_size / (res as f32);

    // Pre-extract triangles with normals for fast point-to-triangle signed distance evaluation
    let mut tris = Vec::with_capacity(mesh.faces.len() * 2);
    for (fi, face) in mesh.faces.iter().enumerate() {
        if face.len() < 3 {
            continue;
        }
        let fn_norm = mesh.face_normal(fi);
        let p0 = mesh.positions[face[0]];
        for i in 1..(face.len() - 1) {
            let p1 = mesh.positions[face[i]];
            let p2 = mesh.positions[face[i + 1]];
            tris.push((p0, p1, p2, fn_norm));
        }
    }

    let total = res * res * res;
    let mut sdf = vec![1.0f32; total];
    let colors = vec![[0.78, 0.80, 0.84]; total];

    let mut grid = VoxelGrid {
        dims: [res, res, res],
        voxel_size,
        sdf: Vec::new(),
        colors,
        meshing_mode: VoxelMeshingMode::SmoothSurfaceNets,
    };

    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let p = grid.cell_pos(x, y, z) + center;
                let mut best_d2 = f32::MAX;
                let mut sign = 1.0f32;

                for &(a, b, c, n) in &tris {
                    let closest = closest_point_on_triangle(p, a, b, c);
                    let diff = p - closest;
                    let d2 = diff.length_squared();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        sign = if diff.dot(n) >= 0.0 { 1.0 } else { -1.0 };
                    }
                }

                let idx = (z * res + y) * res + x;
                sdf[idx] = best_d2.sqrt() * sign;
            }
        }
    }

    grid.sdf = sdf;
    let mut remeshed = grid.extract_surface_nets_mesh();
    for p in &mut remeshed.positions {
        *p += center;
    }
    remeshed.recalculate_normals();
    remeshed
}

fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}
