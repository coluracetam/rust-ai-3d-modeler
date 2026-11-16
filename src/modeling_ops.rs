//! Modelling operations exposed to agents beyond the basic kernel: profile extrusion, lofting,
//! bridging, hole filling, welding, deformers, subdivision smoothing, decimation, scattering,
//! per-face painting and mesh health checks.
use crate::mesh::Mesh;
use glam::{Mat4, Quat, Vec2, Vec3};
use std::collections::{HashMap, HashSet};

fn finish(mut mesh: Mesh, color: [f32; 3]) -> Mesh {
    mesh.colors = vec![color; mesh.positions.len()].into();
    mesh.recalculate_normals();
    mesh.bump_version();
    mesh
}

fn flip_faces(mesh: &mut Mesh) {
    for f in &mut mesh.faces {
        f.reverse();
    }
}

/// Make a closed solid face outward.
fn orient_outward(mesh: &mut Mesh) {
    if crate::modeling::signed_volume(mesh) < 0.0 {
        flip_faces(mesh);
    }
}

fn ensure_colors(mesh: &mut Mesh) {
    if mesh.colors.len() != mesh.positions.len() {
        mesh.colors.resize(mesh.positions.len(), [1.0; 3]);
    }
}

// ---------------------------------------------------------------------------
// Polygon triangulation (ear clipping)
// ---------------------------------------------------------------------------

fn polygon_area(points: &[Vec2]) -> f32 {
    let n = points.len();
    (0..n)
        .map(|i| points[i].perp_dot(points[(i + 1) % n]))
        .sum::<f32>()
        * 0.5
}

/// Triangulate a simple polygon (convex or concave). Triangles keep the polygon's winding.
pub fn triangulate_polygon(points: &[Vec2]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let ccw = polygon_area(points) >= 0.0;
    let mut idx: Vec<usize> = if ccw { (0..n).collect() } else { (0..n).rev().collect() };
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (pa, pb, pc) = (points[a], points[b], points[c]);
            if (pb - pa).perp_dot(pc - pb) <= 1e-12 {
                continue; // reflex or degenerate corner
            }
            let inside = idx.iter().any(|&k| {
                if k == a || k == b || k == c {
                    return false;
                }
                let p = points[k];
                (pb - pa).perp_dot(p - pa) >= 0.0
                    && (pc - pb).perp_dot(p - pb) >= 0.0
                    && (pa - pc).perp_dot(p - pc) >= 0.0
            });
            if !inside {
                tris.push([a, b, c]);
                idx.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Degenerate input: fall back to a fan so the cap is still closed.
            break;
        }
    }
    for i in 1..idx.len() - 1 {
        tris.push([idx[0], idx[i], idx[i + 1]]);
    }
    if !ccw {
        for t in &mut tris {
            t.swap(1, 2);
        }
    }
    tris
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// Extrude a 2D outline (`[x, z]` points on the ground plane) upward along Y by `height`.
/// `top_scale` tapers the top about the outline's centroid; `twist_deg` rotates it.
pub fn extrude_profile(
    profile: &[Vec2],
    height: f32,
    top_scale: f32,
    twist_deg: f32,
    segments: usize,
    cap: bool,
    color: [f32; 3],
) -> Result<Mesh, String> {
    let mut outline: Vec<Vec2> = profile.to_vec();
    if outline.len() >= 2 && outline[0].distance(*outline.last().unwrap()) < 1e-6 {
        outline.pop();
    }
    if outline.len() < 3 {
        return Err("profile needs at least 3 [x, z] points".into());
    }
    if outline.iter().any(|p| !p.is_finite()) || !height.is_finite() || height.abs() < 1e-6 {
        return Err("profile points and height must be finite, and height non-zero".into());
    }
    if polygon_area(&outline).abs() < 1e-9 {
        return Err("profile has no area".into());
    }
    let n = outline.len();
    let segments = if twist_deg.abs() > 1e-3 { segments.clamp(2, 256) } else { segments.clamp(1, 256) };
    let centroid = outline.iter().copied().sum::<Vec2>() / n as f32;
    let mut positions = Vec::with_capacity(n * (segments + 1));
    for s in 0..=segments {
        let t = s as f32 / segments as f32;
        let scale = 1.0 + (top_scale - 1.0) * t;
        let (sin, cos) = (twist_deg.to_radians() * t).sin_cos();
        for p in &outline {
            let d = (*p - centroid) * scale;
            let r = Vec2::new(d.x * cos - d.y * sin, d.x * sin + d.y * cos) + centroid;
            positions.push(Vec3::new(r.x, height * t, r.y));
        }
    }
    let mut faces = Vec::new();
    for s in 0..segments {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (s * n + i, s * n + j);
            faces.push(vec![a, b, b + n, a + n]);
        }
    }
    if cap {
        let top = segments * n;
        for t in triangulate_polygon(&outline) {
            // Opposite to the side walls along every shared edge.
            faces.push(vec![t[0], t[2], t[1]]);
            faces.push(vec![top + t[0], top + t[1], top + t[2]]);
        }
    }
    let mut mesh = Mesh::new(positions, faces);
    if cap {
        orient_outward(&mut mesh);
    }
    Ok(finish(mesh, color))
}

/// Skin a surface through a series of cross-sections, each a ring of the same number of points.
pub fn loft(
    sections: &[Vec<Vec3>],
    cap: bool,
    closed: bool,
    color: [f32; 3],
) -> Result<Mesh, String> {
    if sections.len() < 2 {
        return Err("loft needs at least 2 sections".into());
    }
    let n = sections[0].len();
    if n < 3 {
        return Err("each section needs at least 3 points".into());
    }
    if sections.iter().any(|s| s.len() != n) {
        return Err(format!(
            "every section must have the same number of points (the first has {n})"
        ));
    }
    if sections.iter().flatten().any(|p| !p.is_finite()) {
        return Err("section points must be finite [x, y, z] triples".into());
    }
    let count = sections.len();
    let positions: Vec<Vec3> = sections.iter().flatten().copied().collect();
    let mut faces = Vec::new();
    let spans = if closed { count } else { count - 1 };
    for s in 0..spans {
        let next = (s + 1) % count;
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push(vec![s * n + i, s * n + j, next * n + j, next * n + i]);
        }
    }
    let mut mesh = Mesh::new(positions, faces);
    if cap && !closed {
        for (s, reverse) in [(0, true), (count - 1, false)] {
            let ring: Vec<usize> = (0..n).map(|i| s * n + i).collect();
            let centre = ring.iter().map(|&i| mesh.positions[i]).sum::<Vec3>() / n as f32;
            let c = mesh.positions.len();
            mesh.positions.push(centre);
            for i in 0..n {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                mesh.faces.push(if reverse { vec![c, b, a] } else { vec![c, a, b] });
            }
        }
    }
    if closed || cap {
        orient_outward(&mut mesh);
    }
    Ok(finish(mesh, color))
}

// ---------------------------------------------------------------------------
// Topology edits
// ---------------------------------------------------------------------------

/// Merge vertices closer than `tolerance` and drop faces that collapse. Returns vertices removed.
pub fn weld(mesh: &mut Mesh, tolerance: f32) -> usize {
    let tol = tolerance.max(1e-6);
    let key = |p: Vec3| {
        (
            (p.x / tol).round() as i64,
            (p.y / tol).round() as i64,
            (p.z / tol).round() as i64,
        )
    };
    ensure_colors(mesh);
    let before = mesh.positions.len();
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut remap = vec![0usize; before];
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    for (i, &p) in mesh.positions.iter().enumerate() {
        let id = *map.entry(key(p)).or_insert_with(|| {
            positions.push(p);
            colors.push(mesh.colors[i]);
            positions.len() - 1
        });
        remap[i] = id;
    }
    let mut selected = HashSet::new();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for (fi, face) in mesh.faces.iter().enumerate() {
        let mut f: Vec<usize> = Vec::with_capacity(face.len());
        for &vi in face {
            let v = remap[vi];
            if f.last() != Some(&v) {
                f.push(v);
            }
        }
        if f.len() > 1 && f.first() == f.last() {
            f.pop();
        }
        if f.len() >= 3 {
            if mesh.selected_faces.contains(&fi) {
                selected.insert(faces.len());
            }
            faces.push(f);
        }
    }
    mesh.positions = positions.into();
    mesh.colors = colors.into();
    mesh.faces = faces.into();
    mesh.selected_faces = selected;
    mesh.selected_vertices.clear();
    mesh.selected_edges.clear();
    mesh.recalculate_normals();
    mesh.bump_version();
    before - mesh.positions.len()
}

/// Close open boundaries with new faces. Returns how many holes were filled.
pub fn fill_holes(mesh: &mut Mesh) -> usize {
    weld(mesh, 1e-5);
    let mut directed: HashSet<(usize, usize)> = HashSet::new();
    for face in &mesh.faces {
        for i in 0..face.len() {
            directed.insert((face[i], face[(i + 1) % face.len()]));
        }
    }
    // A boundary edge a->b has no partner b->a; the cap runs the other way round.
    let mut next: HashMap<usize, usize> = HashMap::new();
    for &(a, b) in &directed {
        if !directed.contains(&(b, a)) {
            next.insert(b, a);
        }
    }
    ensure_colors(mesh);
    let mut filled = 0;
    let mut visited: HashSet<usize> = HashSet::new();
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut ring = vec![start];
        visited.insert(start);
        let mut cur = start;
        let mut closed = false;
        while let Some(&n) = next.get(&cur) {
            if n == start {
                closed = true;
                break;
            }
            if !visited.insert(n) {
                break;
            }
            ring.push(n);
            cur = n;
        }
        if !closed || ring.len() < 3 {
            continue;
        }
        if ring.len() <= 4 {
            mesh.faces.push(ring);
        } else {
            let centre = ring.iter().map(|&i| mesh.positions[i]).sum::<Vec3>() / ring.len() as f32;
            let colour = mesh.colors[ring[0]];
            let c = mesh.positions.len();
            mesh.positions.push(centre);
            mesh.colors.push(colour);
            for i in 0..ring.len() {
                mesh.faces.push(vec![c, ring[i], ring[(i + 1) % ring.len()]]);
            }
        }
        filled += 1;
    }
    mesh.recalculate_normals();
    mesh.bump_version();
    filled
}

/// Replace two selected faces (same vertex count) with a tube of quads joining them.
pub fn bridge_selected(mesh: &mut Mesh) -> Result<usize, String> {
    let mut sel: Vec<usize> = mesh.selected_faces.iter().copied().collect();
    sel.sort_unstable();
    if sel.len() != 2 {
        return Err(format!(
            "bridge needs exactly 2 selected faces (got {}); select the two faces that should be joined",
            sel.len()
        ));
    }
    let a = mesh.faces[sel[0]].clone();
    let b = mesh.faces[sel[1]].clone();
    if a.len() != b.len() {
        return Err(format!(
            "bridge needs two faces with the same number of corners ({} vs {})",
            a.len(),
            b.len()
        ));
    }
    let n = a.len();
    let b_rev: Vec<usize> = b.iter().rev().copied().collect();
    let cost = |k: usize| -> f32 {
        (0..n)
            .map(|i| mesh.positions[a[i]].distance(mesh.positions[b_rev[(i + k) % n]]))
            .sum()
    };
    let best = (0..n)
        .min_by(|&x, &y| cost(x).total_cmp(&cost(y)))
        .unwrap_or(0);
    // Remove the higher index first so the lower one stays valid.
    mesh.faces.remove(sel[1]);
    mesh.faces.remove(sel[0]);
    mesh.selected_faces.clear();
    for i in 0..n {
        let j = (i + 1) % n;
        mesh.faces.push(vec![a[i], a[j], b_rev[(j + best) % n], b_rev[(i + best) % n]]);
        mesh.selected_faces.insert(mesh.faces.len() - 1);
    }
    mesh.recalculate_normals();
    mesh.bump_version();
    Ok(n)
}

/// One or more rounds of Catmull-Clark subdivision: quads everywhere, rounded form.
pub fn subdivide_smooth(mesh: &mut Mesh, levels: usize) -> Result<(), String> {
    for _ in 0..levels.clamp(1, 4) {
        if mesh.faces.len() > 400_000 {
            return Err("mesh is too dense to subdivide again".into());
        }
        weld(mesh, 1e-5);
        catmull_clark_once(mesh);
    }
    mesh.selected_faces.clear();
    mesh.recalculate_normals();
    mesh.bump_version();
    Ok(())
}

fn catmull_clark_once(mesh: &mut Mesh) {
    ensure_colors(mesh);
    let vcount = mesh.positions.len();
    let col = |c: [f32; 3]| Vec3::from_array(c);
    let face_pts: Vec<(Vec3, Vec3)> = mesh
        .faces
        .iter()
        .map(|f| {
            let n = f.len() as f32;
            (
                f.iter().map(|&i| mesh.positions[i]).sum::<Vec3>() / n,
                f.iter().map(|&i| col(mesh.colors[i])).sum::<Vec3>() / n,
            )
        })
        .collect();
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); vcount];
    for (fi, f) in mesh.faces.iter().enumerate() {
        for i in 0..f.len() {
            let (a, b) = (f[i], f[(i + 1) % f.len()]);
            edge_faces.entry(Mesh::canonical_edge(a, b)).or_default().push(fi);
            vert_faces[f[i]].push(fi);
        }
    }
    let mut positions = mesh.positions.clone();
    let mut colors: Vec<Vec3> = mesh.colors.iter().map(|&c| col(c)).collect();
    let face_base = positions.len();
    for (p, c) in &face_pts {
        positions.push(*p);
        colors.push(*c);
    }
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
    let mut vert_edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); vcount]; // (other end, boundary)
    let mut keys: Vec<(usize, usize)> = edge_faces.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let adj = &edge_faces[&key];
        let (a, b) = key;
        let mid = (mesh.positions[a] + mesh.positions[b]) * 0.5;
        let boundary = adj.len() != 2;
        let p = if boundary {
            mid
        } else {
            (mesh.positions[a] + mesh.positions[b] + face_pts[adj[0]].0 + face_pts[adj[1]].0) * 0.25
        };
        edge_index.insert(key, positions.len());
        positions.push(p);
        colors.push((col(mesh.colors[a]) + col(mesh.colors[b])) * 0.5);
        vert_edges[a].push((b, boundary));
        vert_edges[b].push((a, boundary));
    }
    for v in 0..vcount {
        let edges = &vert_edges[v];
        if edges.is_empty() {
            continue;
        }
        let p = mesh.positions[v];
        let boundary: Vec<usize> = edges.iter().filter(|e| e.1).map(|e| e.0).collect();
        positions[v] = if boundary.len() >= 2 {
            p * 0.75 + (mesh.positions[boundary[0]] + mesh.positions[boundary[1]]) * 0.125
        } else {
            let n = edges.len() as f32;
            let f = vert_faces[v].iter().map(|&fi| face_pts[fi].0).sum::<Vec3>()
                / vert_faces[v].len().max(1) as f32;
            let r = edges
                .iter()
                .map(|e| (p + mesh.positions[e.0]) * 0.5)
                .sum::<Vec3>()
                / n;
            (f + r * 2.0 + p * (n - 3.0)) / n
        };
    }
    let mut faces = Vec::with_capacity(mesh.faces.iter().map(|f| f.len()).sum());
    for (fi, f) in mesh.faces.iter().enumerate() {
        let n = f.len();
        for i in 0..n {
            let (prev, cur, next) = (f[(i + n - 1) % n], f[i], f[(i + 1) % n]);
            faces.push(vec![
                cur,
                edge_index[&Mesh::canonical_edge(cur, next)],
                face_base + fi,
                edge_index[&Mesh::canonical_edge(prev, cur)],
            ]);
        }
    }
    mesh.positions = positions;
    mesh.colors = colors.into_iter().map(|c| c.to_array()).collect();
    mesh.faces = faces.into();
}

/// Reduce detail by merging vertices that fall in the same grid cell.
/// `ratio` is the fraction of vertices to keep (0.05..1.0). Returns (faces before, faces after).
pub fn decimate(mesh: &mut Mesh, ratio: f32) -> (usize, usize) {
    let before = mesh.faces.len();
    weld(mesh, 1e-5);
    let target = ((mesh.positions.len() as f32) * ratio.clamp(0.02, 1.0)).max(4.0);
    let area: f32 = (0..mesh.faces.len()).map(|i| mesh.face_area(i)).sum();
    if area <= 0.0 || ratio >= 0.999 {
        return (before, mesh.faces.len());
    }
    let mut cell = (area / target).sqrt();
    // Clustering removes more than the estimate on curved parts; tighten until close.
    for _ in 0..6 {
        let mut probe = mesh.clone();
        cluster(&mut probe, cell);
        if (probe.positions.len() as f32) >= target * 0.8 {
            *mesh = probe;
            mesh.recalculate_normals();
            mesh.bump_version();
            return (before, mesh.faces.len());
        }
        cell *= 0.75;
    }
    cluster(mesh, cell);
    mesh.recalculate_normals();
    mesh.bump_version();
    (before, mesh.faces.len())
}

fn cluster(mesh: &mut Mesh, cell: f32) {
    ensure_colors(mesh);
    let key = |p: Vec3| {
        (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        )
    };
    let mut cells: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut sums: Vec<(Vec3, Vec3, f32)> = Vec::new();
    let mut remap = vec![0usize; mesh.positions.len()];
    for (i, &p) in mesh.positions.iter().enumerate() {
        let id = *cells.entry(key(p)).or_insert_with(|| {
            sums.push((Vec3::ZERO, Vec3::ZERO, 0.0));
            sums.len() - 1
        });
        sums[id].0 += p;
        sums[id].1 += Vec3::from_array(mesh.colors[i]);
        sums[id].2 += 1.0;
        remap[i] = id;
    }
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut faces = Vec::new();
    for face in &mesh.faces {
        let mut f: Vec<usize> = Vec::new();
        for &vi in face {
            let v = remap[vi];
            if !f.contains(&v) {
                f.push(v);
            }
        }
        if f.len() < 3 {
            continue;
        }
        let mut sorted = f.clone();
        sorted.sort_unstable();
        if seen.insert(sorted) {
            faces.push(f);
        }
    }
    mesh.positions = sums.iter().map(|s| s.0 / s.2).collect();
    mesh.colors = sums.iter().map(|s| (s.1 / s.2).to_array()).collect();
    mesh.faces = faces.into();
    mesh.selected_faces.clear();
    mesh.selected_vertices.clear();
    mesh.selected_edges.clear();
}

/// Relax the selected faces' vertices toward their neighbours (whole mesh when nothing is selected).
pub fn smooth_vertices(mesh: &mut Mesh, strength: f32, iterations: usize) {
    let targets: HashSet<usize> = if mesh.selected_faces.is_empty() {
        (0..mesh.positions.len()).collect()
    } else {
        mesh.selected_faces
            .iter()
            .filter_map(|&f| mesh.faces.get(f))
            .flatten()
            .copied()
            .collect()
    };
    let key = |p: Vec3| ((p.x * 2000.0).round() as i64, (p.y * 2000.0).round() as i64, (p.z * 2000.0).round() as i64);
    for _ in 0..iterations.clamp(1, 50) {
        // Neighbours by welded position, so split vertices move together.
        let mut neighbours: HashMap<(i64, i64, i64), (Vec3, f32)> = HashMap::new();
        for face in &mesh.faces {
            let n = face.len();
            for i in 0..n {
                let (a, b) = (mesh.positions[face[i]], mesh.positions[face[(i + 1) % n]]);
                let ea = neighbours.entry(key(a)).or_insert((Vec3::ZERO, 0.0));
                ea.0 += b;
                ea.1 += 1.0;
                let eb = neighbours.entry(key(b)).or_insert((Vec3::ZERO, 0.0));
                eb.0 += a;
                eb.1 += 1.0;
            }
        }
        let old = mesh.positions.clone();
        for &vi in &targets {
            if let Some((sum, count)) = neighbours.get(&key(old[vi])) {
                if *count > 0.0 {
                    mesh.positions[vi] = old[vi].lerp(*sum / *count, strength.clamp(0.0, 1.0));
                }
            }
        }
    }
    mesh.recalculate_normals();
    mesh.bump_version();
}

// ---------------------------------------------------------------------------
// Deformers
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Deform {
    /// Curve the part along `axis` by a total angle, curling toward `toward`.
    Bend { angle_deg: f32, toward: usize },
    /// Rotate progressively around `axis` by a total angle.
    Twist { angle_deg: f32 },
    /// Scale the cross-section from 1 at the start of `axis` to `factor` at the end.
    Taper { factor: f32 },
}

/// Deform the selected faces' vertices (the whole mesh when nothing is selected) in local space.
pub fn deform(mesh: &mut Mesh, axis: usize, kind: Deform) -> Result<usize, String> {
    let targets: Vec<usize> = if mesh.selected_faces.is_empty() {
        (0..mesh.positions.len()).collect()
    } else {
        let set: HashSet<usize> = mesh
            .selected_faces
            .iter()
            .filter_map(|&f| mesh.faces.get(f))
            .flatten()
            .copied()
            .collect();
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort_unstable();
        v
    };
    if targets.is_empty() {
        return Err("nothing to deform".into());
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &i in &targets {
        min = min.min(mesh.positions[i]);
        max = max.max(mesh.positions[i]);
    }
    let length = max[axis] - min[axis];
    if length < 1e-6 {
        return Err("the part has no extent along that axis".into());
    }
    let centre = (min + max) * 0.5;
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    match kind {
        Deform::Twist { angle_deg } => {
            for &i in &targets {
                let p = mesh.positions[i];
                let t = (p[axis] - min[axis]) / length;
                let (sin, cos) = (angle_deg.to_radians() * t).sin_cos();
                let (du, dv) = (p[u] - centre[u], p[v] - centre[v]);
                let mut q = p;
                q[u] = centre[u] + du * cos - dv * sin;
                q[v] = centre[v] + du * sin + dv * cos;
                mesh.positions[i] = q;
            }
        }
        Deform::Taper { factor } => {
            for &i in &targets {
                let p = mesh.positions[i];
                let t = (p[axis] - min[axis]) / length;
                let s = 1.0 + (factor - 1.0) * t;
                let mut q = p;
                q[u] = centre[u] + (p[u] - centre[u]) * s;
                q[v] = centre[v] + (p[v] - centre[v]) * s;
                mesh.positions[i] = q;
            }
        }
        Deform::Bend { angle_deg, toward } => {
            if toward == axis || toward > 2 {
                return Err("bend 'toward' must be a different axis from 'axis'".into());
            }
            let angle = angle_deg.to_radians();
            if angle.abs() < 1e-4 {
                return Ok(0);
            }
            let radius = length / angle;
            for &i in &targets {
                let p = mesh.positions[i];
                let a = (p[axis] - min[axis]) / radius;
                let offset = radius - (p[toward] - centre[toward]);
                let mut q = p;
                q[axis] = min[axis] + offset * a.sin();
                q[toward] = centre[toward] + radius - offset * a.cos();
                mesh.positions[i] = q;
            }
        }
    }
    mesh.recalculate_normals();
    mesh.bump_version();
    Ok(targets.len())
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

/// Small deterministic generator so a seed always gives the same scatter.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

pub struct ScatterOptions {
    pub count: usize,
    pub seed: u64,
    pub align_to_normal: bool,
    pub random_yaw: bool,
    pub scale_min: f32,
    pub scale_max: f32,
    pub offset: f32,
    /// Skip faces whose world normal points below this Y value (e.g. 0.5 = only upward faces).
    pub min_normal_y: Option<f32>,
}

/// World-space transforms for copies spread over a surface, weighted by area.
/// `faces` limits the surface to those faces; empty means all of it.
pub fn scatter_transforms(
    surface: &Mesh,
    model: Mat4,
    faces: &HashSet<usize>,
    opts: &ScatterOptions,
) -> Vec<Mat4> {
    let normal_mat = model.inverse().transpose();
    let mut tris: Vec<([Vec3; 3], Vec3)> = Vec::new();
    let mut cumulative: Vec<f32> = Vec::new();
    let mut total = 0.0;
    for (fi, face) in surface.faces.iter().enumerate() {
        if face.len() < 3 || (!faces.is_empty() && !faces.contains(&fi)) {
            continue;
        }
        let n = normal_mat
            .transform_vector3(surface.face_normal(fi))
            .normalize_or_zero();
        if opts.min_normal_y.is_some_and(|m| n.y < m) {
            continue;
        }
        let p0 = model.transform_point3(surface.positions[face[0]]);
        for i in 1..face.len() - 1 {
            let p1 = model.transform_point3(surface.positions[face[i]]);
            let p2 = model.transform_point3(surface.positions[face[i + 1]]);
            let area = (p1 - p0).cross(p2 - p0).length() * 0.5;
            if area > 1e-10 {
                total += area;
                cumulative.push(total);
                tris.push(([p0, p1, p2], n));
            }
        }
    }
    if tris.is_empty() {
        return Vec::new();
    }
    let mut rng = Rng::new(opts.seed);
    let mut out = Vec::with_capacity(opts.count);
    for _ in 0..opts.count {
        let pick = rng.next_f32() * total;
        let i = cumulative.partition_point(|&c| c < pick).min(tris.len() - 1);
        let ([a, b, c], n) = tris[i];
        let (mut u, mut v) = (rng.next_f32(), rng.next_f32());
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let point = a + (b - a) * u + (c - a) * v + n * opts.offset;
        let align = if opts.align_to_normal {
            Quat::from_rotation_arc(Vec3::Y, n)
        } else {
            Quat::IDENTITY
        };
        let yaw = if opts.random_yaw {
            Quat::from_rotation_y(rng.range(0.0, std::f32::consts::TAU))
        } else {
            Quat::IDENTITY
        };
        let scale = rng.range(opts.scale_min, opts.scale_max.max(opts.scale_min));
        out.push(Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            align * yaw,
            point,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Per-face colour
// ---------------------------------------------------------------------------

/// Give `faces` their own vertex colour. Vertices shared with other faces are split so the
/// colour boundary stays crisp. `vertex_color` is in vertex-colour space (it multiplies the
/// part's base colour when rendered). Returns the number of faces painted.
pub fn paint_faces(mesh: &mut Mesh, faces: &HashSet<usize>, vertex_color: [f32; 3], blend: f32) -> usize {
    ensure_colors(mesh);
    if mesh.normals.len() != mesh.positions.len() {
        mesh.recalculate_normals();
    }
    let mut used_outside: HashSet<usize> = HashSet::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        if !faces.contains(&fi) {
            used_outside.extend(face.iter().copied());
        }
    }
    let mut split: HashMap<usize, usize> = HashMap::new();
    let mut painted_verts: HashSet<usize> = HashSet::new();
    let mut painted = 0;
    let mut indices: Vec<usize> = faces.iter().copied().filter(|&f| f < mesh.faces.len()).collect();
    indices.sort_unstable();
    for fi in indices {
        painted += 1;
        for slot in 0..mesh.faces[fi].len() {
            let vi = mesh.faces[fi][slot];
            let target = if used_outside.contains(&vi) {
                *split.entry(vi).or_insert_with(|| {
                    let (p, n, c) = (mesh.positions[vi], mesh.normals[vi], mesh.colors[vi]);
                    mesh.positions.push(p);
                    mesh.normals.push(n);
                    mesh.colors.push(c);
                    mesh.positions.len() - 1
                })
            } else {
                vi
            };
            mesh.faces[fi][slot] = target;
            painted_verts.insert(target);
        }
    }
    let t = blend.clamp(0.0, 1.0);
    for vi in painted_verts {
        let old = Vec3::from_array(mesh.colors[vi]);
        mesh.colors[vi] = old.lerp(Vec3::from_array(vertex_color), t).to_array();
    }
    mesh.bump_version();
    painted
}

// ---------------------------------------------------------------------------
// Health checks
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct MeshReport {
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub degenerate_faces: usize,
    pub components: usize,
    pub signed_volume: f32,
    pub watertight: bool,
    pub inside_out: bool,
    /// Edges whose two faces run the same way round (one of them is flipped).
    pub inconsistent_edges: usize,
}

pub fn check_mesh(mesh: &Mesh) -> MeshReport {
    let key = |p: Vec3| ((p.x * 2000.0).round() as i64, (p.y * 2000.0).round() as i64, (p.z * 2000.0).round() as i64);
    let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();
    let mut directed: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();
    let mut degenerate = 0;
    for (fi, face) in mesh.faces.iter().enumerate() {
        if face.len() < 3 || mesh.face_area(fi) < 1e-10 {
            degenerate += 1;
            continue;
        }
        for i in 0..face.len() {
            let (a, b) = (key(mesh.positions[face[i]]), key(mesh.positions[face[(i + 1) % face.len()]]));
            if a == b {
                continue;
            }
            *edges.entry(if a <= b { (a, b) } else { (b, a) }).or_default() += 1;
            *directed.entry((a, b)).or_default() += 1;
        }
    }
    let boundary = edges.values().filter(|&&c| c == 1).count();
    let non_manifold = edges.values().filter(|&&c| c > 2).count();
    let mut probe = mesh.clone();
    probe.selected_faces = (0..mesh.faces.len()).collect();
    let components = probe.selected_components().len();
    let volume = crate::modeling::signed_volume(mesh);
    let watertight = boundary == 0 && non_manifold == 0 && !mesh.faces.is_empty();
    MeshReport {
        boundary_edges: boundary,
        non_manifold_edges: non_manifold,
        degenerate_faces: degenerate,
        components,
        signed_volume: volume,
        watertight,
        inside_out: watertight && volume < 0.0,
        inconsistent_edges: directed.values().filter(|&&c| c > 1).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed(mesh: &Mesh) -> bool {
        let r = check_mesh(mesh);
        r.watertight && !r.inside_out && r.inconsistent_edges == 0
    }

    #[test]
    fn extruded_concave_profile_is_a_closed_outward_solid() {
        // An L shape: concave, so the caps need real triangulation.
        let l = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let mesh = extrude_profile(&l, 1.5, 1.0, 0.0, 1, true, [0.8; 3]).unwrap();
        assert!(closed(&mesh));
        assert!((crate::modeling::signed_volume(&mesh) - 3.0 * 1.5).abs() < 1e-3);
        // Clockwise input gives the same solid.
        let rev: Vec<Vec2> = l.iter().rev().copied().collect();
        let mesh = extrude_profile(&rev, 1.5, 1.0, 0.0, 1, true, [0.8; 3]).unwrap();
        assert!(closed(&mesh));
        assert!(extrude_profile(&l[..2], 1.0, 1.0, 0.0, 1, true, [0.8; 3]).is_err());
        let twisted = extrude_profile(&l, 2.0, 0.5, 90.0, 8, true, [0.8; 3]).unwrap();
        assert!(closed(&twisted));
    }

    #[test]
    fn loft_between_rings_is_closed() {
        let ring = |y: f32, r: f32| -> Vec<Vec3> {
            (0..8)
                .map(|i| {
                    let a = i as f32 / 8.0 * std::f32::consts::TAU;
                    Vec3::new(a.cos() * r, y, a.sin() * r)
                })
                .collect()
        };
        let mesh = loft(&[ring(0.0, 1.0), ring(1.0, 0.6), ring(2.0, 0.9)], true, false, [0.8; 3]).unwrap();
        assert!(closed(&mesh));
        assert!(loft(&[ring(0.0, 1.0), ring(1.0, 1.0)[..5].to_vec()], true, false, [0.8; 3]).is_err());
    }

    #[test]
    fn weld_fill_and_bridge_repair_topology() {
        let mut cube = Mesh::cube(1.0);
        cube.selected_faces.insert(0);
        cube.delete_selected(crate::mesh::EditElementMode::Face);
        assert!(!check_mesh(&cube).watertight);
        assert_eq!(fill_holes(&mut cube), 1);
        assert!(closed(&cube));

        // Two cubes facing each other along X, joined through their facing sides.
        let mut a = Mesh::cube(1.0);
        let b = crate::modeling::transformed(&Mesh::cube(1.0), Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)));
        crate::modeling::append(&mut a, &b);
        let pick = |m: &Mesh, x: f32, nx: f32| {
            (0..m.faces.len())
                .find(|&f| (m.face_center(f).x - x).abs() < 1e-3 && m.face_normal(f).x * nx > 0.9)
                .unwrap()
        };
        let (fa, fb) = (pick(&a, 0.5, 1.0), pick(&a, 2.5, -1.0));
        a.selected_faces = [fa, fb].into_iter().collect();
        assert_eq!(bridge_selected(&mut a).unwrap(), 4);
        weld(&mut a, 1e-4);
        let r = check_mesh(&a);
        assert!(r.watertight && !r.inside_out && r.components == 1, "{r:?}");
    }

    #[test]
    fn subdivision_decimation_and_deformers_keep_solids_valid() {
        let mut cube = Mesh::cube(2.0);
        subdivide_smooth(&mut cube, 2).unwrap();
        assert_eq!(cube.faces.len(), 6 * 16);
        assert!(closed(&cube));
        // Smoothing rounds the cube toward a ball of radius ~0.88: volume well below 8.
        let v = crate::modeling::signed_volume(&cube);
        assert!(v > 2.4 && v < 4.5, "{v}");

        let mut sphere = Mesh::sphere(1.0, 24, 32);
        let (before, after) = decimate(&mut sphere, 0.3);
        assert!(after < before && after > 20, "{before} -> {after}");

        let mut bar = extrude_profile(
            &[Vec2::new(-0.2, -0.2), Vec2::new(0.2, -0.2), Vec2::new(0.2, 0.2), Vec2::new(-0.2, 0.2)],
            4.0, 1.0, 0.0, 16, true, [0.8; 3],
        )
        .unwrap();
        let volume = crate::modeling::signed_volume(&bar);
        deform(&mut bar, 1, Deform::Twist { angle_deg: 180.0 }).unwrap();
        assert!((crate::modeling::signed_volume(&bar) - volume).abs() < 0.05);
        deform(&mut bar, 1, Deform::Bend { angle_deg: 90.0, toward: 0 }).unwrap();
        let (lo, hi) = bar.bounds();
        // A quarter-circle bend of a 4-long bar reaches about 4 / (pi/2) sideways.
        assert!(hi.x - lo.x > 2.0 && hi.y - lo.y < 3.2, "{lo} {hi}");
        assert!(closed(&bar));
        deform(&mut bar, 1, Deform::Taper { factor: 0.5 }).unwrap();
        assert!(deform(&mut bar, 1, Deform::Bend { angle_deg: 45.0, toward: 1 }).is_err());
    }

    #[test]
    fn painting_faces_splits_shared_vertices_only_where_needed() {
        let mut plane = Mesh::plane(2.0, 2);
        let verts = plane.positions.len();
        let faces: HashSet<usize> = [0].into_iter().collect();
        assert_eq!(paint_faces(&mut plane, &faces, [1.0, 0.0, 0.0], 1.0), 1);
        assert!(plane.positions.len() > verts);
        assert_eq!(plane.face_color(0), [1.0, 0.0, 0.0]);
        assert_eq!(plane.face_color(1), [1.0, 1.0, 1.0]);
        assert_eq!(plane.colors.len(), plane.positions.len());
    }

    #[test]
    fn scatter_is_deterministic_and_stays_on_the_surface() {
        let plane = Mesh::plane(4.0, 2);
        let opts = ScatterOptions {
            count: 50,
            seed: 7,
            align_to_normal: true,
            random_yaw: true,
            scale_min: 0.5,
            scale_max: 1.5,
            offset: 0.0,
            min_normal_y: None,
        };
        let a = scatter_transforms(&plane, Mat4::IDENTITY, &HashSet::new(), &opts);
        let b = scatter_transforms(&plane, Mat4::IDENTITY, &HashSet::new(), &opts);
        assert_eq!(a.len(), 50);
        assert_eq!(a, b);
        for m in &a {
            let p = m.transform_point3(Vec3::ZERO);
            assert!(p.y.abs() < 1e-4 && p.x.abs() <= 2.001 && p.z.abs() <= 2.001);
        }
    }
}
