//! Higher-level modeling operations for agents: bevel, mirror, array, lathe, sweep and
//! constructive solid geometry (union / subtract / intersect).
use crate::mesh::Mesh;
use glam::{Mat4, Quat, Vec2, Vec3};
use std::collections::{HashMap, HashSet};

type QPt = (i32, i32, i32);

fn quantize(p: Vec3) -> QPt {
    (
        (p.x * 500.0).round() as i32,
        (p.y * 500.0).round() as i32,
        (p.z * 500.0).round() as i32,
    )
}

/// Signed volume of a closed mesh; negative means faces are wound inside-out.
pub fn signed_volume(mesh: &Mesh) -> f32 {
    let mut v = 0.0;
    for f in &mesh.faces {
        if f.len() < 3 {
            continue;
        }
        let a = mesh.positions[f[0]];
        for i in 1..f.len() - 1 {
            v += a.dot(mesh.positions[f[i]].cross(mesh.positions[f[i + 1]]));
        }
    }
    v / 6.0
}

fn flip_faces(mesh: &mut Mesh) {
    for f in &mut mesh.faces {
        f.reverse();
    }
}

fn finish(mut mesh: Mesh, color: [f32; 3]) -> Mesh {
    mesh.colors = vec![color; mesh.positions.len()];
    mesh.recalculate_normals();
    mesh
}

// ---------------------------------------------------------------------------
// Bevel
// ---------------------------------------------------------------------------

/// Chamfer the border of the selected face region by `width`.
///
/// The selected region shrinks inward by `width`, its old border drops by `width` along the
/// region normal (taking neighbouring faces with it), and a chamfer strip joins the two.
pub fn bevel_selected(mesh: &mut Mesh, width: f32) -> Result<usize, String> {
    if mesh.selected_faces.is_empty() {
        return Err("No faces selected to bevel".into());
    }
    if !(width.is_finite() && width > 0.0) {
        return Err("Bevel width must be positive".into());
    }
    let selected: Vec<usize> = {
        let mut s: Vec<usize> = mesh
            .selected_faces
            .iter()
            .copied()
            .filter(|&f| f < mesh.faces.len() && mesh.faces[f].len() >= 3)
            .collect();
        s.sort_unstable();
        s
    };
    // Weld by position so split-vertex meshes behave like connected ones.
    let mut rep: HashMap<QPt, usize> = HashMap::new();
    let weld = |vi: usize, rep: &mut HashMap<QPt, usize>, mesh: &Mesh| -> usize {
        *rep.entry(quantize(mesh.positions[vi])).or_insert(vi)
    };
    let mut directed = HashSet::new();
    let mut vertex_normal: HashMap<usize, Vec3> = HashMap::new();
    for &fi in &selected {
        let n = mesh.face_normal(fi);
        let face = mesh.faces[fi].clone();
        for i in 0..face.len() {
            let u = weld(face[i], &mut rep, mesh);
            let v = weld(face[(i + 1) % face.len()], &mut rep, mesh);
            if u != v {
                directed.insert((u, v));
            }
            *vertex_normal.entry(u).or_default() += n;
        }
    }
    let boundary: Vec<(usize, usize)> = directed
        .iter()
        .copied()
        .filter(|&(u, v)| !directed.contains(&(v, u)))
        .collect();
    if boundary.is_empty() {
        return Err("The selection has no open border to bevel; select a region, not a whole closed part".into());
    }
    let normal_of = |w: usize| vertex_normal.get(&w).copied().unwrap_or(Vec3::Y).normalize_or(Vec3::Y);

    // Inward offset for every border vertex (mitred at corners).
    let mut inward: HashMap<usize, (Vec3, Vec<Vec3>)> = HashMap::new();
    for &(u, v) in &boundary {
        let d = (mesh.positions[v] - mesh.positions[u]).normalize_or_zero();
        for w in [u, v] {
            let side = normal_of(w).cross(d).normalize_or_zero();
            let e = inward.entry(w).or_default();
            e.0 += side;
            e.1.push(side);
        }
    }
    let mut new_vertex: HashMap<usize, usize> = HashMap::new();
    for (&w, (sum, sides)) in &inward {
        let dir = sum.normalize_or_zero();
        let cos = sides
            .iter()
            .map(|s| dir.dot(*s))
            .fold(1.0_f32, f32::min)
            .max(0.35);
        let p = mesh.positions[w] + dir * (width / cos);
        let color = mesh.colors.get(w).copied().unwrap_or([0.8, 0.8, 0.8]);
        new_vertex.insert(w, mesh.positions.len());
        mesh.positions.push(p);
        mesh.normals.push(normal_of(w));
        mesh.colors.push(color);
    }
    // Selected faces use the inner ring.
    for &fi in &selected {
        for i in 0..mesh.faces[fi].len() {
            let vi = mesh.faces[fi][i];
            if let Some(w) = rep.get(&quantize(mesh.positions[vi])) {
                if let Some(&nv) = new_vertex.get(w) {
                    mesh.faces[fi][i] = nv;
                }
            }
        }
    }
    // Chamfer strip.
    for &(u, v) in &boundary {
        mesh.faces.push(vec![u, v, new_vertex[&v], new_vertex[&u]]);
    }
    // Drop the old border (and every coincident vertex of neighbouring faces).
    let drops: HashMap<QPt, Vec3> = inward
        .keys()
        .map(|&w| (quantize(mesh.positions[w]), normal_of(w) * width))
        .collect();
    let new_set: HashSet<usize> = new_vertex.values().copied().collect();
    for vi in 0..mesh.positions.len() {
        if new_set.contains(&vi) {
            continue;
        }
        if let Some(d) = drops.get(&quantize(mesh.positions[vi])) {
            mesh.positions[vi] -= *d;
        }
    }
    mesh.recalculate_normals();
    Ok(boundary.len())
}

// ---------------------------------------------------------------------------
// Mirror / array
// ---------------------------------------------------------------------------

pub fn axis_index(axis: &str) -> Result<usize, String> {
    match axis.to_lowercase().as_str() {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        other => Err(format!("Unknown axis '{other}'; use x, y or z")),
    }
}

/// A copy of `mesh` reflected across the local plane through the origin normal to `axis`.
pub fn mirrored(mesh: &Mesh, axis: usize) -> Mesh {
    let mut m = Mesh::new(
        mesh.positions
            .iter()
            .map(|p| {
                let mut q = *p;
                q[axis] = -q[axis];
                q
            })
            .collect(),
        mesh.faces.iter().map(|f| f.iter().rev().copied().collect()).collect(),
    );
    m.colors = mesh.colors.clone();
    m.recalculate_normals();
    m
}

/// Append `other` (already in this mesh's space) to `mesh`.
pub fn append(mesh: &mut Mesh, other: &Mesh) {
    let base = mesh.positions.len();
    mesh.positions.extend_from_slice(&other.positions);
    mesh.normals.extend_from_slice(&other.normals);
    let mut colors = other.colors.clone();
    colors.resize(other.positions.len(), [0.8, 0.8, 0.8]);
    mesh.colors.resize(base, [0.8, 0.8, 0.8]);
    mesh.colors.extend(colors);
    mesh.faces
        .extend(other.faces.iter().map(|f| f.iter().map(|i| i + base).collect::<Vec<_>>()));
    mesh.recalculate_normals();
}

pub fn transformed(mesh: &Mesh, m: Mat4) -> Mesh {
    let mut out = Mesh::new(
        mesh.positions.iter().map(|p| m.transform_point3(*p)).collect(),
        mesh.faces.clone(),
    );
    out.colors = mesh.colors.clone();
    if m.determinant() < 0.0 {
        flip_faces(&mut out);
    }
    out.recalculate_normals();
    out
}

/// Per-copy step transform for arrays: rotate about the origin, then offset.
pub fn array_step(offset: Vec3, rotation_deg: Vec3, scale: f32) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::splat(scale),
        Quat::from_euler(
            glam::EulerRot::XYZ,
            rotation_deg.x.to_radians(),
            rotation_deg.y.to_radians(),
            rotation_deg.z.to_radians(),
        ),
        offset,
    )
}

// ---------------------------------------------------------------------------
// Lathe / sweep
// ---------------------------------------------------------------------------

/// Revolve a `(radius, height)` profile around the local Y axis.
pub fn lathe(
    profile: &[Vec2],
    segments: usize,
    angle_deg: f32,
    cap: bool,
    color: [f32; 3],
) -> Result<Mesh, String> {
    if profile.len() < 2 {
        return Err("A lathe profile needs at least two [radius, height] points".into());
    }
    if profile.iter().any(|p| !p.is_finite() || p.x < -1e-6) {
        return Err("Lathe radii must be finite and not negative".into());
    }
    let segments = segments.clamp(3, 256);
    let full = angle_deg >= 359.999;
    let angle = angle_deg.clamp(1.0, 360.0).to_radians();
    let columns = if full { segments } else { segments + 1 };
    let mut positions = Vec::new();
    // ring[i][j]: vertex for profile point i at angle step j (poles collapse to one vertex).
    let mut ring: Vec<Vec<usize>> = Vec::new();
    for p in profile {
        if p.x < 1e-5 {
            let idx = positions.len();
            positions.push(Vec3::new(0.0, p.y, 0.0));
            ring.push(vec![idx; columns]);
        } else {
            let row = (0..columns)
                .map(|j| {
                    let a = angle * j as f32 / segments as f32;
                    positions.push(Vec3::new(p.x * a.cos(), p.y, -p.x * a.sin()));
                    positions.len() - 1
                })
                .collect();
            ring.push(row);
        }
    }
    let mut faces = Vec::new();
    for i in 0..profile.len() - 1 {
        for j in 0..segments {
            let jn = if full { (j + 1) % columns } else { j + 1 };
            let mut f = vec![ring[i][j], ring[i][jn], ring[i + 1][jn], ring[i + 1][j]];
            f.dedup();
            if f.first() == f.last() && f.len() > 1 {
                f.pop();
            }
            if f.len() >= 3 {
                faces.push(f);
            }
        }
    }
    let closed_ends = |i: usize| profile[i].x < 1e-5;
    if cap && full {
        for (i, top) in [(0, false), (profile.len() - 1, true)] {
            if !closed_ends(i) {
                let mut f: Vec<usize> = ring[i].clone();
                if !top {
                    f.reverse();
                }
                faces.push(f);
            }
        }
    }
    let mut mesh = Mesh::new(positions, faces);
    let closed = full && (cap || (closed_ends(0) && closed_ends(profile.len() - 1)));
    let outward = if closed {
        signed_volume(&mesh) >= 0.0
    } else {
        // Open surface: make the widest band face away from the axis.
        let (fi, _) = (0..mesh.faces.len())
            .map(|fi| (fi, Vec2::new(mesh.face_center(fi).x, mesh.face_center(fi).z).length()))
            .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a });
        let c = mesh.face_center(fi);
        mesh.face_normal(fi).dot(Vec3::new(c.x, 0.0, c.z)) >= 0.0
    };
    if !outward {
        flip_faces(&mut mesh);
    }
    Ok(finish(mesh, color))
}

/// Sweep a closed 2D cross-section along a 3D path using rotation-minimising frames.
pub fn sweep(
    path: &[Vec3],
    section: &[Vec2],
    closed_path: bool,
    cap: bool,
    twist_deg: f32,
    end_scale: f32,
    color: [f32; 3],
) -> Result<Mesh, String> {
    if path.len() < 2 {
        return Err("A sweep path needs at least two points".into());
    }
    if section.len() < 3 {
        return Err("A sweep profile needs at least three points".into());
    }
    if path.iter().any(|p| !p.is_finite()) || section.iter().any(|p| !p.is_finite()) {
        return Err("Sweep points must be finite".into());
    }
    let n = path.len();
    let tangent = |i: usize| -> Vec3 {
        let (a, b) = if closed_path {
            (path[(i + n - 1) % n], path[(i + 1) % n])
        } else {
            (path[i.saturating_sub(1)], path[(i + 1).min(n - 1)])
        };
        (b - a).normalize_or(Vec3::Y)
    };
    let t0 = tangent(0);
    let helper = if t0.dot(Vec3::Y).abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let mut normal = t0.cross(helper).normalize();
    let mut positions = Vec::new();
    let m = section.len();
    for i in 0..n {
        let t = tangent(i);
        if i > 0 {
            // Parallel transport: rotate the previous frame onto the new tangent.
            let prev = tangent(i - 1);
            normal = Quat::from_rotation_arc(prev, t) * normal;
            normal = (normal - t * normal.dot(t)).normalize_or(normal);
        }
        let s = 1.0 + (end_scale - 1.0) * i as f32 / (n - 1) as f32;
        let twist = Quat::from_axis_angle(t, (twist_deg * i as f32 / (n - 1) as f32).to_radians());
        let u = twist * normal;
        let v = t.cross(u);
        for p in section {
            positions.push(path[i] + (u * p.x + v * p.y) * s);
        }
    }
    let rows = if closed_path { n } else { n - 1 };
    let mut faces = Vec::new();
    for i in 0..rows {
        let a = i * m;
        let b = ((i + 1) % n) * m;
        for k in 0..m {
            let kn = (k + 1) % m;
            faces.push(vec![a + k, a + kn, b + kn, b + k]);
        }
    }
    if cap && !closed_path {
        faces.push((0..m).rev().collect());
        faces.push(((n - 1) * m..n * m).collect());
    }
    let mut mesh = Mesh::new(positions, faces);
    let outward = if closed_path || cap {
        signed_volume(&mesh) >= 0.0
    } else {
        let c = mesh.face_center(0);
        mesh.face_normal(0).dot(c - path[0]) >= 0.0
    };
    if !outward {
        flip_faces(&mut mesh);
    }
    Ok(finish(mesh, color))
}

pub fn circle_section(radius: f32, sides: usize) -> Vec<Vec2> {
    let sides = sides.clamp(3, 128);
    (0..sides)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / sides as f32;
            Vec2::new(a.cos(), a.sin()) * radius
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Constructive solid geometry (BSP, after Evan Wallace's csg.js)
// ---------------------------------------------------------------------------

const EPS: f32 = 1e-5;

#[derive(Clone, Copy)]
struct CVert {
    pos: Vec3,
    color: [f32; 3],
}

impl CVert {
    fn lerp(&self, o: &CVert, t: f32) -> CVert {
        CVert {
            pos: self.pos.lerp(o.pos, t),
            color: [
                self.color[0] + (o.color[0] - self.color[0]) * t,
                self.color[1] + (o.color[1] - self.color[1]) * t,
                self.color[2] + (o.color[2] - self.color[2]) * t,
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct Plane {
    n: Vec3,
    w: f32,
}

impl Plane {
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Plane> {
        let n = (b - a).cross(c - a);
        (n.length_squared() > 1e-20).then(|| {
            let n = n.normalize();
            Plane { n, w: n.dot(a) }
        })
    }
    fn flip(&mut self) {
        self.n = -self.n;
        self.w = -self.w;
    }
}

#[derive(Clone)]
struct Poly {
    verts: Vec<CVert>,
    plane: Plane,
}

impl Poly {
    fn new(verts: Vec<CVert>) -> Option<Poly> {
        let plane = Plane::from_points(verts[0].pos, verts[1].pos, verts[2].pos)?;
        Some(Poly { verts, plane })
    }
    fn flip(&mut self) {
        self.verts.reverse();
        self.plane.flip();
    }
}

/// Classify and split `poly` by `plane` into the four output lists.
fn split(
    plane: &Plane,
    poly: &Poly,
    cofront: &mut Vec<Poly>,
    coback: &mut Vec<Poly>,
    front: &mut Vec<Poly>,
    back: &mut Vec<Poly>,
) {
    const COPLANAR: u8 = 0;
    const FRONT: u8 = 1;
    const BACK: u8 = 2;
    let mut kind = 0u8;
    let types: Vec<u8> = poly
        .verts
        .iter()
        .map(|v| {
            let t = plane.n.dot(v.pos) - plane.w;
            let k = if t < -EPS {
                BACK
            } else if t > EPS {
                FRONT
            } else {
                COPLANAR
            };
            kind |= k;
            k
        })
        .collect();
    match kind {
        COPLANAR => {
            if plane.n.dot(poly.plane.n) > 0.0 {
                cofront.push(poly.clone())
            } else {
                coback.push(poly.clone())
            }
        }
        FRONT => front.push(poly.clone()),
        BACK => back.push(poly.clone()),
        _ => {
            let (mut f, mut b) = (Vec::new(), Vec::new());
            let len = poly.verts.len();
            for i in 0..len {
                let j = (i + 1) % len;
                let (ti, tj) = (types[i], types[j]);
                let (vi, vj) = (&poly.verts[i], &poly.verts[j]);
                if ti != BACK {
                    f.push(*vi);
                }
                if ti != FRONT {
                    b.push(*vi);
                }
                if (ti | tj) == (FRONT | BACK) {
                    let t = (plane.w - plane.n.dot(vi.pos)) / plane.n.dot(vj.pos - vi.pos);
                    let v = vi.lerp(vj, t);
                    f.push(v);
                    b.push(v);
                }
            }
            if f.len() >= 3 {
                front.push(Poly {
                    verts: f,
                    plane: poly.plane,
                });
            }
            if b.len() >= 3 {
                back.push(Poly {
                    verts: b,
                    plane: poly.plane,
                });
            }
        }
    }
}

#[derive(Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
    polys: Vec<Poly>,
}

impl Node {
    fn new(polys: Vec<Poly>) -> Node {
        let mut n = Node::default();
        n.build(polys);
        n
    }

    fn invert(&mut self) {
        for p in &mut self.polys {
            p.flip();
        }
        if let Some(pl) = &mut self.plane {
            pl.flip();
        }
        if let Some(f) = &mut self.front {
            f.invert();
        }
        if let Some(b) = &mut self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    fn clip_polys(&self, polys: Vec<Poly>) -> Vec<Poly> {
        let Some(plane) = &self.plane else {
            return polys;
        };
        let (mut front, mut back) = (Vec::new(), Vec::new());
        for p in &polys {
            let (mut cf, mut cb) = (Vec::new(), Vec::new());
            split(plane, p, &mut cf, &mut cb, &mut front, &mut back);
            front.extend(cf);
            back.extend(cb);
        }
        let mut front = match &self.front {
            Some(f) => f.clip_polys(front),
            None => front,
        };
        let back = match &self.back {
            Some(b) => b.clip_polys(back),
            None => Vec::new(),
        };
        front.extend(back);
        front
    }

    fn clip_to(&mut self, other: &Node) {
        self.polys = other.clip_polys(std::mem::take(&mut self.polys));
        if let Some(f) = &mut self.front {
            f.clip_to(other);
        }
        if let Some(b) = &mut self.back {
            b.clip_to(other);
        }
    }

    fn all_polys(&self, out: &mut Vec<Poly>) {
        out.extend(self.polys.iter().cloned());
        if let Some(f) = &self.front {
            f.all_polys(out);
        }
        if let Some(b) = &self.back {
            b.all_polys(out);
        }
    }

    fn build(&mut self, polys: Vec<Poly>) {
        if polys.is_empty() {
            return;
        }
        let plane = *self.plane.get_or_insert(polys[0].plane);
        let (mut front, mut back) = (Vec::new(), Vec::new());
        let (mut cf, mut cb) = (Vec::new(), Vec::new());
        for p in &polys {
            split(&plane, p, &mut cf, &mut cb, &mut front, &mut back);
        }
        self.polys.extend(cf);
        self.polys.extend(cb);
        if !front.is_empty() {
            self.front.get_or_insert_with(Default::default).build(front);
        }
        if !back.is_empty() {
            self.back.get_or_insert_with(Default::default).build(back);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Subtract,
    Intersect,
}

impl BooleanOp {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "union" | "add" => Ok(Self::Union),
            "subtract" | "difference" | "cut" => Ok(Self::Subtract),
            "intersect" | "intersection" => Ok(Self::Intersect),
            other => Err(format!(
                "Unknown boolean operation '{other}'; use union, subtract or intersect"
            )),
        }
    }
}

fn to_polys(mesh: &Mesh, m: Mat4, fallback: [f32; 3]) -> Vec<Poly> {
    let pos: Vec<Vec3> = mesh.positions.iter().map(|p| m.transform_point3(*p)).collect();
    let flip = m.determinant() < 0.0;
    let mut out = Vec::new();
    for f in &mesh.faces {
        if f.len() < 3 || f.iter().any(|&i| i >= pos.len()) {
            continue;
        }
        let v = |i: usize| CVert {
            pos: pos[f[i]],
            color: mesh.colors.get(f[i]).copied().unwrap_or(fallback),
        };
        // Fan-triangulate so non-planar quads cannot confuse the splitter.
        for i in 1..f.len() - 1 {
            let mut tri = vec![v(0), v(i), v(i + 1)];
            if flip {
                tri.reverse();
            }
            if let Some(p) = Poly::new(tri) {
                out.push(p);
            }
        }
    }
    out
}

fn csg(a: Vec<Poly>, b: Vec<Poly>, op: BooleanOp) -> Vec<Poly> {
    let mut a = Node::new(a);
    let mut b = Node::new(b);
    match op {
        BooleanOp::Union => {
            a.clip_to(&b);
            b.clip_to(&a);
            b.invert();
            b.clip_to(&a);
            b.invert();
            let mut bp = Vec::new();
            b.all_polys(&mut bp);
            a.build(bp);
        }
        BooleanOp::Subtract => {
            a.invert();
            a.clip_to(&b);
            b.clip_to(&a);
            b.invert();
            b.clip_to(&a);
            b.invert();
            let mut bp = Vec::new();
            b.all_polys(&mut bp);
            a.build(bp);
            a.invert();
        }
        BooleanOp::Intersect => {
            a.invert();
            b.clip_to(&a);
            b.invert();
            a.clip_to(&b);
            b.clip_to(&a);
            let mut bp = Vec::new();
            b.all_polys(&mut bp);
            a.build(bp);
            a.invert();
        }
    }
    let mut out = Vec::new();
    a.all_polys(&mut out);
    out
}

/// Combine `target` (with world matrix `target_m`) and `tool` (with `tool_m`). The result is in
/// the target's local space. Vertices are shared only between coplanar fragments, so hard
/// edges stay crisp.
pub fn boolean(
    target: &Mesh,
    target_m: Mat4,
    tool: &Mesh,
    tool_m: Mat4,
    op: BooleanOp,
    tool_color: [f32; 3],
) -> Result<Mesh, String> {
    let a = to_polys(target, target_m, [0.8, 0.8, 0.8]);
    let b = to_polys(tool, tool_m, tool_color);
    if a.is_empty() || b.is_empty() {
        return Err("Both parts need faces for a boolean".into());
    }
    // BSP recursion depth grows with polygon count; give it a large stack.
    let polys = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(move || csg(a, b, op))
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "Boolean failed".to_string())?;
    if polys.is_empty() {
        return Err("The boolean produced no geometry (the parts may not overlap as expected)".into());
    }
    let inv = target_m.inverse();
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut faces = Vec::new();
    let mut index: HashMap<(QPt, (i32, i32, i32)), usize> = HashMap::new();
    for p in &polys {
        let nq = {
            let n = p.plane.n * 50.0;
            (n.x.round() as i32, n.y.round() as i32, n.z.round() as i32)
        };
        let mut face = Vec::with_capacity(p.verts.len());
        for v in &p.verts {
            let local = inv.transform_point3(v.pos);
            let key = (quantize(local), nq);
            let i = *index.entry(key).or_insert_with(|| {
                positions.push(local);
                colors.push(v.color);
                positions.len() - 1
            });
            if face.last() != Some(&i) {
                face.push(i);
            }
        }
        if face.len() > 1 && face.first() == face.last() {
            face.pop();
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    let mut mesh = Mesh::new(positions, faces);
    mesh.colors = colors;
    mesh.recalculate_normals();
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn subtract_union_intersect_volumes() {
        let a = Mesh::cube(2.0);
        let b = Mesh::cube(2.0);
        let shift = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let sub = boolean(&a, Mat4::IDENTITY, &b, shift, BooleanOp::Subtract, [1.0; 3]).unwrap();
        assert!(approx(signed_volume(&sub), 4.0, 1e-3), "{}", signed_volume(&sub));
        let uni = boolean(&a, Mat4::IDENTITY, &b, shift, BooleanOp::Union, [1.0; 3]).unwrap();
        assert!(approx(signed_volume(&uni), 12.0, 1e-3), "{}", signed_volume(&uni));
        let int = boolean(&a, Mat4::IDENTITY, &b, shift, BooleanOp::Intersect, [1.0; 3]).unwrap();
        assert!(approx(signed_volume(&int), 4.0, 1e-3), "{}", signed_volume(&int));
        // Result is expressed in the target's local space.
        let moved = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let sub2 = boolean(&a, moved, &b, moved * shift, BooleanOp::Subtract, [1.0; 3]).unwrap();
        assert!(approx(signed_volume(&sub2), 4.0, 1e-3));
    }

    #[test]
    fn bevel_chamfers_a_cube_top() {
        let mut m = Mesh::cube(2.0);
        let top = (0..m.faces.len())
            .find(|&f| m.face_normal(f).y > 0.9)
            .unwrap();
        m.selected_faces.insert(top);
        let before = signed_volume(&m);
        bevel_selected(&mut m, 0.2).unwrap();
        let after = signed_volume(&m);
        // A 45° chamfer of width w on all four top edges of a 2-unit cube removes
        // 4 * (w²/2 * 2) minus the corner overlap; the volume must drop but stay close.
        assert!(after < before && after > before - 0.5, "{before} -> {after}");
        let max_y = m.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
        assert!(approx(max_y, 1.0, 1e-4));
        let top_face = &m.faces[top];
        assert!(top_face.iter().all(|&i| m.positions[i].x.abs() <= 0.8 + 1e-4));
    }

    #[test]
    fn lathe_and_sweep_are_closed_and_outward() {
        let vase = lathe(
            &[Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.6, 1.0), Vec2::new(0.0, 1.2)],
            24,
            360.0,
            true,
            [1.0; 3],
        )
        .unwrap();
        assert!(signed_volume(&vase) > 0.5);
        let cyl = lathe(&[Vec2::new(1.0, 0.0), Vec2::new(1.0, 2.0)], 64, 360.0, true, [1.0; 3]).unwrap();
        assert!(approx(signed_volume(&cyl), std::f32::consts::PI * 2.0, 0.05), "{}", signed_volume(&cyl));
        let tube = sweep(
            &[Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 2.0, 0.0)],
            &circle_section(0.1, 12),
            false,
            true,
            0.0,
            1.0,
            [1.0; 3],
        )
        .unwrap();
        assert!(signed_volume(&tube) > 0.0);
        let ring = sweep(
            &(0..16)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / 16.0;
                    Vec3::new(a.cos(), 0.0, a.sin())
                })
                .collect::<Vec<_>>(),
            &circle_section(0.2, 8),
            true,
            false,
            0.0,
            1.0,
            [1.0; 3],
        )
        .unwrap();
        assert!(signed_volume(&ring) > 0.0);
    }

    #[test]
    fn mirror_keeps_volume_positive() {
        let mut m = Mesh::cube(1.0);
        for p in &mut m.positions {
            p.x += 2.0;
        }
        let r = mirrored(&m, 0);
        assert!(signed_volume(&r) > 0.0);
        assert!(r.positions.iter().all(|p| p.x < 0.0));
    }
}
