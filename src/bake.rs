//! Texture baking for the game-ready exporter: turns dense vertex-painted parts into
//! low-poly meshes with UVs and one colour atlas.
//!
//! 1. Every distinct part (identical copies are baked once and share their atlas space) is
//!    simplified by quadric edge collapse. The collapse order is recorded with its error, so
//!    one error threshold for the whole model decides how far each part goes: flat parts
//!    drop to a handful of triangles and the budget is spent where shape needs it. A shape
//!    is never pushed past a small fraction of its own thickness, so an unreachable budget
//!    leaves boxes as boxes instead of wedges.
//! 2. Triangles nothing can see (the sides of books pressed together, the underside of a
//!    part standing on another) are dropped. Parts only hide each other when the export
//!    merges them; separate parts may be moved apart in the game.
//! 3. The remaining triangles are grouped into charts of similar normal, projected flat,
//!    cropped to their visible area and packed into the atlas. Charts whose source colour
//!    is uniform share a small swatch; the others get texels in proportion to how finely
//!    the original is painted there.
//! 4. Every texel takes the colour of the closest point on the ORIGINAL part (vertex colour
//!    times part colour, linear), found through a bounding volume hierarchy. Empty texels
//!    are filled from their nearest neighbours so filtering and mipmaps never show seams.
use crate::scene::SceneObject;
use glam::{DVec3, Mat4, Vec2, Vec3};
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A simplified part in its own local space, ready to be written with `TEXCOORD_0`.
#[derive(Default, Clone)]
pub struct BakedPart {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

pub struct Bake {
    /// One entry per input object, in order.
    pub parts: Vec<BakedPart>,
    /// Optional simplified collision geometry, before hidden faces are removed.
    pub collision_parts: Vec<BakedPart>,
    /// The atlas as an sRGB-encoded RGB PNG.
    pub png: Vec<u8>,
    pub size: u32,
    /// Atlas texels per unit length on the most finely painted surfaces.
    pub texels_per_unit: f32,
    /// Whether any chart needed more than a flat colour or gentle shading.
    pub painted: bool,
    /// Share of the resolution the painted detail asked for that fitted the atlas (1 = all).
    pub detail: f32,
}

pub const MIN_SIZE: u32 = 64;
pub const MAX_SIZE: u32 = 8192;
pub const DEFAULT_SIZE: u32 = 2048;

/// Empty texels kept around every chart so neighbours never bleed into each other.
const MARGIN: u32 = 2;
/// Side of the square a uniform colour gets in the atlas.
const SWATCH: u32 = 8;
/// Triangles join a chart while their normal is within about 53 degrees of its first one,
/// which bounds the stretch of the flat projection to 1.67.
const CHART_COS: f32 = 0.6;
/// Edges sharper than about 40 degrees stay hard on the simplified mesh.
const CREASE_COS: f32 = 0.766;
/// However small the budget, a connected shape only deviates by this share of its thinnest
/// side (boxes stay boxes) and this share of its diagonal (blobs keep their form).
const THICKNESS_SHARE: f64 = 0.15;
const SIZE_SHARE: f64 = 0.005;
/// Texels per edge of the painted source mesh; more would only store interpolation.
const TEXELS_PER_SOURCE_EDGE: f32 = 2.0;
/// Directions tried from every surface sample before it counts as hidden.
const VISIBILITY_RAYS: usize = 16;

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

/// Runs `f` over `items` on all cores and returns the results in order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(items.len());
    if threads <= 1 {
        return items.iter().enumerate().map(|(i, item)| f(i, item)).collect();
    }
    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            return done;
                        }
                        done.push((i, f(i, &items[i])));
                    }
                })
            })
            .collect();
        for handle in handles {
            for (i, result) in handle.join().expect("bake worker panicked") {
                slots[i] = Some(result);
            }
        }
    });
    slots.into_iter().map(|slot| slot.expect("every item is processed")).collect()
}

pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

fn srgb8(color: [f32; 3]) -> [u8; 3] {
    color.map(|c| (linear_to_srgb(c) * 255.0).round() as u8)
}

/// Barycentric coordinates of the point of triangle `abc` closest to `p`.
fn closest_bary(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let result = (|| {
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return [1.0 - v, v, 0.0];
        }
        let cp = p - c;
        let (d5, d6) = (ab.dot(cp), ac.dot(cp));
        if d6 >= 0.0 && d5 <= d6 {
            return [0.0, 0.0, 1.0];
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return [1.0 - w, 0.0, w];
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return [0.0, 1.0 - w, w];
        }
        let denom = 1.0 / (va + vb + vc);
        let (v, w) = (vb * denom, vc * denom);
        [1.0 - v - w, v, w]
    })();
    if result.iter().all(|x| x.is_finite()) { result } else { [1.0, 0.0, 0.0] }
}

// ---------------------------------------------------------------------------------------
// The original part: triangles, colours and a closest-point index
// ---------------------------------------------------------------------------------------

/// A part as painted, in "sized" space: local space times the absolute part scale, so
/// lengths and areas are world lengths and areas whatever the part's rotation or mirroring.
struct Source {
    positions: Vec<Vec3>,
    /// Viewport colour per vertex: vertex colour times part colour, linear.
    colors: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    scale: Vec3,
    /// Image layers (titles, labels), in the part's local space.
    layers: Vec<crate::texture::TextureLayer>,
}

fn part_scale(obj: &SceneObject) -> Vec3 {
    let scale = obj.transform.scale.abs();
    if scale.is_finite() && scale.min_element() > 1e-9 { scale } else { Vec3::ONE }
}

impl Source {
    fn new(obj: &SceneObject) -> Self {
        let mesh = &obj.mesh;
        let scale = part_scale(obj);
        let positions: Vec<Vec3> = mesh.positions.iter().map(|&p| p * scale).collect();
        let colors = (0..positions.len())
            .map(|i| {
                let vc = mesh.colors.get(i).copied().unwrap_or([1.0; 3]);
                [0, 1, 2].map(|k| (vc[k] * obj.color[k]).clamp(0.0, 1.0))
            })
            .collect();
        let mut tris = Vec::new();
        for face in mesh.faces.iter() {
            if face.len() < 3 || face.iter().any(|&v| v >= positions.len()) {
                continue;
            }
            for i in 1..face.len() - 1 {
                let tri = [face[0] as u32, face[i] as u32, face[i + 1] as u32];
                let [a, b, c] = tri.map(|v| positions[v as usize]);
                let area = (b - a).cross(c - a).length_squared();
                if area > 0.0 && area.is_finite() {
                    tris.push(tri);
                }
            }
        }
        Self { positions, colors, tris, scale, layers: obj.textures.clone() }
    }
}

/// Identical parts (same geometry, colours and size) are baked once.
fn source_key(obj: &SceneObject) -> (u64, usize, usize) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |w: u64| {
        h = (h ^ w).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    };
    let mesh = &obj.mesh;
    for p in mesh.positions.iter() {
        mix(p.x.to_bits() as u64 | (p.y.to_bits() as u64) << 32);
        mix(p.z.to_bits() as u64);
    }
    for c in mesh.colors.iter() {
        mix(c[0].to_bits() as u64 | (c[1].to_bits() as u64) << 32);
        mix(c[2].to_bits() as u64);
    }
    for face in mesh.faces.iter() {
        mix(0xface_0000 + face.len() as u64);
        for &v in face {
            mix(v as u64);
        }
    }
    for v in obj.color.iter().chain(part_scale(obj).to_array().iter()) {
        mix(v.to_bits() as u64);
    }
    // Two books with different titles are different parts even on the same mesh.
    for layer in &obj.textures {
        mix(0x1a7e_0000 + ((layer.image.width as u64) << 32 | layer.image.height as u64));
        for chunk in layer.image.rgba.chunks(8) {
            mix(chunk.iter().fold(0u64, |word, &byte| word << 8 | byte as u64));
        }
        for v in [layer.origin, layer.u_axis, layer.v_axis].iter().flat_map(|v| v.to_array()).chain([layer.depth, layer.opacity]) {
            mix(v.to_bits() as u64);
        }
    }
    (h, mesh.positions.len(), mesh.faces.len())
}

#[derive(Clone, Copy)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first triangle. Inner node: index of the second child (the first follows it).
    a: u32,
    /// Leaf: triangle count. Inner node: 0.
    count: u32,
}

/// Bounding volume hierarchy over triangles, for closest-point and ray queries.
struct Bvh {
    nodes: Vec<Node>,
    /// Triangles in leaf order.
    tris: Vec<[u32; 3]>,
}

impl Bvh {
    const LEAF: usize = 8;

    fn new(positions: &[Vec3], tris: &[[u32; 3]]) -> Self {
        let centroids: Vec<Vec3> = tris
            .iter()
            .map(|t| (positions[t[0] as usize] + positions[t[1] as usize] + positions[t[2] as usize]) / 3.0)
            .collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = Vec::with_capacity(tris.len() / 3 + 1);
        if !order.is_empty() {
            Self::build(positions, tris, &centroids, &mut order, 0, &mut nodes);
        }
        Self { nodes, tris: order.iter().map(|&i| tris[i as usize]).collect() }
    }

    fn build(
        positions: &[Vec3],
        tris: &[[u32; 3]],
        centroids: &[Vec3],
        order: &mut [u32],
        first: usize,
        nodes: &mut Vec<Node>,
    ) {
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let (mut cmin, mut cmax) = (min, max);
        for &t in order.iter() {
            for &v in &tris[t as usize] {
                min = min.min(positions[v as usize]);
                max = max.max(positions[v as usize]);
            }
            cmin = cmin.min(centroids[t as usize]);
            cmax = cmax.max(centroids[t as usize]);
        }
        let index = nodes.len();
        nodes.push(Node { min, max, a: first as u32, count: order.len() as u32 });
        let spread = cmax - cmin;
        if order.len() <= Self::LEAF || spread.max_element() <= 0.0 {
            return;
        }
        let axis = if spread.x >= spread.y && spread.x >= spread.z {
            0
        } else if spread.y >= spread.z {
            1
        } else {
            2
        };
        let half = order.len() / 2;
        order.select_nth_unstable_by(half, |&a, &b| {
            centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
        });
        let (left, right) = order.split_at_mut(half);
        Self::build(positions, tris, centroids, left, first, nodes);
        nodes[index].a = nodes.len() as u32;
        nodes[index].count = 0;
        Self::build(positions, tris, centroids, right, first + half, nodes);
    }

    /// The triangle closest to `p` and the barycentric coordinates of the closest point.
    /// Surfaces facing away from `normal` count as twice as far, so the outside of a thin
    /// sheet is not painted with its inside.
    fn closest(&self, positions: &[Vec3], p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3])> {
        if self.nodes.is_empty() {
            return None;
        }
        let box_distance = |node: &Node| (p - p.clamp(node.min, node.max)).length_squared();
        let mut best = f32::INFINITY;
        let mut hit = None;
        let mut stack = [(0u32, 0.0f32); 64];
        let mut top = 1;
        stack[0] = (0, box_distance(&self.nodes[0]));
        while top > 0 {
            top -= 1;
            let (index, distance) = stack[top];
            if distance >= best {
                continue;
            }
            let node = &self.nodes[index as usize];
            if node.count > 0 {
                for tri in &self.tris[node.a as usize..(node.a + node.count) as usize] {
                    let [a, b, c] = tri.map(|v| positions[v as usize]);
                    let bary = closest_bary(p, a, b, c);
                    let q = a * bary[0] + b * bary[1] + c * bary[2];
                    let mut d = (p - q).length_squared();
                    if (b - a).cross(c - a).dot(normal) < 0.0 {
                        d = d * 4.0 + 1e-12;
                    }
                    if d < best {
                        best = d;
                        hit = Some((*tri, bary));
                    }
                }
                continue;
            }
            let (left, right) = (index + 1, node.a);
            let (dl, dr) = (box_distance(&self.nodes[left as usize]), box_distance(&self.nodes[right as usize]));
            // The nearer child is pushed last so it is visited first.
            let (near, far) = if dl <= dr { ((left, dl), (right, dr)) } else { ((right, dr), (left, dl)) };
            if top + 2 <= stack.len() {
                stack[top] = far;
                stack[top + 1] = near;
                top += 2;
            }
        }
        hit
    }

    /// Whether a ray from `origin` along `direction` meets any triangle, from either side.
    fn blocked(&self, positions: &[Vec3], origin: Vec3, direction: Vec3) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inverse = direction.recip();
        // Distance at which the ray enters a node's box, or infinity when it misses.
        let entry = |node: &Node| {
            let (t0, t1) = ((node.min - origin) * inverse, (node.max - origin) * inverse);
            let near = t0.min(t1).max_element().max(0.0);
            if near <= t0.max(t1).min_element() { near } else { f32::INFINITY }
        };
        if entry(&self.nodes[0]).is_infinite() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let index = stack[top];
            let node = &self.nodes[index as usize];
            if node.count > 0 {
                for tri in &self.tris[node.a as usize..(node.a + node.count) as usize] {
                    let [a, b, c] = tri.map(|v| positions[v as usize]);
                    let (e1, e2) = (b - a, c - a);
                    let h = direction.cross(e2);
                    let det = e1.dot(h);
                    if det.abs() < 1e-20 {
                        continue;
                    }
                    let s = origin - a;
                    let u = s.dot(h) / det;
                    if !(0.0..=1.0).contains(&u) {
                        continue;
                    }
                    let q = s.cross(e1);
                    let v = direction.dot(q) / det;
                    if v < 0.0 || u + v > 1.0 {
                        continue;
                    }
                    if e2.dot(q) / det > 0.0 {
                        return true;
                    }
                }
                continue;
            }
            // Nearest box first: what hides a surface is usually right next to it.
            let (left, right) = (index + 1, node.a);
            let (tl, tr) = (entry(&self.nodes[left as usize]), entry(&self.nodes[right as usize]));
            let (near, far, t_far) = if tl <= tr { (left, right, tr) } else { (right, left, tl) };
            if top + 2 <= stack.len() {
                if t_far.is_finite() {
                    stack[top] = far;
                    top += 1;
                }
                if tl.min(tr).is_finite() {
                    stack[top] = near;
                    top += 1;
                }
            }
        }
        false
    }
}

impl Source {
    /// Colour of the painted surface closest to `p`, and the size of the triangle found
    /// there (the edge length of a square of twice its area).
    fn color_at(&self, index: &Bvh, p: Vec3, normal: Vec3) -> ([f32; 3], f32) {
        let Some((tri, bary)) = index.closest(&self.positions, p, normal) else {
            return ([1.0; 3], f32::INFINITY);
        };
        let [ca, cb, cc] = tri.map(|v| self.colors[v as usize]);
        let [a, b, c] = tri.map(|v| self.positions[v as usize]);
        let mut color = [0, 1, 2].map(|k| ca[k] * bary[0] + cb[k] * bary[1] + cc[k] * bary[2]);
        let mut size = (b - a).cross(c - a).length().sqrt();
        if !self.layers.is_empty() {
            // Layers live in local space; the source is in sized space (local times scale).
            let [la, lb, lc] = [a, b, c].map(|v| v / self.scale);
            let point = la * bary[0] + lb * bary[1] + lc * bary[2];
            let local_normal = (lb - la).cross(lc - la).normalize_or_zero();
            for layer in &self.layers {
                if layer.uv(point, local_normal).is_some() {
                    // An image is as detailed as its pixels, however few triangles carry it.
                    let texel = ((layer.u_axis * self.scale).length() / layer.image.width as f32)
                        .min((layer.v_axis * self.scale).length() / layer.image.height as f32);
                    size = size.min(texel);
                }
            }
            color = crate::texture::apply_layers(&self.layers, color, point, local_normal);
        }
        (color, size)
    }
}

// ---------------------------------------------------------------------------------------
// Simplification: quadric edge collapse with a recorded collapse order
// ---------------------------------------------------------------------------------------

/// Error quadric: `eval(p) = pᵀAp + 2b·p + c`, the summed squared distance to its planes.
#[derive(Clone, Copy, Default)]
struct Quadric {
    a: [f64; 6],
    b: DVec3,
    c: f64,
    /// Summed weight of the planes, so `eval / weight` is a mean squared distance.
    weight: f64,
}

impl Quadric {
    fn plane(normal: DVec3, point: DVec3, weight: f64) -> Self {
        let d = -normal.dot(point);
        let n = normal;
        Self {
            a: [n.x * n.x, n.x * n.y, n.x * n.z, n.y * n.y, n.y * n.z, n.z * n.z].map(|v| v * weight),
            b: n * d * weight,
            c: d * d * weight,
            weight,
        }
    }

    fn add(&mut self, other: &Quadric) {
        for (mine, theirs) in self.a.iter_mut().zip(other.a) {
            *mine += theirs;
        }
        self.b += other.b;
        self.c += other.c;
        self.weight += other.weight;
    }

    fn eval(&self, p: DVec3) -> f64 {
        let [xx, xy, xz, yy, yz, zz] = self.a;
        let ap = DVec3::new(
            xx * p.x + xy * p.y + xz * p.z,
            xy * p.x + yy * p.y + yz * p.z,
            xz * p.x + yz * p.y + zz * p.z,
        );
        p.dot(ap) + 2.0 * self.b.dot(p) + self.c
    }

    /// The point with the least error, pulled gently towards `anchor` wherever the planes
    /// leave it free (along a crease, anywhere on a flat area).
    fn minimum(&self, anchor: DVec3) -> Option<DVec3> {
        let [xx, xy, xz, yy, yz, zz] = self.a;
        let w = 1e-7 * (xx + yy + zz).max(1e-300);
        let (xx, yy, zz) = (xx + w, yy + w, zz + w);
        let rhs = anchor * w - self.b;
        let (c0, c1, c2) = (yy * zz - yz * yz, xz * yz - xy * zz, xy * yz - xz * yy);
        let det = xx * c0 + xy * c1 + xz * c2;
        if !det.is_finite() || det.abs() < 1e-300 {
            return None;
        }
        let inv = 1.0 / det;
        let p = DVec3::new(
            (c0 * rhs.x + c1 * rhs.y + c2 * rhs.z) * inv,
            (c1 * rhs.x + (xx * zz - xz * xz) * rhs.y + (xy * xz - xx * yz) * rhs.z) * inv,
            (c2 * rhs.x + (xy * xz - xx * yz) * rhs.y + (xx * yy - xy * xy) * rhs.z) * inv,
        );
        p.is_finite().then_some(p)
    }
}

struct Collapse {
    from: u32,
    to: u32,
    position: DVec3,
    /// Highest error of this and every earlier collapse of the part, so thresholds cut the
    /// log at one point.
    cost: f64,
    removed: u32,
}

/// A part welded by position (colour seams and hard edges share vertices again).
struct Welded {
    positions: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
}

fn weld(source: &Source) -> Welded {
    let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &p in &source.positions {
        min = min.min(p);
        max = max.max(p);
    }
    // Stay below the shortest edge so welding cannot merge its endpoints.
    let shortest_edge = source.tris.iter().flat_map(|tri| {
        let [a, b, c] = tri.map(|v| source.positions[v as usize].as_dvec3());
        [(b - a).length(), (c - b).length(), (a - c).length()]
    }).fold(f64::INFINITY, f64::min);
    let cell = (((max - min).as_dvec3().length() * 1e-6).min(shortest_edge * 0.25)).max(f64::MIN_POSITIVE);
    let mut lookup: HashMap<[i64; 3], u32> = HashMap::with_capacity(source.positions.len());
    let mut positions = Vec::new();
    let remap: Vec<u32> = source
        .positions
        .iter()
        .map(|p| {
            let p = p.as_dvec3();
            let key = [(p.x / cell).round() as i64, (p.y / cell).round() as i64, (p.z / cell).round() as i64];
            *lookup.entry(key).or_insert_with(|| {
                positions.push(p);
                positions.len() as u32 - 1
            })
        })
        .collect();
    let tris = source
        .tris
        .iter()
        .map(|t| t.map(|v| remap[v as usize]))
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    Welded { positions, tris }
}

struct Entry {
    cost: f64,
    a: u32,
    b: u32,
    versions: (u32, u32),
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost).is_eq()
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    // Reversed: the heap pops the cheapest collapse first.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

struct Simplifier {
    positions: Vec<DVec3>,
    quadrics: Vec<Quadric>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    /// Triangles around every vertex.
    around: Vec<Vec<u32>>,
    /// Bumped when a vertex moves, so queued collapses that used the old one are skipped.
    versions: Vec<u32>,
    marks: Vec<u32>,
    stamp: u32,
    pairs: Vec<(u32, u32)>,
    /// Largest mean squared deviation allowed at each vertex (see `THICKNESS_SHARE`).
    limits: Vec<f64>,
}

const DEAD: u32 = u32::MAX;

impl Simplifier {
    fn new(welded: &Welded) -> Self {
        let count = welded.positions.len();
        let mut quadrics = vec![Quadric::default(); count];
        let mut around = vec![Vec::new(); count];
        let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(welded.tris.len() * 3);
        for (ti, tri) in welded.tris.iter().enumerate() {
            let [a, b, c] = tri.map(|v| welded.positions[v as usize]);
            let cross = (b - a).cross(c - a);
            let area = cross.length() * 0.5;
            if area > 0.0 {
                let q = Quadric::plane(cross / (area * 2.0), a, area);
                for &v in tri {
                    quadrics[v as usize].add(&q);
                }
            }
            for k in 0..3 {
                around[tri[k] as usize].push(ti as u32);
                let (u, v) = (tri[k], tri[(k + 1) % 3]);
                edges.push((u.min(v), u.max(v), ti as u32));
            }
        }
        // Open borders are held in place by a plane through each border edge, square to
        // its triangle; without it sheets would shrink from their edges.
        edges.sort_unstable();
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
                j += 1;
            }
            if j - i == 1 {
                let (u, v, ti) = edges[i];
                let [a, b, c] = welded.tris[ti as usize].map(|x| welded.positions[x as usize]);
                let (pu, pv) = (welded.positions[u as usize], welded.positions[v as usize]);
                let edge = pv - pu;
                if let Some(normal) = edge.cross((b - a).cross(c - a)).try_normalize() {
                    let q = Quadric::plane(normal, pu, edge.length_squared() * 10.0);
                    quadrics[u as usize].add(&q);
                    quadrics[v as usize].add(&q);
                }
            }
            i = j;
        }
        // Each connected shape may only deviate by a share of its own thinnest side.
        let mut parent: Vec<u32> = (0..count as u32).collect();
        fn root(parent: &mut [u32], mut v: u32) -> u32 {
            while parent[v as usize] != v {
                parent[v as usize] = parent[parent[v as usize] as usize];
                v = parent[v as usize];
            }
            v
        }
        for tri in &welded.tris {
            let a = root(&mut parent, tri[0]);
            for &v in &tri[1..] {
                let b = root(&mut parent, v);
                parent[b as usize] = a;
            }
        }
        let mut boxes = vec![(DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)); count];
        for v in 0..count as u32 {
            let r = root(&mut parent, v) as usize;
            boxes[r].0 = boxes[r].0.min(welded.positions[v as usize]);
            boxes[r].1 = boxes[r].1.max(welded.positions[v as usize]);
        }
        let limits = (0..count as u32)
            .map(|v| {
                let (min, max) = boxes[root(&mut parent, v) as usize];
                let size = max - min;
                let thickness = size.min_element().max(size.max_element() * 0.02);
                (thickness * THICKNESS_SHARE).min(size.length() * SIZE_SHARE).powi(2)
            })
            .collect();
        Self {
            limits,
            positions: welded.positions.clone(),
            quadrics,
            tris: welded.tris.clone(),
            alive: vec![true; welded.tris.len()],
            around,
            versions: vec![0; count],
            marks: vec![0; count],
            stamp: 0,
            pairs: Vec::new(),
        }
    }

    /// Error and best position for merging `a` and `b`, and the mean squared distance the
    /// merged vertex would have from the surfaces it stands for.
    fn target(&self, a: u32, b: u32) -> (f64, DVec3, f64) {
        let mut q = self.quadrics[a as usize];
        q.add(&self.quadrics[b as usize]);
        let (pa, pb) = (self.positions[a as usize], self.positions[b as usize]);
        let mid = (pa + pb) * 0.5;
        let length2 = (pb - pa).length_squared();
        let position = match q.minimum(mid) {
            Some(p) if (p - mid).length_squared() <= 4.0 * length2 => p,
            _ => [pa, pb, mid]
                .into_iter()
                .min_by(|x, y| q.eval(*x).total_cmp(&q.eval(*y)))
                .unwrap_or(mid),
        };
        // The tiny length term orders collapses on flat areas: short edges first.
        let error = q.eval(position).max(0.0);
        let deviation = if q.weight > 0.0 { error / q.weight } else { 0.0 };
        (error + 1e-12 * length2 * length2, position, deviation)
    }

    fn other_vertices(tri: [u32; 3], v: u32) -> (u32, u32) {
        if tri[0] == v {
            (tri[1], tri[2])
        } else if tri[1] == v {
            (tri[2], tri[0])
        } else {
            (tri[0], tri[1])
        }
    }

    /// Whether merging `b` into `a` at `position` keeps the surface sound: no pinched or
    /// doubled triangles and no triangle turned over.
    fn valid(&mut self, a: u32, b: u32, position: DVec3) -> bool {
        self.stamp += 2;
        let (near_a, common_mark) = (self.stamp - 1, self.stamp);
        let mut shared = 0;
        self.pairs.clear();
        for &t in &self.around[a as usize] {
            let tri = self.tris[t as usize];
            if tri.contains(&b) {
                shared += 1;
                continue;
            }
            let (x, y) = Self::other_vertices(tri, a);
            self.marks[x as usize] = near_a;
            self.marks[y as usize] = near_a;
            self.pairs.push((x.min(y), x.max(y)));
        }
        for &t in &self.around[a as usize] {
            let tri = self.tris[t as usize];
            if tri.contains(&b) {
                let (x, y) = Self::other_vertices(tri, a);
                let third = if x == b { y } else { x };
                self.marks[third as usize] = near_a;
            }
        }
        if shared == 0 {
            return false;
        }
        self.pairs.sort_unstable();
        let mut common = 0;
        for &t in &self.around[b as usize] {
            let tri = self.tris[t as usize];
            let (x, y) = Self::other_vertices(tri, b);
            for v in [x, y] {
                if v != a && self.marks[v as usize] == near_a {
                    self.marks[v as usize] = common_mark;
                    common += 1;
                }
            }
            if !tri.contains(&a) && self.pairs.binary_search(&(x.min(y), x.max(y))).is_ok() {
                return false;
            }
        }
        if common != shared {
            return false;
        }
        for (moved, skip) in [(a, b), (b, a)] {
            for &t in &self.around[moved as usize] {
                let tri = self.tris[t as usize];
                if tri.contains(&skip) {
                    continue;
                }
                let old = tri.map(|v| self.positions[v as usize]);
                let new = tri.map(|v| if v == moved { position } else { self.positions[v as usize] });
                let n0 = (old[1] - old[0]).cross(old[2] - old[0]);
                let n1 = (new[1] - new[0]).cross(new[2] - new[0]);
                let scale = n0.length() * n1.length();
                if n0.length_squared() > 0.0 && n0.dot(n1) < 0.2 * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Merges `b` into `a` and returns how many triangles that removed.
    fn collapse(&mut self, a: u32, b: u32, position: DVec3) -> u32 {
        let mut removed = 0;
        for t in std::mem::take(&mut self.around[b as usize]) {
            let tri = self.tris[t as usize];
            if tri.contains(&a) {
                self.alive[t as usize] = false;
                removed += 1;
                for v in tri {
                    if v != b {
                        self.around[v as usize].retain(|&x| x != t);
                    }
                }
            } else {
                self.tris[t as usize] = tri.map(|v| if v == b { a } else { v });
                self.around[a as usize].push(t);
            }
        }
        self.positions[a as usize] = position;
        let q = self.quadrics[b as usize];
        self.quadrics[a as usize].add(&q);
        self.versions[a as usize] += 1;
        self.versions[b as usize] = DEAD;
        removed
    }

    fn push(&self, heap: &mut BinaryHeap<Entry>, a: u32, b: u32) {
        let (cost, _, deviation) = self.target(a, b);
        if deviation > self.limits[a as usize] {
            return;
        }
        heap.push(Entry { cost, a, b, versions: (self.versions[a as usize], self.versions[b as usize]) });
    }

    /// Collapses edges, cheapest first, until nothing more can go or the error passes
    /// `max_cost`, and returns the order they were done in.
    fn run(&mut self, max_cost: f64) -> Vec<Collapse> {
        let mut log = Vec::new();
        let mut highest = 0.0_f64;
        loop {
            // A rejected collapse can become possible once its surroundings change, so
            // the queue is rebuilt until a full pass makes no progress.
            let mut edges: Vec<(u32, u32)> = Vec::new();
            for (t, tri) in self.tris.iter().enumerate() {
                if self.alive[t] {
                    for k in 0..3 {
                        let (u, v) = (tri[k], tri[(k + 1) % 3]);
                        edges.push((u.min(v), u.max(v)));
                    }
                }
            }
            edges.sort_unstable();
            edges.dedup();
            let mut heap = BinaryHeap::with_capacity(edges.len());
            for (u, v) in edges {
                self.push(&mut heap, u, v);
            }
            let before = log.len();
            while let Some(entry) = heap.pop() {
                if entry.cost > max_cost {
                    break;
                }
                let (a, b) = (entry.a, entry.b);
                if entry.versions != (self.versions[a as usize], self.versions[b as usize]) {
                    continue;
                }
                let (_, position, _) = self.target(a, b);
                if !self.valid(a, b, position) {
                    continue;
                }
                let removed = self.collapse(a, b, position);
                highest = highest.max(entry.cost);
                log.push(Collapse { from: b, to: a, position, cost: highest, removed });
                self.stamp += 1;
                let stamp = self.stamp;
                for i in 0..self.around[a as usize].len() {
                    let tri = self.tris[self.around[a as usize][i] as usize];
                    for v in tri {
                        if v != a && self.marks[v as usize] != stamp {
                            self.marks[v as usize] = stamp;
                            self.push(&mut heap, a, v);
                        }
                    }
                }
            }
            if log.len() == before {
                return log;
            }
        }
    }
}

/// The simplified part after the first `count` collapses of `log`.
struct LowMesh {
    positions: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
}

fn replay(welded: &Welded, log: &[Collapse], count: usize) -> LowMesh {
    let mut target: Vec<u32> = (0..welded.positions.len() as u32).collect();
    let mut positions = welded.positions.clone();
    for collapse in &log[..count.min(log.len())] {
        target[collapse.from as usize] = collapse.to;
        positions[collapse.to as usize] = collapse.position;
    }
    for v in 0..target.len() {
        let mut root = v as u32;
        while target[root as usize] != root {
            root = target[root as usize];
        }
        let mut walk = v as u32;
        while target[walk as usize] != root {
            walk = std::mem::replace(&mut target[walk as usize], root);
        }
    }
    let mut compact = vec![u32::MAX; positions.len()];
    let mut out = LowMesh { positions: Vec::new(), tris: Vec::new() };
    for tri in &welded.tris {
        let tri = tri.map(|v| target[v as usize]);
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
            continue;
        }
        out.tris.push(tri.map(|v| {
            if compact[v as usize] == u32::MAX {
                compact[v as usize] = out.positions.len() as u32;
                out.positions.push(positions[v as usize].as_vec3());
            }
            compact[v as usize]
        }));
    }
    out
}

// ---------------------------------------------------------------------------------------
// Charts: groups of triangles unwrapped by one flat projection
// ---------------------------------------------------------------------------------------

struct Chart {
    tris: Vec<u32>,
    /// Corner positions on the chart's plane, in world units from its lower-left corner.
    corners: Vec<[Vec2; 3]>,
    extent: Vec2,
}

fn triangle_normal(low: &LowMesh, tri: u32) -> Vec3 {
    let [a, b, c] = low.tris[tri as usize].map(|v| low.positions[v as usize]);
    (b - a).cross(c - a)
}

/// Which triangles of a simplified part can be seen at all, and the surface points found
/// visible on each (with the spacing they were taken at).
struct Seen {
    keep: Vec<bool>,
    points: Vec<Vec<(Vec3, f32)>>,
}

/// Something that can hide a surface: triangles in some space and the way into it.
struct Occluder<'a> {
    matrix: Mat4,
    bvh: &'a Bvh,
    positions: &'a [Vec3],
}

fn ray_directions() -> Vec<Vec3> {
    // Evenly spread over the hemisphere, leaving out the last few degrees above the
    // surface: a slit seen only at a grazing angle is not worth texture space.
    (0..VISIBILITY_RAYS)
        .map(|i| {
            let z = 0.1 + 0.9 * (i as f32 + 0.5) / VISIBILITY_RAYS as f32;
            let angle = i as f32 * 2.399_963;
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * angle.cos(), r * angle.sin(), z)
        })
        .collect()
}

/// Samples triangle `tri` about every `spacing` and keeps the samples from which some ray
/// leaves the model without meeting anything, in at least one of the placements.
fn visible_points(
    low: &LowMesh,
    tri: u32,
    occluders: &[Occluder],
    rays: &[Vec3],
    spacing: f32,
    lift: f32,
) -> Vec<(Vec3, f32)> {
    let [a, b, c] = low.tris[tri as usize].map(|v| low.positions[v as usize]);
    let longest = (b - a).length().max((c - b).length()).max((a - c).length());
    let steps = ((longest / spacing).ceil() as usize).clamp(1, 16);
    let frames: Vec<Option<(Vec3, Vec3, Vec3)>> = occluders
        .iter()
        .map(|occluder| {
            let [wa, wb, wc] = [a, b, c].map(|p| occluder.matrix.transform_point3(p));
            let flip = if occluder.matrix.determinant() < 0.0 { -1.0 } else { 1.0 };
            let normal = ((wb - wa).cross(wc - wa) * flip).try_normalize()?;
            let tangent = normal.any_orthonormal_vector();
            Some((normal, tangent, normal.cross(tangent)))
        })
        .collect();
    let mut out = Vec::new();
    let mut test = |p: Vec3| {
        let seen = occluders.iter().zip(&frames).any(|(occluder, frame)| {
            let Some((normal, tangent, bitangent)) = frame else { return true };
            let origin = occluder.matrix.transform_point3(p) + *normal * lift;
            rays.iter().any(|ray| {
                let direction = *tangent * ray.x + *bitangent * ray.y + *normal * ray.z;
                !occluder.bvh.blocked(occluder.positions, origin, direction)
            })
        });
        if seen {
            out.push((p, longest / steps as f32));
        }
    };
    let n = steps as f32;
    let at = |u: f32, v: f32| a + (b - a) * u + (c - a) * v;
    for i in 0..steps {
        for j in 0..steps - i {
            test(at((i as f32 + 1.0 / 3.0) / n, (j as f32 + 1.0 / 3.0) / n));
            if j + 1 < steps - i {
                test(at((i as f32 + 2.0 / 3.0) / n, (j as f32 + 2.0 / 3.0) / n));
            }
        }
    }
    // A face is often seen only as a strip along one edge (a book standing next to a
    // slightly shorter one), so the edges get their own samples just inside the border.
    let centre = (a + b + c) / 3.0;
    for (p, q) in [(a, b), (b, c), (c, a)] {
        for i in 0..steps {
            let edge = p + (q - p) * ((i as f32 + 0.5) / n);
            let inward = centre - edge;
            test(edge + inward * (2.0 * lift / inward.length().max(1e-12)).min(0.5));
        }
    }
    out
}

fn build_charts(low: &LowMesh, seen: &Seen) -> Vec<Chart> {
    let count = low.tris.len();
    let normals: Vec<Vec3> = (0..count as u32).map(|t| triangle_normal(low, t)).collect();
    let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(count * 3);
    for (t, tri) in low.tris.iter().enumerate().filter(|(t, _)| seen.keep[*t]) {
        for k in 0..3 {
            let (u, v) = (tri[k], tri[(k + 1) % 3]);
            edges.push((u.min(v), u.max(v), t as u32));
        }
    }
    edges.sort_unstable();
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); count];
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        for x in i..j {
            for y in i..j {
                if x != y {
                    neighbours[edges[x].2 as usize].push(edges[y].2);
                }
            }
        }
        i = j;
    }
    // The largest triangles seed charts first, so charts grow from the big flat areas.
    let mut seeds: Vec<u32> = (0..count as u32).filter(|&t| seen.keep[t as usize]).collect();
    seeds.sort_unstable_by(|&a, &b| {
        normals[b as usize].length_squared().total_cmp(&normals[a as usize].length_squared())
    });
    let mut assigned = vec![false; count];
    let mut charts = Vec::new();
    for seed in seeds {
        if assigned[seed as usize] {
            continue;
        }
        let normal = normals[seed as usize].try_normalize().unwrap_or(Vec3::Y);
        let mut tris = vec![seed];
        assigned[seed as usize] = true;
        let mut next = 0;
        while next < tris.len() {
            let t = tris[next];
            next += 1;
            for &n in &neighbours[t as usize] {
                if assigned[n as usize] {
                    continue;
                }
                let facing = normals[n as usize].try_normalize().is_none_or(|x| x.dot(normal) >= CHART_COS);
                if facing {
                    assigned[n as usize] = true;
                    tris.push(n);
                }
            }
        }
        charts.push(flatten(low, tris, normal, seen));
    }
    charts
}

/// Projects a chart along `normal`, turns it to the tightest bounding rectangle and crops
/// that to the part of the chart that can be seen. What lies outside the crop still gets
/// texture coordinates, but nothing ever looks at them.
fn flatten(low: &LowMesh, tris: Vec<u32>, normal: Vec3, seen: &Seen) -> Chart {
    let tangent = normal.any_orthonormal_vector();
    let bitangent = normal.cross(tangent);
    let project = |p: Vec3| Vec2::new(p.dot(tangent), p.dot(bitangent));
    let flat: Vec<[Vec2; 3]> =
        tris.iter().map(|&t| low.tris[t as usize].map(|v| project(low.positions[v as usize]))).collect();
    let bounds = |axis: Vec2| {
        let side = axis.perp();
        let (mut min, mut max) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for corner in flat.iter().flatten() {
            let p = Vec2::new(corner.dot(axis), corner.dot(side));
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    };
    // Candidate directions come from the chart's own edges; boxes and panels line up.
    let step = (flat.len() / 64).max(1);
    let mut best = (Vec2::X, f32::INFINITY);
    for tri in flat.iter().step_by(step) {
        for k in 0..3 {
            let Some(axis) = (tri[(k + 1) % 3] - tri[k]).try_normalize() else { continue };
            let (min, max) = bounds(axis);
            let size = max - min;
            if size.x * size.y < best.1 {
                best = (axis, size.x * size.y);
            }
        }
    }
    let mut axis = best.0;
    let (min, max) = bounds(axis);
    if (max - min).y > (max - min).x {
        axis = axis.perp();
    }
    let side = axis.perp();
    let (mut min, mut max) = bounds(axis);
    let (mut seen_min, mut seen_max) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for &t in &tris {
        for &(p, spacing) in &seen.points[t as usize] {
            let flat = project(p);
            let p = Vec2::new(flat.dot(axis), flat.dot(side));
            seen_min = seen_min.min(p - spacing);
            seen_max = seen_max.max(p + spacing);
        }
    }
    if seen_min.x <= seen_max.x {
        min = min.max(seen_min);
        max = max.min(seen_max);
        // The crop may have made the chart taller than wide.
        if (max - min).y > (max - min).x {
            let turn = |p: Vec2| Vec2::new(p.y, -p.x);
            let (a, b) = (turn(min), turn(max));
            (min, max) = (a.min(b), a.max(b));
            axis = side;
        }
    }
    let side = axis.perp();
    let corners = flat.iter().map(|tri| tri.map(|c| Vec2::new(c.dot(axis), c.dot(side)) - min)).collect();
    Chart { tris, corners, extent: (max - min).max(Vec2::ZERO) }
}

/// A chart's texels: colour and how far the texel centre is from the chart (0 inside it,
/// infinite where the chart does not reach).
struct Tile {
    width: u32,
    height: u32,
    colors: Vec<[f32; 3]>,
    distance: Vec<f32>,
    /// Size of the source triangles found under the texels inside the chart.
    source_sizes: Vec<f32>,
}

fn tile_size(extent: Vec2, density: f32, margin: u32) -> (u32, u32) {
    let side = |length: f32| ((length * density).ceil() as u32).max(1).saturating_add(2 * margin);
    (side(extent.x), side(extent.y))
}

/// Colours a chart at `density` texels per unit. Texels up to one texel outside the chart
/// take the colour of its nearest edge so filtering at the border reads real surface.
/// `samples` squared points are averaged per texel.
fn sample_chart(
    low: &LowMesh,
    chart: &Chart,
    source: &Source,
    index: &Bvh,
    density: f32,
    margin: u32,
    samples: u32,
) -> Tile {
    let (width, height) = tile_size(chart.extent, density, margin);
    let texels = (width * height) as usize;
    let mut owner = vec![u32::MAX; texels];
    let mut distance = vec![f32::INFINITY; texels];
    let place = |corner: Vec2| (corner * density + Vec2::splat(margin as f32)).extend(0.0);
    for (i, corners) in chart.corners.iter().enumerate() {
        let [a, b, c] = corners.map(place);
        let (min, max) = (a.min(b).min(c) - 1.0, a.max(b).max(c) + 1.0);
        let x_range = (min.x.floor().max(0.0) as u32)..(max.x.ceil().max(0.0) as u32).min(width);
        for y in (min.y.floor().max(0.0) as u32)..(max.y.ceil().max(0.0) as u32).min(height) {
            for x in x_range.clone() {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let bary = closest_bary(p, a, b, c);
                let d = (a * bary[0] + b * bary[1] + c * bary[2] - p).length();
                let d = if d < 1e-4 { 0.0 } else { d };
                let texel = (y * width + x) as usize;
                if d <= 1.0 && d < distance[texel] {
                    distance[texel] = d;
                    owner[texel] = i as u32;
                }
            }
        }
    }
    let mut colors = vec![[0.0; 3]; texels];
    let mut source_sizes = Vec::new();
    for texel in 0..texels {
        let i = owner[texel];
        if i == u32::MAX {
            continue;
        }
        let [a, b, c] = chart.corners[i as usize].map(place);
        let tri = chart.tris[i as usize];
        let [pa, pb, pc] = low.tris[tri as usize].map(|v| low.positions[v as usize]);
        let normal = (pb - pa).cross(pc - pa);
        let (x, y) = ((texel as u32 % width) as f32, (texel as u32 / width) as f32);
        let samples = if distance[texel] > 0.0 { 1 } else { samples.max(1) };
        let mut sum = [0.0; 3];
        for sy in 0..samples {
            for sx in 0..samples {
                let offset = |s: u32| (s as f32 + 0.5) / samples as f32;
                let bary = closest_bary(Vec3::new(x + offset(sx), y + offset(sy), 0.0), a, b, c);
                let (color, source_size) =
                    source.color_at(index, pa * bary[0] + pb * bary[1] + pc * bary[2], normal);
                for k in 0..3 {
                    sum[k] += color[k];
                }
                if samples == 1 && distance[texel] == 0.0 {
                    source_sizes.push(source_size);
                }
            }
        }
        colors[texel] = sum.map(|v| v / (samples * samples) as f32);
    }
    Tile { width, height, colors, distance, source_sizes }
}

/// How much atlas a chart needs.
#[derive(Clone, Copy, PartialEq)]
enum Detail {
    /// One colour everywhere: the chart points at a shared swatch.
    Flat([u8; 3]),
    /// Gentle shading only: half the resolution is plenty.
    Smooth,
    Painted,
}

fn chart_detail(tile: &Tile) -> Detail {
    let inside = |i: usize| tile.distance[i] == 0.0;
    let any_inside = (0..tile.colors.len()).any(inside);
    let used = |i: usize| if any_inside { inside(i) } else { tile.distance[i].is_finite() };
    let encoded: Vec<[f32; 3]> = tile.colors.iter().map(|c| c.map(|v| linear_to_srgb(v) * 255.0)).collect();
    let (mut min, mut max, mut sum, mut count) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3], [0.0f32; 3], 0.0);
    let mut step = 0.0_f32;
    let width = tile.width as usize;
    for i in (0..encoded.len()).filter(|&i| used(i)) {
        count += 1.0;
        for k in 0..3 {
            min[k] = min[k].min(encoded[i][k]);
            max[k] = max[k].max(encoded[i][k]);
            sum[k] += tile.colors[i][k];
        }
        for other in [(i % width + 1 < width).then_some(i + 1), (i + width < encoded.len()).then_some(i + width)] {
            if let Some(other) = other.filter(|&o| used(o)) {
                for k in 0..3 {
                    step = step.max((encoded[i][k] - encoded[other][k]).abs());
                }
            }
        }
    }
    if count == 0.0 {
        return Detail::Flat([255; 3]);
    }
    let range = (0..3).map(|k| max[k] - min[k]).fold(0.0, f32::max);
    if range <= 2.5 {
        Detail::Flat(srgb8(sum.map(|v| v / count)))
    } else if step <= 2.5 {
        Detail::Smooth
    } else {
        Detail::Painted
    }
}

/// Places rectangles in rows, tallest first. `None` when they do not fit.
fn pack(sizes: &[(u32, u32)], atlas: u32) -> Option<Vec<(u32, u32)>> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_unstable_by(|&a, &b| sizes[b].1.cmp(&sizes[a].1).then(sizes[b].0.cmp(&sizes[a].0)));
    let mut places = vec![(0, 0); sizes.len()];
    let (mut x, mut y, mut row) = (0u32, 0u32, 0u32);
    for i in order {
        let (w, h) = sizes[i];
        if w > atlas || h > atlas {
            return None;
        }
        if x + w > atlas {
            x = 0;
            y += row;
            row = 0;
        }
        if y + h > atlas {
            return None;
        }
        places[i] = (x, y);
        x += w;
        row = row.max(h);
    }
    Some(places)
}

// ---------------------------------------------------------------------------------------
// The whole bake
// ---------------------------------------------------------------------------------------

struct Unique {
    source: Source,
    welded: Welded,
    log: Vec<Collapse>,
    /// How many of the input objects are this part.
    copies: usize,
}

struct Unwrapped {
    low: LowMesh,
    charts: Vec<Chart>,
    index: Bvh,
}

/// Simplifies `objects` to about `max_triangles` (in total; `None` keeps every triangle the
/// shape needs) and bakes their colours into an atlas of at most `size` x `size` pixels
/// (smaller when that holds all the detail the parts have). `merged` says the
/// parts will be joined into one mesh, so faces hidden by OTHER parts can go too.
pub fn bake(
    objects: &[&SceneObject],
    max_triangles: Option<usize>,
    mut size: u32,
    merged: bool,
    keep_collision: bool,
) -> Result<Bake, String> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&size) {
        return Err(format!("The texture size must be between {MIN_SIZE} and {MAX_SIZE} pixels"));
    }
    // Identical parts are baked once.
    let keys = parallel_map(objects, |_, obj| {
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let matrix = obj.transform.matrix();
        for &p in obj.mesh.positions.iter() {
            let p = matrix.transform_point3(p);
            min = min.min(p);
            max = max.max(p);
        }
        (source_key(obj), min, max)
    });
    let mut unique_of: HashMap<(u64, usize, usize), usize> = HashMap::new();
    let mut representatives: Vec<Vec<usize>> = Vec::new();
    let mut object_unique = Vec::with_capacity(objects.len());
    let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for (i, (key, part_min, part_max)) in keys.iter().enumerate() {
        if part_min.is_finite() && part_max.is_finite() {
            min = min.min(*part_min);
            max = max.max(*part_max);
        }
        let unique = *unique_of.entry(*key).or_insert_with(|| {
            representatives.push(Vec::new());
            representatives.len() - 1
        });
        representatives[unique].push(i);
        object_unique.push(unique);
    }
    let diagonal = if min.is_finite() && max.is_finite() { (max - min).length() } else { 1.0 };
    // Collapses below this error do not change the shape visibly; they are always taken.
    let free = 1e-12 * (diagonal as f64).powi(4);

    let budgeted = max_triangles.is_some();
    let uniques: Vec<Unique> = parallel_map(&representatives, |_, copies| {
        let source = Source::new(objects[copies[0]]);
        let welded = weld(&source);
        let log = Simplifier::new(&welded).run(if budgeted { f64::INFINITY } else { free });
        Unique { source, welded, log, copies: copies.len() }
    });

    // One error threshold for the whole model: take the cheapest collapses anywhere until
    // the budget is met.
    let mut order: Vec<(f64, u32, u32)> = uniques
        .iter()
        .enumerate()
        .flat_map(|(i, u)| u.log.iter().map(move |c| (c.cost, i as u32, c.removed * u.copies as u32)))
        .collect();
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let placement = |object: usize| {
        objects[object].transform.matrix() * Mat4::from_scale(part_scale(objects[object]).recip())
    };
    let rays = ray_directions();
    let budget = max_triangles.unwrap_or(usize::MAX);
    let mut aim = budget;
    let mut pass = 0;
    let (lows, seen) = loop {
        let mut total: usize = uniques.iter().map(|u| u.welded.tris.len() * u.copies).sum();
        let mut taken = vec![0usize; uniques.len()];
        let mut squeezed = false;
        for &(cost, unique, removed) in &order {
            if cost > free {
                if total <= aim {
                    break;
                }
                squeezed = true;
            }
            taken[unique as usize] += 1;
            total = total.saturating_sub(removed as usize);
        }
        let lows: Vec<LowMesh> =
            parallel_map(&uniques, |i, unique| replay(&unique.welded, &unique.log, taken[i]));

        // Drop what cannot be seen. Merged parts hide each other where they stand in the
        // model; a separate part is only hidden by itself.
        let mut world_positions: Vec<Vec3> = Vec::new();
        let mut world_tris: Vec<[u32; 3]> = Vec::new();
        if merged {
            for (object, &unique) in object_unique.iter().enumerate() {
                let matrix = placement(object);
                let base = world_positions.len() as u32;
                world_positions.extend(lows[unique].positions.iter().map(|&p| matrix.transform_point3(p)));
                world_tris.extend(lows[unique].tris.iter().map(|t| t.map(|v| v + base)));
            }
        }
        let world = Bvh::new(&world_positions, &world_tris);
        let own: Vec<Bvh> = if merged {
            Vec::new()
        } else {
            parallel_map(&lows, |_, low| Bvh::new(&low.positions, &low.tris))
        };
        let triangles: Vec<(usize, u32)> = lows
            .iter()
            .enumerate()
            .flat_map(|(u, low)| (0..low.tris.len() as u32).map(move |t| (u, t)))
            .collect();
        let mut found = parallel_map(&triangles, |_, &(u, t)| {
            let occluders: Vec<Occluder> = if merged {
                representatives[u]
                    .iter()
                    .map(|&object| Occluder { matrix: placement(object), bvh: &world, positions: &world_positions })
                    .collect()
            } else {
                vec![Occluder { matrix: Mat4::IDENTITY, bvh: &own[u], positions: &lows[u].positions }]
            };
            visible_points(&lows[u], t, &occluders, &rays, diagonal * 0.005, diagonal * 1e-4)
        })
        .into_iter();
        let seen: Vec<Seen> = lows
            .iter()
            .map(|low| {
                let points: Vec<Vec<(Vec3, f32)>> = found.by_ref().take(low.tris.len()).collect();
                Seen { keep: points.iter().map(|p| !p.is_empty()).collect(), points }
            })
            .collect();
        // The budget is for what gets exported. When hidden faces took a good part of it,
        // simplify once more with the budget raised by that share.
        let kept: usize = seen
            .iter()
            .zip(&uniques)
            .map(|(seen, unique)| seen.keep.iter().filter(|&&k| k).count() * unique.copies)
            .sum();
        pass += 1;
        if pass == 1 && squeezed && kept > 0 && kept < budget - budget / 10 {
            aim = (budget as f64 * total as f64 / kept as f64 * 0.97) as usize;
            continue;
        }
        break (lows, seen);
    };

    let lows: Vec<(LowMesh, Seen)> = lows.into_iter().zip(seen).collect();
    let charts = parallel_map(&lows, |i, (low, seen)| {
        (build_charts(low, seen), Bvh::new(&uniques[i].source.positions, &uniques[i].source.tris))
    });
    let unwrapped: Vec<Unwrapped> = lows
        .into_iter()
        .zip(charts)
        .map(|((low, _), (charts, index))| Unwrapped { low, charts, index })
        .collect();

    // Every chart of every distinct part, largest first so workers share the load evenly.
    let mut jobs: Vec<(usize, usize)> = unwrapped
        .iter()
        .enumerate()
        .flat_map(|(u, part)| (0..part.charts.len()).map(move |c| (u, c)))
        .collect();
    let area = |&(u, c): &(usize, usize)| {
        let extent = unwrapped[u].charts[c].extent;
        extent.x * extent.y
    };
    jobs.sort_unstable_by(|a, b| area(b).total_cmp(&area(a)));
    let total_area: f32 = jobs.iter().map(area).sum();
    let atlas_texels = (size as f32) * (size as f32);
    // A first look at half the texels the charts would get if all were painted alike:
    // is the chart one colour, and how finely is the original painted there?
    let estimate = if total_area > 0.0 { (atlas_texels * 0.6 / total_area).sqrt() } else { 1.0 };
    let probe = estimate * 0.5;
    let details: Vec<(Detail, f32)> = parallel_map(&jobs, |_, &(u, c)| {
        let part = &unwrapped[u];
        let chart = &part.charts[c];
        // Small charts are still looked at closely enough to see whether they are painted.
        let span = chart.extent.max_element().max(1e-9);
        let density = probe.max(8.0 / span).min(probe * 64.0).min(size as f32 / span);
        let mut tile = sample_chart(&part.low, chart, &uniques[u].source, &part.index, density, 0, 1);
        let detail = chart_detail(&tile);
        let sizes = &mut tile.source_sizes;
        let wanted = if sizes.is_empty() {
            density
        } else {
            // The finer third of the chart sets the pace, so lettering on a mostly plain
            // surface stays sharp.
            let third = sizes.len() / 3;
            let (_, fine, _) = sizes.select_nth_unstable_by(third, f32::total_cmp);
            TEXELS_PER_SOURCE_EDGE / fine.max(1e-9)
        };
        (detail, if detail == Detail::Smooth { wanted * 0.5 } else { wanted })
    });

    // Swatches for the flat colours, then the densest layout of the other charts that fits.
    let mut swatches: Vec<[u8; 3]> = details
        .iter()
        .filter_map(|(d, _)| if let Detail::Flat(color) = d { Some(*color) } else { None })
        .collect();
    swatches.sort_unstable();
    swatches.dedup();
    let layout = |share: f32, size: u32| {
        let mut sizes: Vec<(u32, u32)> = vec![(SWATCH, SWATCH); swatches.len()];
        for (job, (detail, wanted)) in jobs.iter().zip(&details) {
            if !matches!(detail, Detail::Flat(_)) {
                sizes.push(tile_size(unwrapped[job.0].charts[job.1].extent, wanted * share, MARGIN));
            }
        }
        pack(&sizes, size)
    };
    // `share` 1 gives every chart the texels its source detail asks for.
    let mut share = 0.0_f32;
    let mut places = layout(share, size)
        .ok_or("The parts have too many separate surfaces for this texture size; use a larger one")?;
    if let Some(found) = layout(1.0, size) {
        (places, share) = (found, 1.0);
        // `size` is the most the texture may take; a smaller one that holds the same
        // detail is better for the game.
        while size / 2 >= MIN_SIZE {
            let Some(found) = layout(1.0, size / 2) else { break };
            (places, size) = (found, size / 2);
        }
    } else {
        let (mut low, mut high) = (0.0_f32, 1.0_f32);
        for _ in 0..20 {
            let middle = (low + high) * 0.5;
            match layout(middle, size) {
                Some(found) => (places, low) = (found, middle),
                None => high = middle,
            }
        }
        share = low;
    }
    let painted = details.iter().any(|(d, _)| *d == Detail::Painted);
    let density = details
        .iter()
        .filter(|(d, _)| *d == Detail::Painted)
        .map(|(_, wanted)| wanted * share)
        .fold(0.0, f32::max);

    // Fill the atlas.
    let side = size as usize;
    let mut pixels = vec![[0u8; 3]; side * side];
    let mut filled = vec![false; side * side];
    for (swatch, &(x, y)) in swatches.iter().zip(&places) {
        for row in y..y + SWATCH {
            for column in x..x + SWATCH {
                let texel = row as usize * side + column as usize;
                pixels[texel] = *swatch;
                filled[texel] = true;
            }
        }
    }
    let mut next_place = swatches.len();
    // Where each chart sits: the texel of its lower-left corner and its density, or the
    // centre of its swatch.
    let mut chart_places: Vec<Vec<(Vec2, f32)>> =
        unwrapped.iter().map(|part| vec![(Vec2::ZERO, 0.0); part.charts.len()]).collect();
    let mut to_sample: Vec<(usize, usize, f32, (u32, u32))> = Vec::new();
    for (job, (detail, wanted)) in jobs.iter().zip(&details) {
        match detail {
            Detail::Flat(color) => {
                let (x, y) = places[swatches.binary_search(color).expect("swatch exists")];
                let centre = Vec2::new(x as f32, y as f32) + SWATCH as f32 * 0.5;
                chart_places[job.0][job.1] = (centre, 0.0);
            }
            _ => {
                let (x, y) = places[next_place];
                next_place += 1;
                let chart_density = wanted * share;
                chart_places[job.0][job.1] =
                    (Vec2::new((x + MARGIN) as f32, (y + MARGIN) as f32), chart_density);
                to_sample.push((job.0, job.1, chart_density, (x, y)));
            }
        }
    }
    // With fewer texels than the source has detail, two by two samples per texel keep
    // fine painted lines from flickering in and out.
    let samples = if share < 0.99 { 2 } else { 1 };
    let tiles = parallel_map(&to_sample, |_, &(u, c, chart_density, _)| {
        let part = &unwrapped[u];
        sample_chart(&part.low, &part.charts[c], &uniques[u].source, &part.index, chart_density, MARGIN, samples)
    });
    for (tile, &(_, _, _, (x, y))) in tiles.iter().zip(&to_sample) {
        for row in 0..tile.height {
            for column in 0..tile.width {
                let from = (row * tile.width + column) as usize;
                if tile.distance[from].is_finite() {
                    let texel = (y + row) as usize * side + (x + column) as usize;
                    pixels[texel] = srgb8(tile.colors[from]);
                    filled[texel] = true;
                }
            }
        }
    }
    drop(tiles);
    dilate(&mut pixels, &mut filled, side);

    let mut png = Vec::new();
    {
        use image::ImageEncoder;
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            &mut png,
            image::codecs::png::CompressionType::Fast,
            image::codecs::png::FilterType::Adaptive,
        );
        encoder
            .write_image(bytemuck::cast_slice(&pixels), size, size, image::ExtendedColorType::Rgb8)
            .map_err(|e| format!("Could not encode the texture: {e}"))?;
    }

    // Vertices: one per (position, chart, shading normal).
    let built: Vec<BakedPart> = parallel_map(&unwrapped, |u, part| {
        build_part(part, &chart_places[u], uniques[u].source.scale, size as f32)
    });
    let collision_parts = if keep_collision {
        object_unique.iter().map(|&u| {
            let low = &unwrapped[u].low;
            let scale = uniques[u].source.scale;
            BakedPart {
                positions: low.positions.iter().map(|&p| (p / scale).to_array()).collect(),
                indices: low.tris.iter().flatten().copied().collect(),
                ..Default::default()
            }
        }).collect()
    } else {
        Vec::new()
    };
    Ok(Bake {
        collision_parts,
        parts: object_unique.iter().map(|&u| built[u].clone()).collect(),
        png,
        size,
        texels_per_unit: density,
        painted,
        detail: share,
    })
}

/// Spreads colours into every empty texel from its nearest filled one.
fn dilate(pixels: &mut [[u8; 3]], filled: &mut [bool], side: usize) {
    let mut frontier: Vec<u32> = (0..filled.len() as u32).filter(|&i| filled[i as usize]).collect();
    if frontier.is_empty() {
        pixels.fill([255; 3]);
        return;
    }
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &texel in &frontier {
            let (x, y) = (texel as usize % side, texel as usize / side);
            let neighbours = [
                (x > 0).then(|| texel as usize - 1),
                (x + 1 < side).then(|| texel as usize + 1),
                (y > 0).then(|| texel as usize - side),
                (y + 1 < side).then(|| texel as usize + side),
            ];
            for neighbour in neighbours.into_iter().flatten() {
                if !filled[neighbour] {
                    filled[neighbour] = true;
                    pixels[neighbour] = pixels[texel as usize];
                    next.push(neighbour as u32);
                }
            }
        }
        frontier = next;
    }
}

fn build_part(part: &Unwrapped, places: &[(Vec2, f32)], scale: Vec3, size: f32) -> BakedPart {
    let low = &part.low;
    let normals: Vec<Vec3> = (0..low.tris.len() as u32).map(|t| triangle_normal(low, t)).collect();
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); low.positions.len()];
    for (t, tri) in low.tris.iter().enumerate() {
        for &v in tri {
            around[v as usize].push(t as u32);
        }
    }
    let mut out = BakedPart::default();
    let mut lookup: HashMap<(u32, u32, [i16; 3]), u32> = HashMap::new();
    for (c, chart) in part.charts.iter().enumerate() {
        let (origin, density) = places[c];
        for (i, &t) in chart.tris.iter().enumerate() {
            let face = normals[t as usize].try_normalize().unwrap_or(Vec3::Y);
            for k in 0..3 {
                let v = low.tris[t as usize][k];
                // Smooth across gentle edges, hard across sharp ones; weighted by area.
                let mut normal = Vec3::ZERO;
                for &other in &around[v as usize] {
                    let n = normals[other as usize];
                    if n.try_normalize().is_some_and(|unit| unit.dot(face) >= CREASE_COS) {
                        normal += n;
                    }
                }
                // Back to the part's local space, where normals scale inversely to lengths.
                let normal = (normal.try_normalize().unwrap_or(face) * scale).try_normalize().unwrap_or(face);
                let key = (v, c as u32, normal.to_array().map(|x| (x * 16000.0) as i16));
                let index = *lookup.entry(key).or_insert_with(|| {
                    out.positions.push((low.positions[v as usize] / scale).to_array());
                    out.normals.push(normal.to_array());
                    out.uvs.push(((origin + chart.corners[i][k] * density) / size).to_array());
                    out.positions.len() as u32 - 1
                });
                out.indices.push(index);
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::scene::Scene;

    /// A densely tessellated unit cube whose +X side is painted with red and blue stripes.
    pub fn striped_block() -> Scene {
        let mut mesh = Mesh::cube(1.0);
        for _ in 0..4 {
            mesh.selected_faces = (0..mesh.faces.len()).collect();
            mesh.subdivide_or_loop_cut();
        }
        let colors: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .map(|p| {
                if p.x < 0.49 {
                    [0.8, 0.8, 0.8]
                } else if ((p.y + 0.5) * 4.0).floor() as i32 % 2 == 0 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect();
        mesh.colors = colors.into();
        let mut scene = Scene::empty();
        scene.add_mesh_object("Block", mesh, [1.0, 1.0, 1.0]);
        scene
    }

    #[test]
    fn flat_parts_simplify_to_their_shape() {
        let scene = striped_block();
        let source = Source::new(&scene.objects[0]);
        assert!(source.tris.len() > 2000);
        let welded = weld(&source);
        let log = Simplifier::new(&welded).run(1e-12);
        let low = replay(&welded, &log, log.len());
        assert_eq!(low.tris.len(), 12, "a cube needs 12 triangles");
        for p in &low.positions {
            assert!(p.abs().max_element() < 0.5 + 1e-4 && p.abs().min_element() > 0.5 - 1e-4, "{p}");
        }
        let seen = Seen { keep: vec![true; 12], points: vec![Vec::new(); 12] };
        assert_eq!(build_charts(&low, &seen).len(), 6);
    }

    #[test]
    fn closest_point_reads_the_painted_colour() {
        let scene = striped_block();
        let source = Source::new(&scene.objects[0]);
        let index = Bvh::new(&source.positions, &source.tris);
        let (red, size) = source.color_at(&index, Vec3::new(0.6, -0.4, 0.1), Vec3::X);
        let (blue, _) = source.color_at(&index, Vec3::new(0.6, -0.1, 0.1), Vec3::X);
        let (grey, _) = source.color_at(&index, Vec3::new(-0.7, 0.0, 0.0), Vec3::NEG_X);
        assert!((size - 1.0 / 16.0).abs() < 1e-4, "{size}");
        // A ray from inside the block is stopped; one from outside pointing away is not.
        assert!(index.blocked(&source.positions, Vec3::ZERO, Vec3::new(0.3, 0.2, 0.9)));
        assert!(!index.blocked(&source.positions, Vec3::new(0.0, 0.6, 0.0), Vec3::new(0.3, 0.2, 0.9)));
        assert!(red[0] > 0.9 && red[2] < 0.1, "{red:?}");
        assert!(blue[2] > 0.9 && blue[0] < 0.1, "{blue:?}");
        assert!((grey[0] - 0.8).abs() < 1e-3);
    }

    #[test]
    fn hidden_faces_are_dropped_only_between_merged_parts() {
        let mut scene = striped_block();
        scene.add_mesh_object("Crate", Mesh::cube(1.0), [0.5, 0.4, 0.3]);
        scene.objects[1].transform.translation = Vec3::new(1.0, 0.0, 0.0);
        let parts: Vec<&SceneObject> = scene.objects.iter().collect();
        let triangles = |merged: bool| -> Vec<usize> {
            let baked = bake(&parts, None, 256, merged, false).unwrap();
            baked.parts.iter().map(|p| p.indices.len() / 3).collect()
        };
        // The two sides pressed together cannot be seen once the parts are one mesh.
        assert_eq!(triangles(true), [10, 10]);
        assert_eq!(triangles(false), [12, 12]);
    }

    #[test]
    fn baking_preserves_small_geometry() {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Small cube", Mesh::cube(1e-6), [1.0; 3]);
        let baked = bake(&[&scene.objects[0]], None, 64, false, false).unwrap();
        assert_eq!(baked.parts[0].indices.len(), 36);
    }

    #[test]
    fn charts_are_cropped_to_what_can_be_seen() {
        let low = LowMesh {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            tris: vec![[0, 1, 2], [0, 2, 3]],
        };
        let all = Seen { keep: vec![true; 2], points: vec![Vec::new(); 2] };
        let whole = &build_charts(&low, &all)[0];
        assert!((whole.extent - Vec2::ONE).length() < 1e-5);
        // Only a strip along one edge is visible: the chart shrinks to it.
        let strip = Seen {
            keep: vec![true; 2],
            points: vec![vec![(Vec3::new(0.3, 0.05, 0.0), 0.05), (Vec3::new(0.9, 0.1, 0.0), 0.05)], Vec::new()],
        };
        let cropped = &build_charts(&low, &strip)[0];
        assert_eq!(cropped.tris.len(), 2);
        assert!((cropped.extent - Vec2::new(0.7, 0.15)).length() < 1e-4, "{}", cropped.extent);
        // A triangle nothing sees is left out.
        let half = Seen { keep: vec![true, false], points: strip.points.clone() };
        assert_eq!(build_charts(&low, &half)[0].tris, [0]);
    }

    #[test]
    fn bake_shares_identical_parts_and_respects_the_budget() {
        let mut scene = striped_block();
        let copy = scene.objects[0].clone();
        scene.objects.push(copy);
        scene.objects[1].transform.translation = Vec3::new(3.0, 0.0, 0.0);
        scene.add_mesh_object("Ball", Mesh::sphere(1.0, 64, 48), [0.2, 0.6, 0.3]);
        let parts: Vec<&SceneObject> = scene.objects.iter().collect();
        let baked = bake(&parts, Some(400), 256, false, false).unwrap();
        let triangles: usize = baked.parts.iter().map(|p| p.indices.len() / 3).sum();
        assert!(triangles <= 400 && triangles > 200, "{triangles}");
        // The blocks stay boxes and the sphere takes the rest of the budget.
        assert_eq!(baked.parts[0].indices.len(), 36);
        assert_eq!(baked.parts[0].uvs, baked.parts[1].uvs);
        for part in &baked.parts {
            assert_eq!(part.uvs.len(), part.positions.len());
            assert!(part.uvs.iter().flatten().all(|&v| (0.0..=1.0).contains(&v)));
            assert!(part.normals.iter().all(|n| (Vec3::from(*n).length() - 1.0).abs() < 1e-3));
        }
        assert!(bake(&parts, None, 16, false, false).is_err());
    }
}
