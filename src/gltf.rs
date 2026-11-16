//! Game-ready binary glTF 2.0 (`.glb`) export for Godot and Blender: named nodes, PBR
//! materials, vertex colours, flat or smooth normals, optional merging by material and
//! Godot collision hints.
//!
//! Colours are written as they are: the viewport treats part and vertex colours as linear
//! values, which is what glTF expects for `baseColorFactor` and `COLOR_0`.
use crate::scene::{ProceduralPattern, SceneObject};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;

/// Collision geometry for Godot, written as extra nodes whose name suffix tells Godot's
/// importer to turn them into a `StaticBody3D` (`-convcolonly` convex, `-colonly` exact).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Collision {
    #[default]
    None,
    /// One box around the whole model.
    Box,
    /// One convex hull around the whole model (Godot computes the hull).
    Convex,
    /// One box per part, following each part's position and rotation.
    PartBoxes,
    /// The exact triangles. Static geometry only.
    Mesh,
}

impl Collision {
    pub const ALL: [Collision; 5] =
        [Collision::None, Collision::Box, Collision::Convex, Collision::PartBoxes, Collision::Mesh];

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_lowercase().as_str() {
            "" | "none" | "off" => Ok(Collision::None),
            "box" => Ok(Collision::Box),
            "convex" | "hull" => Ok(Collision::Convex),
            "part_boxes" | "boxes" => Ok(Collision::PartBoxes),
            "mesh" | "trimesh" | "exact" => Ok(Collision::Mesh),
            other => Err(format!(
                "Unknown collision '{other}'. Use none, box, convex, part_boxes or mesh."
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Collision::None => "none",
            Collision::Box => "box",
            Collision::Convex => "convex",
            Collision::PartBoxes => "part_boxes",
            Collision::Mesh => "mesh",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Collision::None => "No collision",
            Collision::Box => "One box",
            Collision::Convex => "Convex hull",
            Collision::PartBoxes => "Box per part",
            Collision::Mesh => "Exact mesh",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GltfOptions {
    /// Name of the root node (the scene root in Godot, the parent empty in Blender).
    pub root_name: String,
    /// Join parts that share a material into one mesh: far fewer draw calls in a game.
    /// Off keeps every part as its own named node, which is easier to edit in Blender.
    pub merge: bool,
    pub collision: Collision,
    /// Uniform scale applied to everything (the app works in metres, like Godot).
    pub scale: f32,
    /// Move the model so the centre of its base sits at the origin.
    pub center_base: bool,
    /// Triangle budget for the whole model. Dense parts are simplified to fit; `None`
    /// exports every triangle.
    pub max_triangles: Option<usize>,
    /// Write for Godot's importer: mark the single root (`GODOT_single_root`) and lead every
    /// vertex-coloured mesh with one flat triangle (see `Writer::mesh_primitives`).
    pub godot: bool,
}

impl GltfOptions {
    /// Options from tool arguments (`merge`, `collision`, `scale`, `origin`), starting from
    /// `self` so each caller keeps its own defaults.
    pub fn with_args(mut self, args: &Value) -> Result<Self, String> {
        if let Some(merge) = args["merge"].as_bool() {
            self.merge = merge;
        }
        if let Some(collision) = args["collision"].as_str() {
            self.collision = Collision::from_name(collision)?;
        }
        if let Some(scale) = args["scale"].as_f64() {
            self.scale = scale as f32;
        }
        match args["origin"].as_str().map(|s| s.trim().to_lowercase()) {
            None => {}
            Some(o) if o == "keep" || o == "scene" => self.center_base = false,
            Some(o) if o == "base" || o == "base_center" => self.center_base = true,
            Some(o) => return Err(format!("Unknown origin '{o}'. Use keep or base.")),
        }
        if let Some(budget) = args["max_triangles"].as_u64() {
            self.max_triangles = (budget > 0).then_some(budget as usize);
        }
        match args["target"].as_str().map(|s| s.trim().to_lowercase()) {
            None => {}
            Some(t) if t == "godot" => self.godot = true,
            Some(t) if t == "blender" || t == "generic" => self.godot = false,
            Some(t) => return Err(format!("Unknown target '{t}'. Use godot or blender.")),
        }
        if let Some(name) = args["root_name"].as_str().map(str::trim).filter(|n| !n.is_empty()) {
            self.root_name = name.to_string();
        }
        Ok(self)
    }
}

impl Default for GltfOptions {
    fn default() -> Self {
        Self {
            root_name: "Model".into(),
            merge: false,
            collision: Collision::None,
            scale: 1.0,
            center_base: false,
            max_triangles: None,
            godot: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExportReport {
    pub parts: usize,
    pub meshes: usize,
    pub materials: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub collision_shapes: usize,
    /// Triangles before simplifying to the budget (equal to `triangles` when nothing was cut).
    pub source_triangles: usize,
    /// World-space size of the exported model, after scaling.
    pub size: [f32; 3],
    pub warnings: Vec<String>,
}

impl ExportReport {
    pub fn to_json(&self) -> Value {
        json!({
            "parts": self.parts,
            "meshes": self.meshes,
            "materials": self.materials,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "collision_shapes": self.collision_shapes,
            "source_triangles": self.source_triangles,
            "size": self.size,
            "warnings": self.warnings,
        })
    }
}

/// Triangles of one part (or one merged group) ready to be written.
#[derive(Default)]
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl Geometry {
    /// Adds `other`, moving it by `matrix`.
    fn append(&mut self, other: &Geometry, matrix: Mat4) {
        let base = self.positions.len() as u32;
        let normal_matrix = matrix.inverse().transpose();
        let flip = matrix.determinant() < 0.0;
        self.positions
            .extend(other.positions.iter().map(|&p| matrix.transform_point3(p.into()).to_array()));
        self.normals.extend(other.normals.iter().map(|&n| {
            normal_matrix.transform_vector3(n.into()).try_normalize().unwrap_or(Vec3::Y).to_array()
        }));
        self.colors.extend_from_slice(&other.colors);
        for tri in other.indices.chunks_exact(3) {
            // A mirrored part needs its triangles turned to stay outward-facing.
            let tri = if flip { [tri[0], tri[2], tri[1]] } else { [tri[0], tri[1], tri[2]] };
            self.indices.extend(tri.iter().map(|i| base + i));
        }
    }

    fn uniform_color(&self) -> Option<[f32; 3]> {
        let first = *self.colors.first()?;
        self.colors
            .iter()
            .all(|c| (0..3).all(|i| (c[i] - first[i]).abs() < 1e-4))
            .then_some(first)
    }
}

/// A part's triangles in its local space, with the colours the viewport shows (vertex
/// colour times part colour) and flat or smooth normals.
fn part_geometry(obj: &SceneObject) -> Geometry {
    let mesh = &obj.mesh;
    let mut out = Geometry::default();
    let color = |vi: usize| {
        let vc = mesh.colors.get(vi).copied().unwrap_or([1.0; 3]);
        [0, 1, 2].map(|i| (vc[i] * obj.color[i]).clamp(0.0, 1.0))
    };
    // Smooth parts share vertices between faces; only the ones a face uses are written.
    let mut remap: Vec<u32> = vec![u32::MAX; if obj.flat_shading { 0 } else { mesh.positions.len() }];
    for (fi, face) in mesh.faces.iter().enumerate() {
        if face.len() < 3 || face.iter().any(|&v| v >= mesh.positions.len()) {
            continue;
        }
        let face_normal = mesh.face_normal(fi).try_normalize().unwrap_or(Vec3::Y);
        let corners: Vec<u32> = face
            .iter()
            .map(|&vi| {
                if !obj.flat_shading && remap[vi] != u32::MAX {
                    return remap[vi];
                }
                let index = out.positions.len() as u32;
                out.positions.push(mesh.positions[vi].to_array());
                let normal = if obj.flat_shading {
                    face_normal
                } else {
                    mesh.normals.get(vi).and_then(|n| n.try_normalize()).unwrap_or(face_normal)
                };
                out.normals.push(normal.to_array());
                out.colors.push(color(vi));
                if !obj.flat_shading {
                    remap[vi] = index;
                }
                index
            })
            .collect();
        for i in 1..corners.len() - 1 {
            out.indices.extend([corners[0], corners[i], corners[i + 1]]);
        }
    }
    out
}

fn box_geometry(min: Vec3, max: Vec3) -> Geometry {
    let corner = |i: usize| {
        [
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        ]
    };
    Geometry {
        positions: (0..8).map(corner).collect(),
        indices: vec![
            0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
        ],
        ..Default::default()
    }
}

fn bounds(points: impl Iterator<Item = Vec3>) -> Option<(Vec3, Vec3)> {
    points.fold(None, |acc: Option<(Vec3, Vec3)>, p| match acc {
        Some((min, max)) => Some((min.min(p), max.max(p))),
        None => Some((p, p)),
    })
}

/// Material properties that decide whether two parts can share a material.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MaterialKey {
    roughness: i32,
    metallic: i32,
    emission: [i32; 3],
    /// `None` = the colour comes from the vertices.
    color: Option<[i32; 3]>,
}

fn quantize(v: f32) -> i32 {
    (v * 1000.0).round() as i32
}

impl MaterialKey {
    /// `step` rounds roughness and metallic: merged exports use a coarse step so parts that
    /// look the same share one material (one draw call) instead of dozens.
    fn of(obj: &SceneObject, color: Option<[f32; 3]>, step: f32) -> Self {
        let strength = obj.emission_strength.max(0.0);
        let rounded = |v: f32| quantize((v.clamp(0.0, 1.0) / step).round() * step);
        Self {
            roughness: rounded(obj.roughness),
            metallic: rounded(obj.metallic),
            emission: obj.emission_color.map(|c| quantize(c.max(0.0) * strength)),
            color: color.map(|c| c.map(quantize)),
        }
    }
}

struct Writer {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    materials: Vec<Value>,
    material_index: HashMap<MaterialKey, usize>,
    meshes: Vec<Value>,
    nodes: Vec<Value>,
    emissive_strength: bool,
    report: ExportReport,
}

impl Writer {
    fn view(&mut self, bytes: &[u8], target: u32) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        self.views.push(json!({
            "buffer": 0,
            "byteOffset": self.bin.len(),
            "byteLength": bytes.len(),
            "target": target,
        }));
        self.bin.extend_from_slice(bytes);
        self.views.len() - 1
    }

    fn vec3_accessor(&mut self, data: &[[f32; 3]], with_bounds: bool) -> usize {
        let view = self.view(bytemuck::cast_slice(data), 34962);
        let mut accessor = json!({
            "bufferView": view,
            "componentType": 5126,
            "count": data.len(),
            "type": "VEC3",
        });
        if with_bounds {
            let (min, max) = bounds(data.iter().map(|&p| p.into())).unwrap_or_default();
            accessor["min"] = json!(min.to_array());
            accessor["max"] = json!(max.to_array());
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn material(&mut self, key: MaterialKey) -> usize {
        if let Some(&index) = self.material_index.get(&key) {
            return index;
        }
        let unq = |v: i32| v as f32 / 1000.0;
        let color = key.color.map_or([1.0; 3], |c| c.map(unq));
        let mut material = json!({
            "name": format!("Material {}", self.materials.len() + 1),
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], 1.0],
                "metallicFactor": unq(key.metallic),
                "roughnessFactor": unq(key.roughness),
            },
        });
        let emission = key.emission.map(unq);
        let peak = emission.iter().copied().fold(0.0_f32, f32::max);
        if peak > 0.0 {
            // glTF caps emissiveFactor at 1; brighter glows use the strength extension.
            let strength = peak.max(1.0);
            material["emissiveFactor"] = json!(emission.map(|c| c / strength));
            if strength > 1.0 {
                material["extensions"] = json!({ "KHR_materials_emissive_strength": { "emissiveStrength": strength } });
                self.emissive_strength = true;
            }
        }
        self.materials.push(material);
        self.material_index.insert(key, self.materials.len() - 1);
        self.materials.len() - 1
    }

    /// Writes a mesh's primitives from (material, geometry, has vertex colours) groups.
    ///
    /// Godot 4's importer only turns on "vertex colour as albedo" for a surface when an
    /// EARLIER surface of the same mesh had `COLOR_0`, so the first coloured surface would
    /// import white. For Godot the mesh therefore starts with one triangle taken from that
    /// surface, written with white vertex colours and its real colour in the material: it
    /// looks right whether or not the importer enables vertex colours for it, and every
    /// surface after it gets them.
    fn mesh_primitives(&mut self, groups: &[(MaterialKey, &Geometry, bool)], godot: bool) -> Vec<Value> {
        let mut primitives = Vec::new();
        let lead = groups.iter().position(|(_, g, colored)| godot && *colored && g.indices.len() >= 3);
        if let Some(index) = lead {
            let (key, geometry, _) = &groups[index];
            let corners = [0, 1, 2].map(|i| geometry.indices[i] as usize);
            let mut average = [0.0_f32; 3];
            for &corner in &corners {
                for (sum, value) in average.iter_mut().zip(geometry.colors[corner]) {
                    *sum += value / 3.0;
                }
            }
            let triangle = Geometry {
                positions: corners.map(|c| geometry.positions[c]).to_vec(),
                normals: corners.map(|c| geometry.normals[c]).to_vec(),
                colors: vec![[1.0; 3]; 3],
                indices: vec![0, 1, 2],
            };
            let material = self.material(MaterialKey { color: Some(average.map(quantize)), ..*key });
            primitives.push(self.primitive(&triangle, &triangle.indices, Some((material, true))));
        }
        for (i, (key, geometry, colored)) in groups.iter().enumerate() {
            let indices = if lead == Some(i) { &geometry.indices[3..] } else { &geometry.indices[..] };
            if indices.is_empty() {
                continue;
            }
            let material = self.material(*key);
            primitives.push(self.primitive(geometry, indices, Some((material, *colored))));
        }
        primitives
    }

    /// Writes one primitive. `material` is `None` for collision shapes.
    fn primitive(&mut self, geometry: &Geometry, indices: &[u32], material: Option<(usize, bool)>) -> Value {
        let position = self.vec3_accessor(&geometry.positions, true);
        let mut attributes = json!({ "POSITION": position });
        let mut primitive = json!({ "mode": 4 });
        if let Some((material, vertex_colors)) = material {
            attributes["NORMAL"] = json!(self.vec3_accessor(&geometry.normals, false));
            if vertex_colors {
                attributes["COLOR_0"] = json!(self.vec3_accessor(&geometry.colors, false));
            }
            primitive["material"] = json!(material);
            self.report.vertices += geometry.positions.len();
            self.report.triangles += indices.len() / 3;
        }
        let view = self.view(bytemuck::cast_slice(indices), 34963);
        self.accessors.push(json!({
            "bufferView": view,
            "componentType": 5125,
            "count": indices.len(),
            "type": "SCALAR",
        }));
        primitive["indices"] = json!(self.accessors.len() - 1);
        primitive["attributes"] = attributes;
        primitive
    }

    fn mesh(&mut self, name: &str, primitives: Vec<Value>) -> usize {
        self.meshes.push(json!({ "name": name, "primitives": primitives }));
        self.meshes.len() - 1
    }

    fn node(&mut self, node: Value) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn collision_node(&mut self, name: &str, geometry: &Geometry, matrix: Option<Mat4>) -> usize {
        let primitive = self.primitive(geometry, &geometry.indices, None);
        let mesh = self.mesh(name, vec![primitive]);
        let mut node = json!({ "name": name, "mesh": mesh });
        if let Some(matrix) = matrix {
            set_transform(&mut node, matrix);
        }
        self.report.collision_shapes += 1;
        self.node(node)
    }
}

fn set_transform(node: &mut Value, matrix: Mat4) {
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    node["translation"] = json!(translation.to_array());
    node["rotation"] = json!(rotation.normalize().to_array());
    node["scale"] = json!(scale.to_array());
}

/// Builds the `.glb` bytes for `objects`.
pub fn build_glb(objects: &[&SceneObject], options: &GltfOptions) -> Result<(Vec<u8>, ExportReport), String> {
    if !options.scale.is_finite() || options.scale <= 0.0 {
        return Err("The export scale must be above 0".into());
    }
    let triangles = |obj: &SceneObject| -> usize {
        obj.mesh.faces.iter().map(|f| f.len().saturating_sub(2)).sum()
    };
    let source_triangles: usize = objects.iter().map(|o| triangles(o)).sum();
    // Fit the triangle budget by simplifying the dense parts; small parts are left alone
    // because clustering would only wreck their shape for a handful of triangles.
    const KEEP_BELOW: usize = 300;
    let mut simplified: Vec<SceneObject> = Vec::new();
    if let Some(budget) = options.max_triangles.filter(|&b| b < source_triangles) {
        let small: usize = objects.iter().map(|o| triangles(o)).filter(|&t| t < KEEP_BELOW).sum();
        let dense = (source_triangles - small).max(1);
        let ratio = (budget.saturating_sub(small) as f32 / dense as f32).clamp(0.02, 1.0);
        simplified = objects
            .iter()
            .map(|obj| {
                let mut copy = (*obj).clone();
                if triangles(obj) >= KEEP_BELOW {
                    crate::modeling_ops::decimate(&mut copy.mesh, ratio);
                }
                copy
            })
            .collect();
    }
    let objects: Vec<&SceneObject> =
        if simplified.is_empty() { objects.to_vec() } else { simplified.iter().collect() };
    let parts: Vec<(&SceneObject, Geometry)> = objects
        .iter()
        .map(|obj| (*obj, part_geometry(obj)))
        .filter(|(_, geometry)| !geometry.indices.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("There is no geometry to export".into());
    }

    // World placement of every part: scale, then (optionally) the base centre at the origin.
    let scale = Mat4::from_scale(Vec3::splat(options.scale));
    let world_points = |placement: Mat4| {
        parts.iter().flat_map(move |(obj, geometry)| {
            let matrix = placement * obj.transform.matrix();
            geometry.positions.iter().map(move |&p| matrix.transform_point3(p.into()))
        })
    };
    let mut placement = scale;
    if options.center_base {
        let (min, max) = bounds(world_points(scale)).unwrap();
        let base = Vec3::new((min.x + max.x) * 0.5, min.y, (min.z + max.z) * 0.5);
        placement = Mat4::from_translation(-base) * scale;
    }
    let (min, max) = bounds(world_points(placement)).unwrap();

    let mut w = Writer {
        bin: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
        materials: Vec::new(),
        material_index: HashMap::new(),
        meshes: Vec::new(),
        nodes: Vec::new(),
        emissive_strength: false,
        report: ExportReport {
            parts: parts.len(),
            source_triangles,
            size: (max - min).to_array(),
            ..Default::default()
        },
    };
    let patterned = parts.iter().filter(|(obj, _)| obj.pattern != ProceduralPattern::None).count();
    if patterned > 0 {
        w.report.warnings.push(format!(
            "{patterned} part(s) use a procedural pattern (bricks, wood grain…). Patterns are viewport shaders and are not exported; those parts keep their base colour."
        ));
    }

    // The root is a plain node so Godot gets a Node3D with the meshes and bodies under it.
    // It is node 0, which is where `GODOT_single_root` expects the scene root.
    w.nodes.push(Value::Null);
    let mut children: Vec<usize> = Vec::new();
    if options.merge {
        // One mesh; parts that share a material become one primitive (one draw call).
        let mut groups: Vec<(MaterialKey, Geometry)> = Vec::new();
        for (obj, geometry) in &parts {
            let key = MaterialKey::of(obj, None, 0.1);
            let index = groups.iter().position(|(k, _)| *k == key).unwrap_or_else(|| {
                groups.push((key, Geometry::default()));
                groups.len() - 1
            });
            groups[index].1.append(geometry, placement * obj.transform.matrix());
        }
        let groups: Vec<(MaterialKey, &Geometry, bool)> =
            groups.iter().map(|(key, geometry)| (*key, geometry, true)).collect();
        let primitives = w.mesh_primitives(&groups, options.godot);
        let mesh = w.mesh(&options.root_name, primitives);
        children.push(w.node(json!({ "name": format!("{} Mesh", options.root_name), "mesh": mesh })));
    } else {
        // Every part keeps its name and transform; "Group/Part" names become a parent node.
        let mut group_nodes: Vec<(String, Vec<usize>)> = Vec::new();
        for (obj, geometry) in &parts {
            let matrix = placement * obj.transform.matrix();
            let uniform = geometry.uniform_color();
            let key = MaterialKey::of(obj, uniform, 0.001);
            let (group, name) = match obj.name.split_once('/') {
                Some((group, name)) if !group.trim().is_empty() && !name.trim().is_empty() => {
                    (Some(group.trim()), name.trim())
                }
                _ => (None, obj.name.trim()),
            };
            let name = if name.is_empty() { "Part" } else { name };
            // Mirrored parts are baked: engines disagree on negative node scales.
            let node = if matrix.determinant() < 0.0 {
                let mut baked = Geometry::default();
                baked.append(geometry, matrix);
                let primitives = w.mesh_primitives(&[(key, &baked, uniform.is_none())], options.godot);
                let mesh = w.mesh(name, primitives);
                w.node(json!({ "name": name, "mesh": mesh }))
            } else {
                let primitives = w.mesh_primitives(&[(key, geometry, uniform.is_none())], options.godot);
                let mesh = w.mesh(name, primitives);
                let mut node = json!({ "name": name, "mesh": mesh });
                set_transform(&mut node, matrix);
                w.node(node)
            };
            match group {
                Some(group) => match group_nodes.iter_mut().find(|(g, _)| g == group) {
                    Some((_, list)) => list.push(node),
                    None => group_nodes.push((group.to_string(), vec![node])),
                },
                None => children.push(node),
            }
        }
        for (group, list) in group_nodes {
            children.push(w.node(json!({ "name": group, "children": list })));
        }
    }
    w.report.meshes = w.meshes.len();

    match options.collision {
        Collision::None => {}
        Collision::Box => {
            children.push(w.collision_node("Collision-convcolonly", &box_geometry(min, max), None));
        }
        Collision::Convex | Collision::Mesh => {
            let mut all = Geometry::default();
            for (obj, geometry) in &parts {
                all.append(geometry, placement * obj.transform.matrix());
            }
            all.normals.clear();
            all.colors.clear();
            let name = if options.collision == Collision::Convex {
                "Collision-convcolonly"
            } else {
                "Collision-colonly"
            };
            children.push(w.collision_node(name, &all, None));
        }
        Collision::PartBoxes => {
            let mut boxes = Vec::new();
            for (obj, geometry) in &parts {
                let (min, max) = bounds(geometry.positions.iter().map(|&p| p.into())).unwrap();
                let name = obj.name.rsplit('/').next().unwrap_or("Part").trim();
                let matrix = placement * obj.transform.matrix();
                boxes.push(w.collision_node(
                    &format!("{name}-convcolonly"),
                    &box_geometry(min, max),
                    Some(matrix),
                ));
            }
            children.push(w.node(json!({ "name": "Collision", "children": boxes })));
        }
    }

    w.nodes[0] = json!({ "name": options.root_name, "children": children });
    w.report.materials = w.materials.len();
    if w.report.triangles > 300_000 {
        w.report.warnings.push(format!(
            "{} triangles is heavy for a game asset; set a triangle budget (max_triangles) to simplify it.",
            w.report.triangles
        ));
    }
    if let Some(budget) = options.max_triangles {
        if w.report.triangles > budget + budget / 4 {
            w.report.warnings.push(format!(
                "Simplified from {source_triangles} to {} triangles; that is as far as the parts simplify, so the {budget} budget was not reached.",
                w.report.triangles
            ));
        }
    }

    while w.bin.len() % 4 != 0 {
        w.bin.push(0);
    }
    let mut document = json!({
        "asset": { "version": "2.0", "generator": "Rust 3D Modeler" },
        "scene": 0,
        "scenes": [{ "name": options.root_name, "nodes": [0] }],
        "nodes": w.nodes,
        "meshes": w.meshes,
        "materials": w.materials,
        "buffers": [{ "byteLength": w.bin.len() }],
        "bufferViews": w.views,
        "accessors": w.accessors,
    });
    let mut extensions = Vec::new();
    if w.emissive_strength {
        extensions.push("KHR_materials_emissive_strength");
    }
    if options.godot {
        // Use our root as the scene root instead of wrapping it in another node.
        extensions.push("GODOT_single_root");
    }
    if !extensions.is_empty() {
        document["extensionsUsed"] = json!(extensions);
    }
    let mut json_bytes =
        serde_json::to_vec(&document).map_err(|e| format!("Failed to serialize glTF JSON: {e}"))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + w.bin.len();
    if total > u32::MAX as usize {
        return Err("The model is too large for one .glb file (4 GB limit)".into());
    }
    let mut glb = Vec::with_capacity(total);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(w.bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    glb.extend_from_slice(&w.bin);
    Ok((glb, w.report))
}

pub fn export_glb(objects: &[&SceneObject], path: &Path, options: &GltfOptions) -> Result<ExportReport, String> {
    let (glb, report) = build_glb(objects, options)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("Could not create {}: {e}", parent.display()))?;
    }
    // Written in one step so an engine watching the folder never imports half a file.
    crate::project::atomic_write(path, &glb, false)?;
    Ok(report)
}

/// The Godot project a path belongs to, and the path's `res://` address in it.
pub fn godot_resource_path(path: &Path) -> Option<(std::path::PathBuf, String)> {
    let absolute = std::path::absolute(path).ok()?;
    let project = absolute.ancestors().skip(1).find(|dir| dir.join("project.godot").is_file())?;
    let relative = absolute.strip_prefix(project).ok()?;
    let relative: Vec<String> =
        relative.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    Some((project.to_path_buf(), format!("res://{}", relative.join("/"))))
}

/// File and folder names the way Godot projects spell them: `Oak bookshelf` -> `oak_bookshelf`.
pub fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    let out = out.trim_end_matches('_').to_string();
    if out.is_empty() { "asset".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::scene::Scene;

    fn document(glb: &[u8]) -> Value {
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize, glb.len());
        let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        serde_json::from_slice(&glb[20..20 + len]).unwrap()
    }

    fn shelf() -> Scene {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Shelf/Frame", Mesh::cube(1.0), [0.6, 0.4, 0.2]);
        scene.add_mesh_object("Shelf/Board", Mesh::cube(0.5), [0.5, 0.3, 0.2]);
        scene.add_mesh_object("Lamp", Mesh::cube(0.25), [0.9, 0.9, 0.6]);
        scene.objects[0].flat_shading = true;
        scene.objects[1].transform.translation = Vec3::new(0.0, 2.0, 0.0);
        scene.objects[2].transform.translation = Vec3::new(3.0, 0.0, 0.0);
        scene.objects[2].roughness = 0.2;
        scene.objects[2].emission_strength = 4.0;
        scene
    }

    #[test]
    fn separate_parts_keep_names_groups_and_materials() {
        let scene = shelf();
        let parts: Vec<&SceneObject> = scene.objects.iter().collect();
        let (glb, report) = build_glb(&parts, &GltfOptions::default()).unwrap();
        let doc = document(&glb);
        let names: Vec<&str> = doc["nodes"].as_array().unwrap().iter().map(|n| n["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["Model", "Frame", "Board", "Lamp", "Shelf"]);
        assert_eq!(doc["nodes"][4]["children"], json!([1, 2]));
        assert_eq!(doc["nodes"][0]["children"], json!([3, 4]));
        assert_eq!(doc["scenes"][0]["nodes"], json!([0]));
        assert_eq!(doc["nodes"][2]["translation"], json!([0.0, 2.0, 0.0]));
        assert_eq!((report.parts, report.meshes, report.materials, report.triangles), (3, 3, 3, 36));
        // Flat shading unshares vertices (4 per quad); smooth parts keep theirs.
        assert_eq!(doc["accessors"][0]["count"], 24);
        // Uniform parts carry their colour in the material, not per vertex.
        assert!(doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].is_null());
        let lamp = &doc["materials"][2];
        let strength = lamp["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"].as_f64().unwrap();
        assert!((strength - 3.6).abs() < 1e-5);
        assert_eq!(doc["extensionsUsed"], json!(["KHR_materials_emissive_strength"]));
    }

    #[test]
    fn merged_export_joins_parts_by_material_and_adds_collision() {
        let mut scene = shelf();
        scene.objects[2].emission_strength = 0.0;
        scene.objects[2].roughness = scene.objects[0].roughness;
        // A mirrored part must stay outward-facing once baked.
        scene.objects[1].transform.scale = Vec3::new(-1.0, 1.0, 1.0);
        let parts: Vec<&SceneObject> = scene.objects.iter().collect();
        let options = GltfOptions {
            root_name: "Oak shelf".into(),
            merge: true,
            collision: Collision::Box,
            scale: 2.0,
            center_base: true,
            max_triangles: None,
            godot: false,
        };
        let (glb, report) = build_glb(&parts, &options).unwrap();
        let doc = document(&glb);
        assert_eq!((report.meshes, report.materials, report.collision_shapes), (1, 1, 1));
        let primitives = doc["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(primitives.len(), 1);
        assert!(primitives[0]["attributes"]["COLOR_0"].is_u64());
        assert_eq!(doc["nodes"][2]["name"], "Collision-convcolonly");
        assert_eq!(doc["nodes"][0]["name"], "Oak shelf");
        assert!(doc["extensionsUsed"].is_null());

        // For Godot the mesh leads with one flat triangle so every real surface gets its
        // vertex colours, and the root is marked as the scene root.
        let (godot_glb, godot) = build_glb(&parts, &GltfOptions { godot: true, ..options.clone() }).unwrap();
        let godot_doc = document(&godot_glb);
        let leading = godot_doc["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(leading.len(), 2);
        let count =
            |p: &Value| godot_doc["accessors"][p["indices"].as_u64().unwrap() as usize]["count"].as_u64().unwrap();
        assert_eq!((count(&leading[0]), count(&leading[1])), (3, 36 * 3 - 3));
        assert_eq!(godot.triangles, report.triangles);
        assert_eq!(godot_doc["extensionsUsed"], json!(["GODOT_single_root"]));
        // Scaled by 2 with the base centre at the origin.
        let position = &doc["accessors"][primitives[0]["attributes"]["POSITION"].as_u64().unwrap() as usize];
        assert_eq!(position["min"][1], 0.0);
        let (min_x, max_x) = (position["min"][0].as_f64().unwrap(), position["max"][0].as_f64().unwrap());
        assert!((min_x + max_x).abs() < 1e-5);
        assert!((report.size[0] - (max_x - min_x) as f32).abs() < 1e-5);

        for collision in [Collision::Convex, Collision::Mesh, Collision::PartBoxes] {
            let (glb, report) = build_glb(&parts, &GltfOptions { collision, ..options.clone() }).unwrap();
            let names: Vec<String> = document(&glb)["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n["name"].as_str().unwrap().to_string())
                .collect();
            match collision {
                Collision::Convex => assert!(names.contains(&"Collision-convcolonly".to_string())),
                Collision::Mesh => assert!(names.contains(&"Collision-colonly".to_string())),
                _ => {
                    assert_eq!(report.collision_shapes, 3);
                    assert!(names.contains(&"Frame-convcolonly".to_string()));
                }
            }
        }
    }

    #[test]
    fn triangle_budget_simplifies_dense_parts_only() {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Ball", Mesh::sphere(1.0, 48, 32), [0.8, 0.2, 0.2]);
        scene.add_mesh_object("Crate", Mesh::cube(1.0), [0.5, 0.4, 0.3]);
        let parts: Vec<&SceneObject> = scene.objects.iter().collect();
        let (_, full) = build_glb(&parts, &GltfOptions::default()).unwrap();
        assert_eq!(full.triangles, full.source_triangles);
        let options = GltfOptions { max_triangles: Some(full.triangles / 4), ..Default::default() };
        let (glb, cut) = build_glb(&parts, &options).unwrap();
        assert_eq!(cut.source_triangles, full.triangles);
        assert!(cut.triangles < full.triangles / 2, "{} of {}", cut.triangles, full.triangles);
        // The cube is far below the per-part floor and keeps all 12 triangles.
        let doc = document(&glb);
        let cube = &doc["meshes"][1]["primitives"][0];
        assert_eq!(doc["accessors"][cube["indices"].as_u64().unwrap() as usize]["count"], 36);
    }

    #[test]
    fn names_and_godot_paths() {
        assert_eq!(snake_case("Oak bookshelf (v2)"), "oak_bookshelf_v2");
        assert_eq!(snake_case("  "), "asset");
        assert!(Collision::from_name("hull").is_ok() && Collision::from_name("sphere").is_err());
        let project = std::env::temp_dir().join(format!("rust_modeler_godot_{}", std::process::id()));
        std::fs::create_dir_all(project.join("assets")).unwrap();
        std::fs::write(project.join("project.godot"), "").unwrap();
        let (root, res) = godot_resource_path(&project.join("assets/furniture/chair.glb")).unwrap();
        assert_eq!(root, project);
        assert_eq!(res, "res://assets/furniture/chair.glb");
        let _ = std::fs::remove_dir_all(&project);
    }
}
