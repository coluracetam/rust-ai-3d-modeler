use crate::camera::Ray;
use crate::mesh::{EditElementMode, Mesh, RayHit};
use crate::voxel::VoxelGrid;
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionMode {
    Object,
    EditMesh(EditElementMode),
    Sculpt,
    Voxel,
}

impl InteractionMode {
    pub fn label(&self) -> &'static str {
        match self {
            InteractionMode::Object => "Pin & Region Select",
            InteractionMode::EditMesh(_) => "Paint Section (Brush)",
            InteractionMode::Sculpt => "Sculpt Mode (AI)",
            InteractionMode::Voxel => "Voxel Mode (AI)",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation_deg: Vec3,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation_deg: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn rotation_quat(&self) -> Quat {
        Quat::from_euler(
            EulerRot::XYZ,
            self.rotation_deg.x.to_radians(),
            self.rotation_deg.y.to_radians(),
            self.rotation_deg.z.to_radians(),
        )
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation_quat(),
            self.translation,
        )
    }

    pub fn inverse_matrix(&self) -> Mat4 {
        self.matrix().inverse()
    }

    pub fn transform_ray_to_local(&self, world_ray: &Ray) -> Ray {
        let inv = self.inverse_matrix();
        let local_origin = inv.transform_point3(world_ray.origin);
        let local_dir = inv.transform_vector3(world_ray.direction);
        Ray::new(local_origin, local_dir)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProceduralPattern {
    #[default]
    None,
    Checker,
    Bricks,
    StoneBlocks,
    WoodGrain,
    MarbleVeins,
    HexGrid,
    RuneCircuit,
}

impl ProceduralPattern {
    pub fn label(&self) -> &'static str {
        match self {
            ProceduralPattern::None => "Solid / Vertex Color",
            ProceduralPattern::Checker => "Checker Tiles",
            ProceduralPattern::Bricks => "Masonry Bricks",
            ProceduralPattern::StoneBlocks => "Ashlar Stone Blocks",
            ProceduralPattern::WoodGrain => "Wood Grain",
            ProceduralPattern::MarbleVeins => "Veined Marble",
            ProceduralPattern::HexGrid => "Arcane Hex Grid",
            ProceduralPattern::RuneCircuit => "Rune Circuit Lines",
        }
    }

    pub fn shader_id(&self) -> f32 {
        match self {
            ProceduralPattern::None => 0.0,
            ProceduralPattern::Checker => 1.0,
            ProceduralPattern::Bricks => 2.0,
            ProceduralPattern::StoneBlocks => 3.0,
            ProceduralPattern::WoodGrain => 4.0,
            ProceduralPattern::MarbleVeins => 5.0,
            ProceduralPattern::HexGrid => 6.0,
            ProceduralPattern::RuneCircuit => 7.0,
        }
    }

    pub fn from_name(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "checker" | "tiles" | "checkerboard" => ProceduralPattern::Checker,
            "brick" | "bricks" | "masonry" => ProceduralPattern::Bricks,
            "stone" | "stone_blocks" | "ashlar" | "cobble" => ProceduralPattern::StoneBlocks,
            "wood" | "wood_grain" | "timber" => ProceduralPattern::WoodGrain,
            "marble" | "marble_veins" | "veins" => ProceduralPattern::MarbleVeins,
            "hex" | "hex_grid" | "hexagon" => ProceduralPattern::HexGrid,
            "rune" | "circuit" | "rune_circuit" | "tech" => ProceduralPattern::RuneCircuit,
            _ => ProceduralPattern::None,
        }
    }

    pub fn all() -> &'static [ProceduralPattern] {
        &[
            ProceduralPattern::None,
            ProceduralPattern::Checker,
            ProceduralPattern::Bricks,
            ProceduralPattern::StoneBlocks,
            ProceduralPattern::WoodGrain,
            ProceduralPattern::MarbleVeins,
            ProceduralPattern::HexGrid,
            ProceduralPattern::RuneCircuit,
        ]
    }
}

fn default_secondary_color() -> [f32; 3] {
    [0.25, 0.25, 0.28]
}

fn default_texture_scale() -> f32 {
    2.5
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub locked: bool,
    pub transform: Transform,
    pub mesh: Mesh,
    pub voxel_grid: Option<VoxelGrid>,
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub flat_shading: bool,
    #[serde(default)]
    pub pattern: ProceduralPattern,
    #[serde(default = "default_secondary_color")]
    pub secondary_color: [f32; 3],
    #[serde(default = "default_texture_scale")]
    pub texture_scale: f32,
    #[serde(default)]
    pub emission_color: [f32; 3],
    #[serde(default)]
    pub emission_strength: f32,
}

impl SceneObject {
    pub fn new_mesh(id: u64, name: impl Into<String>, mesh: Mesh, color: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            visible: true,
            locked: false,
            transform: Transform::default(),
            mesh,
            voxel_grid: None,
            color,
            roughness: 0.45,
            metallic: 0.05,
            flat_shading: false,
            pattern: ProceduralPattern::None,
            secondary_color: [color[0] * 0.45, color[1] * 0.45, color[2] * 0.48],
            texture_scale: 2.5,
            emission_color: color,
            emission_strength: 0.0,
        }
    }

    pub fn new_voxel(id: u64, name: impl Into<String>, grid: VoxelGrid, color: [f32; 3]) -> Self {
        let mesh = grid.extract_mesh();
        Self {
            id,
            name: name.into(),
            visible: true,
            locked: false,
            transform: Transform::default(),
            mesh,
            voxel_grid: Some(grid),
            color,
            roughness: 0.55,
            metallic: 0.0,
            flat_shading: false,
            pattern: ProceduralPattern::None,
            secondary_color: [color[0] * 0.45, color[1] * 0.45, color[2] * 0.48],
            texture_scale: 2.5,
            emission_color: color,
            emission_strength: 0.0,
        }
    }

    pub fn rebuild_voxel_mesh(&mut self) {
        if let Some(grid) = &self.voxel_grid {
            self.mesh = grid.extract_mesh();
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldRayHit {
    pub object_idx: usize,
    pub world_point: Vec3,
    pub world_normal: Vec3,
    pub local_hit: RayHit,
    pub world_distance: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommentTargetSection {
    pub object_id: u64,
    pub object_name: String,
    pub world_anchor: Vec3,
    pub local_anchor: Vec3,
    pub surface_normal: Vec3,
    pub selected_faces: Vec<usize>,
    pub local_bounds_min: Vec3,
    pub local_bounds_max: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialComment {
    pub id: u64,
    #[serde(default)]
    pub is_general: bool,
    pub object_id: u64,
    pub object_name: String,
    pub world_anchor: Vec3,
    pub local_anchor: Vec3,
    pub surface_normal: Vec3,
    pub selection_radius: f32,
    pub selected_faces: Vec<usize>,
    pub local_bounds_min: Vec3,
    pub local_bounds_max: Vec3,
    #[serde(default)]
    pub targets: Vec<CommentTargetSection>,
    pub comment: String,
    pub resolved: bool,
    pub agent_reply: Option<String>,
}

fn default_next_comment_id() -> u64 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub selected_object: Option<usize>,
    pub next_id: u64,
    pub mode: InteractionMode,
    #[serde(default)]
    pub comments: Vec<SpatialComment>,
    #[serde(default = "default_next_comment_id")]
    pub next_comment_id: u64,
    #[serde(default)]
    pub active_comment_id: Option<u64>,
    /// Tracks the latest clicked/painted surface anchor `(object_id, world_point, local_point, world_normal)`
    #[serde(default)]
    pub pending_anchor: Option<(u64, Vec3, Vec3, Vec3)>,
}

impl Default for Scene {
    fn default() -> Self {
        let mut scene = Self {
            objects: Vec::new(),
            selected_object: None,
            next_id: 1,
            mode: InteractionMode::Object,
            comments: Vec::new(),
            next_comment_id: 1,
            active_comment_id: None,
            pending_anchor: None,
        };
        // Start with a high-detail Sculpt/Voxel Sphere and a Reference Cube or single default sculptable sphere
        let grid = VoxelGrid::new_sphere(32, 3.2, 1.15);
        scene.add_voxel_object("Voxel Clay Sphere", grid, [0.85, 0.72, 0.62]);
        scene
    }
}

impl Scene {
    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn add_mesh_object(
        &mut self,
        name: impl Into<String>,
        mesh: Mesh,
        color: [f32; 3],
    ) -> usize {
        let id = self.alloc_id();
        let obj = SceneObject::new_mesh(id, name, mesh, color);
        self.objects.push(obj);
        let idx = self.objects.len() - 1;
        self.selected_object = Some(idx);
        idx
    }

    pub fn add_voxel_object(
        &mut self,
        name: impl Into<String>,
        grid: VoxelGrid,
        color: [f32; 3],
    ) -> usize {
        let id = self.alloc_id();
        let obj = SceneObject::new_voxel(id, name, grid, color);
        self.objects.push(obj);
        let idx = self.objects.len() - 1;
        self.selected_object = Some(idx);
        idx
    }

    pub fn active_object(&self) -> Option<&SceneObject> {
        self.selected_object.and_then(|i| self.objects.get(i))
    }

    pub fn active_object_mut(&mut self) -> Option<&mut SceneObject> {
        self.selected_object.and_then(|i| self.objects.get_mut(i))
    }

    pub fn duplicate_selected(&mut self) -> Option<usize> {
        let idx = self.selected_object?;
        let mut clone = self.objects.get(idx)?.clone();
        clone.id = self.alloc_id();
        clone.name = format!("{} Copy", clone.name);
        clone.transform.translation += Vec3::new(0.6, 0.0, 0.6);
        self.objects.push(clone);
        let new_idx = self.objects.len() - 1;
        self.selected_object = Some(new_idx);
        Some(new_idx)
    }

    pub fn delete_selected_object(&mut self) -> bool {
        let Some(idx) = self.selected_object else {
            return false;
        };
        if idx >= self.objects.len() {
            return false;
        }
        self.objects.remove(idx);
        if self.objects.is_empty() {
            self.selected_object = None;
        } else {
            self.selected_object = Some(idx.min(self.objects.len() - 1));
        }
        true
    }

    /// Raycast against all visible objects in the scene and return the closest hit in world space.
    pub fn raycast(&self, world_ray: &Ray) -> Option<WorldRayHit> {
        let mut best: Option<WorldRayHit> = None;

        for (i, obj) in self.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let local_ray = obj.transform.transform_ray_to_local(world_ray);
            if let Some(local_hit) = obj.mesh.ray_intersect(&local_ray) {
                let model = obj.transform.matrix();
                let world_pt = model.transform_point3(local_hit.point);
                let normal_mat = model.inverse().transpose();
                let world_n = normal_mat
                    .transform_vector3(local_hit.normal)
                    .normalize_or_zero();
                let world_dist = (world_pt - world_ray.origin).length();

                if best.as_ref().map_or(true, |b| world_dist < b.world_distance) {
                    best = Some(WorldRayHit {
                        object_idx: i,
                        world_point: world_pt,
                        world_normal: world_n,
                        local_hit,
                        world_distance: world_dist,
                    });
                }
            }
        }

        best
    }

    /// Raycast specifically against the currently selected object.
    pub fn raycast_active(&self, world_ray: &Ray) -> Option<WorldRayHit> {
        let idx = self.selected_object?;
        let obj = self.objects.get(idx)?;
        if !obj.visible {
            return None;
        }
        let local_ray = obj.transform.transform_ray_to_local(world_ray);
        let local_hit = obj.mesh.ray_intersect(&local_ray)?;
        let model = obj.transform.matrix();
        let world_pt = model.transform_point3(local_hit.point);
        let normal_mat = model.inverse().transpose();
        let world_n = normal_mat
            .transform_vector3(local_hit.normal)
            .normalize_or_zero();
        let world_dist = (world_pt - world_ray.origin).length();

        Some(WorldRayHit {
            object_idx: idx,
            world_point: world_pt,
            world_normal: world_n,
            local_hit,
            world_distance: world_dist,
        })
    }

    /// Clears face/vertex/edge selections across all objects in the scene.
    pub fn clear_all_selections(&mut self) {
        for obj in &mut self.objects {
            if !obj.mesh.selected_faces.is_empty()
                || !obj.mesh.selected_vertices.is_empty()
                || !obj.mesh.selected_edges.is_empty()
            {
                obj.mesh.clear_selection();
            }
        }
        self.pending_anchor = None;
    }

    /// Returns true if any object has selected faces or if `selected_object` is set.
    pub fn has_any_selection(&self) -> bool {
        self.selected_object.is_some()
            || self
                .objects
                .iter()
                .any(|o| !o.mesh.selected_faces.is_empty())
    }

    /// Collects a `CommentTargetSection` for every object that currently has `selected_faces`
    /// (or falls back to the active object if no specific faces are highlighted).
    pub fn collect_current_target_sections(
        &self,
        selection_radius: f32,
    ) -> Vec<CommentTargetSection> {
        let mut sections = Vec::new();

        for obj in &self.objects {
            if obj.mesh.selected_faces.is_empty() {
                continue;
            }
            let model = obj.transform.matrix();
            let normal_mat = model.inverse().transpose();
            let mut selected_faces: Vec<usize> = obj.mesh.selected_faces.iter().copied().collect();
            selected_faces.sort_unstable();

            let (local_anchor, local_normal, b_min, b_max) =
                if let Some((c, n, min_p, max_p)) = obj.mesh.selected_faces_bounds_and_normal() {
                    (c, n, min_p, max_p)
                } else {
                    let (min_p, max_p) = obj.mesh.bounds();
                    ((min_p + max_p) * 0.5, Vec3::Y, min_p, max_p)
                };
            let world_anchor = model.transform_point3(local_anchor);
            let surface_normal = normal_mat
                .transform_vector3(local_normal)
                .normalize_or_zero();

            sections.push(CommentTargetSection {
                object_id: obj.id,
                object_name: obj.name.clone(),
                world_anchor,
                local_anchor,
                surface_normal,
                selected_faces,
                local_bounds_min: b_min,
                local_bounds_max: b_max,
            });
        }

        if sections.is_empty() {
            if let Some(obj) = self.active_object() {
                let model = obj.transform.matrix();
                let normal_mat = model.inverse().transpose();
                let (min_p, max_p) = obj.mesh.bounds();
                let (local_anchor, local_normal, b_min, b_max) =
                    if let Some((anchor_obj_id, _wp, lp, wn)) = self.pending_anchor {
                        if anchor_obj_id == obj.id {
                            (
                                lp,
                                obj.transform
                                    .inverse_matrix()
                                    .transform_vector3(wn)
                                    .normalize_or_zero(),
                                (lp - Vec3::splat(selection_radius)).max(min_p),
                                (lp + Vec3::splat(selection_radius)).min(max_p),
                            )
                        } else {
                            ((min_p + max_p) * 0.5, Vec3::Y, min_p, max_p)
                        }
                    } else {
                        ((min_p + max_p) * 0.5, Vec3::Y, min_p, max_p)
                    };
                let world_anchor = if let Some((anchor_obj_id, wp, _lp, _wn)) = self.pending_anchor
                {
                    if anchor_obj_id == obj.id {
                        wp
                    } else {
                        model.transform_point3(local_anchor)
                    }
                } else {
                    model.transform_point3(local_anchor)
                };
                let surface_normal = normal_mat
                    .transform_vector3(local_normal)
                    .normalize_or_zero();
                sections.push(CommentTargetSection {
                    object_id: obj.id,
                    object_name: obj.name.clone(),
                    world_anchor,
                    local_anchor,
                    surface_normal,
                    selected_faces: Vec::new(),
                    local_bounds_min: b_min,
                    local_bounds_max: b_max,
                });
            }
        }

        sections
    }

    /// Creates a new `SpatialComment` attached to all currently selected sections/objects.
    pub fn add_comment_for_selection(
        &mut self,
        comment_text: impl Into<String>,
        selection_radius: f32,
    ) -> Option<SpatialComment> {
        let targets = self.collect_current_target_sections(selection_radius);
        let first = targets.first()?.clone();

        let (obj_id, obj_name, world_anchor, surface_normal) = if targets.len() == 1 {
            (
                first.object_id,
                first.object_name.clone(),
                first.world_anchor,
                first.surface_normal,
            )
        } else {
            let names: Vec<&str> = targets.iter().map(|t| t.object_name.as_str()).collect();
            let combined_name = format!("{} Parts ({})", targets.len(), names.join(", "));
            let avg_world = targets
                .iter()
                .fold(Vec3::ZERO, |acc, t| acc + t.world_anchor)
                / (targets.len() as f32);
            let avg_norm = targets
                .iter()
                .fold(Vec3::ZERO, |acc, t| acc + t.surface_normal)
                .normalize_or_zero();
            (
                first.object_id,
                combined_name,
                avg_world,
                if avg_norm.length_squared() > 1e-4 {
                    avg_norm
                } else {
                    Vec3::Y
                },
            )
        };

        let cid = self.next_comment_id.max(1);
        self.next_comment_id = cid + 1;

        let item = SpatialComment {
            id: cid,
            is_general: false,
            object_id: obj_id,
            object_name: obj_name,
            world_anchor,
            local_anchor: first.local_anchor,
            surface_normal,
            selection_radius,
            selected_faces: first.selected_faces.clone(),
            local_bounds_min: first.local_bounds_min,
            local_bounds_max: first.local_bounds_max,
            targets,
            comment: comment_text.into(),
            resolved: false,
            agent_reply: None,
        };

        self.comments.push(item.clone());
        self.active_comment_id = Some(cid);
        Some(item)
    }

    /// Creates a general scene-wide comment not tied to a specific 3D surface pin or object section.
    pub fn add_general_comment(&mut self, comment_text: impl Into<String>) -> SpatialComment {
        let cid = self.next_comment_id.max(1);
        self.next_comment_id = cid + 1;

        let item = SpatialComment {
            id: cid,
            is_general: true,
            object_id: 0,
            object_name: "General Scene".to_string(),
            world_anchor: Vec3::ZERO,
            local_anchor: Vec3::ZERO,
            surface_normal: Vec3::Y,
            selection_radius: 0.0,
            selected_faces: Vec::new(),
            local_bounds_min: Vec3::ZERO,
            local_bounds_max: Vec3::ZERO,
            targets: Vec::new(),
            comment: comment_text.into(),
            resolved: false,
            agent_reply: None,
        };

        self.comments.push(item.clone());
        self.active_comment_id = Some(cid);
        item
    }

    /// Selects the object(s) and restores all highlighted face regions associated with comment `comment_id`.
    pub fn focus_comment(&mut self, comment_id: u64) -> Option<Vec3> {
        let comment = self.comments.iter().find(|c| c.id == comment_id)?.clone();
        self.active_comment_id = Some(comment_id);

        if comment.is_general || comment.object_id == 0 {
            self.pending_anchor = None;
            return Some(Vec3::ZERO);
        }

        self.clear_all_selections();

        if !comment.targets.is_empty() {
            for target in &comment.targets {
                if let Some(idx) = self.objects.iter().position(|o| o.id == target.object_id) {
                    self.selected_object = Some(idx);
                    let obj = &mut self.objects[idx];
                    for &fi in &target.selected_faces {
                        if fi < obj.mesh.faces.len() {
                            obj.mesh.selected_faces.insert(fi);
                        }
                    }
                    obj.mesh.bump_version();
                }
            }
            if let Some(first) = comment.targets.first() {
                self.pending_anchor = Some((
                    first.object_id,
                    first.world_anchor,
                    first.local_anchor,
                    first.surface_normal,
                ));
            }
        } else if let Some(idx) = self.objects.iter().position(|o| o.id == comment.object_id) {
            self.selected_object = Some(idx);
            let obj = &mut self.objects[idx];
            for &fi in &comment.selected_faces {
                if fi < obj.mesh.faces.len() {
                    obj.mesh.selected_faces.insert(fi);
                }
            }
            obj.mesh.bump_version();
            self.pending_anchor = Some((
                comment.object_id,
                comment.world_anchor,
                comment.local_anchor,
                comment.surface_normal,
            ));
        }

        Some(comment.world_anchor)
    }
}
