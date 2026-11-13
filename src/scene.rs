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
        Mat4::from_scale_rotation_translation(self.scale, self.rotation_quat(), self.translation)
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

/// A world-space box the user places to mark a region. Unlike face indices it survives
/// topology changes and can mark empty space ("add something here").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionVolume {
    pub center: Vec3,
    pub half_extents: Vec3,
    #[serde(default)]
    pub label: String,
}

impl SelectionVolume {
    pub const MIN_HALF: f32 = 0.005;

    pub fn new(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            center,
            half_extents: half_extents.abs().max(Vec3::splat(Self::MIN_HALF)),
            label: String::new(),
        }
    }

    pub fn from_min_max(a: Vec3, b: Vec3) -> Self {
        let (min, max) = (a.min(b), a.max(b));
        Self::new((min + max) * 0.5, (max - min) * 0.5)
    }

    pub fn min(&self) -> Vec3 {
        self.center - self.half_extents
    }

    pub fn max(&self) -> Vec3 {
        self.center + self.half_extents
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min()).all() && p.cmple(self.max()).all()
    }

    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min(), self.max());
        [
            Vec3::new(a.x, a.y, a.z),
            Vec3::new(b.x, a.y, a.z),
            Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, a.y, b.z),
            Vec3::new(a.x, b.y, a.z),
            Vec3::new(b.x, b.y, a.z),
            Vec3::new(b.x, b.y, b.z),
            Vec3::new(a.x, b.y, b.z),
        ]
    }

    /// Corner index pairs of the 12 edges.
    pub const EDGES: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    /// Centre of the face on `axis` (0..3), on the positive or negative side.
    pub fn face_handle(&self, axis: usize, positive: bool) -> Vec3 {
        let mut p = self.center;
        p[axis] += if positive { self.half_extents[axis] } else { -self.half_extents[axis] };
        p
    }

    /// Move one face along its axis to `value`, keeping the opposite face in place.
    pub fn move_face(&mut self, axis: usize, positive: bool, value: f32) {
        let (mut lo, mut hi) = (self.min()[axis], self.max()[axis]);
        if positive {
            hi = value.max(lo + 2.0 * Self::MIN_HALF);
        } else {
            lo = value.min(hi - 2.0 * Self::MIN_HALF);
        }
        self.center[axis] = (lo + hi) * 0.5;
        self.half_extents[axis] = (hi - lo) * 0.5;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommentTargetSection {
    /// Name the user gave this region (e.g. "left tower roof"); empty when unnamed.
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub topology_hash: Option<u64>,
    #[serde(default)]
    pub face_signatures: Vec<u64>,
    pub object_id: u64,
    pub object_name: String,
    pub world_anchor: Vec3,
    pub local_anchor: Vec3,
    pub surface_normal: Vec3,
    pub selected_faces: Vec<usize>,
    pub local_bounds_min: Vec3,
    pub local_bounds_max: Vec3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommentStatus {
    #[default]
    Open,
    InProgress,
    ReadyForReview,
    Accepted,
}
impl CommentStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::InProgress => "In progress",
            Self::ReadyForReview => "Ready for review",
            Self::Accepted => "Accepted",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommentReply {
    pub author: String,
    pub text: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedSelection {
    pub name: String,
    pub targets: Vec<CommentTargetSection>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedView {
    pub name: String,
    pub camera: crate::camera::OrbitCamera,
}
fn default_project_name() -> String {
    "Untitled project".into()
}
fn default_units() -> String {
    "m".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialComment {
    #[serde(default)]
    pub status: CommentStatus,
    #[serde(default)]
    pub replies: Vec<CommentReply>,
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
    /// World-space box marking the region. When present the comment follows the box, so it
    /// never needs reattachment and may cover empty space.
    #[serde(default)]
    pub volume: Option<SelectionVolume>,
    pub comment: String,
    pub resolved: bool,
    pub agent_reply: Option<String>,
    /// Still being written by the user; hidden from agents until sent.
    #[serde(default)]
    pub draft: bool,
    /// Reference images (absolute paths) the user attached for the agent.
    #[serde(default)]
    pub attachments: Vec<String>,
}

fn default_next_comment_id() -> u64 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    #[serde(default = "default_project_name")]
    pub project_name: String,
    #[serde(default)]
    pub project_path: Option<String>,
    #[serde(default)]
    pub last_camera: Option<crate::camera::OrbitCamera>,
    #[serde(default = "default_units")]
    pub units: String,
    #[serde(default)]
    pub revisions: Vec<crate::review::Revision>,
    #[serde(default = "default_next_comment_id")]
    pub next_revision_id: u64,
    #[serde(default)]
    pub saved_selections: Vec<SavedSelection>,
    #[serde(default)]
    pub saved_views: Vec<SavedView>,
    /// Incremented each time the user sends comments to the agent (`wait_for_feedback`).
    #[serde(default)]
    pub feedback_batch: u64,
    #[serde(skip)]
    pub preview_revision: Option<(u64, bool)>,
    #[serde(skip)]
    pub highlight_changes: bool,
    #[serde(skip)]
    pub clip_plane: Option<[f32; 4]>,
    /// Identifies which project this in-memory scene belongs to. Bumped whenever a different
    /// project replaces the scene, so undo can tell "same project, earlier state" apart from
    /// "a different project" even when neither has a file path yet.
    #[serde(skip)]
    pub project_generation: u64,
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
    /// The 3D box selector, when placed.
    #[serde(default)]
    pub selection_volume: Option<SelectionVolume>,
    /// Names for the current selection's regions, keyed by (object id, lowest face index).
    #[serde(skip)]
    pub region_labels: std::collections::HashMap<(u64, usize), String>,
}

impl Default for Scene {
    fn default() -> Self {
        let mut scene = Self {
            project_name: "Untitled project".into(),
            project_path: None,
            last_camera: None,
            units: default_units(),
            revisions: Vec::new(),
            next_revision_id: 1,
            saved_selections: Vec::new(),
            saved_views: Vec::new(),
            feedback_batch: 0,
            preview_revision: None,
            highlight_changes: false,
            clip_plane: None,
            project_generation: 0,
            objects: Vec::new(),
            selected_object: None,
            next_id: 1,
            mode: InteractionMode::Object,
            comments: Vec::new(),
            next_comment_id: 1,
            active_comment_id: None,
            pending_anchor: None,
            selection_volume: None,
            region_labels: Default::default(),
        };
        // Start with a high-detail Sculpt/Voxel Sphere and a Reference Cube or single default sculptable sphere
        let grid = VoxelGrid::new_sphere(32, 3.2, 1.15);
        scene.add_voxel_object("Voxel Clay Sphere", grid, [0.85, 0.72, 0.62]);
        scene
    }
}

impl Scene {
    /// Replace this scene with a blank, unsaved project: no objects, comments, revisions,
    /// saved selections or views. Returns the scene that was replaced.
    pub fn start_new_project(&mut self, name: &str) -> Scene {
        let mut fresh = Scene::empty();
        let name = name.trim();
        if !name.is_empty() {
            fresh.project_name = name.to_string();
        }
        fresh.project_generation = self.project_generation + 1;
        std::mem::replace(self, fresh)
    }

    pub fn empty() -> Self {
        // Avoid constructing the default voxel object for display snapshots.
        Self {
            project_name: "Untitled project".into(),
            project_path: None,
            last_camera: None,
            units: default_units(),
            revisions: Vec::new(),
            next_revision_id: 1,
            saved_selections: Vec::new(),
            saved_views: Vec::new(),
            feedback_batch: 0,
            preview_revision: None,
            highlight_changes: false,
            clip_plane: None,
            project_generation: 0,
            objects: Vec::new(),
            selected_object: None,
            next_id: 1,
            mode: InteractionMode::Object,
            comments: Vec::new(),
            next_comment_id: 1,
            active_comment_id: None,
            pending_anchor: None,
            selection_volume: None,
            region_labels: Default::default(),
        }
    }

    pub fn validate(&mut self) -> Result<(), String> {
        fn check(objects: &[SceneObject]) -> Result<(), String> {
            let mut ids = std::collections::HashSet::new();
            for o in objects {
                if !ids.insert(o.id) {
                    return Err("Duplicate object IDs in project".into());
                }
                if o.mesh.positions.iter().any(|p| !p.is_finite())
                    || o.mesh
                        .faces
                        .iter()
                        .any(|f| f.len() < 3 || f.iter().any(|&v| v >= o.mesh.positions.len()))
                {
                    return Err(format!("Invalid mesh in '{}'", o.name));
                }
                if let Some(grid) = &o.voxel_grid {
                    let count = grid.dims.iter().try_fold(1usize, |n, &d| {
                        if (2..=512).contains(&d) {
                            n.checked_mul(d)
                        } else {
                            None
                        }
                    });
                    if count != Some(grid.sdf.len())
                        || grid.sdf.len() != grid.colors.len()
                        || !grid.voxel_size.is_finite()
                        || grid.voxel_size <= 0.0
                    {
                        return Err("Invalid voxel grid".into());
                    }
                }
                if !o.transform.rotation_deg.is_finite()
                    || !o.transform.translation.is_finite()
                    || !o.transform.scale.is_finite()
                    || o.transform.scale.abs().min_element() < 1e-8
                {
                    return Err("Invalid object transform".into());
                }
            }
            Ok(())
        }
        check(&self.objects)?;
        for r in &self.revisions {
            check(&r.before)?;
            if let Some(after) = &r.after {
                check(after)?;
            }
        }
        self.selected_object = self.selected_object.filter(|&i| i < self.objects.len());
        self.next_id = self.next_id.max(
            self.objects
                .iter()
                .map(|o| o.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        self.next_comment_id = self.next_comment_id.max(
            self.comments
                .iter()
                .map(|c| c.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        self.next_revision_id = self.next_revision_id.max(
            self.revisions
                .iter()
                .map(|r| r.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        for o in &mut self.objects {
            o.mesh.selected_faces.retain(|&i| i < o.mesh.faces.len());
            o.mesh.recalculate_normals();
        }
        for c in &mut self.comments {
            if c.resolved {
                c.status = CommentStatus::Accepted;
            }
        }
        // Migrates full-scene revision snapshots from older projects.
        self.prune_revision_geometry();
        Ok(())
    }

    /// Unsent comments the user is still writing.
    pub fn draft_count(&self) -> usize {
        self.comments.iter().filter(|c| c.draft && !c.resolved).count()
    }

    /// Release all draft comments to the agent as one batch. Returns how many were sent.
    pub fn send_drafts(&mut self) -> usize {
        let mut sent = 0;
        for c in &mut self.comments {
            if c.draft {
                c.draft = false;
                sent += 1;
            }
        }
        if sent > 0 {
            self.feedback_batch += 1;
        }
        sent
    }

    pub fn set_comment_status(&mut self, id: u64, status: CommentStatus) -> Result<(), String> {
        let c = self
            .comments
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or("Comment not found")?;
        c.status = status;
        c.resolved = status == CommentStatus::Accepted;
        Ok(())
    }

    pub fn unit_factor(&self) -> f32 {
        match self.units.as_str() {
            "mm" => 1000.0,
            "cm" => 100.0,
            "in" => 39.37008,
            _ => 1.0,
        }
    }

    pub fn target_faces(&self, target: &CommentTargetSection) -> Option<Vec<usize>> {
        let obj = self.objects.iter().find(|o| o.id == target.object_id)?;
        if target.selected_faces.is_empty() {
            return Some(Vec::new());
        }
        if target.topology_hash == Some(obj.mesh.topology_hash())
            && target
                .selected_faces
                .iter()
                .all(|&i| i < obj.mesh.faces.len())
        {
            return Some(target.selected_faces.clone());
        }
        if target.face_signatures.len() != target.selected_faces.len() {
            return None;
        }
        let mut map = std::collections::HashMap::<u64, Vec<usize>>::new();
        for i in 0..obj.mesh.faces.len() {
            map.entry(obj.mesh.face_signature(i)).or_default().push(i);
        }
        target
            .face_signatures
            .iter()
            .map(|hash| map.get(hash).filter(|ids| ids.len() == 1).map(|ids| ids[0]))
            .collect()
    }

    pub fn refreshed_target(&self, target: &CommentTargetSection) -> Option<CommentTargetSection> {
        let faces = self.target_faces(target)?;
        let object = self.objects.iter().find(|o| o.id == target.object_id)?;
        let mut result = target.clone();
        result.topology_hash = Some(object.mesh.topology_hash());
        result.face_signatures = faces
            .iter()
            .map(|&i| object.mesh.face_signature(i))
            .collect();
        if !faces.is_empty() {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            let mut center = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            for &fi in &faces {
                center += object.mesh.face_center(fi);
                normal += object.mesh.face_normal(fi);
                for &vi in &object.mesh.faces[fi] {
                    min = min.min(object.mesh.positions[vi]);
                    max = max.max(object.mesh.positions[vi]);
                }
            }
            result.local_anchor = center / faces.len() as f32;
            result.local_bounds_min = min;
            result.local_bounds_max = max;
            result.surface_normal = object
                .transform
                .matrix()
                .inverse()
                .transpose()
                .transform_vector3(normal.normalize_or_zero())
                .normalize_or_zero();
        }
        result.world_anchor = object
            .transform
            .matrix()
            .transform_point3(result.local_anchor);
        result.object_name = object.name.clone();
        result.selected_faces = faces;
        Some(result)
    }

    pub fn comment_needs_reattachment(&self, comment: &SpatialComment) -> bool {
        !comment.is_general
            && comment.volume.is_none()
            && (comment.targets.is_empty()
                || comment
                    .targets
                    .iter()
                    .any(|t| self.target_faces(t).is_none()))
    }

    pub fn reattach_comment(&mut self, id: u64, radius: f32) -> Result<(), String> {
        let targets = self.collect_current_target_sections(radius);
        let first = targets
            .first()
            .ok_or("Select a replacement region first")?
            .clone();
        let c = self
            .comments
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or("Comment not found")?;
        c.is_general = false;
        c.object_id = first.object_id;
        c.object_name = first.object_name.clone();
        c.local_anchor = first.local_anchor;
        c.world_anchor = first.world_anchor;
        c.surface_normal = first.surface_normal;
        c.selected_faces = first.selected_faces.clone();
        c.local_bounds_min = first.local_bounds_min;
        c.local_bounds_max = first.local_bounds_max;
        c.targets = targets;
        Ok(())
    }

    pub fn current_comment_anchor(&self, comment: &SpatialComment) -> Vec3 {
        let points: Vec<Vec3> = comment
            .targets
            .iter()
            .filter_map(|t| {
                let obj = self.objects.iter().find(|o| o.id == t.object_id)?;
                let faces = self.target_faces(t)?;
                let local = if faces.is_empty() {
                    t.local_anchor
                } else {
                    faces.iter().map(|&i| obj.mesh.face_center(i)).sum::<Vec3>()
                        / faces.len() as f32
                };
                Some(obj.transform.matrix().transform_point3(local))
            })
            .collect();
        if let (true, Some(v)) = (points.is_empty(), &comment.volume) {
            v.center
        } else if points.is_empty() {
            comment.world_anchor
        } else {
            points.iter().sum::<Vec3>() / points.len() as f32
        }
    }

    pub fn restore_targets(&mut self, targets: &[CommentTargetSection]) -> Result<(), String> {
        let selections: Option<Vec<_>> = targets
            .iter()
            .map(|t| self.target_faces(t).map(|faces| (t.object_id, faces)))
            .collect();
        let selections = selections
            .ok_or("Geometry changed: select the intended region and reattach this comment")?;
        self.clear_all_selections();
        self.selected_object = None;
        for (id, faces) in selections {
            if let Some(i) = self.objects.iter().position(|o| o.id == id) {
                self.selected_object = Some(i);
                self.objects[i].visible = true;
                self.objects[i].mesh.selected_faces.extend(faces);
                self.objects[i].mesh.bump_version();
            }
        }
        Ok(())
    }

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
            if let Some(local_hit) = obj.mesh.ray_intersect_filtered(&local_ray, |p| {
                self.clip_plane.is_none_or(|plane| {
                    let world = obj.transform.matrix().transform_point3(p);
                    Vec3::from_array([plane[0], plane[1], plane[2]]).dot(world) + plane[3] >= 0.0
                })
            }) {
                let model = obj.transform.matrix();
                let world_pt = model.transform_point3(local_hit.point);
                let normal_mat = model.inverse().transpose();
                let world_n = normal_mat
                    .transform_vector3(local_hit.normal)
                    .normalize_or_zero();
                let world_dist = (world_pt - world_ray.origin).length();

                if best
                    .as_ref()
                    .map_or(true, |b| world_dist < b.world_distance)
                {
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
        let local_hit = obj.mesh.ray_intersect_filtered(&local_ray, |p| {
            self.clip_plane.is_none_or(|plane| {
                let world = obj.transform.matrix().transform_point3(p);
                Vec3::from_array([plane[0], plane[1], plane[2]]).dot(world) + plane[3] >= 0.0
            })
        })?;
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

    /// Forget cached adjacency/bounds after commands that may edit geometry in place.
    pub fn invalidate_mesh_caches(&self) {
        for obj in &self.objects {
            obj.mesh.invalidate_caches();
        }
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
        self.region_labels.clear();
    }

    /// Faces of every visible part whose centre lies inside `volume`, per object index.
    pub fn faces_in_volume(
        &self,
        volume: &SelectionVolume,
    ) -> std::collections::HashMap<usize, std::collections::HashSet<usize>> {
        let (min, max) = (volume.min(), volume.max());
        let mut hits = std::collections::HashMap::new();
        for (oi, obj) in self.objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let model = obj.transform.matrix();
            // Skip parts whose bounds miss the box entirely.
            let (lo, hi) = obj.mesh.bounds();
            let mut wmin = Vec3::splat(f32::MAX);
            let mut wmax = Vec3::splat(f32::MIN);
            for i in 0..8 {
                let c = Vec3::new(
                    if i & 1 == 0 { lo.x } else { hi.x },
                    if i & 2 == 0 { lo.y } else { hi.y },
                    if i & 4 == 0 { lo.z } else { hi.z },
                );
                let w = model.transform_point3(c);
                wmin = wmin.min(w);
                wmax = wmax.max(w);
            }
            if wmin.cmpgt(max).any() || wmax.cmplt(min).any() {
                continue;
            }
            let faces = obj.mesh.faces_in_world_box(model, min, max);
            if !faces.is_empty() {
                hits.insert(oi, faces);
            }
        }
        hits
    }

    /// Make the face selection exactly the faces inside the 3D box selector.
    /// Returns (faces, parts) selected.
    pub fn apply_volume_selection(&mut self) -> (usize, usize) {
        let Some(volume) = self.selection_volume.clone() else {
            return (0, 0);
        };
        let hits = self.faces_in_volume(&volume);
        for obj in &mut self.objects {
            if !obj.mesh.selected_faces.is_empty() {
                obj.mesh.clear_selection();
            }
        }
        self.region_labels.clear();
        let mut faces = 0;
        let mut best: Option<(usize, usize)> = None;
        for (&oi, set) in &hits {
            faces += set.len();
            self.objects[oi].mesh.apply_face_set_selection(set, true, false);
            if best.is_none_or(|(n, _)| set.len() > n) {
                best = Some((set.len(), oi));
            }
        }
        self.selected_object = best.map(|(_, oi)| oi);
        self.pending_anchor = None;
        (faces, hits.len())
    }

    /// Targets of a comment as they stand now: refreshed indices where the stored faces can
    /// still be found, otherwise (for comments with a box) whatever lies inside the box.
    pub fn resolved_targets(&self, comment: &SpatialComment) -> Vec<CommentTargetSection> {
        let refreshed: Option<Vec<_>> = comment
            .targets
            .iter()
            .map(|t| self.refreshed_target(t))
            .collect();
        match (refreshed, &comment.volume) {
            (Some(list), _) => list,
            (None, Some(volume)) => {
                let mut out = Vec::new();
                let mut hits: Vec<_> = self.faces_in_volume(volume).into_iter().collect();
                hits.sort_by_key(|(oi, _)| *oi);
                for (oi, faces) in hits {
                    let obj = &self.objects[oi];
                    let mut probe = obj.mesh.clone();
                    probe.selected_faces = faces;
                    for group in probe.selected_components() {
                        out.push(self.section_for_faces(obj, group, String::new()));
                    }
                }
                out
            }
            (None, None) => comment.targets.clone(),
        }
    }

    fn section_for_faces(
        &self,
        obj: &SceneObject,
        selected_faces: Vec<usize>,
        label: String,
    ) -> CommentTargetSection {
        let model = obj.transform.matrix();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut center = Vec3::ZERO;
        let mut normal = Vec3::ZERO;
        for &fi in &selected_faces {
            center += obj.mesh.face_center(fi);
            normal += obj.mesh.face_normal(fi);
            for &vi in &obj.mesh.faces[fi] {
                min = min.min(obj.mesh.positions[vi]);
                max = max.max(obj.mesh.positions[vi]);
            }
        }
        center /= selected_faces.len().max(1) as f32;
        CommentTargetSection {
            label,
            topology_hash: Some(obj.mesh.topology_hash()),
            face_signatures: selected_faces
                .iter()
                .map(|&i| obj.mesh.face_signature(i))
                .collect(),
            object_id: obj.id,
            object_name: obj.name.clone(),
            world_anchor: model.transform_point3(center),
            local_anchor: center,
            surface_normal: model
                .inverse()
                .transpose()
                .transform_vector3(normal.normalize_or_zero())
                .normalize_or_zero(),
            selected_faces,
            local_bounds_min: min,
            local_bounds_max: max,
        }
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
            let model = obj.transform.matrix();
            let normal_mat = model.inverse().transpose();
            for selected_faces in obj.mesh.selected_components() {
                let mut min = Vec3::splat(f32::MAX);
                let mut max = Vec3::splat(f32::MIN);
                let mut center = Vec3::ZERO;
                let mut normal = Vec3::ZERO;
                for &fi in &selected_faces {
                    center += obj.mesh.face_center(fi);
                    normal += obj.mesh.face_normal(fi);
                    for &vi in &obj.mesh.faces[fi] {
                        min = min.min(obj.mesh.positions[vi]);
                        max = max.max(obj.mesh.positions[vi]);
                    }
                }
                center /= selected_faces.len() as f32;
                sections.push(CommentTargetSection {
                    label: selected_faces
                        .first()
                        .and_then(|&f| self.region_labels.get(&(obj.id, f)))
                        .cloned()
                        .unwrap_or_default(),
                    topology_hash: Some(obj.mesh.topology_hash()),
                    face_signatures: selected_faces
                        .iter()
                        .map(|&i| obj.mesh.face_signature(i))
                        .collect(),
                    object_id: obj.id,
                    object_name: obj.name.clone(),
                    world_anchor: model.transform_point3(center),
                    local_anchor: center,
                    surface_normal: normal_mat
                        .transform_vector3(normal.normalize_or_zero())
                        .normalize_or_zero(),
                    selected_faces,
                    local_bounds_min: min,
                    local_bounds_max: max,
                });
            }
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
                    label: String::new(),
                    topology_hash: Some(obj.mesh.topology_hash()),
                    face_signatures: Vec::new(),
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
        let has_faces = self.objects.iter().any(|o| !o.mesh.selected_faces.is_empty());
        if let (false, Some(volume)) = (has_faces, self.selection_volume.clone()) {
            // The box covers no geometry: a comment about that space itself.
            let cid = self.next_comment_id.max(1);
            self.next_comment_id = cid + 1;
            let item = SpatialComment {
                status: CommentStatus::Open,
                replies: Vec::new(),
                id: cid,
                is_general: false,
                object_id: 0,
                object_name: if volume.label.is_empty() {
                    "Empty space".to_string()
                } else {
                    volume.label.clone()
                },
                world_anchor: volume.center,
                local_anchor: volume.center,
                surface_normal: Vec3::Y,
                selection_radius,
                selected_faces: Vec::new(),
                local_bounds_min: volume.min(),
                local_bounds_max: volume.max(),
                targets: Vec::new(),
                volume: Some(volume),
                comment: comment_text.into(),
                resolved: false,
                agent_reply: None,
                draft: false,
                attachments: Vec::new(),
            };
            self.comments.push(item.clone());
            self.active_comment_id = Some(cid);
            return Some(item);
        }
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
            status: CommentStatus::Open,
            replies: Vec::new(),
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
            volume: self.selection_volume.clone(),
            comment: comment_text.into(),
            resolved: false,
            agent_reply: None,
            draft: false,
            attachments: Vec::new(),
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
            status: CommentStatus::Open,
            replies: Vec::new(),
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
            volume: None,
            comment: comment_text.into(),
            resolved: false,
            agent_reply: None,
            draft: false,
            attachments: Vec::new(),
        };

        self.comments.push(item.clone());
        self.active_comment_id = Some(cid);
        item
    }

    /// Selects the object(s) and restores all highlighted face regions associated with comment `comment_id`.
    pub fn focus_comment(&mut self, comment_id: u64) -> Option<Vec3> {
        let comment = self.comments.iter().find(|c| c.id == comment_id)?.clone();
        if comment.is_general {
            self.active_comment_id = Some(comment_id);
            return Some(Vec3::ZERO);
        }
        if let Some(volume) = &comment.volume {
            // The box is the source of truth: reselect whatever is inside it now.
            self.selection_volume = Some(volume.clone());
            if comment.targets.is_empty() || self.restore_targets(&comment.targets).is_err() {
                self.apply_volume_selection();
            }
            self.active_comment_id = Some(comment_id);
            return Some(volume.center);
        }
        self.restore_targets(&comment.targets).ok()?;
        let anchor = self.current_comment_anchor(&comment);
        self.active_comment_id = Some(comment_id);
        if let Some(first) = comment.targets.first() {
            self.pending_anchor = Some((
                first.object_id,
                anchor,
                first.local_anchor,
                first.surface_normal,
            ));
        }
        Some(anchor)
    }
}
