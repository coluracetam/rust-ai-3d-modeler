//! Asset library: reusable parts kept on disk as `Section/Subsection/Asset`, shared by every
//! project. Each asset is `<name>.asset.json` (its parts), `<name>.meta.json` (a small summary
//! read when listing) and an optional `<name>.png` thumbnail. Sections and subsections are
//! plain folders, so the library can also be organised with a file manager.
use crate::camera::OrbitCamera;
use crate::history::UndoHistory;
use crate::ipc::{IpcRequest, IpcResponse, parse_vec3};
use crate::renderer::ViewportShading;
use crate::scene::{Scene, SceneObject};
use glam::{EulerRot, Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_SUBSECTION: &str = "General";
const ASSET_EXT: &str = ".asset.json";
const META_EXT: &str = ".meta.json";
const THUMB_EXT: &str = ".png";
const TRASH_DIR: &str = ".trash";

/// Tools that never change the scene (they only touch library files).
pub const READ_ONLY: &[&str] = &[
    "library_list",
    "library_show",
    "library_save_asset",
    "library_create_section",
    "library_delete_asset",
    "library_delete_section",
    "library_move_asset",
];
pub const GEOMETRY: &[&str] = &["library_insert_asset"];
/// Tools worth running off the GUI thread (file parsing, thumbnail rendering).
pub const HEAVY: &[&str] = &["library_save_asset", "library_insert_asset"];

#[derive(Serialize, Deserialize)]
pub struct Asset {
    pub format_version: u32,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Parts in asset space: the centre of the asset's base sits at the origin.
    pub objects: Vec<SceneObject>,
}

/// The summary stored next to an asset so listing never parses geometry.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AssetMeta {
    #[serde(default)]
    description: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    parts: usize,
    #[serde(default)]
    faces: usize,
    #[serde(default)]
    size: [f32; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct AssetInfo {
    pub name: String,
    pub section: String,
    pub subsection: String,
    pub description: String,
    pub tags: Vec<String>,
    pub parts: usize,
    pub faces: usize,
    /// Bounding-box size in scene units.
    pub size: [f32; 3],
    pub path: String,
    pub thumbnail: Option<String>,
    /// Seconds since the Unix epoch; changes when the asset is saved again.
    pub modified: u64,
}

impl AssetInfo {
    /// `Section/Subsection/Name`, the reference accepted by the library tools.
    pub fn reference(&self) -> String {
        format!("{}/{}/{}", self.section, self.subsection, self.name)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Subsection {
    pub name: String,
    pub assets: Vec<AssetInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Section {
    pub name: String,
    pub subsections: Vec<Subsection>,
}

/// Library folder: `$RUST_MODELER_LIBRARY`, else `$XDG_DATA_HOME/rust_modeler/library`,
/// else `~/.local/share/rust_modeler/library`.
pub fn root() -> PathBuf {
    let env = |key: &str| std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = env("RUST_MODELER_LIBRARY") {
        return dir;
    }
    env("XDG_DATA_HOME")
        .or_else(|| env("HOME").map(|home| home.join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rust_modeler/library")
}

/// Section, subsection and asset names become file names, so keep them to one safe path part.
pub fn clean_name(name: &str, what: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(format!("Give the {what} a name"));
    }
    if name.chars().count() > 80 {
        return Err(format!("The {what} name is too long (80 characters at most)"));
    }
    if name.starts_with('.')
        || name.ends_with('.')
        || name
            .chars()
            .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
    {
        return Err(format!(
            "The {what} name '{name}' cannot start or end with a dot or contain / \\ : * ? \" < > |"
        ));
    }
    Ok(name.to_string())
}

fn sorted_dirs(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.file_name().to_str()?.to_string(), e.path())))
        .filter(|(name, _)| !name.starts_with('.'))
        .collect();
    out.sort_by_key(|(name, _)| name.to_lowercase());
    out
}

fn asset_info(dir: &Path, section: &str, subsection: &str, name: &str) -> AssetInfo {
    let path = dir.join(format!("{name}{ASSET_EXT}"));
    let meta: AssetMeta = fs::read(dir.join(format!("{name}{META_EXT}")))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let thumbnail = dir.join(format!("{name}{THUMB_EXT}"));
    let modified = fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    AssetInfo {
        name: name.to_string(),
        section: section.to_string(),
        subsection: subsection.to_string(),
        description: meta.description,
        tags: meta.tags,
        parts: meta.parts,
        faces: meta.faces,
        size: meta.size,
        path: path.to_string_lossy().into_owned(),
        thumbnail: thumbnail
            .exists()
            .then(|| thumbnail.to_string_lossy().into_owned()),
        modified,
    }
}

/// Every section, subsection and asset, sorted by name. A missing library is an empty one.
pub fn list(root: &Path) -> Vec<Section> {
    sorted_dirs(root)
        .into_iter()
        .map(|(section, section_dir)| Section {
            subsections: sorted_dirs(&section_dir)
                .into_iter()
                .map(|(subsection, dir)| {
                    let mut names: Vec<String> = fs::read_dir(&dir)
                        .into_iter()
                        .flatten()
                        .flatten()
                        .filter_map(|e| {
                            let file = e.file_name().to_str()?.to_string();
                            Some(file.strip_suffix(ASSET_EXT)?.to_string())
                        })
                        .filter(|name| !name.is_empty() && !name.starts_with('.'))
                        .collect();
                    names.sort_by_key(|n| n.to_lowercase());
                    Subsection {
                        assets: names
                            .iter()
                            .map(|name| asset_info(&dir, &section, &subsection, name))
                            .collect(),
                        name: subsection,
                    }
                })
                .collect(),
            name: section,
        })
        .collect()
}

/// `parent/name`, reusing an existing folder that differs only in letter case so "chairs"
/// and "Chairs" never become two sections.
fn folder(parent: &Path, name: &str) -> PathBuf {
    sorted_dirs(parent)
        .into_iter()
        .find(|(existing, _)| existing.to_lowercase() == name.to_lowercase())
        .map_or_else(|| parent.join(name), |(_, path)| path)
}

pub fn create_section(root: &Path, section: &str, subsection: Option<&str>) -> Result<PathBuf, String> {
    let mut dir = folder(root, &clean_name(section, "section")?);
    if let Some(sub) = subsection {
        dir = folder(&dir, &clean_name(sub, "subsection")?);
    }
    fs::create_dir_all(&dir).map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// The section and subsection names as spelled on disk for a `root/Section/Subsection` folder.
fn folder_names(dir: &Path, section: String, subsection: String) -> (String, String) {
    let name = |p: Option<&Path>| p.and_then(|p| p.file_name()).and_then(|n| n.to_str()).map(str::to_string);
    (name(dir.parent()).unwrap_or(section), name(Some(dir)).unwrap_or(subsection))
}

/// Removes an empty subsection, or a section whose subsections are all empty.
pub fn delete_section(root: &Path, section: &str, subsection: Option<&str>) -> Result<(), String> {
    let section_dir = folder(root, &clean_name(section, "section")?);
    let dirs = match subsection {
        Some(sub) => vec![folder(&section_dir, &clean_name(sub, "subsection")?)],
        None => {
            let mut dirs: Vec<PathBuf> = sorted_dirs(&section_dir).into_iter().map(|(_, p)| p).collect();
            dirs.push(section_dir.clone());
            dirs
        }
    };
    if !dirs.last().is_some_and(|d| d.is_dir()) {
        return Err("That section does not exist".into());
    }
    // Check everything first so a refused delete leaves the section untouched.
    for dir in &dirs {
        let occupied = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .flatten()
            .any(|e| !e.path().is_dir() || (subsection.is_some() || *dir != section_dir));
        if occupied {
            return Err("Only empty sections can be deleted; move or delete its assets first".into());
        }
    }
    for dir in &dirs {
        fs::remove_dir(dir).map_err(|e| format!("Could not delete {}: {e}", dir.display()))?;
    }
    Ok(())
}

/// Finds an asset from `Section/Subsection/Name`, or from a bare name when it is unambiguous.
pub fn find_asset(root: &Path, reference: &str) -> Result<AssetInfo, String> {
    let parts: Vec<&str> = reference.split('/').map(str::trim).collect();
    let all = || list(root).into_iter().flat_map(|s| s.subsections).flat_map(|s| s.assets);
    let same = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
    let found: Vec<AssetInfo> = match parts.as_slice() {
        [section, subsection, name] => all()
            .filter(|a| same(&a.section, section) && same(&a.subsection, subsection) && same(&a.name, name))
            .collect(),
        [name] => all().filter(|a| same(&a.name, name)).collect(),
        _ => return Err("Name the asset as 'Section/Subsection/Name'".into()),
    };
    match found.len() {
        0 => Err(format!("No library asset named '{reference}'")),
        1 => Ok(found.into_iter().next().unwrap()),
        _ => Err(format!(
            "Several assets are named '{reference}': {}. Use the full 'Section/Subsection/Name'.",
            found.iter().map(AssetInfo::reference).collect::<Vec<_>>().join(", ")
        )),
    }
}

fn bounds_of(objects: &[SceneObject]) -> Option<(Vec3, Vec3)> {
    objects
        .iter()
        .filter_map(crate::agent_tools::world_bounds)
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

pub struct SaveOptions<'a> {
    pub section: &'a str,
    pub subsection: &'a str,
    pub name: &'a str,
    pub description: &'a str,
    pub tags: Vec<String>,
    pub overwrite: bool,
}

/// Saves the parts at `indices` as one asset and returns its listing entry.
pub fn save_asset(root: &Path, scene: &Scene, indices: &[usize], options: SaveOptions) -> Result<AssetInfo, String> {
    let section = clean_name(options.section, "section")?;
    let subsection = clean_name(options.subsection, "subsection")?;
    let name = clean_name(options.name, "asset")?;
    let mut objects: Vec<SceneObject> = indices
        .iter()
        .filter_map(|&i| scene.objects.get(i))
        .filter(|o| !o.mesh.faces.is_empty())
        .cloned()
        .collect();
    let Some((min, max)) = bounds_of(&objects) else {
        return Err("Choose at least one part with geometry to save".into());
    };
    let base = Vec3::new((min.x + max.x) * 0.5, min.y, (min.z + max.z) * 0.5);
    for (i, object) in objects.iter_mut().enumerate() {
        object.id = i as u64 + 1;
        object.visible = true;
        object.locked = false;
        object.mesh.clear_selection();
        object.transform.translation -= base;
    }
    let dir = create_section(root, &section, Some(&subsection))?;
    let (section, subsection) = folder_names(&dir, section, subsection);
    let path = dir.join(format!("{name}{ASSET_EXT}"));
    if path.exists() && !options.overwrite {
        return Err(format!(
            "'{section}/{subsection}/{name}' already exists; pick another name or overwrite it"
        ));
    }
    let meta = AssetMeta {
        description: options.description.trim().to_string(),
        tags: options.tags,
        parts: objects.len(),
        faces: objects.iter().map(|o| o.mesh.faces.len()).sum(),
        size: (max - min).to_array(),
    };
    let asset = Asset {
        format_version: 1,
        name: name.clone(),
        description: meta.description.clone(),
        tags: meta.tags.clone(),
        objects,
    };
    let write = |path: &Path, bytes: Result<Vec<u8>, serde_json::Error>| {
        crate::project::atomic_write(path, &bytes.map_err(|e| e.to_string())?, false)
    };
    write(&path, serde_json::to_vec(&asset))?;
    write(&dir.join(format!("{name}{META_EXT}")), serde_json::to_vec(&meta))?;
    // A thumbnail from an earlier version would show the wrong model.
    let _ = fs::remove_file(dir.join(format!("{name}{THUMB_EXT}")));
    Ok(asset_info(&dir, &section, &subsection, &name))
}

pub fn load_asset(info: &AssetInfo) -> Result<Asset, String> {
    let bytes = fs::read(&info.path).map_err(|e| format!("Could not open {}: {e}", info.path))?;
    let mut asset: Asset =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid asset file {}: {e}", info.path))?;
    if asset.format_version != 1 {
        return Err("Unsupported asset version".into());
    }
    // Reuse the project checks (finite positions, face indices, voxel grids, transforms).
    let mut check = Scene::empty();
    for (i, object) in asset.objects.iter_mut().enumerate() {
        object.id = i as u64 + 1;
    }
    check.objects = std::mem::take(&mut asset.objects);
    check.validate().map_err(|e| format!("Invalid asset file {}: {e}", info.path))?;
    asset.objects = check.objects;
    if asset.objects.is_empty() {
        return Err(format!("Asset file {} has no parts", info.path));
    }
    Ok(asset)
}

/// Renders the asset's thumbnail next to it. Needs a GPU adapter, so callers treat failure as
/// "no thumbnail".
pub fn render_thumbnail(info: &AssetInfo) -> Result<String, String> {
    let asset = load_asset(info)?;
    let mut scene = Scene::empty();
    scene.objects = asset.objects;
    let bounds = bounds_of(&scene.objects).ok_or("Asset has no geometry")?;
    let (width, height) = (480, 360);
    let mut camera = OrbitCamera {
        aspect: width as f32 / height as f32,
        yaw: 0.7,
        pitch: 0.45,
        ..Default::default()
    };
    crate::agent_tools::frame_camera(&mut camera, bounds, 1.15);
    let path = Path::new(&info.path)
        .with_file_name(format!("{}{THUMB_EXT}", info.name));
    crate::renderer::render_scene_headless_to_png(
        &camera,
        &scene,
        ViewportShading::StudioLit,
        false,
        false,
        width,
        height,
        &path,
    )?;
    Ok(path.to_string_lossy().into_owned())
}

/// Moves an asset to the library's `.trash` folder (recoverable with a file manager).
pub fn delete_asset(root: &Path, info: &AssetInfo) -> Result<(), String> {
    let trash = root.join(TRASH_DIR);
    fs::create_dir_all(&trash).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let prefix = format!("{stamp} {} - {} - ", info.section, info.subsection);
    move_files(info, &trash, &prefix, &info.name)
}

fn move_files(info: &AssetInfo, to_dir: &Path, prefix: &str, new_name: &str) -> Result<(), String> {
    let from_dir = Path::new(&info.path).parent().unwrap_or(Path::new("."));
    // The asset file goes last so a failure part-way never leaves a listed asset without data.
    for ext in [THUMB_EXT, META_EXT, ASSET_EXT] {
        let from = from_dir.join(format!("{}{ext}", info.name));
        if !from.exists() {
            continue;
        }
        let to = to_dir.join(format!("{prefix}{new_name}{ext}"));
        fs::rename(&from, &to).map_err(|e| format!("Could not move {}: {e}", from.display()))?;
    }
    Ok(())
}

/// Moves and/or renames an asset.
pub fn move_asset(root: &Path, info: &AssetInfo, section: &str, subsection: &str, name: &str) -> Result<AssetInfo, String> {
    let section = clean_name(section, "section")?;
    let subsection = clean_name(subsection, "subsection")?;
    let name = clean_name(name, "asset")?;
    let dir = create_section(root, &section, Some(&subsection))?;
    let (section, subsection) = folder_names(&dir, section, subsection);
    let target = dir.join(format!("{name}{ASSET_EXT}"));
    if target == Path::new(&info.path) {
        return Ok(info.clone());
    }
    if target.exists() {
        return Err(format!("'{section}/{subsection}/{name}' already exists"));
    }
    let _ = fs::remove_file(dir.join(format!("{name}{THUMB_EXT}")));
    move_files(info, &dir, "", &name)?;
    Ok(asset_info(&dir, &section, &subsection, &name))
}

/// A group name not used by any part yet: `Chair`, then `Chair 2`, `Chair 3`…
fn unique_group(scene: &Scene, base: &str) -> String {
    let taken = |candidate: &str| {
        let prefix = format!("{}/", candidate.to_lowercase());
        scene.objects.iter().any(|o| {
            let name = o.name.to_lowercase();
            name == candidate.to_lowercase() || name.starts_with(&prefix)
        })
    };
    (1..)
        .map(|n| if n == 1 { base.to_string() } else { format!("{base} {n}") })
        .find(|candidate| !taken(candidate))
        .unwrap()
}

/// Adds a copy of the asset to the scene with its base centre at `position`, turned
/// `yaw_deg` around the vertical axis and scaled uniformly. Returns the new part indices.
pub fn insert_asset(scene: &mut Scene, asset: &Asset, position: Vec3, yaw_deg: f32, scale: f32) -> Vec<usize> {
    let group = unique_group(scene, &asset.name);
    let turn = Quat::from_rotation_y(yaw_deg.to_radians());
    let single = asset.objects.len() == 1;
    let mut used = std::collections::HashSet::new();
    let mut indices = Vec::new();
    for source in &asset.objects {
        let mut object = source.clone();
        object.id = scene.alloc_id();
        object.name = if single {
            group.clone()
        } else {
            // Keep the part's own name, without the group it had in its original scene.
            let part = source.name.rsplit('/').next().unwrap_or("").trim();
            let part = if part.is_empty() { "Part" } else { part };
            let mut name = format!("{group}/{part}");
            let mut n = 2;
            while !used.insert(name.to_lowercase()) {
                name = format!("{group}/{part} {n}");
                n += 1;
            }
            name
        };
        let (x, y, z) = (turn * object.transform.rotation_quat()).to_euler(EulerRot::XYZ);
        object.transform.rotation_deg = Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
        object.transform.translation = position + turn * (object.transform.translation * scale);
        object.transform.scale *= scale;
        scene.objects.push(object);
        indices.push(scene.objects.len() - 1);
    }
    scene.selected_object = indices.first().copied();
    indices
}

/// Parts named by `all` (the whole model), `object_ids`, `names` or `group` (a `Group/Part`
/// prefix); with none of those, the current selection.
pub fn parts_from_args(scene: &Scene, args: &Value) -> Result<Vec<usize>, String> {
    let mut indices = Vec::new();
    let mut explicit = false;
    if args["all"].as_bool().unwrap_or(false) {
        explicit = true;
        indices = visible_parts(scene);
        if indices.is_empty() {
            return Err("The scene has no visible parts to save".into());
        }
    }
    for id in args["object_ids"].as_array().into_iter().flatten() {
        explicit = true;
        let id = id.as_u64().ok_or("object_ids must be a list of part ids")?;
        indices.push(
            scene
                .objects
                .iter()
                .position(|o| o.id == id)
                .ok_or(format!("No part with id {id}"))?,
        );
    }
    for name in args["names"].as_array().into_iter().flatten() {
        explicit = true;
        let name = name.as_str().ok_or("names must be a list of part names")?;
        indices.push(
            scene
                .objects
                .iter()
                .position(|o| o.name.eq_ignore_ascii_case(name))
                .ok_or(format!("No part named '{name}'"))?,
        );
    }
    if let Some(group) = args["group"].as_str() {
        explicit = true;
        let group = group.trim().trim_end_matches('/').to_lowercase();
        let prefix = format!("{group}/");
        let before = indices.len();
        indices.extend(scene.objects.iter().enumerate().filter_map(|(i, o)| {
            let name = o.name.to_lowercase();
            (name == group || name.starts_with(&prefix)).then_some(i)
        }));
        if indices.len() == before {
            return Err(format!("No parts in group '{group}'"));
        }
    }
    if !explicit {
        indices = selected_parts(scene);
    }
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return Err("Select the parts to save, or pass all, object_ids, names or group".into());
    }
    Ok(indices)
}

/// The whole model: every visible part that has geometry.
pub fn visible_parts(scene: &Scene) -> Vec<usize> {
    (0..scene.objects.len())
        .filter(|&i| scene.objects[i].visible && !scene.objects[i].mesh.faces.is_empty())
        .collect()
}

/// The active part plus every part with selected faces.
pub fn selected_parts(scene: &Scene) -> Vec<usize> {
    (0..scene.objects.len())
        .filter(|&i| scene.selected_object == Some(i) || !scene.objects[i].mesh.selected_faces.is_empty())
        .collect()
}

fn info_json(info: &AssetInfo) -> Value {
    let mut value = serde_json::to_value(info).unwrap_or(Value::Null);
    value["asset"] = json!(info.reference());
    value
}

/// Library tools. `None` means the tool is not one of these.
pub fn dispatch(
    req: &IpcRequest,
    scene: &mut Scene,
    history: &mut UndoHistory,
    ui_state: &mut crate::ui::UiState,
) -> Option<IpcResponse> {
    let args = &req.args;
    let root = root();
    let text = |key: &str| args[key].as_str().map(str::trim).filter(|s| !s.is_empty());
    let result: Result<IpcResponse, String> = (|| match req.tool.as_str() {
        "library_list" => {
            let search = text("search").map(str::to_lowercase);
            let mut sections = list(&root);
            if let Some(section) = text("section") {
                sections.retain(|s| s.name.eq_ignore_ascii_case(section));
            }
            if let Some(search) = &search {
                for section in &mut sections {
                    for sub in &mut section.subsections {
                        sub.assets.retain(|a| {
                            a.name.to_lowercase().contains(search)
                                || a.description.to_lowercase().contains(search)
                                || a.tags.iter().any(|t| t.to_lowercase().contains(search))
                        });
                    }
                    section.subsections.retain(|s| !s.assets.is_empty());
                }
                sections.retain(|s| !s.subsections.is_empty());
            }
            let count: usize = sections.iter().flat_map(|s| &s.subsections).map(|s| s.assets.len()).sum();
            let tree: Vec<Value> = sections
                .iter()
                .map(|section| {
                    json!({
                        "name": section.name,
                        "subsections": section.subsections.iter().map(|sub| json!({
                            "name": sub.name,
                            "assets": sub.assets.iter().map(info_json).collect::<Vec<_>>(),
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            Ok(IpcResponse::success(
                format!("{count} asset(s) in {} section(s)", sections.len()),
                json!({ "library_path": root.to_string_lossy(), "sections": tree }),
            ))
        }
        "library_show" => {
            ui_state.library.open = true;
            ui_state.library.scanned = None;
            Ok(IpcResponse::success("Opened the asset library view", json!({})))
        }
        "library_save_asset" => {
            let indices = parts_from_args(scene, args)?;
            let name = text("name").ok_or("Missing 'name' for the asset")?;
            let mut info = save_asset(
                &root,
                scene,
                &indices,
                SaveOptions {
                    section: text("section").ok_or("Missing 'section'")?,
                    subsection: text("subsection").unwrap_or(DEFAULT_SUBSECTION),
                    name,
                    description: text("description").unwrap_or(""),
                    tags: args["tags"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|t| t.as_str().map(str::to_string))
                        .collect(),
                    overwrite: args["overwrite"].as_bool().unwrap_or(false),
                },
            )?;
            if args["thumbnail"].as_bool().unwrap_or(true) {
                info.thumbnail = render_thumbnail(&info).ok();
            }
            Ok(IpcResponse::success(
                format!("Saved {} part(s) to the library as '{}'", info.parts, info.reference()),
                info_json(&info),
            ))
        }
        "library_insert_asset" => {
            let info = find_asset(&root, text("asset").ok_or("Missing 'asset'")?)?;
            let asset = load_asset(&info)?;
            let position = parse_vec3(&args["position"], Vec3::ZERO);
            let yaw = args["rotation_y_deg"].as_f64().unwrap_or(0.0) as f32;
            let scale = args["scale"].as_f64().unwrap_or(1.0) as f32;
            if !position.is_finite() || !yaw.is_finite() || !scale.is_finite() || scale <= 1e-4 {
                return Err("position, rotation_y_deg and scale must be finite, with scale above 0".into());
            }
            history.record("Insert library asset", scene);
            let indices = insert_asset(scene, &asset, position, yaw, scale);
            let objects: Vec<Value> = indices
                .iter()
                .map(|&i| json!({ "object_id": scene.objects[i].id, "name": scene.objects[i].name }))
                .collect();
            let bounds = bounds_of(&indices.iter().map(|&i| scene.objects[i].clone()).collect::<Vec<_>>())
                .unwrap_or((position, position));
            Ok(IpcResponse::success(
                format!("Inserted '{}' ({} part(s))", info.reference(), indices.len()),
                json!({
                    "asset": info.reference(),
                    "objects": objects,
                    "world_bounds_min": bounds.0.to_array(),
                    "world_bounds_max": bounds.1.to_array(),
                }),
            ))
        }
        "library_create_section" => {
            let section = text("section").ok_or("Missing 'section'")?;
            let dir = create_section(&root, section, text("subsection"))?;
            Ok(IpcResponse::success(
                format!("Library section ready: {}", dir.strip_prefix(&root).unwrap_or(&dir).display()),
                json!({ "path": dir.to_string_lossy() }),
            ))
        }
        "library_delete_section" => {
            let section = text("section").ok_or("Missing 'section'")?;
            delete_section(&root, section, text("subsection"))?;
            Ok(IpcResponse::success("Deleted the empty library section", json!({})))
        }
        "library_delete_asset" => {
            let info = find_asset(&root, text("asset").ok_or("Missing 'asset'")?)?;
            delete_asset(&root, &info)?;
            Ok(IpcResponse::success(
                format!("Moved '{}' to the library trash", info.reference()),
                json!({ "trash": root.join(TRASH_DIR).to_string_lossy() }),
            ))
        }
        "library_move_asset" => {
            let info = find_asset(&root, text("asset").ok_or("Missing 'asset'")?)?;
            let moved = move_asset(
                &root,
                &info,
                text("section").unwrap_or(&info.section),
                text("subsection").unwrap_or(&info.subsection),
                text("name").unwrap_or(&info.name),
            )?;
            Ok(IpcResponse::success(
                format!("'{}' is now '{}'", info.reference(), moved.reference()),
                info_json(&moved),
            ))
        }
        _ => Err(String::new()),
    })();
    match result {
        Ok(response) => Some(response),
        Err(e) if e.is_empty() => None,
        Err(e) => Some(IpcResponse::error(e)),
    }
}

/// Tests that point `RUST_MODELER_LIBRARY` at a temporary folder hold this while they run.
#[cfg(test)]
pub static TEST_ROOT: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_modeler_library_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn chair_scene() -> Scene {
        let mut scene = Scene::empty();
        scene.add_mesh_object("Chair/Seat", Mesh::cube(1.0), [0.6, 0.4, 0.2]);
        scene.add_mesh_object("Chair/Leg", Mesh::cube(0.2), [0.5, 0.3, 0.2]);
        scene.add_mesh_object("Lamp", Mesh::cube(0.5), [0.9, 0.9, 0.6]);
        scene.objects[0].transform.translation = Vec3::new(4.0, 2.0, -3.0);
        scene.objects[1].transform.translation = Vec3::new(4.0, 1.0, -3.0);
        scene.objects[0].mesh.selected_faces.insert(2);
        scene
    }

    fn options<'a>(section: &'a str, subsection: &'a str, name: &'a str) -> SaveOptions<'a> {
        SaveOptions { section, subsection, name, description: "", tags: vec![], overwrite: false }
    }

    #[test]
    fn save_list_insert_round_trip() {
        let root = temp_root("round_trip");
        let scene = chair_scene();
        let info = save_asset(&root, &scene, &[0, 1], options("Furniture", "Chairs", "Oak chair")).unwrap();
        assert_eq!((info.parts, info.faces), (2, 12));
        assert_eq!(info.reference(), "Furniture/Chairs/Oak chair");
        assert!(save_asset(&root, &scene, &[0, 1], options("furniture", "CHAIRS", "Oak chair")).is_err());
        let lamp = save_asset(&root, &scene, &[2], options("furniture", "Lighting", "Lamp")).unwrap();
        assert_eq!(lamp.reference(), "Furniture/Lighting/Lamp");

        let tree = list(&root);
        assert_eq!(tree.len(), 1);
        let subs: Vec<&str> = tree[0].subsections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(subs, ["Chairs", "Lighting"]);
        assert_eq!(tree[0].subsections[0].assets[0].name, "Oak chair");

        let found = find_asset(&root, "oak chair").unwrap();
        let asset = load_asset(&found).unwrap();
        assert!(asset.objects.iter().all(|o| o.mesh.selected_faces.is_empty()));
        // The base centre of the saved parts is the asset origin.
        let (min, max) = bounds_of(&asset.objects).unwrap();
        assert!(min.y.abs() < 1e-5 && (min.x + max.x).abs() < 1e-5 && (min.z + max.z).abs() < 1e-5);

        let mut target = chair_scene();
        let indices = insert_asset(&mut target, &asset, Vec3::new(10.0, 0.0, 0.0), 90.0, 2.0);
        let names: Vec<&str> = indices.iter().map(|&i| target.objects[i].name.as_str()).collect();
        // "Oak chair" is free; the parts keep their own names under the new group.
        assert_eq!(names, ["Oak chair/Seat", "Oak chair/Leg"]);
        let again = insert_asset(&mut target, &asset, Vec3::ZERO, 0.0, 1.0);
        assert_eq!(target.objects[again[0]].name, "Oak chair 2/Seat");
        let placed: Vec<SceneObject> = indices.iter().map(|&i| target.objects[i].clone()).collect();
        let (min, max) = bounds_of(&placed).unwrap();
        assert!((min.y).abs() < 1e-4 && ((min.x + max.x) * 0.5 - 10.0).abs() < 1e-4);
        assert!(((max.y - min.y) - 2.0 * info.size[1]).abs() < 1e-4);
        target.validate().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sections_move_and_delete() {
        let root = temp_root("sections");
        let scene = chair_scene();
        create_section(&root, "Props", Some("Empty")).unwrap();
        assert!(create_section(&root, "../escape", None).is_err());
        assert!(clean_name(".hidden", "section").is_err());
        let info = save_asset(&root, &scene, &[2], options("Props", "Lights", "Lamp")).unwrap();
        assert!(delete_section(&root, "Props", None).is_err());
        assert!(root.join("Props/Empty").is_dir(), "a refused delete removes nothing");
        delete_section(&root, "Props", Some("Empty")).unwrap();

        let moved = move_asset(&root, &info, "Lighting", "Table", "Desk lamp").unwrap();
        assert_eq!(moved.reference(), "Lighting/Table/Desk lamp");
        assert!(find_asset(&root, "Lamp").is_err());
        assert_eq!(load_asset(&moved).unwrap().objects.len(), 1);
        delete_section(&root, "Props", None).unwrap();

        delete_asset(&root, &moved).unwrap();
        assert!(find_asset(&root, "Desk lamp").is_err());
        assert_eq!(fs::read_dir(root.join(TRASH_DIR)).unwrap().count(), 2);
        delete_section(&root, "Lighting", None).unwrap();
        assert!(list(&root).is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tools_save_insert_and_undo() {
        let root = temp_root("tools");
        let _guard = TEST_ROOT.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("RUST_MODELER_LIBRARY", &root) };
        let mut scene = chair_scene();
        let mut camera = OrbitCamera::default();
        let mut history = UndoHistory::default();
        let mut ui = crate::ui::UiState::default();
        let (mut shading, mut wire, mut grid) = (ViewportShading::StudioLit, false, true);
        let mut run = |scene: &mut Scene, tool: &str, args: Value| {
            crate::ipc::execute_ipc_command(
                &IpcRequest { tool: tool.into(), args },
                scene,
                &mut camera,
                &mut history,
                &mut ui,
                &mut shading,
                &mut wire,
                &mut grid,
            )
        };
        let saved = run(
            &mut scene,
            "library_save_asset",
            json!({ "name": "Oak chair", "section": "Furniture", "group": "Chair", "thumbnail": false, "tags": ["wood"] }),
        );
        assert!(saved.ok, "{}", saved.message);
        assert_eq!(saved.data["asset"], "Furniture/General/Oak chair");
        assert_eq!(scene.objects.len(), 3, "saving leaves the scene alone");

        let listed = run(&mut scene, "library_list", json!({ "search": "WOOD" }));
        assert_eq!(listed.data["sections"][0]["subsections"][0]["assets"][0]["parts"], 2);
        assert!(run(&mut scene, "library_list", json!({ "search": "metal" })).data["sections"].as_array().unwrap().is_empty());

        let inserted = run(&mut scene, "library_insert_asset", json!({ "asset": "oak chair", "position": [1, 0, 2] }));
        assert!(inserted.ok, "{}", inserted.message);
        assert_eq!(scene.objects.len(), 5);
        assert_eq!(inserted.data["objects"][0]["name"], "Oak chair/Seat");
        assert!(run(&mut scene, "undo", json!({})).ok);
        assert_eq!(scene.objects.len(), 3);

        assert!(!run(&mut scene, "library_insert_asset", json!({ "asset": "Missing" })).ok);
        assert!(!run(&mut scene, "library_delete_section", json!({ "section": "Furniture" })).ok);
        assert!(run(&mut scene, "library_delete_asset", json!({ "asset": "Furniture/General/Oak chair" })).ok);
        assert!(run(&mut scene, "library_delete_section", json!({ "section": "Furniture" })).ok);
        unsafe { std::env::remove_var("RUST_MODELER_LIBRARY") };
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parts_come_from_arguments_or_the_selection() {
        let mut scene = chair_scene();
        assert_eq!(parts_from_args(&scene, &json!({ "group": "Chair" })).unwrap(), [0, 1]);
        assert_eq!(parts_from_args(&scene, &json!({ "names": ["lamp"] })).unwrap(), [2]);
        scene.objects[1].visible = false;
        assert_eq!(parts_from_args(&scene, &json!({ "all": true })).unwrap(), [0, 2]);
        scene.objects[1].visible = true;
        // Active part (the lamp, added last) plus the seat with a selected face.
        assert_eq!(parts_from_args(&scene, &json!({})).unwrap(), [0, 2]);
        assert!(parts_from_args(&scene, &json!({ "object_ids": [99] })).is_err());
        scene.clear_all_selections();
        scene.selected_object = None;
        assert!(parts_from_args(&scene, &json!({})).is_err());
    }
}
