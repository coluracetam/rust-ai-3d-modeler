//! Versioned project files and crash-safe persistence shared by the GUI and agents.
use crate::{camera::OrbitCamera, scene::Scene};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
pub struct Project {
    pub format_version: u32,
    pub scene: Scene,
    pub camera: OrbitCamera,
}

pub fn absolute_path(path: &str) -> Result<String, String> {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    Ok(absolute.to_string_lossy().into_owned())
}

pub fn backup_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_os_string();
    p.push(".bak");
    p.into()
}

pub fn atomic_write(path: &Path, bytes: &[u8], backup: bool) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("Choose a file path")?
        .to_string_lossy();
    let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut file = fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        if backup && path.exists() {
            fs::copy(path, backup_path(path))?;
        }
        fs::rename(&temp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(|e| format!("Could not save {}: {e}", path.display()))
}

pub fn save(path: &Path, scene: &Scene, camera: &OrbitCamera) -> Result<(), String> {
    let project = Project {
        format_version: 1,
        scene: scene.clone(),
        camera: camera.clone(),
    };
    let data = serde_json::to_vec(&project).map_err(|e| e.to_string())?;
    atomic_write(path, &data, true)
}

pub fn load(path: &Path) -> Result<Project, String> {
    let data = fs::read(path).map_err(|e| format!("Could not open {}: {e}", path.display()))?;
    if let Ok(mut project) = serde_json::from_slice::<Project>(&data) {
        if project.format_version != 1 {
            return Err("Unsupported project version".into());
        }
        project.scene.validate()?;
        if !project.camera.target.is_finite()
            || !project.camera.distance.is_finite()
            || project.camera.distance <= 0.0
            || !project.camera.yaw.is_finite()
            || !project.camera.pitch.is_finite()
            || !(0.01..3.13).contains(&project.camera.fov_y_radians)
            || project.camera.z_near <= 0.0
            || project.camera.z_far <= project.camera.z_near
        {
            return Err("Invalid project camera".into());
        }
        project.scene.preview_revision = None;
        // Shared project files may reference images relative to the project directory.
        let parent = path.parent().unwrap_or(Path::new("."));
        for revision in &mut project.scene.revisions {
            for screenshot in &mut revision.screenshots {
                if Path::new(screenshot).is_relative() {
                    *screenshot = parent.join(&*screenshot).to_string_lossy().into_owned();
                }
            }
        }
        return Ok(project);
    }
    let mut scene: Scene =
        serde_json::from_slice(&data).map_err(|e| format!("Invalid project: {e}"))?;
    scene.validate()?;
    let mut camera = scene.last_camera.clone().unwrap_or_default();
    if scene.last_camera.is_none() {
        crate::ui::frame_all(&scene, &mut camera);
    }
    Ok(Project {
        format_version: 1,
        scene,
        camera,
    })
}

pub fn autosave_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_os_string();
    p.push(".autosave");
    p.into()
}

// ---------------------------------------------------------------------------------------
// The projects folder: where projects made in the app live, so nobody has to type a path.

/// `$XDG_DATA_HOME/rust_modeler`, else `~/.local/share/rust_modeler`.
pub fn data_dir() -> PathBuf {
    let env = |key: &str| std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    env("XDG_DATA_HOME")
        .or_else(|| env("HOME").map(|home| home.join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rust_modeler")
}

/// Projects folder: `$RUST_MODELER_PROJECTS`, else `projects` in the app's data folder.
pub fn projects_dir() -> PathBuf {
    std::env::var_os("RUST_MODELER_PROJECTS")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("projects"))
}

const PROJECT_EXT: &str = ".spatial.json";
const RECENT_FILE: &str = "recent.json";

#[derive(Clone, Debug, Serialize)]
pub struct ProjectInfo {
    /// File name without `.spatial.json`.
    pub name: String,
    pub path: String,
    /// Seconds since the Unix epoch.
    pub modified: u64,
    pub size: u64,
    pub thumbnail: Option<String>,
    /// Lives in the projects folder (deleting it moves the file to the folder's trash);
    /// otherwise it is a file elsewhere that was opened or saved from the app.
    pub managed: bool,
}

fn project_info(path: &Path, managed: bool) -> Option<ProjectInfo> {
    let meta = fs::metadata(path).ok().filter(|m| m.is_file())?;
    let file = path.file_name()?.to_string_lossy();
    let name = file.strip_suffix(PROJECT_EXT).or(file.strip_suffix(".json")).unwrap_or(&file);
    let thumbnail = thumbnail_path(path);
    Some(ProjectInfo {
        name: name.to_string(),
        path: path.to_string_lossy().into_owned(),
        modified: meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs()),
        size: meta.len(),
        thumbnail: thumbnail.exists().then(|| thumbnail.to_string_lossy().into_owned()),
        managed,
    })
}

fn recent_paths(dir: &Path) -> Vec<String> {
    fs::read(dir.join(RECENT_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// Every project in `dir`, plus projects elsewhere that were opened or saved from the app,
/// most recently changed first.
pub fn list_projects(dir: &Path) -> Vec<ProjectInfo> {
    let mut projects: Vec<ProjectInfo> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(PROJECT_EXT) && !n.starts_with('.'))
        })
        .filter_map(|path| project_info(&path, true))
        .collect();
    for path in recent_paths(dir) {
        if !projects.iter().any(|p| p.path == path) {
            projects.extend(project_info(Path::new(&path), false));
        }
    }
    projects.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.name.cmp(&b.name)));
    projects
}

/// Keeps a project saved or opened outside the projects folder on the Projects screen.
pub fn remember_recent(dir: &Path, path: &str) {
    if Path::new(path).parent() == Some(dir) {
        return;
    }
    let mut recent = recent_paths(dir);
    recent.retain(|p| p != path);
    recent.insert(0, path.to_string());
    recent.truncate(30);
    if fs::create_dir_all(dir).is_ok() {
        if let Ok(bytes) = serde_json::to_vec_pretty(&recent) {
            let _ = fs::write(dir.join(RECENT_FILE), bytes);
        }
    }
}

/// A free `<name>.spatial.json` in `dir`: `Castle`, then `Castle 2`, `Castle 3`…
pub fn new_project_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let name = name.trim();
    let name = if name.is_empty() { "Untitled project" } else { name };
    let name = crate::library::clean_name(name, "project")?;
    Ok((1..)
        .map(|n| if n == 1 { name.clone() } else { format!("{name} {n}") })
        .map(|candidate| dir.join(format!("{candidate}{PROJECT_EXT}")))
        .find(|path| !path.exists())
        .unwrap())
}

/// Where a project's preview picture is kept (one folder for all projects, so files saved
/// elsewhere do not get pictures dropped next to them).
pub fn thumbnail_path(project: &Path) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    project.hash(&mut hasher);
    projects_dir().join(".thumbnails").join(format!("{:016x}.png", hasher.finish()))
}

/// Renders the project's preview from the current camera. Needs a GPU adapter; callers
/// treat failure as "no picture".
pub fn render_thumbnail(scene: &Scene, camera: &OrbitCamera, project: &Path) -> Result<(), String> {
    if scene.objects.iter().all(|o| !o.visible || o.mesh.faces.is_empty()) {
        return Err("Nothing to draw".into());
    }
    let path = thumbnail_path(project);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let (width, height) = (480, 360);
    let mut camera = camera.clone();
    camera.aspect = width as f32 / height as f32;
    crate::renderer::render_scene_headless_to_png(
        &camera,
        scene,
        crate::renderer::ViewportShading::StudioLit,
        false,
        false,
        width,
        height,
        &path,
    )
}

/// Removes a project from the Projects screen. A project in the projects folder moves to
/// its `.trash` (with its backup and autosave); a file elsewhere is only forgotten.
pub fn delete_project(dir: &Path, info: &ProjectInfo) -> Result<(), String> {
    let path = Path::new(&info.path);
    if !info.managed {
        let mut recent = recent_paths(dir);
        recent.retain(|p| *p != info.path);
        let bytes = serde_json::to_vec_pretty(&recent).map_err(|e| e.to_string())?;
        return fs::write(dir.join(RECENT_FILE), bytes).map_err(|e| e.to_string());
    }
    let trash = dir.join(".trash");
    fs::create_dir_all(&trash).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let file = path.file_name().ok_or("Not a project file")?.to_string_lossy().into_owned();
    // The project file goes last so a failure part-way never hides a project that still exists.
    for source in [autosave_path(path), backup_path(path), path.to_path_buf()] {
        if source.exists() {
            let suffix = source.to_string_lossy()[info.path.len()..].to_string();
            fs::rename(&source, trash.join(format!("{stamp} {file}{suffix}")))
                .map_err(|e| format!("Could not move {}: {e}", source.display()))?;
        }
    }
    let _ = fs::remove_file(thumbnail_path(path));
    Ok(())
}

/// Saves the project being left so switching never loses work: to its own file, or, when it
/// has content but was never saved, to a new file in the projects folder. Returns that file.
pub fn save_before_switching(scene: &mut Scene, camera: &OrbitCamera, thumbnail: bool) -> Result<Option<String>, String> {
    let path = match &scene.project_path {
        Some(path) => PathBuf::from(path),
        None if scene.objects.is_empty() && scene.comments.is_empty() => return Ok(None),
        None => {
            let dir = projects_dir();
            fs::create_dir_all(&dir).map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
            new_project_path(&dir, &scene.project_name)?
        }
    };
    let text = path.to_string_lossy().into_owned();
    let previous = scene.project_path.replace(text.clone());
    if let Err(e) = save(&path, scene, camera) {
        scene.project_path = previous;
        return Err(e);
    }
    remember_recent(&projects_dir(), &text);
    if thumbnail {
        let _ = render_thumbnail(scene, camera, &path);
    }
    Ok(Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    /// The GUI's new / switch flow: every project gets a file, and the one being left is saved.
    #[test]
    fn switching_projects_saves_the_one_being_left() {
        use crate::ipc::{IpcRequest, execute_ipc_command};
        use serde_json::json;
        let _guard = crate::library::TEST_ROOT.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("rust_modeler_switch_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        unsafe { std::env::set_var("RUST_MODELER_PROJECTS", &dir) };

        let mut scene = Scene::empty();
        scene.project_name = "Castle".into();
        scene.add_mesh_object("Tower/Wall", Mesh::cube(1.0), [0.5; 3]);
        scene.add_mesh_object("Tower/Roof", Mesh::cube(1.0), [0.5; 3]);
        scene.add_mesh_object("Gate", Mesh::cube(1.0), [0.5; 3]);
        let mut camera = OrbitCamera::default();
        let mut history = crate::history::UndoHistory::default();
        let mut ui = crate::ui::UiState::default();
        let mut run = |scene: &mut Scene, tool: &str, args: serde_json::Value| {
            execute_ipc_command(
                &IpcRequest { tool: tool.into(), args },
                scene,
                &mut camera,
                &mut history,
                &mut ui,
                &mut crate::renderer::ViewportShading::StudioLit,
                &mut false,
                &mut true,
            )
        };

        // Deleting a group is one undo step.
        let deleted = run(&mut scene, "delete_object", json!({ "group": "Tower" }));
        assert!(deleted.ok && deleted.message.contains("'Tower' (2 parts)"), "{}", deleted.message);
        assert_eq!(scene.objects.len(), 1);
        assert!(run(&mut scene, "undo", json!({})).ok);
        assert_eq!(scene.objects.len(), 3);
        let ids = json!([scene.objects[0].id, scene.objects[2].id]);
        assert!(run(&mut scene, "delete_object", json!({ "object_ids": ids })).ok);
        assert_eq!(scene.objects[0].name, "Tower/Roof");

        // The never-saved Castle is kept in the projects folder when a new project starts.
        let started = run(&mut scene, "new_project", json!({ "name": "Bookshelf", "save": true, "save_current": true }));
        assert!(started.ok, "{}", started.message);
        let castle = dir.join("Castle.spatial.json");
        let shelf = dir.join("Bookshelf.spatial.json");
        assert!(castle.is_file() && shelf.is_file());
        assert_eq!(scene.project_path.as_deref(), shelf.to_str());
        assert!(scene.objects.is_empty());

        // Work in the new project is saved when switching back; the same name gets a new file.
        scene.add_mesh_object("Shelf", Mesh::cube(1.0), [0.5; 3]);
        let opened = run(&mut scene, "open_project", json!({ "path": castle, "save_current": true }));
        assert!(opened.ok, "{}", opened.message);
        assert_eq!((scene.project_name.as_str(), scene.objects.len()), ("Castle", 1));
        assert_eq!(load(&shelf).unwrap().scene.objects.len(), 1);
        let again = run(&mut scene, "new_project", json!({ "name": "Bookshelf", "save": true, "save_current": true }));
        assert!(again.data["project_path"].as_str().unwrap().ends_with("Bookshelf 2.spatial.json"));

        let listed = run(&mut scene, "list_projects", json!({}));
        assert_eq!(listed.data["projects"].as_array().unwrap().len(), 3);
        assert_eq!(listed.data["current_project_path"], again.data["project_path"]);
        // Without the GUI's flags the tools behave as before: unsaved, nothing written.
        let plain = run(&mut scene, "new_project", json!({ "name": "Scratch" }));
        assert!(plain.ok && scene.project_path.is_none());
        assert_eq!(list_projects(&dir).len(), 3);

        unsafe { std::env::remove_var("RUST_MODELER_PROJECTS") };
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn projects_folder_lists_names_and_trashes() {
        let dir = std::env::temp_dir().join(format!("rust_modeler_projects_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(list_projects(&dir).is_empty());

        let mut scene = Scene::empty();
        scene.add_mesh_object("Tower", Mesh::cube(1.0), [0.5; 3]);
        let camera = OrbitCamera::default();
        let first = new_project_path(&dir, "Castle").unwrap();
        assert!(first.ends_with("Castle.spatial.json"));
        save(&first, &scene, &camera).unwrap();
        save(&first, &scene, &camera).unwrap(); // writes the .bak
        let second = new_project_path(&dir, "Castle").unwrap();
        assert!(second.ends_with("Castle 2.spatial.json"));
        assert!(new_project_path(&dir, "../escape").is_err());
        assert!(new_project_path(&dir, "  ").unwrap().ends_with("Untitled project.spatial.json"));

        // A project saved elsewhere shows up once it is remembered, and is only forgotten.
        let outside = std::env::temp_dir().join(format!("rust_modeler_outside_{}.spatial.json", std::process::id()));
        save(&outside, &scene, &camera).unwrap();
        remember_recent(&dir, &outside.to_string_lossy());
        remember_recent(&dir, &first.to_string_lossy());
        let listed = list_projects(&dir);
        assert_eq!(listed.len(), 2);
        let castle = listed.iter().find(|p| p.name == "Castle").unwrap();
        assert!(castle.managed);
        let other = listed.iter().find(|p| !p.managed).unwrap();
        delete_project(&dir, other).unwrap();
        assert!(outside.exists());
        delete_project(&dir, castle).unwrap();
        assert!(!first.exists() && list_projects(&dir).is_empty());
        assert_eq!(fs::read_dir(dir.join(".trash")).unwrap().count(), 2);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_file(&outside);
        let _ = fs::remove_file(backup_path(&outside));
    }
}
