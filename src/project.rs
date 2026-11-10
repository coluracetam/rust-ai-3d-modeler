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
