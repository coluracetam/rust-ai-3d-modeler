use crate::{
    camera::{OrbitCamera, Ray},
    history::UndoHistory,
    ipc::{IpcRequest, execute_ipc_command},
    mesh::Mesh,
    project,
    renderer::ViewportShading,
    scene::{CommentStatus, Scene},
    ui::{UiState, measurement_text},
};
use glam::Vec3;
use serde_json::{Value, json};

fn scene() -> Scene {
    let mut s = Scene::empty();
    s.add_mesh_object("Review part", Mesh::cube(2.0), [0.6, 0.7, 0.8]);
    s.objects[0].mesh.selected_faces.insert(5);
    s
}

struct Editor {
    scene: Scene,
    camera: OrbitCamera,
    history: UndoHistory,
    ui: UiState,
}
impl Editor {
    fn new() -> Self {
        Self {
            scene: scene(),
            camera: OrbitCamera::default(),
            history: UndoHistory::default(),
            ui: UiState::default(),
        }
    }
    fn call(&mut self, tool: &str, args: Value) -> crate::ipc::IpcResponse {
        execute_ipc_command(
            &IpcRequest {
                tool: tool.into(),
                args,
            },
            &mut self.scene,
            &mut self.camera,
            &mut self.history,
            &mut self.ui,
            &mut ViewportShading::StudioLit,
            &mut false,
            &mut true,
        )
    }
}

#[test]
fn comment_remaps_reordered_faces_and_rejects_remeshing() {
    let mut scene = scene();
    let c = scene
        .add_comment_for_selection("Change this face", 0.3)
        .unwrap();
    scene.objects[0].mesh.faces.remove(0);
    scene.objects[0].mesh.recalculate_normals();
    assert!(!scene.comment_needs_reattachment(&c));
    scene.focus_comment(c.id).unwrap();
    assert_eq!(
        scene.objects[0]
            .mesh
            .selected_faces
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![4]
    );
    scene.objects[0].mesh.subdivide_or_loop_cut();
    assert!(scene.comment_needs_reattachment(&c));
    assert!(scene.focus_comment(c.id).is_none());
    scene.objects[0].mesh.selected_faces.clear();
    scene.objects[0].mesh.selected_faces.insert(0);
    scene.reattach_comment(c.id, 0.3).unwrap();
    assert!(scene.focus_comment(c.id).is_some());
}

#[test]
fn comment_anchor_follows_transform_and_deformation() {
    let mut scene = scene();
    let comment = scene.add_comment_for_selection("Track me", 0.3).unwrap();
    let old = scene.current_comment_anchor(&comment);
    scene.objects[0].transform.translation = Vec3::new(4.0, 2.0, -1.0);
    assert!(
        (scene.current_comment_anchor(&comment) - old - Vec3::new(4.0, 2.0, -1.0)).length() < 1e-6
    );
    scene.objects[0]
        .mesh
        .translate_selected(Vec3::X, crate::mesh::EditElementMode::Face);
    assert!(
        (scene.current_comment_anchor(&comment) - old - Vec3::new(5.0, 2.0, -1.0)).length() < 1e-6
    );
}

#[test]
fn disconnected_regions_in_one_object_remain_independent() {
    let mut s = scene();
    let cube = Mesh::cube(2.0);
    let offset = s.objects[0].mesh.positions.len();
    s.objects[0]
        .mesh
        .positions
        .extend(cube.positions.iter().map(|p| *p + Vec3::X * 5.0));
    s.objects[0].mesh.faces.extend(
        cube.faces
            .iter()
            .map(|f| f.iter().map(|v| v + offset).collect::<Vec<_>>()),
    );
    s.objects[0].mesh.selected_faces = [0, 6].into_iter().collect();
    let sections = s.collect_current_target_sections(0.3);
    assert_eq!(sections.len(), 2);
    assert_eq!(sections[0].object_id, sections[1].object_id);
    assert_eq!(sections[0].selected_faces, vec![0]);
    assert_eq!(sections[1].selected_faces, vec![6]);
}

#[test]
fn revision_preview_is_read_only_and_revert_preserves_feedback() {
    let mut e = Editor::new();
    let c = e
        .scene
        .add_comment_for_selection("Make taller", 0.3)
        .unwrap();
    let before = e.scene.objects[0].mesh.positions.clone();
    let r = e.call(
        "begin_revision",
        json!({"label":"Taller roof","comment_id":c.id}),
    );
    assert!(r.ok);
    let id = r.data["revision_id"].as_u64().unwrap();
    assert!(
        e.call(
            "edit_mesh",
            json!({"operation":"translate_selected","delta":[0,2,0]})
        )
        .ok
    );
    assert!(
        e.call(
            "edit_mesh",
            json!({"operation":"translate_selected","delta":[0,1,0]})
        )
        .ok
    );
    assert_eq!(e.history.undo_stack.len(), 1, "all operations are grouped");
    assert!(
        e.call(
            "finish_revision",
            json!({"revision_id":id,"summary":"Raised the selected roof"})
        )
        .ok
    );
    assert_eq!(e.scene.comments[0].status, CommentStatus::ReadyForReview);
    let after = e.scene.objects[0].mesh.positions.clone();
    assert_ne!(before, after);
    assert!(
        e.call("preview_revision", json!({"revision_id":id,"before":true}))
            .ok
    );
    assert_eq!(e.scene.display_scene().objects[0].mesh.positions, before);
    assert_eq!(e.scene.objects[0].mesh.positions, after);
    assert!(!e.call("add_primitive", json!({"shape":"cube"})).ok);
    e.scene.add_general_comment("Keep the trim blue");
    assert!(e.call("revert_revision", json!({"revision_id":id})).ok);
    assert_eq!(e.scene.objects[0].mesh.positions, before);
    assert_eq!(e.scene.comments.len(), 2);
    assert_eq!(e.scene.comments[0].status, CommentStatus::Open);
}

#[test]
fn revision_conflicts_and_cancel_are_safe() {
    let mut e = Editor::new();
    let r = e.call("begin_revision", json!({"label":"Change"}));
    let id = r.data["revision_id"].as_u64().unwrap();
    assert!(!e.call("begin_revision", json!({"label":"Overlap"})).ok);
    e.scene.objects[0].transform.translation.x = 2.0;
    assert!(
        e.call(
            "finish_revision",
            json!({"revision_id":id,"summary":"Moved"})
        )
        .ok
    );
    e.scene.objects[0].transform.translation.x = 3.0;
    assert!(!e.call("revert_revision", json!({"revision_id":id})).ok);
    assert_eq!(e.scene.objects[0].transform.translation.x, 3.0);
    e.scene.objects[0].transform.translation.x = 2.0;
    assert!(e.call("accept_revision", json!({"revision_id":id})).ok);
    let r = e.call("begin_revision", json!({"label":"Cancel me"}));
    let id = r.data["revision_id"].as_u64().unwrap();
    e.scene.objects[0].transform.translation.x = 9.0;
    e.scene.objects[0].visible = false;
    assert!(e.call("cancel_revision", json!({"revision_id":id})).ok);
    assert_eq!(e.scene.objects[0].transform.translation.x, 2.0);
    assert!(!e.scene.objects[0].visible);
}

#[test]
fn undo_geometry_keeps_new_comments_and_unique_ids() {
    let mut e = Editor::new();
    assert!(e.call("add_primitive", json!({"shape":"cube"})).ok);
    let id = e.scene.objects.last().unwrap().id;
    e.scene.add_general_comment("Do not lose this");
    assert!(e.call("undo", json!({})).ok);
    assert_eq!(e.scene.comments.len(), 1);
    assert!(e.call("add_primitive", json!({"shape":"cube"})).ok);
    assert!(e.scene.objects.last().unwrap().id > id);
}

#[test]
fn projects_roundtrip_with_backup_and_autosave_recovery() {
    let dir = std::env::temp_dir().join(format!("spatial-project-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("project.spatial.json");
    let mut e = Editor::new();
    e.scene.add_general_comment("Persist me");
    e.camera.distance = 14.0;
    e.scene
        .begin_revision("Unfinished work".into(), None)
        .unwrap();
    project::save(&path, &e.scene, &e.camera).unwrap();
    e.scene.project_name = "Changed title".into();
    project::save(&path, &e.scene, &e.camera).unwrap();
    assert_eq!(
        project::load(&project::backup_path(&path))
            .unwrap()
            .scene
            .project_name,
        "Untitled project"
    );
    let restored = project::load(&path).unwrap();
    assert_eq!(restored.scene.comments.len(), 1);
    assert_eq!(restored.scene.revisions.len(), 1);
    assert_eq!(restored.camera.distance, 14.0);
    project::save(&project::autosave_path(&path), &e.scene, &e.camera).unwrap();
    assert!(
        e.call("open_project", json!({"path":path,"recover":true}))
            .ok
            == false,
        "active revision cannot be overwritten"
    );
    e.scene.review_revision(1, false).unwrap();
    assert!(
        e.call("open_project", json!({"path":path,"recover":true}))
            .ok
    );
    let previous = std::fs::read(&path).unwrap();
    let bad_destination = dir.join("missing/scene.json");
    assert!(project::save(&bad_destination, &e.scene, &e.camera).is_err());
    assert_eq!(previous, std::fs::read(&path).unwrap());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn clipping_picks_visible_back_surface_and_units_are_correct() {
    let mut s = scene();
    s.clip_plane = Some([0.0, 0.0, -1.0, 0.0]);
    let hit = s
        .raycast(&Ray::new(Vec3::new(0.0, 0.0, 5.0), -Vec3::Z))
        .unwrap();
    assert!(hit.world_point.z < 0.0);
    let text = measurement_text(&[Vec3::ZERO, Vec3::X, Vec3::X + Vec3::Y], 1000.0, "mm").unwrap();
    assert!(text.contains("1000.000 mm"));
    assert!(text.contains("90.0°"));
}

#[test]
fn legacy_scene_deserialization_and_invalid_project_validation() {
    let source =
        std::fs::read_to_string("assets/scenes/clockwork_astral_cathedral_scene.json").unwrap();
    let mut s: Scene = serde_json::from_str(&source).unwrap();
    s.validate().unwrap();
    assert_eq!(s.project_name, "Untitled project");
    s.objects[0].mesh.faces[0][0] = usize::MAX;
    assert!(s.validate().is_err());
}

#[test]
fn accepting_and_undoing_revision_updates_lifecycle_without_losing_replies() {
    let mut e = Editor::new();
    let c = e
        .scene
        .add_comment_for_selection("Raise roof", 0.3)
        .unwrap();
    let r = e.call(
        "begin_revision",
        json!({"label":"Raise roof","comment_id":c.id}),
    );
    let id = r.data["revision_id"].as_u64().unwrap();
    assert!(
        e.call(
            "edit_mesh",
            json!({"operation":"translate_selected","delta":[0,1,0]})
        )
        .ok
    );
    assert!(
        e.call(
            "finish_revision",
            json!({"revision_id":id,"summary":"Raised"})
        )
        .ok
    );
    assert!(!e.call("resolve_comment", json!({"comment_id":c.id})).ok);
    assert!(
        !e.call(
            "edit_mesh",
            json!({"operation":"translate_selected","delta":[0,1,0]})
        )
        .ok
    );
    assert!(
        e.call(
            "update_comment",
            json!({"comment_id":c.id,"reply":"Keep the color","author":"You"})
        )
        .ok
    );
    assert!(e.call("accept_revision", json!({"revision_id":id})).ok);
    assert!(e.call("undo", json!({})).ok);
    assert_eq!(e.scene.comments[0].status, CommentStatus::Open);
    assert_eq!(e.scene.comments[0].replies.len(), 1);
    assert!(e.call("redo", json!({})).ok);
    assert_eq!(e.scene.comments[0].status, CommentStatus::Accepted);
    assert_eq!(e.scene.comments[0].replies.len(), 1);
}

#[test]
fn cancelled_background_result_cannot_change_live_geometry() {
    let mut app = crate::ModelerApp::with_scene(scene());
    let before = app.scene.objects[0].mesh.positions.clone();
    let (tx, receiver) = std::sync::mpsc::channel();
    let (reply, reply_rx) = std::sync::mpsc::channel();
    app.background = Some(crate::BackgroundOperation {
        receiver,
        reply,
        tool: "voxel_remesh".into(),
        started: std::time::Instant::now(),
        selected_id: Some(app.scene.objects[0].id),
    });
    assert!(app.cancel_operation().ok);
    assert!(!reply_rx.recv().unwrap().ok);
    let mut changed = app.scene.clone();
    changed.objects.clear();
    assert!(
        tx.send(crate::BackgroundResult {
            response: crate::ipc::IpcResponse::success("Finished late", json!({})),
            scene: changed,
            history: UndoHistory::default()
        })
        .is_err()
    );
    app.poll_operation();
    assert_eq!(app.scene.objects[0].mesh.positions, before);
}

#[test]
fn shared_project_resolves_screenshots_relative_to_its_directory() {
    let dir = std::env::temp_dir().join(format!("spatial-relative-project-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("images")).unwrap();
    let path = dir.join("shared.spatial.json");
    let screenshot = dir.join("images/after.png");
    std::fs::write(&screenshot, b"image-placeholder").unwrap();
    let mut scene = scene();
    let id = scene.begin_revision("Shared review".into(), None).unwrap();
    scene
        .finish_revision(id, "Updated".into(), vec!["images/after.png".into()])
        .unwrap();
    project::save(&path, &scene, &OrbitCamera::default()).unwrap();
    let loaded = project::load(&path).unwrap();
    assert_eq!(
        std::path::Path::new(&loaded.scene.revisions[0].screenshots[0]),
        screenshot
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn new_project_starts_blank_and_undo_restores_unsaved_work() {
    let mut e = Editor::new();
    e.scene.project_name = "Chair study".into();
    e.scene.add_general_comment("Keep this feedback");
    e.scene.saved_views.push(crate::scene::SavedView {
        name: "Hero".into(),
        camera: OrbitCamera::default(),
    });
    e.camera.distance = 42.0;
    e.camera.aspect = 1.75;
    e.ui.clip_enabled = true;
    e.ui.measure_points.push(Vec3::ONE);

    let r = e.call("new_project", json!({"name":"  Fresh start  "}));
    assert!(r.ok, "{}", r.message);
    assert_eq!(e.scene.project_name, "Fresh start");
    assert!(e.scene.project_path.is_none());
    assert!(e.scene.objects.is_empty());
    assert!(e.scene.comments.is_empty());
    assert!(e.scene.revisions.is_empty());
    assert!(e.scene.saved_views.is_empty());
    assert_eq!(e.camera.distance, OrbitCamera::default().distance);
    assert_eq!(e.camera.aspect, 1.75, "viewport aspect survives the reset");
    assert!(!e.ui.clip_enabled && e.ui.measure_points.is_empty());
    assert!(e.ui.scene_dirty_for_disk, "the blank project is persisted");

    // Neither project has a file path, but undo must still bring back the whole previous project.
    assert!(e.call("undo", json!({})).ok);
    assert_eq!(e.scene.project_name, "Chair study");
    assert_eq!(e.scene.objects.len(), 1);
    assert_eq!(e.scene.comments.len(), 1);
    assert_eq!(e.scene.saved_views.len(), 1);

    assert!(e.call("redo", json!({})).ok);
    assert_eq!(e.scene.project_name, "Fresh start");
    assert!(e.scene.objects.is_empty() && e.scene.comments.is_empty());

    // Unnamed new projects get the default title.
    assert!(e.call("new_project", json!({})).ok);
    assert_eq!(e.scene.project_name, "Untitled project");
}

#[test]
fn new_project_keeps_saved_file_and_undo_reopens_it() {
    let dir = std::env::temp_dir().join(format!("spatial-new-project-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kept.spatial.json");
    let mut e = Editor::new();
    assert!(e.call("save_project", json!({"path":path})).ok);
    let saved = std::fs::read(&path).unwrap();

    assert!(e.call("new_project", json!({})).ok);
    assert!(e.scene.project_path.is_none());
    assert_eq!(
        saved,
        std::fs::read(&path).unwrap(),
        "the saved file is untouched"
    );

    assert!(e.call("undo", json!({})).ok);
    assert_eq!(
        e.scene.project_path.as_deref(),
        Some(path.to_string_lossy().as_ref())
    );
    assert_eq!(e.scene.objects.len(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn new_project_waits_for_pending_revisions() {
    let mut e = Editor::new();
    let id = e.scene.begin_revision("Agent work".into(), None).unwrap();
    assert!(!e.call("new_project", json!({})).ok, "in-progress revision");
    e.scene
        .finish_revision(id, "Done".into(), Vec::new())
        .unwrap();
    assert!(
        !e.call("new_project", json!({})).ok,
        "revision awaiting review"
    );
    assert_eq!(e.scene.objects.len(), 1);
    e.scene.review_revision(id, true).unwrap();
    assert!(e.call("new_project", json!({})).ok);
    assert!(e.scene.revisions.is_empty());
}
