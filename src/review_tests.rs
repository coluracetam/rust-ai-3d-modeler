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
fn cached_topology_detects_a_face_reorder_without_a_version_bump() {
    let mut scene = scene();
    let comment = scene
        .add_comment_for_selection("Track this face", 0.3)
        .unwrap();
    let anchor = scene.current_comment_anchor(&comment);

    scene.objects[0].mesh.faces.swap(0, 5);

    assert_eq!(scene.target_faces(&comment.targets[0]), Some(vec![0]));
    assert_eq!(scene.current_comment_anchor(&comment), anchor);

    let mesh = &mut scene.objects[0].mesh;
    let duplicate = mesh.faces[0].clone();
    mesh.faces[2] = duplicate;
    assert!(scene.comment_needs_reattachment(&comment));
}

#[test]
fn comment_geometry_cache_tracks_target_edits_and_ignores_selection() {
    let mut scene = scene();
    let mut comment = scene
        .add_comment_for_selection("Track this face", 0.3)
        .unwrap();
    let mesh = &mut scene.objects[0].mesh;
    mesh.faces.swap(0, 5);
    let expected_anchor = mesh.face_center(1);
    let new_signature = mesh.face_signature(1);
    let original = scene.comment_geometry(&comment);

    scene.objects[0].mesh.clear_selection();
    assert!(std::sync::Arc::ptr_eq(
        &original,
        &scene.comment_geometry(&comment)
    ));

    comment.targets[0].face_signatures[0] = new_signature;
    let updated = scene.comment_geometry(&comment);
    assert_eq!(updated.anchor, expected_anchor);
    assert!(!updated.needs_reattachment);
    assert_eq!(
        scene.current_comment_anchor(&scene.comments[0]),
        original.anchor
    );
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
fn background_export_keeps_inspection_changes() {
    let mut app = crate::ModelerApp::with_scene(scene());
    let snapshot = app.scene.clone();
    let (tx, receiver) = std::sync::mpsc::channel();
    let (reply, reply_rx) = std::sync::mpsc::channel();
    app.background = Some(crate::BackgroundOperation {
        receiver,
        reply,
        tool: "export_model".into(),
        started: std::time::Instant::now(),
        selected_id: app.scene.active_object().map(|object| object.id),
    });
    app.scene.objects[0].mesh.selected_faces.insert(0);
    assert!(!app.cancel_operation().ok);
    assert!(!crate::ipc::allowed_during_operation("export_model"));
    assert!(!crate::ipc::allowed_during_operation("library_export"));
    tx.send(crate::BackgroundResult {
        response: crate::ipc::IpcResponse::success("Exported model", json!({})),
        scene: snapshot,
        history: UndoHistory::default(),
    }).unwrap();
    app.poll_operation();
    assert!(reply_rx.recv().unwrap().ok);
    assert!(app.scene.objects[0].mesh.selected_faces.contains(&0));
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

#[test]
fn finished_revisions_keep_only_changed_objects_and_revert_in_place() {
    let mut e = Editor::new();
    e.scene
        .add_mesh_object("Untouched", Mesh::cube(1.0), [0.2, 0.2, 0.2]);
    e.scene
        .add_mesh_object("Deleted", Mesh::cube(1.0), [0.3, 0.3, 0.3]);
    let order: Vec<u64> = e.scene.objects.iter().map(|o| o.id).collect();
    let r = e.call("begin_revision", json!({"label":"Edit"}));
    let id = r.data["revision_id"].as_u64().unwrap();
    e.scene.objects[0].transform.translation.y = 4.0;
    e.scene.objects.remove(2);
    assert!(e.call("add_primitive", json!({"shape":"sphere"})).ok);
    assert!(
        e.call("finish_revision", json!({"revision_id":id,"summary":"Done"}))
            .ok
    );
    let rev = e.scene.revisions.iter().find(|r| r.id == id).unwrap();
    assert!(rev.partial);
    assert_eq!(rev.before.len(), 2, "moved + deleted");
    assert_eq!(rev.after.as_ref().unwrap().len(), 2, "moved + added");
    assert_eq!(e.scene.changed_objects(id).len(), 3);
    // Comparison views rebuild the full scene from live parts.
    e.scene.preview_revision = Some((id, true));
    let before: Vec<u64> = e.scene.display_scene().objects.iter().map(|o| o.id).collect();
    assert_eq!(before, order);
    e.scene.preview_revision = None;
    assert!(e.call("revert_revision", json!({"revision_id":id})).ok);
    let ids: Vec<u64> = e.scene.objects.iter().map(|o| o.id).collect();
    assert_eq!(ids, order);
    assert_eq!(e.scene.objects[0].transform.translation.y, 0.0);
}

#[test]
fn old_reviewed_revisions_drop_geometry() {
    let mut e = Editor::new();
    for i in 0..4 {
        let r = e.call("begin_revision", json!({"label": format!("R{i}")}));
        let id = r.data["revision_id"].as_u64().unwrap();
        e.scene.objects[0].transform.translation.x = i as f32 + 1.0;
        assert!(e.call("finish_revision", json!({"revision_id":id,"summary":"s"})).ok);
        assert!(e.call("accept_revision", json!({"revision_id":id})).ok);
    }
    let pruned: Vec<bool> = e.scene.revisions.iter().map(|r| r.geometry_pruned).collect();
    assert_eq!(pruned, vec![true, true, false, false]);
    assert!(e.call("get_revisions", json!({})).ok);
}

#[test]
fn drafts_are_hidden_until_sent_and_change_requests_reopen_comments() {
    let mut e = Editor::new();
    let c = e.scene.add_comment_for_selection("Taller", 0.3).unwrap();
    e.scene.comments.iter_mut().for_each(|c| c.draft = true);
    let r = e.call("get_comments", json!({}));
    assert_eq!(r.data["count"], 0);
    assert_eq!(r.data["drafts_pending"], 1);
    let state = e.call("get_feedback_state", json!({}));
    assert_eq!(state.data["open_comments"], 0);
    assert_eq!(e.call("send_comments", json!({})).data["sent"], 1);
    assert_eq!(e.scene.feedback_batch, 1);
    assert_eq!(e.call("get_comments", json!({})).data["count"], 1);

    // wait_for_feedback returns at once for waiting comments, and times out otherwise.
    let resp = crate::mcp::wait_for_feedback(&json!({}), |req| e.call(&req.tool, req.args.clone()));
    assert_eq!(resp.data["woke"], true);
    let resp = crate::mcp::wait_for_feedback(
        &json!({"since_batch": 1, "timeout_seconds": 1}),
        |req| e.call(&req.tool, req.args.clone()),
    );
    assert_eq!(resp.data["woke"], false);

    let r = e.call("begin_revision", json!({"label":"Taller","comment_id":c.id}));
    let id = r.data["revision_id"].as_u64().unwrap();
    e.scene.objects[0].transform.scale.y = 2.0;
    assert!(e.call("finish_revision", json!({"revision_id":id,"summary":"Scaled"})).ok);
    // Notes must go through while the user is looking at Before / After, not only Current.
    assert!(e.call("preview_revision", json!({"revision_id":id,"before":true})).ok);
    e.ui.request_changes_for = Some(id);
    e.ui.request_notes = "Only the top half".into();
    e.scene.objects[0].transform.scale.y = 3.0;
    assert!(
        !e.call("request_changes", json!({"revision_id":id,"notes":"Only the top half"}))
            .ok
    );
    assert_eq!(e.ui.request_changes_for, Some(id));
    assert_eq!(e.ui.request_notes, "Only the top half");
    e.scene.objects[0].transform.scale.y = 2.0;
    assert!(
        e.call("request_changes", json!({"revision_id":id,"notes":"Only the top half"}))
            .ok
    );
    assert!(e.scene.preview_revision.is_none(), "the comparison closes with the review");
    assert!(e.ui.request_changes_for.is_none());
    assert!(e.ui.request_notes.is_empty());
    assert_eq!(e.scene.objects[0].transform.scale.y, 1.0);
    let comment = e.scene.comments.iter().find(|x| x.id == c.id).unwrap();
    assert_eq!(comment.status, CommentStatus::Open);
    assert_eq!(comment.replies.last().unwrap().text, "Only the top half");
    assert_eq!(e.scene.feedback_batch, 2);
    let resp = crate::mcp::wait_for_feedback(
        &json!({"since_batch": 1, "timeout_seconds": 1}),
        |req| e.call(&req.tool, req.args.clone()),
    );
    assert_eq!(resp.data["woke"], true);
}

#[test]
fn modeling_tools_through_the_dispatcher() {
    let mut e = Editor::new();
    let base = e.scene.objects[0].id;
    assert!(e.call("add_primitive", json!({"shape":"cube","size":1.0,"position":[1.0,0.0,0.0]})).ok);
    let cutter = e.scene.objects.last().unwrap().id;
    let r = e.call("boolean", json!({"object_id":base,"tool_object_id":cutter,"operation":"subtract"}));
    assert!(r.ok, "{}", r.message);
    assert_eq!(e.scene.objects.len(), 1);
    let v = crate::modeling::signed_volume(&e.scene.objects[0].mesh);
    assert!((v - 7.5).abs() < 1e-2, "{v}");
    assert!(e.call("array_object", json!({"object_id":base,"count":4,"offset":[3.0,0.0,0.0]})).ok);
    assert_eq!(e.scene.objects.len(), 4);
    assert!((e.scene.objects[3].transform.translation.x - 9.0).abs() < 1e-4);
    assert!(e.call("mirror_object", json!({"object_id":base,"axis":"x","merge":false})).ok);
    assert_eq!(e.scene.objects.len(), 5);
    let lathe = e.call("add_lathe", json!({"profile":[[0.0,0.0],[0.5,0.0],[0.3,1.0],[0.0,1.2]]}));
    assert!(lathe.ok, "{}", lathe.message);
    let sweep = e.call("add_sweep", json!({"path":[[0,0,0],[0,1,0],[1,2,0]],"radius":0.05}));
    assert!(sweep.ok, "{}", sweep.message);
    let id = sweep.data["object_id"].as_u64().unwrap();
    let r = e.call("edit_mesh", json!({"object_id":id,"operation":"bevel","select_faces":[0],"width":0.01}));
    assert!(r.ok, "{}", r.message);
    assert!(e.call("undo", json!({})).ok);
}

#[test]
fn clear_selection_tool_clears_everything() {
    let mut e = Editor::new();
    e.scene.selected_object = Some(0);
    assert!(e.scene.has_any_selection());
    let r = e.call("clear_selection", json!({}));
    assert!(r.ok, "{}", r.message);
    assert_eq!(r.data["cleared_faces"], 1);
    assert!(!e.scene.has_any_selection());
    assert!(e.scene.objects[0].mesh.selected_faces.is_empty());
}

#[test]
fn volume_selection_follows_the_box_and_survives_topology_changes() {
    use crate::scene::SelectionVolume;
    let mut s = Scene::empty();
    let a = s.add_mesh_object("Left", Mesh::cube(1.0), [0.7; 3]);
    let b = s.add_mesh_object("Right", Mesh::cube(1.0), [0.7; 3]);
    s.objects[b].transform.translation = Vec3::new(3.0, 0.0, 0.0);

    // A box around the top of the left cube only.
    s.selection_volume = Some(SelectionVolume::from_min_max(
        Vec3::new(-0.6, 0.4, -0.6),
        Vec3::new(0.6, 0.6, 0.6),
    ));
    let (faces, parts) = s.apply_volume_selection();
    assert_eq!((faces, parts), (1, 1));
    assert!(s.objects[b].mesh.selected_faces.is_empty());
    let top = *s.objects[a].mesh.selected_faces.iter().next().unwrap();
    assert!(s.objects[a].mesh.face_normal(top).y > 0.9);

    let id = s.add_comment_for_selection("Raise the lid", 0.3).unwrap().id;
    assert!(s.comments[0].volume.is_some());
    // Change the topology: stored face indices are now stale, the box is not.
    s.objects[a].mesh.select_all(crate::mesh::EditElementMode::Face);
    assert!(s.objects[a].mesh.subdivide_or_loop_cut());
    s.clear_all_selections();
    let comment = s.comments[0].clone();
    assert!(!s.comment_needs_reattachment(&comment));
    let targets = s.resolved_targets(&comment);
    assert!(!targets.is_empty());
    assert!(targets.iter().all(|t| t.object_id == s.objects[a].id));
    assert!(s.focus_comment(id).is_some());
    assert!(!s.objects[a].mesh.selected_faces.is_empty());

    // A box over empty space is a valid target on its own.
    s.clear_all_selections();
    s.selection_volume = Some(SelectionVolume::new(Vec3::new(0.0, 5.0, 0.0), Vec3::splat(0.5)));
    assert_eq!(s.apply_volume_selection(), (0, 0));
    let empty = s.add_comment_for_selection("Add a balcony here", 0.3).unwrap();
    assert!(empty.targets.is_empty() && empty.volume.is_some() && !empty.is_general);
    assert!(!s.comment_needs_reattachment(&empty));
    assert_eq!(s.current_comment_anchor(&empty), Vec3::new(0.0, 5.0, 0.0));
}

#[test]
fn face_paths_inversion_similarity_and_region_names() {
    use crate::mesh::SimilarBy;
    let mut plane = Mesh::plane(4.0, 4);
    let total = plane.faces.len();
    assert_eq!(total, 16);
    // A path between opposite corners is edge-connected and includes both ends.
    let path = plane.face_path(0, total - 1).unwrap();
    assert_eq!((path[0], *path.last().unwrap()), (0, total - 1));
    assert!(path.len() >= 7);
    for pair in path.windows(2) {
        assert!(plane.face_neighbors(pair[0]).contains(&pair[1]));
    }

    plane.selected_faces.insert(0);
    plane.invert_selection();
    assert_eq!(plane.selected_faces.len(), total - 1);
    assert!(!plane.selected_faces.contains(&0));

    // Every face of a flat grid points the same way and has the same area.
    plane.selected_faces = [3].into_iter().collect();
    assert_eq!(plane.select_similar(SimilarBy::Normal, 5.0), total - 1);
    plane.selected_faces = [3].into_iter().collect();
    assert_eq!(plane.select_similar(SimilarBy::Area, 0.1), total - 1);

    // On a cube only the clicked side matches by direction.
    let mut cube = Mesh::cube(1.0);
    cube.selected_faces = [0].into_iter().collect();
    assert_eq!(cube.select_similar(SimilarBy::Normal, 5.0), 0);

    // Region names travel with the comment target.
    let mut s = Scene::empty();
    let i = s.add_mesh_object("Tower", Mesh::cube(1.0), [0.7; 3]);
    s.objects[i].mesh.selected_faces.insert(2);
    let id = s.objects[i].id;
    s.region_labels.insert((id, 2), "north wall".into());
    let c = s.add_comment_for_selection("Add a window", 0.3).unwrap();
    assert_eq!(c.targets[0].label, "north wall");
}

#[test]
fn agent_tools_build_query_select_and_verify() {
    let mut e = Editor::new();
    e.scene = Scene::empty();
    let ok = |r: crate::ipc::IpcResponse| {
        assert!(r.ok, "{}", r.message);
        r.data
    };

    // Build: an L-shaped slab, a raw-vertex pyramid and a lofted post.
    let slab = ok(e.call(
        "add_extrude",
        json!({"name": "Slab", "profile": [[0,0],[4,0],[4,2],[2,2],[2,4],[0,4]], "height": 0.5}),
    ));
    let slab_id = slab["object_id"].as_u64().unwrap();
    let pyramid = ok(e.call(
        "add_mesh",
        json!({
            "name": "Pyramid",
            "vertices": [[-0.5,0,-0.5],[0.5,0,-0.5],[0.5,0,0.5],[-0.5,0,0.5],[0,1,0]],
            // Deliberately inside-out: the tool fixes the winding.
            "faces": [[3,2,1,0],[1,4,0],[2,4,1],[3,4,2],[0,4,3]],
            "position": [8, 0, 0]
        }),
    ));
    assert_eq!(pyramid["watertight"], true);
    assert_eq!(pyramid["winding_fixed"], true);
    let pyramid_id = pyramid["object_id"].as_u64().unwrap();
    assert!(!e.call("add_mesh", json!({"vertices": [[0,0,0],[1,0,0],[0,1,0]], "faces": [[0,1,7]]})).ok);

    // Stack the pyramid on the slab and confirm with a measurement.
    ok(e.call("align_object", json!({"object_id": pyramid_id, "target_object_id": slab_id, "x": "min", "z": "min", "y": "after"})));
    let m = ok(e.call("measure", json!({"object_id": pyramid_id, "other_object_id": slab_id})));
    assert!(m["axis_gap"][1].as_f64().unwrap().abs() < 1e-4, "{m}");

    // Select the slab's top by a world-space box, then read what was selected.
    let sel = ok(e.call(
        "select_faces",
        json!({"object_id": slab_id, "mode": "box", "min": [-1, 0.45, -1], "max": [5, 0.55, 5]}),
    ));
    let top_faces = sel["selected_faces"].as_u64().unwrap();
    assert!(top_faces >= 4, "{sel}");
    let q = ok(e.call("query_geometry", json!({"object_id": slab_id})));
    assert_eq!(q["scope"], "selection");
    assert_eq!(q["face_count"].as_u64().unwrap(), top_faces);
    assert!((q["total_area"].as_f64().unwrap() - 12.0).abs() < 1e-3, "{q}");
    assert!(q["faces"][0]["normal"][1].as_f64().unwrap() > 0.99);
    assert!((q["object_size"][0].as_f64().unwrap() - 4.0).abs() < 1e-3);

    // Paint only those faces, then grow / invert through the selection tool.
    let painted = ok(e.call("paint_faces", json!({"object_id": slab_id, "color": [0.9, 0.2, 0.1]})));
    assert_eq!(painted["painted_faces"].as_u64().unwrap(), top_faces);
    let similar = ok(e.call("select_faces", json!({"object_id": slab_id, "mode": "similar", "by": "color", "op": "add"})));
    assert_eq!(similar["selected_faces"].as_u64().unwrap(), top_faces);
    let inverted = ok(e.call("select_faces", json!({"object_id": slab_id, "mode": "invert"})));
    assert!(inverted["selected_faces"].as_u64().unwrap() > 0);
    ok(e.call("select_faces", json!({"object_id": slab_id, "mode": "none"})));

    // Scatter studs over the slab's upward faces, deform a post, run extra mesh edits in a batch.
    let stud = ok(e.call("add_primitive", json!({"shape": "cube", "size": 0.1, "name": "Stud"})));
    let scattered = ok(e.call(
        "scatter_object",
        json!({"object_id": stud["object_id"], "target_object_id": slab_id, "count": 12, "seed": 3, "min_normal_y": 0.9}),
    ));
    assert_eq!(scattered["copies"], 12);
    let batch = ok(e.call(
        "batch",
        json!({"commands": [
            {"tool": "add_loft", "args": {"name": "Post", "sections": [
                [[-0.2,0,-0.2],[0.2,0,-0.2],[0.2,0,0.2],[-0.2,0,0.2]],
                [[-0.2,1,-0.2],[0.2,1,-0.2],[0.2,1,0.2],[-0.2,1,0.2]],
                [[-0.1,2,-0.1],[0.1,2,-0.1],[0.1,2,0.1],[-0.1,2,0.1]]
            ]}},
            {"tool": "edit_mesh", "args": {"name": "Post", "operation": "subdivide_smooth", "levels": 1}},
            {"tool": "deform_object", "args": {"name": "Post", "mode": "twist", "axis": "y", "angle_deg": 90}},
            {"tool": "check_mesh", "args": {"name": "Post"}}
        ]}),
    ));
    assert_eq!(batch["ran"], 4);
    assert_eq!(batch["results"][3]["data"]["parts"][0]["watertight"], true, "{batch}");
    assert_eq!(batch["results"][3]["data"]["parts"][0]["issues"].as_array().unwrap().len(), 0);
    // A failing command stops the batch and reports where.
    let stopped = e.call("batch", json!({"commands": [{"tool": "deform_object", "args": {"name": "Post", "mode": "wobble"}}, {"tool": "clear_selection"}]}));
    assert!(!stopped.ok);
    assert_eq!(stopped.data["ran"], 1);

    let report = ok(e.call("check_mesh", json!({})));
    assert!(report["parts"].as_array().unwrap().len() >= 4);
    let info = ok(e.call("get_scene_info", json!({})));
    assert!(info["objects"][0]["world_bounds_max"][0].as_f64().is_some());

    // One call, four angles. Rendering needs a GPU adapter; skip the pixel check without one.
    let path = std::env::temp_dir().join(format!("rust_modeler_views_{}.png", std::process::id()));
    let shot = e.call("capture_views", json!({"path": path, "cell_width": 200, "cell_height": 150, "frame": "object", "object_id": slab_id}));
    if shot.ok {
        let img = image::open(&path).unwrap();
        assert_eq!((img.width(), img.height()), (400, 300));
        assert_eq!(shot.data["cells"].as_array().unwrap().len(), 4);
        let _ = std::fs::remove_file(&path);
    } else {
        eprintln!("capture_views skipped: {}", shot.message);
    }
}
