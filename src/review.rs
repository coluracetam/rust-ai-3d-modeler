use crate::scene::{CommentStatus, Scene, SceneObject};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionStatus {
    InProgress,
    ReadyForReview,
    Accepted,
    Reverted,
}

/// Reviewed revisions older than this many keep only their summary, not geometry.
pub const KEEP_REVIEWED_GEOMETRY: usize = 2;
/// Revision metadata kept in the project (unresolved work is never evicted).
pub const MAX_REVISIONS: usize = 50;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Revision {
    pub id: u64,
    pub label: String,
    pub comment_id: Option<u64>,
    pub status: RevisionStatus,
    /// While in progress: every object at the start. Once finished (`partial`), only the
    /// changed objects as they were before the revision.
    pub before: Vec<SceneObject>,
    /// Once finished (`partial`), only the changed objects as they are after the revision.
    pub after: Option<Vec<SceneObject>>,
    #[serde(default)]
    pub screenshots: Vec<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub changed_object_ids: Option<Vec<u64>>,
    /// `before`/`after` hold only changed objects; unchanged ones come from the live scene.
    #[serde(default)]
    pub partial: bool,
    /// Object order before the revision, so a revert restores deleted parts in place.
    #[serde(default)]
    pub object_order: Vec<u64>,
    /// Geometry was dropped to bound project size; only metadata remains.
    #[serde(default)]
    pub geometry_pruned: bool,
    /// The user's notes when they asked for changes instead of accepting.
    #[serde(default)]
    pub change_request: Option<String>,
}

fn hash_floats<H: Hasher>(h: &mut H, it: impl IntoIterator<Item = f32>) {
    for f in it {
        f.to_bits().hash(h);
    }
}

/// Excludes ephemeral selection, visibility and locks from conflict detection.
pub fn object_hash(object: &SceneObject) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    object.id.hash(&mut h);
    object.name.hash(&mut h);
    let t = &object.transform;
    hash_floats(
        &mut h,
        [t.translation, t.rotation_deg, t.scale]
            .iter()
            .flat_map(|v| v.to_array()),
    );
    let m = &object.mesh;
    hash_floats(&mut h, m.positions.iter().flat_map(|v| v.to_array()));
    hash_floats(&mut h, m.normals.iter().flat_map(|v| v.to_array()));
    hash_floats(&mut h, m.colors.iter().flatten().copied());
    m.faces.hash(&mut h);
    if let Some(g) = &object.voxel_grid {
        g.dims.hash(&mut h);
        hash_floats(&mut h, [g.voxel_size]);
        hash_floats(&mut h, g.sdf.iter().copied());
        hash_floats(&mut h, g.colors.iter().flatten().copied());
        format!("{:?}", g.meshing_mode).hash(&mut h);
    }
    hash_floats(&mut h, object.color);
    hash_floats(
        &mut h,
        [object.roughness, object.metallic, object.texture_scale, object.emission_strength],
    );
    object.flat_shading.hash(&mut h);
    for layer in &object.textures {
        layer.name.hash(&mut h);
        (layer.image.width, layer.image.height).hash(&mut h);
        layer.image.rgba.hash(&mut h);
        hash_floats(
            &mut h,
            [layer.origin, layer.u_axis, layer.v_axis].iter().flat_map(|v| v.to_array()).chain([layer.depth, layer.opacity]),
        );
    }
    format!("{:?}", object.pattern).hash(&mut h);
    hash_floats(&mut h, object.secondary_color);
    hash_floats(&mut h, object.emission_color);
    h.finish()
}

pub fn objects_match(a: &[SceneObject], b: &[SceneObject]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.id == b.id && object_hash(a) == object_hash(b))
}

impl Revision {
    /// Replace full snapshots by the changed objects only.
    pub fn compact(&mut self) {
        if self.partial || self.geometry_pruned {
            return;
        }
        let Some(after) = self.after.as_mut() else {
            return;
        };
        let changed: Vec<u64> = match &self.changed_object_ids {
            Some(c) => c.clone(),
            None => {
                let c = after
                    .iter()
                    .filter(|o| {
                        self.before
                            .iter()
                            .find(|b| b.id == o.id)
                            .is_none_or(|b| object_hash(b) != object_hash(o))
                    })
                    .map(|o| o.id)
                    .chain(
                        self.before
                            .iter()
                            .filter(|o| !after.iter().any(|a| a.id == o.id))
                            .map(|o| o.id),
                    )
                    .collect();
                self.changed_object_ids = Some(c);
                self.changed_object_ids.clone().unwrap()
            }
        };
        self.object_order = self.before.iter().map(|o| o.id).collect();
        self.before.retain(|o| changed.contains(&o.id));
        after.retain(|o| changed.contains(&o.id));
        self.partial = true;
    }

    fn changed(&self) -> &[u64] {
        self.changed_object_ids.as_deref().unwrap_or(&[])
    }

    /// The scene objects after this revision, with unchanged parts taken from `live`.
    pub fn after_objects(&self, live: &[SceneObject]) -> Vec<SceneObject> {
        let Some(after) = &self.after else {
            return live.to_vec();
        };
        if !self.partial {
            return after.clone();
        }
        let changed = self.changed();
        let mut out: Vec<SceneObject> = live
            .iter()
            .filter_map(|o| {
                if changed.contains(&o.id) {
                    after.iter().find(|a| a.id == o.id).cloned()
                } else {
                    Some(o.clone())
                }
            })
            .collect();
        for a in after {
            if !out.iter().any(|o| o.id == a.id) {
                out.push(a.clone());
            }
        }
        out
    }

    /// The scene objects before this revision, with unchanged parts taken from `live`.
    pub fn before_objects(&self, live: &[SceneObject]) -> Vec<SceneObject> {
        if !self.partial {
            return self.before.clone();
        }
        let changed = self.changed();
        let mut out: Vec<SceneObject> = self
            .object_order
            .iter()
            .filter_map(|id| {
                self.before
                    .iter()
                    .find(|b| b.id == *id)
                    .or_else(|| live.iter().find(|o| o.id == *id && !changed.contains(id)))
                    .cloned()
            })
            .collect();
        // Parts added later by other work stay visible.
        for o in live {
            if !changed.contains(&o.id) && !self.object_order.contains(&o.id) {
                out.push(o.clone());
            }
        }
        out
    }

    /// Whether `live` still shows exactly this revision's result for the parts it changed.
    pub fn matches_live(&self, live: &[SceneObject]) -> bool {
        let Some(after) = &self.after else {
            return true;
        };
        if !self.partial {
            return objects_match(live, after);
        }
        self.changed().iter().all(|id| {
            match (
                live.iter().find(|o| o.id == *id),
                after.iter().find(|o| o.id == *id),
            ) {
                (Some(l), Some(a)) => object_hash(l) == object_hash(a),
                (None, None) => true,
                _ => false,
            }
        })
    }

    pub fn can_compare(&self) -> bool {
        !self.geometry_pruned
    }
}

impl Scene {
    pub fn begin_revision(
        &mut self,
        label: String,
        comment_id: Option<u64>,
    ) -> Result<u64, String> {
        if self.revisions.iter().any(|r| {
            matches!(
                r.status,
                RevisionStatus::InProgress | RevisionStatus::ReadyForReview
            )
        }) {
            return Err("Accept, revert or cancel the current revision first".into());
        }
        if let Some(id) = comment_id {
            if !self.comments.iter().any(|c| c.id == id) {
                return Err("Comment not found".into());
            }
            self.set_comment_status(id, CommentStatus::InProgress)?;
        }
        let id = self.next_revision_id.max(1);
        self.next_revision_id = id + 1;
        // Bound retained history; unresolved work is never evicted.
        while self.revisions.len() >= MAX_REVISIONS {
            self.revisions.remove(0);
        }
        self.revisions.push(Revision {
            id,
            label,
            comment_id,
            status: RevisionStatus::InProgress,
            before: self.objects.clone(),
            after: None,
            screenshots: Vec::new(),
            summary: String::new(),
            changed_object_ids: None,
            partial: false,
            object_order: Vec::new(),
            geometry_pruned: false,
            change_request: None,
        });
        Ok(id)
    }

    pub fn finish_revision(
        &mut self,
        id: u64,
        summary: String,
        screenshots: Vec<String>,
    ) -> Result<(), String> {
        let changed = self.changed_objects(id);
        let revision = self
            .revisions
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or("Revision not found")?;
        if revision.status != RevisionStatus::InProgress {
            return Err("Revision is not in progress".into());
        }
        revision.after = Some(self.objects.clone());
        revision.changed_object_ids = Some(changed);
        revision.status = RevisionStatus::ReadyForReview;
        revision.summary = summary.clone();
        revision.screenshots = screenshots;
        revision.compact();
        let comment_id = revision.comment_id;
        if let Some(cid) = comment_id {
            self.set_comment_status(cid, CommentStatus::ReadyForReview)?;
            if let Some(c) = self.comments.iter_mut().find(|c| c.id == cid) {
                c.agent_reply = Some(summary);
            }
        }
        self.preview_revision = None;
        Ok(())
    }

    pub fn review_revision(&mut self, id: u64, accept: bool) -> Result<(), String> {
        let revision = self
            .revisions
            .iter()
            .find(|r| r.id == id)
            .ok_or("Revision not found")?;
        if !matches!(
            revision.status,
            RevisionStatus::InProgress | RevisionStatus::ReadyForReview
        ) {
            return Err("Revision has already been reviewed".into());
        }
        if accept && revision.status != RevisionStatus::ReadyForReview {
            return Err("Finish the revision before accepting it".into());
        }
        if revision.after.is_some() {
            if !revision.matches_live(&self.objects) {
                return Err(
                    "The scene changed after this revision; reverting would overwrite newer work"
                        .into(),
                );
            }
        }
        let cid = revision.comment_id;
        if !accept && revision.after.is_none() {
            let changed = self.changed_objects(id);
            let r = self.revisions.iter_mut().find(|r| r.id == id).unwrap();
            r.after = Some(self.objects.clone());
            r.changed_object_ids = Some(changed);
        }
        if !accept {
            let visibility: std::collections::HashMap<_, _> = self
                .objects
                .iter()
                .map(|o| (o.id, (o.visible, o.locked)))
                .collect();
            self.objects = self
                .revisions
                .iter()
                .find(|r| r.id == id)
                .unwrap()
                .before_objects(&self.objects);
            for o in &mut self.objects {
                if let Some(&(visible, locked)) = visibility.get(&o.id) {
                    o.visible = visible;
                    o.locked = locked;
                }
            }
            self.clear_all_selections();
            self.selected_object = None;
        }
        self.revisions
            .iter_mut()
            .find(|r| r.id == id)
            .unwrap()
            .status = if accept {
            RevisionStatus::Accepted
        } else {
            RevisionStatus::Reverted
        };
        if let Some(cid) = cid {
            self.set_comment_status(
                cid,
                if accept {
                    CommentStatus::Accepted
                } else {
                    CommentStatus::Open
                },
            )?;
        }
        self.preview_revision = None;
        self.prune_revision_geometry();
        Ok(())
    }

    /// Ask for changes: revert the revision and reopen its comment with the user's notes.
    pub fn request_revision_changes(&mut self, id: u64, notes: String) -> Result<(), String> {
        self.review_revision(id, false)?;
        let r = self.revisions.iter_mut().find(|r| r.id == id).unwrap();
        let notes = notes.trim().to_string();
        r.change_request = (!notes.is_empty()).then(|| notes.clone());
        if let (Some(cid), false) = (r.comment_id, notes.is_empty()) {
            if let Some(c) = self.comments.iter_mut().find(|c| c.id == cid) {
                c.replies.push(crate::scene::CommentReply {
                    author: "user".into(),
                    text: notes,
                });
            }
        }
        // Wakes agents waiting in `wait_for_feedback`.
        self.feedback_batch += 1;
        Ok(())
    }

    /// Shrink old snapshots: finished revisions keep only changed objects, and reviewed
    /// revisions beyond the most recent few keep only metadata.
    pub fn prune_revision_geometry(&mut self) {
        for r in &mut self.revisions {
            if r.status != RevisionStatus::InProgress {
                r.compact();
            }
        }
        let mut kept = 0;
        for r in self.revisions.iter_mut().rev() {
            if !matches!(r.status, RevisionStatus::Accepted | RevisionStatus::Reverted)
                || r.geometry_pruned
            {
                continue;
            }
            if kept < KEEP_REVIEWED_GEOMETRY {
                kept += 1;
                continue;
            }
            r.before.clear();
            r.after = None;
            r.geometry_pruned = true;
        }
        if self
            .preview_revision
            .is_some_and(|(id, _)| self.revisions.iter().any(|r| r.id == id && r.geometry_pruned))
        {
            self.preview_revision = None;
        }
    }

    pub fn changed_objects(&self, id: u64) -> Vec<u64> {
        let Some(r) = self.revisions.iter().find(|r| r.id == id) else {
            return Vec::new();
        };
        if let Some(changed) = &r.changed_object_ids {
            return changed.clone();
        }
        if r.geometry_pruned {
            return Vec::new();
        }
        let after = r.after.as_deref().unwrap_or(&self.objects);
        after
            .iter()
            .filter(|o| {
                r.before
                    .iter()
                    .find(|b| b.id == o.id)
                    .is_none_or(|b| object_hash(b) != object_hash(o))
            })
            .map(|o| o.id)
            .chain(
                r.before
                    .iter()
                    .filter(|o| !after.iter().any(|a| a.id == o.id))
                    .map(|o| o.id),
            )
            .collect()
    }

    /// Read-only scene showing a revision's before or after state, if it still has geometry.
    pub fn revision_view(&self, id: u64, before: bool) -> Option<Scene> {
        let r = self.revisions.iter().find(|r| r.id == id)?;
        if r.geometry_pruned {
            return None;
        }
        let objects = if before {
            r.before_objects(&self.objects)
        } else {
            r.after_objects(&self.objects)
        };
        let mut display = Scene::empty();
        display.objects = objects;
        // Comparisons show geometry only; selection tint would read as a material change.
        for o in &mut display.objects {
            if !o.mesh.selected_faces.is_empty() {
                o.mesh.selected_faces.clear();
                o.mesh.bump_version();
            }
        }
        display.clip_plane = self.clip_plane;
        if self.highlight_changes {
            let changed = self.changed_objects(id);
            for o in &mut display.objects {
                if changed.contains(&o.id) {
                    o.color = [0.2, 0.85, 0.7];
                    o.emission_color = o.color;
                    o.emission_strength = 0.15;
                }
            }
        }
        Some(display)
    }

    pub fn display_scene(&self) -> std::borrow::Cow<'_, Scene> {
        match self
            .preview_revision
            .and_then(|(id, before)| self.revision_view(id, before))
        {
            Some(view) => std::borrow::Cow::Owned(view),
            None => std::borrow::Cow::Borrowed(self),
        }
    }
}
