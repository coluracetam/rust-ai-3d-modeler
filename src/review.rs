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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Revision {
    pub id: u64,
    pub label: String,
    pub comment_id: Option<u64>,
    pub status: RevisionStatus,
    pub before: Vec<SceneObject>,
    pub after: Option<Vec<SceneObject>>,
    #[serde(default)]
    pub screenshots: Vec<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub changed_object_ids: Option<Vec<u64>>,
}

/// Excludes ephemeral selection, visibility and locks from conflict detection.
pub fn object_hash(object: &SceneObject) -> u64 {
    let mut value = serde_json::to_value(object).unwrap();
    value.as_object_mut().unwrap().remove("visible");
    value.as_object_mut().unwrap().remove("locked");
    if let Some(mesh) = value.get_mut("mesh").and_then(|v| v.as_object_mut()) {
        mesh.remove("selected_faces");
    }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    value.to_string().hash(&mut hash);
    hash.finish()
}

pub fn objects_match(a: &[SceneObject], b: &[SceneObject]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.id == b.id && object_hash(a) == object_hash(b))
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
        // Bound retained geometry; unresolved work is never evicted.
        if self.revisions.len() >= 8 {
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
        if let Some(after) = &revision.after {
            if !objects_match(&self.objects, after) {
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
                .before
                .clone();
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
        Ok(())
    }

    pub fn changed_objects(&self, id: u64) -> Vec<u64> {
        let Some(r) = self.revisions.iter().find(|r| r.id == id) else {
            return Vec::new();
        };
        if let Some(changed) = &r.changed_object_ids {
            return changed.clone();
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

    pub fn display_scene(&self) -> std::borrow::Cow<'_, Scene> {
        let Some((id, before)) = self.preview_revision else {
            return std::borrow::Cow::Borrowed(self);
        };
        let Some(r) = self.revisions.iter().find(|r| r.id == id) else {
            return std::borrow::Cow::Borrowed(self);
        };
        let objects = if before {
            &r.before
        } else {
            r.after.as_ref().unwrap_or(&self.objects)
        };
        let mut display = Scene::empty();
        display.objects = objects.clone();
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
        std::borrow::Cow::Owned(display)
    }
}
