use crate::scene::Scene;

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub label: String,
    pub scene_snapshot: Scene,
}

#[derive(Clone)]
pub struct UndoHistory {
    pub undo_stack: Vec<HistoryEntry>,
    pub redo_stack: Vec<HistoryEntry>,
    pub max_entries: usize,
}

impl Default for UndoHistory {
    fn default() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_entries: 40,
        }
    }
}

impl UndoHistory {
    pub fn record(&mut self, label: impl Into<String>, current_scene: &Scene) {
        if current_scene
            .revisions
            .iter()
            .any(|r| r.status == crate::review::RevisionStatus::InProgress)
        {
            return;
        }
        if self.undo_stack.len() >= self.max_entries {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(HistoryEntry {
            label: label.into(),
            scene_snapshot: current_scene.clone(),
        });
        self.redo_stack.clear();
        trim_memory(&mut self.undo_stack);
    }

    pub fn undo(&mut self, current_scene: &mut Scene) -> Option<String> {
        let entry = self.undo_stack.pop()?;
        let action_label = entry.label.clone();
        self.redo_stack.push(HistoryEntry {
            label: action_label.clone(),
            scene_snapshot: current_scene.clone(),
        });
        restore(current_scene, entry.scene_snapshot);
        trim_memory(&mut self.undo_stack);
        trim_memory(&mut self.redo_stack);
        Some(action_label)
    }

    pub fn redo(&mut self, current_scene: &mut Scene) -> Option<String> {
        let entry = self.redo_stack.pop()?;
        let action_label = entry.label.clone();
        self.undo_stack.push(HistoryEntry {
            label: action_label.clone(),
            scene_snapshot: current_scene.clone(),
        });
        restore(current_scene, entry.scene_snapshot);
        trim_memory(&mut self.undo_stack);
        trim_memory(&mut self.redo_stack);
        Some(action_label)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

fn restore(current: &mut Scene, mut restored: Scene) {
    if restored.project_path == current.project_path {
        restored.comments = current.comments.clone();
        restored.next_comment_id = current.next_comment_id;
        restored.saved_selections = current.saved_selections.clone();
        restored.saved_views = current.saved_views.clone();
        restored.project_name = current.project_name.clone();
        restored.units = current.units.clone();
        use crate::{review::RevisionStatus, scene::CommentStatus};
        let mut statuses = Vec::new();
        for r in &current.revisions {
            if restored.revisions.iter().all(|other| other.id != r.id) {
                if let Some(id) = r.comment_id {
                    statuses.push((id, CommentStatus::Open));
                }
            }
        }
        for r in &restored.revisions {
            if current
                .revisions
                .iter()
                .find(|other| other.id == r.id)
                .is_none_or(|other| other.status != r.status)
            {
                if let Some(id) = r.comment_id {
                    statuses.push((
                        id,
                        match r.status {
                            RevisionStatus::Accepted => CommentStatus::Accepted,
                            RevisionStatus::ReadyForReview => CommentStatus::ReadyForReview,
                            RevisionStatus::InProgress => CommentStatus::InProgress,
                            RevisionStatus::Reverted => CommentStatus::Open,
                        },
                    ));
                }
            }
        }
        for (id, status) in statuses {
            let _ = restored.set_comment_status(id, status);
        }
    }
    restored.next_id = restored.next_id.max(current.next_id);
    restored.next_revision_id = restored.next_revision_id.max(current.next_revision_id);
    restored.preview_revision = None;
    *current = restored;
}

// Snapshot histories include voxel grids and revision geometry; bound retained memory too.
fn trim_memory(stack: &mut Vec<HistoryEntry>) {
    fn objects_size(objects: &[crate::scene::SceneObject]) -> usize {
        objects
            .iter()
            .map(|o| {
                (o.mesh.positions.len() + o.mesh.normals.len()) * std::mem::size_of::<glam::Vec3>()
                    + o.mesh.colors.len() * 12
                    + o.mesh
                        .faces
                        .iter()
                        .map(|f| f.len() * std::mem::size_of::<usize>() + 24)
                        .sum::<usize>()
                    + o.voxel_grid
                        .as_ref()
                        .map(|g| g.sdf.len() * 4 + g.colors.len() * 12)
                        .unwrap_or(0)
            })
            .sum()
    }
    let mut total = 0usize;
    let mut keep_from = stack.len();
    for (i, entry) in stack.iter().enumerate().rev() {
        let scene = &entry.scene_snapshot;
        let bytes = objects_size(&scene.objects)
            + scene
                .revisions
                .iter()
                .map(|r| {
                    objects_size(&r.before) + r.after.as_ref().map(|a| objects_size(a)).unwrap_or(0)
                })
                .sum::<usize>();
        if keep_from != stack.len() && total.saturating_add(bytes) > 128 * 1024 * 1024 {
            break;
        }
        total = total.saturating_add(bytes);
        keep_from = i;
    }
    if keep_from > 0 {
        stack.drain(..keep_from);
    }
}
