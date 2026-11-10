use crate::scene::Scene;

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub label: String,
    pub scene_snapshot: Scene,
}

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
        if self.undo_stack.len() >= self.max_entries {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(HistoryEntry {
            label: label.into(),
            scene_snapshot: current_scene.clone(),
        });
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, current_scene: &mut Scene) -> Option<String> {
        let entry = self.undo_stack.pop()?;
        let action_label = entry.label.clone();
        self.redo_stack.push(HistoryEntry {
            label: action_label.clone(),
            scene_snapshot: current_scene.clone(),
        });
        *current_scene = entry.scene_snapshot;
        Some(action_label)
    }

    pub fn redo(&mut self, current_scene: &mut Scene) -> Option<String> {
        let entry = self.redo_stack.pop()?;
        let action_label = entry.label.clone();
        self.undo_stack.push(HistoryEntry {
            label: action_label.clone(),
            scene_snapshot: current_scene.clone(),
        });
        *current_scene = entry.scene_snapshot;
        Some(action_label)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}
