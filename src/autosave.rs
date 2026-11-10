//! Debounced background persistence for the live GUI scene.
//!
//! Serializing, writing, fsyncing and backing up a large scene can take hundreds of milliseconds,
//! so GUI changes only schedule a save. Changes made within `DELAY` share one write, which runs on
//! a worker thread from a snapshot. `flush` waits for the worker and writes anything still pending,
//! so nothing is lost when the window closes.
use crate::camera::OrbitCamera;
use crate::scene::Scene;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const DELAY: Duration = Duration::from_millis(400);
const RETRY_AFTER_ERROR: Duration = Duration::from_secs(5);
/// How often the event loop checks for the worker's result while a save is in flight.
pub const POLL: Duration = Duration::from_millis(100);

type SaveFn = fn(&mut Scene, &OrbitCamera) -> Result<(), String>;

pub struct SceneSaver {
    tx: Option<mpsc::Sender<(Scene, OrbitCamera)>>,
    results: mpsc::Receiver<Result<(), String>>,
    worker: Option<JoinHandle<()>>,
    in_flight: bool,
    due: Option<Instant>,
    save: SaveFn,
}

impl Default for SceneSaver {
    fn default() -> Self {
        Self::new(crate::mcp::persist_scene)
    }
}

impl SceneSaver {
    pub fn new(save: SaveFn) -> Self {
        let (tx, rx) = mpsc::channel::<(Scene, OrbitCamera)>();
        let (result_tx, results) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("scene-autosave".into())
            .spawn(move || {
                while let Ok((mut scene, camera)) = rx.recv() {
                    let _ = result_tx.send(save(&mut scene, &camera));
                }
            })
            .ok();
        Self {
            tx: worker.as_ref().map(|_| tx),
            results,
            worker,
            in_flight: false,
            due: None,
            save,
        }
    }

    /// Schedule a save; bursts of changes within `DELAY` share one write.
    pub fn request(&mut self) {
        self.due.get_or_insert_with(|| Instant::now() + DELAY);
    }

    #[cfg(test)]
    pub fn pending(&self) -> bool {
        self.due.is_some() || self.in_flight
    }

    /// Collect finished saves and start a due one. Returns errors to show to the user.
    pub fn poll(&mut self, scene: &mut Scene, camera: &OrbitCamera) -> Vec<String> {
        let mut errors = Vec::new();
        while let Ok(result) = self.results.try_recv() {
            self.in_flight = false;
            if let Err(e) = result {
                errors.push(e);
                self.due
                    .get_or_insert_with(|| Instant::now() + RETRY_AFTER_ERROR);
            }
        }
        if self.in_flight || self.due.is_none_or(|due| Instant::now() < due) {
            return errors;
        }
        self.due = None;
        scene.last_camera = Some(camera.clone());
        match &self.tx {
            Some(tx) if tx.send((scene.clone(), camera.clone())).is_ok() => self.in_flight = true,
            // No worker thread: save inline rather than not at all.
            _ => {
                if let Err(e) = (self.save)(scene, camera) {
                    errors.push(e);
                }
            }
        }
        errors
    }

    /// When the event loop should wake up next for saving work.
    pub fn next_wake(&self) -> Option<Instant> {
        if self.in_flight {
            Some(Instant::now() + POLL)
        } else {
            self.due
        }
    }

    /// Finish any in-flight save and write pending changes now (on exit).
    pub fn flush(&mut self, scene: &mut Scene, camera: &OrbitCamera) -> Result<(), String> {
        self.tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let mut result = Ok(());
        while let Ok(r) = self.results.try_recv() {
            if r.is_err() {
                self.due = Some(Instant::now());
            }
            result = result.and(r);
        }
        self.in_flight = false;
        if self.due.take().is_some() {
            scene.last_camera = Some(camera.clone());
            result = (self.save)(scene, camera);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SAVES: AtomicUsize = AtomicUsize::new(0);
    fn counting_save(scene: &mut Scene, _: &OrbitCamera) -> Result<(), String> {
        SAVES.fetch_add(1, Ordering::SeqCst);
        if scene.project_name == "fail" {
            Err("disk full".into())
        } else {
            Ok(())
        }
    }

    fn wait_for(saver: &mut SceneSaver, scene: &mut Scene, camera: &OrbitCamera) -> Vec<String> {
        let start = Instant::now();
        let mut errors = Vec::new();
        while saver.pending() && start.elapsed() < Duration::from_secs(5) {
            errors.extend(saver.poll(scene, camera));
            std::thread::sleep(Duration::from_millis(5));
        }
        errors
    }

    // One test so the shared counter is not raced by parallel tests.
    #[test]
    fn coalesces_bursts_reports_errors_and_flushes() {
        let camera = OrbitCamera::default();
        let mut scene = Scene::empty();
        let mut saver = SceneSaver::new(counting_save);

        for _ in 0..20 {
            saver.request();
            assert!(saver.poll(&mut scene, &camera).is_empty());
        }
        assert_eq!(
            SAVES.load(Ordering::SeqCst),
            0,
            "saves wait for the debounce delay"
        );
        assert!(wait_for(&mut saver, &mut scene, &camera).is_empty());
        assert_eq!(
            SAVES.load(Ordering::SeqCst),
            1,
            "a burst of changes is one write"
        );
        assert!(scene.last_camera.is_some());

        scene.project_name = "fail".into();
        saver.request();
        saver.due = Some(Instant::now());
        let start = Instant::now();
        let mut errors = Vec::new();
        while errors.is_empty() && start.elapsed() < Duration::from_secs(5) {
            errors.extend(saver.poll(&mut scene, &camera));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(errors, vec!["disk full".to_string()]);
        assert!(saver.pending(), "a failed save is retried later");

        scene.project_name = "ok".into();
        let before = SAVES.load(Ordering::SeqCst);
        assert!(saver.flush(&mut scene, &camera).is_ok());
        assert_eq!(
            SAVES.load(Ordering::SeqCst),
            before + 1,
            "flush writes pending changes"
        );
        assert!(!saver.pending());
        // After flush the worker is gone; later saves still happen inline.
        saver.request();
        saver.due = Some(Instant::now());
        assert!(saver.poll(&mut scene, &camera).is_empty());
        assert_eq!(SAVES.load(Ordering::SeqCst), before + 2);
    }
}
