use futures::channel::mpsc;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};

pub struct DatasetWatcher {
    images_dir: PathBuf,
    watcher: Option<RecommendedWatcher>,
    event_rx: Option<mpsc::Receiver<notify::Result<Event>>>,
    images_tx: mpsc::UnboundedSender<Vec<PathBuf>>,
}

impl DatasetWatcher {
    pub fn new(images_dir: PathBuf, images_tx: mpsc::UnboundedSender<Vec<PathBuf>>) -> Self {
        Self {
            images_dir,
            watcher: None,
            event_rx: None,
            images_tx,
        }
    }

    pub fn rebuild(&mut self) -> Result<(), Option<String>> {
        match Self::create_async_watcher(&self.images_dir) {
            Ok((watcher, rx)) => {
                self.watcher = Some(watcher);
                self.event_rx = Some(rx);
                if let Err(err) = self.send_image_snapshot() {
                    return Err(Some(err));
                }
                Ok(())
            }
            Err(err) => {
                self.watcher = None;
                self.event_rx = None;
                Err(Some(err.to_string()))
            }
        }
    }

    pub async fn event(&mut self) -> Option<notify::Result<Event>> {
        use futures::StreamExt;

        if let Some(stream) = self.event_rx.as_mut() {
            stream.next().await
        } else {
            futures::future::pending().await
        }
    }

    pub fn send_image_snapshot(&self) -> Result<(), String> {
        let images = self.gather_images();
        self.images_tx
            .unbounded_send(images)
            .map_err(|_| "Unable to send image list".to_owned())
    }

    pub fn images_dir(&self) -> &Path {
        &self.images_dir
    }

    fn create_async_watcher(
        path: &Path,
    ) -> notify::Result<(RecommendedWatcher, mpsc::Receiver<notify::Result<Event>>)> {
        if !path.exists()
            && let Err(err) = std::fs::create_dir_all(path)
        {
            eprintln!("Failed to create directory {}: {err}", path.display());
        }
        let (mut tx, rx) = mpsc::channel(128);
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                futures::executor::block_on(async {
                    use futures::SinkExt;
                    let _ = tx.send(res).await;
                });
            },
            Config::default(),
        )?;
        watcher.watch(path, RecursiveMode::Recursive)?;
        Ok((watcher, rx))
    }

    fn gather_images(&self) -> Vec<PathBuf> {
        gather_images(self.images_dir())
    }
}

pub fn gather_images(images_dir: &Path) -> Vec<PathBuf> {
    collect_images(images_dir)
}

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() {
        return files;
    }
    let mut queue = VecDeque::from([dir.to_path_buf()]);
    while let Some(path) = queue.pop_front() {
        let Ok(read_dir) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                queue.push_back(entry_path);
            } else if is_supported_image(&entry_path) {
                let rel = entry_path.strip_prefix(dir).unwrap_or(&entry_path);
                files.push(rel.to_path_buf());
            }
        }
    }
    files.sort();
    files
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
}

pub fn matches_event(event: &Event) -> bool {
    use notify::event::{CreateKind, EventKind, ModifyKind, RemoveKind};
    match &event.kind {
        EventKind::Create(kind) => matches!(
            kind,
            CreateKind::Any | CreateKind::File | CreateKind::Folder
        ),
        EventKind::Modify(kind) => matches!(kind, ModifyKind::Data(_) | ModifyKind::Name(_)),
        EventKind::Remove(kind) => matches!(
            kind,
            RemoveKind::Any | RemoveKind::File | RemoveKind::Folder
        ),
        _ => false,
    }
}
