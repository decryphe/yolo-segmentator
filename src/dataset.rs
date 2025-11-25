use crate::watcher::{self, DatasetWatcher as FsWatcher};
use futures::channel::{mpsc, oneshot};
use serde_yaml_ng as serde_yaml;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

#[derive(Clone)]
pub struct Dataset {
    pub root: PathBuf,
    pub yaml_name: String,
    pub classes: Vec<ClassEntry>,
    pub new_class_name: String,
    pub segments: Vec<SegmentEntry>,
    pub images: DatasetImages,
}

#[derive(Default, Clone)]
pub struct ClassEntry {
    pub id: usize,
    pub name: String,
}

#[derive(Clone)]
pub struct SegmentPoint {
    pub x: f32,
    pub y: f32,
}

impl Default for SegmentPoint {
    fn default() -> Self {
        Self { x: 0.5, y: 0.5 }
    }
}

#[derive(Clone)]
pub struct SegmentEntry {
    pub id: usize,
    pub class_index: usize,
    pub polygon: Vec<SegmentPoint>,
}

impl SegmentEntry {
    pub fn new(id: usize, class_index: usize) -> Self {
        Self {
            id,
            class_index,
            polygon: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSplit {
    Train,
    Val,
    Test,
}

#[derive(Clone, Default)]
pub struct DatasetImages {
    pub train: Vec<ImageEntry>,
    pub val: Vec<ImageEntry>,
    pub test: Vec<ImageEntry>,
}

impl DatasetImages {
    pub fn from_entries(entries: Vec<ImageEntry>) -> Self {
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();
        for entry in entries {
            match entry.split {
                ImageSplit::Train => train.push(entry),
                ImageSplit::Val => val.push(entry),
                ImageSplit::Test => test.push(entry),
            }
        }
        Self { train, val, test }
    }
}

#[derive(Clone)]
pub struct ImageEntry {
    pub split: ImageSplit,
    pub path: String,
    pub has_labels: bool,
}

impl Dataset {
    pub fn new(root: PathBuf) -> Self {
        let classes = Self::default_classes();
        let yaml_name = Self::dataset_name_from_path(&root);
        let mut dataset = Self {
            root,
            yaml_name,
            classes,
            new_class_name: String::new(),
            segments: Vec::new(),
            images: DatasetImages::default(),
        };
        dataset.refresh_image_lists();
        dataset
    }

    pub fn load(root: PathBuf) -> Result<Self, String> {
        let yaml_path = root.join("dataset.yaml");
        let contents = std::fs::read_to_string(&yaml_path)
            .map_err(|err| format!("Unable to read dataset.yaml: {err}"))?;
        let doc: YamlDoc = serde_yaml_ng::from_str(&contents)
            .map_err(|err| format!("Unable to parse dataset.yaml: {err}"))?;
        let mut dataset = Self::from((root, doc));
        dataset.refresh_image_lists();
        Ok(dataset)
    }

    fn default_classes() -> Vec<ClassEntry> {
        vec![
            ClassEntry {
                id: 0,
                name: "Person".to_owned(),
            },
            ClassEntry {
                id: 1,
                name: "Car".to_owned(),
            },
        ]
    }

    pub fn name(&self) -> String {
        self.yaml_name.clone()
    }

    pub fn initialize_on_disk(&self) -> Result<(), String> {
        self.ensure_directories()?;
        self.save_metadata()
    }

    fn ensure_directories(&self) -> Result<(), String> {
        if !self.root.exists() {
            std::fs::create_dir_all(&self.root)
                .map_err(|err| format!("Unable to create dataset directory: {err}"))?;
        }
        let images_dir = self.root.join("images");
        for split in ["train", "val", "test"] {
            std::fs::create_dir_all(images_dir.join(split))
                .map_err(|err| format!("Unable to create images/{split} subdirectory: {err}"))?;
        }
        Ok(())
    }

    pub fn save_metadata(&self) -> Result<(), String> {
        let yaml_doc = YamlDoc::from(self);
        let serialized = serde_yaml::to_string(&yaml_doc)
            .map_err(|err| format!("Unable to serialize dataset.yaml: {err}"))?;
        let yaml_path = self.root.join("dataset.yaml");
        std::fs::write(&yaml_path, serialized)
            .map_err(|err| format!("Unable to write dataset.yaml: {err}"))?;
        Ok(())
    }

    pub fn next_class_id(&self) -> usize {
        self.classes
            .iter()
            .map(|c| c.id)
            .max()
            .map_or(0, |max| max + 1)
    }

    pub fn next_segment_id(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.id)
            .max()
            .map_or(0, |max| max + 1)
    }

    fn dataset_name_from_path(path: &Path) -> String {
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset")
            .to_owned()
    }

    pub fn refresh_image_lists(&mut self) {
        let entries = watcher::gather_images(&self.root.join("images"));
        self.images = DatasetImages::from_entries(entries);
    }

    pub fn load_segments_for_image(&mut self, image_path: &Path) -> Result<usize, String> {
        let txt_path = image_path.with_extension("txt");
        self.segments.clear();
        if !txt_path.exists() {
            return Ok(0);
        }
        let contents = std::fs::read_to_string(&txt_path)
            .map_err(|err| format!("Unable to read {}: {err}", txt_path.display()))?;
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut parts = trimmed.split_whitespace();
            let Some(class_part) = parts.next() else {
                continue;
            };
            let Ok(class_index) = class_part.parse::<usize>() else {
                continue;
            };
            let mut coords = Vec::new();
            for part in parts {
                if let Ok(value) = part.parse::<f32>() {
                    coords.push(value);
                }
            }
            if coords.len() < 6 || coords.len() % 2 != 0 {
                continue;
            }
            let mut polygon = Vec::new();
            for chunk in coords.chunks(2) {
                polygon.push(SegmentPoint {
                    x: chunk[0],
                    y: chunk[1],
                });
            }
            if polygon.len() < 3 {
                continue;
            }
            self.segments.push(SegmentEntry {
                id: self.segments.len(),
                class_index,
                polygon,
            });
        }
        Ok(self.segments.len())
    }

    pub fn flattened_image_paths(&self) -> Vec<PathBuf> {
        let mut result = Vec::new();
        for (split, entries) in [
            ("train", &self.images.train),
            ("val", &self.images.val),
            ("test", &self.images.test),
        ] {
            for entry in entries {
                result.push(self.root.join("images").join(split).join(&entry.path));
            }
        }
        result
    }
}

async fn worker_loop(
    dataset: Arc<RwLock<Option<Dataset>>>,
    change_rx: mpsc::UnboundedReceiver<()>,
    shutdown_rx: oneshot::Receiver<()>,
    ctx: eframe::egui::Context,
) {
    use futures::{FutureExt, StreamExt};

    let mut change_rx = change_rx.fuse();
    let mut shutdown_rx = shutdown_rx.fuse();
    let (image_tx, mut image_rx) = mpsc::unbounded();
    let mut watcher: Option<FsWatcher> = None;

    loop {
        futures::select! {
            change = change_rx.next() => {
                if change.is_none() {
                    break;
                }
                watcher = configure_dataset_watcher(&dataset, image_tx.clone());
                ctx.request_repaint();
            }
            entries = image_rx.next() => {
                match entries {
                    Some(images) => {
                        if let Err(err) = update_dataset_images(&dataset, images) {
                            eprintln!("Failed to update dataset images: {err}");
                        }
                        ctx.request_repaint();
                    }
                    None => break,
                }
            }
            _ = shutdown_rx => {
                break;
            }
            event = async {
                if let Some(watcher) = watcher.as_mut() {
                    watcher.event().await
                } else {
                    futures::future::pending().await
                }
            }.fuse() => {
                match event {
                    Some(Ok(event)) => {
                        if watcher::matches_event(&event) {
                            println!("Filesystem event: {event:?}");
                            if let Some(err) = watcher
                                .as_ref()
                                .and_then(|active| active.send_image_snapshot().err())
                            {
                                eprintln!("Failed to refresh dataset images: {err}");
                            }
                        }
                    }
                    Some(Err(err)) => eprintln!("Filesystem watch error: {err}"),
                    None => {
                        println!("Filesystem watcher channel closed; rebuilding");
                        watcher = configure_dataset_watcher(&dataset, image_tx.clone());
                    }
                }
            }
        }
    }
}

fn configure_dataset_watcher(
    dataset: &Arc<RwLock<Option<Dataset>>>,
    images_tx: mpsc::UnboundedSender<Vec<ImageEntry>>,
) -> Option<FsWatcher> {
    let images_dir = {
        let guard = dataset.read().ok()?;
        let ds = guard.as_ref()?;
        if let Err(err) = ds.ensure_directories() {
            eprintln!("Failed to ensure dataset directories: {err}");
        }
        ds.root.join("images")
    };

    let mut watcher = FsWatcher::new(images_dir, images_tx);
    match watcher.rebuild() {
        Ok(()) => Some(watcher),
        Err(err) => {
            if let Some(err) = err {
                eprintln!("Unable to configure watcher: {err}");
            }
            None
        }
    }
}

fn update_dataset_images(
    dataset: &Arc<RwLock<Option<Dataset>>>,
    entries: Vec<ImageEntry>,
) -> Result<(), String> {
    let mut guard = dataset
        .write()
        .map_err(|_| "Dataset lock poisoned".to_owned())?;
    if let Some(ds) = guard.as_mut() {
        ds.images = DatasetImages::from_entries(entries);
    }
    Ok(())
}

impl Drop for DirectoryWatcher {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl From<(PathBuf, YamlDoc)> for Dataset {
    fn from((root, doc): (PathBuf, YamlDoc)) -> Self {
        let YamlDoc { names, .. } = doc;
        let mut classes: Vec<ClassEntry> = names
            .into_iter()
            .map(|(id, name)| ClassEntry { id, name })
            .collect();
        classes.sort_by_key(|class| class.id);
        let mut dataset = Self {
            yaml_name: Self::dataset_name_from_path(&root),
            root,
            classes,
            new_class_name: String::new(),
            segments: Vec::new(),
            images: DatasetImages::default(),
        };
        dataset.refresh_image_lists();
        dataset
    }
}

pub struct DirectoryWatcher {
    change_tx: mpsc::UnboundedSender<()>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl DirectoryWatcher {
    pub fn new(dataset: Arc<RwLock<Option<Dataset>>>, ctx: eframe::egui::Context) -> Self {
        let (change_tx, change_rx) = mpsc::unbounded();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let handle = std::thread::spawn(move || {
            futures::executor::block_on(worker_loop(dataset, change_rx, shutdown_rx, ctx));
        });
        Self {
            change_tx,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    pub fn notify_dataset_changed(&mut self) {
        let _ = self.change_tx.unbounded_send(());
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
struct YamlDoc {
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "YamlDoc::default_path")]
    path: String,
    #[serde(default = "YamlDoc::default_train")]
    train: String,
    #[serde(default = "YamlDoc::default_val")]
    val: String,
    #[serde(default = "YamlDoc::default_test")]
    test: String,
    #[serde(default)]
    names: BTreeMap<usize, String>,
}

impl YamlDoc {
    fn default_path() -> String {
        ".".to_owned()
    }

    fn default_train() -> String {
        "images/train".to_owned()
    }

    fn default_val() -> String {
        "images/val".to_owned()
    }

    fn default_test() -> String {
        "images/test".to_owned()
    }
}

impl From<&Dataset> for YamlDoc {
    fn from(dataset: &Dataset) -> Self {
        let mut names = BTreeMap::new();
        for class in &dataset.classes {
            names.insert(class.id, class.name.clone());
        }
        Self {
            name: Some(dataset.name()),
            path: Self::default_path(),
            train: Self::default_train(),
            val: Self::default_val(),
            test: Self::default_test(),
            names,
        }
    }
}
