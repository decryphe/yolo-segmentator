use crate::watcher::{self, DatasetWatcher as FsWatcher};
use eframe::egui;
use futures::channel::{mpsc, oneshot};
use image::RgbaImage;
use serde_yaml_ng as serde_yaml;
use std::{
    collections::BTreeMap,
    ffi::OsStr,
    fmt::{self, Write as FmtWrite},
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

pub struct Dataset {
    pub root: PathBuf,
    pub yaml_name: String,
    pub classes: Vec<ClassEntry>,
    pub new_class_name: String,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImagePurpose {
    Train,
    Val,
    Test,
}

impl ImagePurpose {
    pub const ALL: [ImagePurpose; 3] = [ImagePurpose::Train, ImagePurpose::Val, ImagePurpose::Test];

    pub fn as_dir(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Train => "Training",
            Self::Val => "Validation",
            Self::Test => "Test",
        }
    }

    pub fn from_component(component: &OsStr) -> Option<Self> {
        component.to_str().and_then(|value| match value {
            "train" => Some(Self::Train),
            "val" => Some(Self::Val),
            "test" => Some(Self::Test),
            _ => None,
        })
    }
}

impl fmt::Display for ImagePurpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_dir())
    }
}

#[derive(Default)]
pub struct DatasetImages {
    pub train: Vec<ImageEntry>,
    pub val: Vec<ImageEntry>,
    pub test: Vec<ImageEntry>,
}

impl DatasetImages {
    pub fn from_paths(images_root: &Path, paths: Vec<PathBuf>, existing: DatasetImages) -> Self {
        let mut previous: BTreeMap<(ImagePurpose, PathBuf), ImageEntry> = BTreeMap::new();
        for (split, entries) in [
            (ImagePurpose::Train, existing.train),
            (ImagePurpose::Val, existing.val),
            (ImagePurpose::Test, existing.test),
        ] {
            for entry in entries {
                previous.insert((split, entry.path.clone()), entry);
            }
        }
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();
        for relative in paths {
            let mut components = relative.components();
            let Some(split_component) = components.next() else {
                continue;
            };
            let Some(split) = ImagePurpose::from_component(split_component.as_os_str()) else {
                continue;
            };
            let remainder = components.as_path();
            if remainder.as_os_str().is_empty() {
                continue;
            }
            let entry_path = remainder.to_path_buf();
            let full_path = images_root.join(&relative);
            let has_labels = full_path.with_extension("txt").exists();
            let key = (split, entry_path.clone());
            let mut entry = previous
                .remove(&key)
                .unwrap_or_else(|| ImageEntry::new(entry_path, has_labels));
            entry.has_labels = has_labels;
            match split {
                ImagePurpose::Train => train.push(entry),
                ImagePurpose::Val => val.push(entry),
                ImagePurpose::Test => test.push(entry),
            }
        }
        train.sort_by(|a, b| a.path.cmp(&b.path));
        val.sort_by(|a, b| a.path.cmp(&b.path));
        test.sort_by(|a, b| a.path.cmp(&b.path));
        Self { train, val, test }
    }

    pub fn view(&self) -> DatasetImagesView {
        DatasetImagesView {
            train: self.train.iter().map(ImageListEntry::from).collect(),
            val: self.val.iter().map(ImageListEntry::from).collect(),
            test: self.test.iter().map(ImageListEntry::from).collect(),
        }
    }

    fn entry_mut(&mut self, split: ImagePurpose, relative: &Path) -> Option<&mut ImageEntry> {
        let list = match split {
            ImagePurpose::Train => &mut self.train,
            ImagePurpose::Val => &mut self.val,
            ImagePurpose::Test => &mut self.test,
        };
        list.iter_mut().find(|entry| entry.path == relative)
    }

    fn entry(&self, split: ImagePurpose, relative: &Path) -> Option<&ImageEntry> {
        let list = match split {
            ImagePurpose::Train => &self.train,
            ImagePurpose::Val => &self.val,
            ImagePurpose::Test => &self.test,
        };
        list.iter().find(|entry| entry.path == relative)
    }
}

#[derive(Clone, Default)]
pub struct DatasetImagesView {
    pub train: Vec<ImageListEntry>,
    pub val: Vec<ImageListEntry>,
    pub test: Vec<ImageListEntry>,
}

impl DatasetImagesView {
    pub fn list(&self, split: ImagePurpose) -> &[ImageListEntry] {
        match split {
            ImagePurpose::Train => &self.train,
            ImagePurpose::Val => &self.val,
            ImagePurpose::Test => &self.test,
        }
    }
}

#[derive(Clone)]
pub struct ImageListEntry {
    pub path: PathBuf,
    pub has_labels: bool,
}

impl From<&ImageEntry> for ImageListEntry {
    fn from(entry: &ImageEntry) -> Self {
        Self {
            path: entry.path.clone(),
            has_labels: entry.has_labels,
        }
    }
}

#[derive(Clone)]
pub struct ImageReference {
    pub split: ImagePurpose,
    pub relative_path: PathBuf,
    pub full_path: PathBuf,
}

impl ImageReference {
    pub fn matches(&self, split: ImagePurpose, relative: &Path) -> bool {
        self.split == split && self.relative_path == relative
    }
}

pub struct ImageEntry {
    pub path: PathBuf,
    pub has_labels: bool,
    pub loaded: Option<LoadedImage>,
}

impl ImageEntry {
    fn new(path: PathBuf, has_labels: bool) -> Self {
        Self {
            path,
            has_labels,
            loaded: None,
        }
    }

    fn ensure_loaded(
        &mut self,
        dataset_root: &Path,
        split: ImagePurpose,
    ) -> Result<&mut LoadedImage, String> {
        if self.loaded.is_none() {
            let full_path = dataset_root
                .join("images")
                .join(split.as_dir())
                .join(&self.path);
            self.loaded = Some(LoadedImage::load(full_path)?);
        }
        Ok(self.loaded.as_mut().expect("loaded image must exist"))
    }
}

pub struct LoadedImage {
    image_path: PathBuf,
    pub pixels: RgbaImage,
    pub segments: Vec<SegmentEntry>,
    pub dirty: bool,
}

impl LoadedImage {
    fn load(image_path: PathBuf) -> Result<Self, String> {
        let pixels = image::open(&image_path)
            .map_err(|err| format!("Failed to load image {}: {err}", image_path.display()))?
            .to_rgba8();
        let segments = Self::load_segments(&image_path.with_extension("txt"))?;
        Ok(Self {
            image_path,
            pixels,
            segments,
            dirty: false,
        })
    }

    fn load_segments(txt_path: &Path) -> Result<Vec<SegmentEntry>, String> {
        if !txt_path.exists() {
            return Ok(Vec::new());
        }
        let contents = std::fs::read_to_string(txt_path)
            .map_err(|err| format!("Unable to read {}: {err}", txt_path.display()))?;
        let mut segments = Vec::new();
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut parts = trimmed.split_whitespace();
            let Some(class_index) = parts.next().and_then(|u| u.parse::<usize>().ok()) else {
                continue;
            };

            let polygon: Vec<_> = parts
                .filter_map(|p| p.parse::<f32>().ok())
                .collect::<Vec<_>>()
                .chunks_exact(2)
                .filter_map(|p| match p {
                    [x, y] => Some(SegmentPoint { x: *x, y: *y }),
                    _ => None,
                })
                .collect();

            if polygon.len() < 3 {
                continue;
            }
            segments.push(SegmentEntry {
                id: segments.len(),
                class_index,
                polygon,
            });
        }
        Ok(segments)
    }

    fn save_segments(&self) -> Result<usize, String> {
        let mut lines = Vec::new();
        for segment in &self.segments {
            if segment.polygon.len() < 3 {
                continue;
            }
            let mut line = segment.class_index.to_string();
            for point in &segment.polygon {
                let _ = write!(&mut line, " {:.6} {:.6}", point.x, point.y);
            }
            lines.push(line);
        }

        let txt_path = self.image_path.with_extension("txt");
        if let Some(parent) = txt_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create directory {}: {err}", parent.display()))?;
        }
        std::fs::write(&txt_path, lines.join("\n"))
            .map_err(|err| format!("Failed to write {}: {err}", txt_path.display()))?;
        Ok(lines.len())
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn next_segment_id(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.id)
            .max()
            .map_or(0, |max| max + 1)
    }

    pub fn as_color_image(&self) -> egui::ColorImage {
        let size = [self.pixels.width() as usize, self.pixels.height() as usize];
        egui::ColorImage::from_rgba_unmultiplied(size, self.pixels.as_raw())
    }
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

    fn dataset_name_from_path(path: &Path) -> String {
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("dataset")
            .to_owned()
    }

    pub fn refresh_image_lists(&mut self) {
        let images_dir = self.root.join("images");
        let entries = watcher::gather_images(&images_dir);
        let previous = std::mem::take(&mut self.images);
        self.images = DatasetImages::from_paths(&images_dir, entries, previous);
    }

    pub fn images_view(&self) -> DatasetImagesView {
        self.images.view()
    }

    pub fn ensure_image_loaded(
        &mut self,
        split: ImagePurpose,
        relative: &Path,
    ) -> Result<&mut LoadedImage, String> {
        let entry = self
            .images
            .entry_mut(split, relative)
            .ok_or_else(|| format!("Unable to find image {split}/{}", relative.display()))?;
        entry.ensure_loaded(&self.root, split)
    }

    pub fn segments_snapshot(
        &self,
        split: ImagePurpose,
        relative: &Path,
    ) -> Option<Vec<SegmentEntry>> {
        self.images
            .entry(split, relative)
            .and_then(|entry| entry.loaded.as_ref().map(|img| img.segments.clone()))
    }

    pub fn save_loaded_image(
        &mut self,
        split: ImagePurpose,
        relative: &Path,
    ) -> Result<usize, String> {
        let entry = self
            .images
            .entry_mut(split, relative)
            .ok_or_else(|| format!("Unable to find image {split}/{}", relative.display()))?;
        let loaded = entry
            .loaded
            .as_mut()
            .ok_or_else(|| "Image is not loaded".to_owned())?;
        let count = loaded.save_segments()?;
        loaded.clear_dirty();
        entry.has_labels = count > 0;
        Ok(count)
    }

    pub fn flattened_image_refs(&self) -> Vec<ImageReference> {
        let mut result = Vec::new();
        for (split, entries) in [
            (ImagePurpose::Train, &self.images.train),
            (ImagePurpose::Val, &self.images.val),
            (ImagePurpose::Test, &self.images.test),
        ] {
            for entry in entries {
                let full_path = self
                    .root
                    .join("images")
                    .join(split.as_dir())
                    .join(&entry.path);
                result.push(ImageReference {
                    split,
                    relative_path: entry.path.clone(),
                    full_path,
                });
            }
        }
        result
    }

    pub fn image_reference(&self, split: ImagePurpose, relative: &Path) -> ImageReference {
        ImageReference {
            split,
            relative_path: relative.to_path_buf(),
            full_path: self.root.join("images").join(split.as_dir()).join(relative),
        }
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
    let (image_tx, mut image_rx) = mpsc::unbounded::<Vec<PathBuf>>();
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
                        if watcher::matches_event(&event)
                            && let Some(err) = watcher
                                .as_ref()
                                .and_then(|active| active.send_image_snapshot().err())
                        {
                            eprintln!("Failed to refresh dataset images: {err}");
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
    images_tx: mpsc::UnboundedSender<Vec<PathBuf>>,
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
    entries: Vec<PathBuf>,
) -> Result<(), String> {
    let mut guard = dataset
        .write()
        .map_err(|_| "Dataset lock poisoned".to_owned())?;
    if let Some(ds) = guard.as_mut() {
        let images_dir = ds.root.join("images");
        let previous = std::mem::take(&mut ds.images);
        ds.images = DatasetImages::from_paths(&images_dir, entries, previous);
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
