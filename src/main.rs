#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::egui;
use futures::{
    FutureExt, SinkExt, StreamExt,
    channel::{mpsc, oneshot},
    executor,
};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_yaml_ng as serde_yaml;
use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Write as FmtWrite,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    thread,
};

#[derive(Default, Clone)]
struct ClassEntry {
    id: usize,
    name: String,
}

#[derive(Clone)]
struct SegmentPoint {
    x: f32,
    y: f32,
}

impl Default for SegmentPoint {
    fn default() -> Self {
        Self { x: 0.5, y: 0.5 }
    }
}

#[derive(Clone)]
struct SegmentEntry {
    id: usize,
    class_index: usize,
    polygon: Vec<SegmentPoint>,
}

impl SegmentEntry {
    fn new(id: usize, class_index: usize) -> Self {
        Self {
            id,
            class_index,
            polygon: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct Dataset {
    root: PathBuf,
    yaml_name: Option<String>,
    classes: Vec<ClassEntry>,
    new_class_name: String,
    segments: Vec<SegmentEntry>,
    images: DatasetImages,
}

#[derive(Clone, Default)]
struct DatasetImages {
    train: Vec<ImageEntry>,
    val: Vec<ImageEntry>,
    test: Vec<ImageEntry>,
}

#[derive(Clone)]
struct ImageEntry {
    path: String,
    has_labels: bool,
}

struct DatasetWorker {
    change_tx: mpsc::UnboundedSender<()>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
}

struct DatasetWatcher {
    dataset: Arc<RwLock<Option<Dataset>>>,
    watcher: Option<RecommendedWatcher>,
    event_rx: Option<mpsc::Receiver<notify::Result<Event>>>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct DatasetYamlDoc {
    #[serde(default)]
    name: Option<String>,
    #[serde(default = "DatasetYamlDoc::default_path")]
    path: String,
    #[serde(default = "DatasetYamlDoc::default_train")]
    train: String,
    #[serde(default = "DatasetYamlDoc::default_val")]
    val: String,
    #[serde(default = "DatasetYamlDoc::default_test")]
    test: String,
    #[serde(default)]
    names: BTreeMap<usize, String>,
}

impl DatasetYamlDoc {
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

impl From<&Dataset> for DatasetYamlDoc {
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

impl Dataset {
    fn new(root: PathBuf) -> Self {
        let classes = Self::default_classes();
        let dataset_name = Some(Self::dataset_name_from_path(&root));
        let mut dataset = Self {
            root,
            yaml_name: dataset_name,
            classes,
            new_class_name: String::new(),
            segments: Vec::new(),
            images: DatasetImages::default(),
        };
        dataset.refresh_image_lists();
        dataset
    }

    fn load(root: PathBuf) -> Result<Self, String> {
        let yaml_path = root.join("dataset.yaml");
        let contents = fs::read_to_string(&yaml_path)
            .map_err(|err| format!("Unable to read dataset.yaml: {err}"))?;
        let doc: DatasetYamlDoc = serde_yaml::from_str(&contents)
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

    fn name(&self) -> String {
        self.yaml_name
            .clone()
            .unwrap_or_else(|| Self::dataset_name_from_path(&self.root))
    }

    fn initialize_on_disk(&self) -> Result<(), String> {
        self.ensure_directories()?;
        self.save_metadata()
    }

    fn ensure_directories(&self) -> Result<(), String> {
        if !self.root.exists() {
            fs::create_dir_all(&self.root)
                .map_err(|err| format!("Unable to create dataset directory: {err}"))?;
        }
        let images_dir = self.root.join("images");
        for split in ["train", "val", "test"] {
            fs::create_dir_all(images_dir.join(split))
                .map_err(|err| format!("Unable to create images/{split} subdirectory: {err}"))?;
        }
        Ok(())
    }

    fn save_metadata(&self) -> Result<(), String> {
        let yaml_doc = DatasetYamlDoc::from(self);
        let serialized = serde_yaml::to_string(&yaml_doc)
            .map_err(|err| format!("Unable to serialize dataset.yaml: {err}"))?;
        let yaml_path = self.root.join("dataset.yaml");
        fs::write(&yaml_path, serialized)
            .map_err(|err| format!("Unable to write dataset.yaml: {err}"))?;
        Ok(())
    }

    fn next_class_id(&self) -> usize {
        self.classes
            .iter()
            .map(|c| c.id)
            .max()
            .map_or(0, |max| max + 1)
    }

    fn next_segment_id(&self) -> usize {
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

    fn refresh_image_lists(&mut self) {
        self.images = Self::gather_images(&self.root);
    }

    fn load_segments_for_image(&mut self, image_path: &Path) -> Result<usize, String> {
        let txt_path = image_path.with_extension("txt");
        self.segments.clear();
        if !txt_path.exists() {
            return Ok(0);
        }
        let contents = fs::read_to_string(&txt_path)
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

    fn gather_images(root: &Path) -> DatasetImages {
        DatasetImages {
            train: Self::collect_images(&root.join("images/train")),
            val: Self::collect_images(&root.join("images/val")),
            test: Self::collect_images(&root.join("images/test")),
        }
    }

    fn collect_images(dir: &Path) -> Vec<ImageEntry> {
        let mut files = Vec::new();
        if !dir.exists() {
            return files;
        }
        let mut queue = VecDeque::from([dir.to_path_buf()]);
        while let Some(path) = queue.pop_front() {
            let Ok(read_dir) = fs::read_dir(&path) else {
                continue;
            };
            for entry in read_dir.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    queue.push_back(entry_path);
                } else if is_supported_image(&entry_path) {
                    let rel = entry_path
                        .strip_prefix(dir)
                        .unwrap_or(&entry_path)
                        .display()
                        .to_string();
                    let has_labels = entry_path.with_extension("txt").exists();
                    files.push(ImageEntry {
                        path: rel,
                        has_labels,
                    });
                }
            }
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files
    }

    fn flattened_image_paths(&self) -> Vec<PathBuf> {
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

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
}

impl DatasetWorker {
    fn new(dataset: Arc<RwLock<Option<Dataset>>>, ctx: egui::Context) -> Self {
        let (change_tx, change_rx) = mpsc::unbounded();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let handle = thread::spawn(move || {
            futures::executor::block_on(worker_loop(dataset, change_rx, shutdown_rx, ctx));
        });
        Self {
            change_tx,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    fn notify_dataset_changed(&mut self) {
        let _ = self.change_tx.unbounded_send(());
    }
}

enum PendingAction {
    SaveOnly,
    Navigate(PathBuf),
}

#[derive(Clone, Copy)]
enum NavDirection {
    Previous,
    Next,
}

impl DatasetWatcher {
    fn new(dataset: Arc<RwLock<Option<Dataset>>>) -> Self {
        Self {
            dataset,
            watcher: None,
            event_rx: None,
        }
    }

    fn rebuild(&mut self) {
        match self.configure_dataset_watcher() {
            Ok((watcher, rx)) => {
                println!("Watching dataset images for changes");
                self.watcher = Some(watcher);
                self.event_rx = Some(rx);
                if let Err(err) = self.refresh_dataset_images() {
                    eprintln!("Failed to refresh dataset images: {err}");
                }
            }
            Err(err) => {
                if let Some(err) = err {
                    eprintln!("Unable to configure watcher: {err}");
                }
                self.watcher = None;
                self.event_rx = None;
            }
        }
    }

    fn configure_dataset_watcher(
        &self,
    ) -> Result<(RecommendedWatcher, mpsc::Receiver<notify::Result<Event>>), Option<String>> {
        let path = {
            let guard = self
                .dataset
                .read()
                .map_err(|_| Some("Dataset lock poisoned".to_owned()))?;
            let Some(ds) = guard.as_ref() else {
                return Err(None);
            };
            if let Err(err) = ds.ensure_directories() {
                eprintln!("Failed to ensure dataset directories: {err}");
            }
            ds.root.join("images")
        };

        Self::create_async_watcher(&path).map_err(|err| Some(err.to_string()))
    }

    fn create_async_watcher(
        path: &Path,
    ) -> notify::Result<(RecommendedWatcher, mpsc::Receiver<notify::Result<Event>>)> {
        if !path.exists()
            && let Err(err) = fs::create_dir_all(path)
        {
            eprintln!("Failed to create directory {}: {err}", path.display());
        }
        let (mut tx, rx) = mpsc::channel(128);
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                executor::block_on(async {
                    let _ = tx.send(res).await;
                });
            },
            Config::default(),
        )?;
        watcher.watch(path, RecursiveMode::Recursive)?;
        println!("Configured watcher on {}", path.display());
        Ok((watcher, rx))
    }

    async fn event(&mut self) -> Option<notify::Result<Event>> {
        if let Some(stream) = self.event_rx.as_mut() {
            stream.next().await
        } else {
            futures::future::pending().await
        }
    }

    fn refresh_dataset_images(&self) -> Result<(), String> {
        let root = {
            let guard = self
                .dataset
                .read()
                .map_err(|_| "Dataset lock poisoned".to_owned())?;
            match guard.as_ref() {
                Some(dataset) => dataset.root.clone(),
                None => return Ok(()),
            }
        };
        let images = Dataset::gather_images(&root);
        let mut guard = self
            .dataset
            .write()
            .map_err(|_| "Dataset lock poisoned".to_owned())?;
        if let Some(dataset) = guard.as_mut() {
            dataset.images = images;
        }
        Ok(())
    }
}

async fn worker_loop(
    dataset: Arc<RwLock<Option<Dataset>>>,
    change_rx: mpsc::UnboundedReceiver<()>,
    shutdown_rx: oneshot::Receiver<()>,
    ctx: egui::Context,
) {
    let mut change_rx = change_rx.fuse();
    let mut shutdown_rx = shutdown_rx.fuse();
    let mut watcher = DatasetWatcher::new(dataset);

    loop {
        futures::select! {
            change = change_rx.next() => {
                if change.is_none() {
                    break;
                }
                watcher.rebuild();
                ctx.request_repaint();
            }
            _ = shutdown_rx => {
                break;
            }
            event = watcher.event().fuse() => {
                match event {
                    Some(Ok(event)) => {
                        if matches_event(&event) {
                            println!("Filesystem event: {event:?}");
                            if let Err(err) = watcher.refresh_dataset_images() {
                                eprintln!("Failed to refresh dataset images: {err}");
                            }
                            ctx.request_repaint();
                        }
                    }
                    Some(Err(err)) => eprintln!("Filesystem watch error: {err}"),
                    None => {
                        println!("Filesystem watcher channel closed; rebuilding");
                        watcher.rebuild();
                    }
                }
            }
        }
    }
}

fn matches_event(event: &Event) -> bool {
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

impl Drop for DatasetWorker {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl From<(PathBuf, DatasetYamlDoc)> for Dataset {
    fn from((root, doc): (PathBuf, DatasetYamlDoc)) -> Self {
        let DatasetYamlDoc { name, names, .. } = doc;
        let mut classes: Vec<ClassEntry> = names
            .into_iter()
            .map(|(id, name)| ClassEntry { id, name })
            .collect();
        classes.sort_by_key(|class| class.id);
        let mut dataset = Self {
            yaml_name: name.or_else(|| Some(Self::dataset_name_from_path(&root))),
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

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 880.0]),
        ..Default::default()
    };
    eframe::run_native(
        "YOLO Segmentator",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(MyApp::new(cc.egui_ctx.clone())))
        }),
    )
}

struct MyApp {
    dataset: Arc<RwLock<Option<Dataset>>>,
    worker: DatasetWorker,
    status_message: Option<String>,
    current_image: Option<(PathBuf, egui::TextureHandle)>,
    selected_segment_id: Option<usize>,
    data_dirty: bool,
    show_save_prompt: bool,
    pending_action: Option<PendingAction>,
}

impl MyApp {
    fn new(ctx: egui::Context) -> Self {
        let dataset = Arc::new(RwLock::new(None));
        let worker = DatasetWorker::new(dataset.clone(), ctx);
        Self {
            dataset,
            worker,
            status_message: None,
            current_image: None,
            selected_segment_id: None,
            data_dirty: false,
            show_save_prompt: false,
            pending_action: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.toolbar(ui);
        });

        if self.has_dataset() && !self.show_save_prompt {
            let mut nav_request = None;
            ctx.input(|input| {
                if input.key_pressed(egui::Key::ArrowLeft) {
                    nav_request = Some(NavDirection::Previous);
                } else if input.key_pressed(egui::Key::ArrowRight) {
                    nav_request = Some(NavDirection::Next);
                }
            });
            if let Some(dir) = nav_request {
                self.navigate_image(dir, ctx);
            }
        }

        egui::TopBottomPanel::bottom("status_bar")
            .resizable(false)
            .show(ctx, |ui| {
                if let Some(status) = &self.status_message {
                    ui.horizontal(|ui| {
                        ui.label(status);
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.label("Ready");
                    });
                }
            });

        if self.has_dataset() {
            egui::SidePanel::left("class_panel")
                .resizable(true)
                .default_width(220.0)
                .show(ctx, |ui| self.class_panel(ui));

            egui::SidePanel::right("segment_panel")
                .resizable(true)
                .default_width(260.0)
                .show(ctx, |ui| self.segment_panel(ui));

            egui::CentralPanel::default().show(ctx, |ui| {
                self.center_panel(ui);
            });
        }

        if self.show_save_prompt {
            egui::Window::new("Save Changes?")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("Would you like to save the changes?");
                    ui.horizontal(|ui| {
                        if ui.button("Yes").clicked() {
                            match self.save_current_image_segments() {
                                Ok(()) => {
                                    self.data_dirty = false;
                                    self.show_save_prompt = false;
                                    self.execute_pending_action(ctx);
                                }
                                Err(err) => {
                                    self.status_message = Some(err);
                                }
                            }
                        }
                        if ui.button("No").clicked() {
                            self.data_dirty = false;
                            self.show_save_prompt = false;
                            self.execute_pending_action(ctx);
                        }
                    });
                });
        }
    }
}

impl MyApp {
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.horizontal_wrapped(|ui| {
                if self.has_dataset() {
                    if ui.button("Close Dataset").clicked() {
                        self.handle_close_dataset();
                    }
                } else if ui.button("New Dataset").clicked() {
                    self.handle_new_dataset();
                }

                if ui.button("Open Directory").clicked() {
                    self.handle_open_dataset();
                }

                if ui.button("Save").clicked() {
                    self.handle_save_dataset();
                }

                if ui.button("Previous").clicked() {
                    self.navigate_image(NavDirection::Previous, ui.ctx());
                }
                if ui.button("Next").clicked() {
                    self.navigate_image(NavDirection::Next, ui.ctx());
                }
            });
        });
    }

    fn class_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Dataset");
        let dataset_overview = {
            let guard = self.dataset.read().expect("dataset lock poisoned");
            guard
                .as_ref()
                .map(|dataset| (dataset.root.clone(), dataset.name(), dataset.images.clone()))
        };
        let Some((root_path, dataset_name, images)) = dataset_overview else {
            ui.label("(No dataset open)");
            return;
        };
        ui.label(format!("Path: {}", root_path.display()));
        ui.label(format!("Name: {dataset_name}"));
        ui.separator();

        ui.heading("Classes");
        ui.separator();

        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open or create a dataset to edit classes.");
            return;
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            if dataset.classes.is_empty() {
                ui.label("No classes defined yet");
            }
            egui::Grid::new("class_grid")
                .striped(true)
                .num_columns(2)
                .show(ui, |ui| {
                    for entry in &mut dataset.classes {
                        ui.label(format!("ID {:02}", entry.id));
                        ui.text_edit_singleline(&mut entry.name);
                        ui.end_row();
                    }
                });
        });

        ui.separator();
        ui.vertical_centered(|ui| {
            ui.label("Add New Class");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut dataset.new_class_name);
                let name_ready = !dataset.new_class_name.trim().is_empty();
                if ui
                    .add_enabled(name_ready, egui::Button::new("Add"))
                    .clicked()
                    && name_ready
                {
                    let new_id = dataset.next_class_id();
                    dataset.classes.push(ClassEntry {
                        id: new_id,
                        name: dataset.new_class_name.trim().to_owned(),
                    });
                    dataset.new_class_name.clear();
                }
            });
        });
        drop(dataset_guard);

        ui.separator();
        ui.heading("Images");
        for (title, list, split_dir) in [
            ("Training", images.train.as_slice(), "train"),
            ("Validation", images.val.as_slice(), "val"),
            ("Test", images.test.as_slice(), "test"),
        ] {
            ui.label(egui::RichText::new(title).strong());
            if list.is_empty() {
                ui.label("  (no images)");
            } else {
                egui::Grid::new(format!("image_grid_{title}"))
                    .striped(true)
                    .num_columns(3)
                    .show(ui, |ui| {
                        for entry in list {
                            let full_path =
                                root_path.join("images").join(split_dir).join(&entry.path);
                            ui.label(&entry.path);
                            ui.label(if entry.has_labels { "" } else { "(new)" });
                            if ui.small_button("Load").clicked() {
                                self.load_image(ui.ctx(), &full_path);
                            }
                            ui.end_row();
                        }
                    });
            }
            ui.add_space(4.0);
        }
    }

    fn handle_new_dataset(&mut self) {
        let Some(selected_dir) = FileDialog::new()
            .set_title("Create or select dataset folder")
            .pick_folder()
        else {
            self.status_message = Some("Dataset creation canceled".to_owned());
            return;
        };

        let dataset = Dataset::new(selected_dir);
        let dataset_root = dataset.root.clone();
        match dataset.initialize_on_disk() {
            Ok(()) => match Dataset::load(dataset_root.clone()) {
                Ok(loaded) => {
                    let dataset_path = dataset_root.display().to_string();
                    {
                        let mut guard = self.dataset.write().expect("dataset lock poisoned");
                        *guard = Some(loaded);
                    }
                    self.current_image = None;
                    self.selected_segment_id = None;
                    self.data_dirty = false;
                    self.show_save_prompt = false;
                    self.worker.notify_dataset_changed();
                    self.status_message = Some(format!(
                        "New dataset created at {dataset_path} and ready for editing"
                    ));
                }
                Err(err) => {
                    self.status_message = Some(format!(
                        "Created dataset but failed to reload metadata: {err}"
                    ));
                }
            },
            Err(err) => {
                self.status_message = Some(format!("Failed to create dataset: {err}"));
            }
        }
    }

    fn handle_open_dataset(&mut self) {
        let Some(selected_dir) = FileDialog::new()
            .set_title("Select dataset folder")
            .pick_folder()
        else {
            self.status_message = Some("Dataset loading canceled".to_owned());
            return;
        };

        match Dataset::load(selected_dir.clone()) {
            Ok(dataset) => {
                let dataset_path = selected_dir.display().to_string();
                {
                    let mut guard = self.dataset.write().expect("dataset lock poisoned");
                    *guard = Some(dataset);
                }
                self.current_image = None;
                self.selected_segment_id = None;
                self.data_dirty = false;
                self.show_save_prompt = false;
                self.worker.notify_dataset_changed();
                self.status_message = Some(format!("Loaded dataset from {dataset_path}"));
            }
            Err(err) => {
                self.status_message = Some(format!("Failed to load dataset: {err}"));
            }
        }
    }

    fn handle_close_dataset(&mut self) {
        let closed_path = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            guard
                .take()
                .map(|dataset| dataset.root.display().to_string())
        };

        if let Some(path) = closed_path {
            self.status_message = Some(format!("Closed dataset {path}"));
        } else {
            self.status_message = Some("No dataset open".to_owned());
        }
        self.current_image = None;
        self.selected_segment_id = None;
        self.data_dirty = false;
        self.show_save_prompt = false;
        self.worker.notify_dataset_changed();
    }

    fn handle_save_dataset(&mut self) {
        if self.data_dirty {
            self.pending_action = Some(PendingAction::SaveOnly);
            self.show_save_prompt = true;
        } else if let Err(err) = self.save_current_image_segments() {
            self.status_message = Some(err);
        }
    }

    fn segment_panel(&mut self, ui: &mut egui::Ui) {
        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open a dataset to manage segments");
            return;
        };

        ui.horizontal(|ui| {
            ui.heading("Segments");
            if ui.button("New Segment").clicked() {
                let default_class = dataset.classes.first().map_or(0, |c| c.id);
                let next_id = dataset.next_segment_id();
                dataset
                    .segments
                    .push(SegmentEntry::new(next_id, default_class));
                self.selected_segment_id = Some(next_id);
                self.data_dirty = true;
                self.status_message = Some(format!("Created segment #{next_id}"));
            }
        });
        ui.separator();

        if dataset.segments.is_empty() {
            ui.label("No segments created yet");
            return;
        }

        let classes_snapshot = dataset.classes.clone();
        let mut segment_to_remove = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("segment_grid")
                .striped(true)
                .num_columns(5)
                .show(ui, |ui| {
                    for (idx, segment) in dataset.segments.iter_mut().enumerate() {
                        ui.label(format!("#{:02}", segment.id));
                        let mut class_changed = false;
                        egui::ComboBox::from_id_salt(("segment_class", segment.id))
                            .selected_text(Self::class_label_from_classes(
                                segment.class_index,
                                &classes_snapshot,
                            ))
                            .show_ui(ui, |ui| {
                                for class in &classes_snapshot {
                                    if ui
                                        .selectable_value(
                                            &mut segment.class_index,
                                            class.id,
                                            format!("{} - {}", class.id, class.name),
                                        )
                                        .clicked()
                                    {
                                        class_changed = true;
                                    }
                                }
                            });
                        if class_changed {
                            self.data_dirty = true;
                        }
                        ui.label(format!("Pts: {}", segment.polygon.len()));
                        if ui.small_button("Edit").clicked() {
                            self.selected_segment_id = Some(segment.id);
                        }
                        if ui.small_button("Delete").clicked() {
                            segment_to_remove = Some(idx);
                        }
                        ui.end_row();
                    }
                });
        });

        if let Some(idx) = segment_to_remove.filter(|&i| i < dataset.segments.len()) {
            let removed = dataset.segments.remove(idx);
            if self.selected_segment_id == Some(removed.id) {
                self.selected_segment_id = None;
            }
            self.data_dirty = true;
        }
    }

    fn class_label_from_classes(class_id: usize, classes: &[ClassEntry]) -> String {
        classes.iter().find(|c| c.id == class_id).map_or_else(
            || format!("{class_id} - Unknown"),
            |c| format!("{} - {}", c.id, c.name),
        )
    }

    #[allow(clippy::cast_precision_loss)]
    fn center_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Image");
        ui.separator();
        if let Some((path, texture)) = &self.current_image {
            let segments_snapshot = self
                .dataset
                .read()
                .ok()
                .and_then(|guard| guard.as_ref().map(|d| d.segments.clone()))
                .unwrap_or_default();
            if let Some(selected) = self.selected_segment_id {
                ui.label(format!("Editing segment #{selected}"));
            } else {
                ui.label("Select a segment to edit");
            }
            ui.label(path.display().to_string());
            let texture_id = texture.id();
            let texture_size = texture.size_vec2();
            let available = ui.available_size();
            let scale = (available.x / texture_size.x)
                .min(available.y / texture_size.y)
                .clamp(0.01, 1.0);
            let final_size = texture_size * scale;
            let image_widget = egui::widgets::Image::new((texture_id, texture_size))
                .fit_to_exact_size(final_size)
                .sense(egui::Sense::click());
            let response = ui.add(image_widget);
            let rect = response.rect;
            let painter = ui.painter_at(rect);
            for segment in &segments_snapshot {
                let points: Vec<egui::Pos2> = segment
                    .polygon
                    .iter()
                    .map(|pt| {
                        egui::Pos2::new(
                            rect.left() + pt.x * rect.width(),
                            rect.top() + pt.y * rect.height(),
                        )
                    })
                    .collect();
                let is_selected = Some(segment.id) == self.selected_segment_id;
                let (fill, stroke_color) = Self::segment_colors(segment.id, is_selected);
                match points.len() {
                    0 | 1 => {}
                    2 => {
                        painter.add(egui::epaint::Shape::line_segment(
                            [points[0], points[1]],
                            egui::Stroke::new(2.0, stroke_color),
                        ));
                        let midpoint = egui::pos2(
                            (points[0].x + points[1].x) * 0.5,
                            (points[0].y + points[1].y) * 0.5,
                        );
                        painter.text(
                            midpoint,
                            egui::Align2::CENTER_CENTER,
                            format!("#{}", segment.id),
                            egui::FontId::proportional(14.0),
                            stroke_color,
                        );
                    }
                    _ => {
                        painter.add(egui::epaint::PathShape::convex_polygon(
                            points.clone(),
                            fill,
                            egui::epaint::Stroke::new(2.0, stroke_color),
                        ));
                        let (sum_x, sum_y) = points
                            .iter()
                            .fold((0.0, 0.0), |acc, p| (acc.0 + p.x, acc.1 + p.y));
                        let len = points.len() as f32;
                        let centroid = egui::pos2(sum_x / len, sum_y / len);
                        painter.text(
                            centroid,
                            egui::Align2::CENTER_CENTER,
                            format!("#{}", segment.id),
                            egui::FontId::proportional(14.0),
                            stroke_color,
                        );
                    }
                }

                for point in &points {
                    let rect = egui::Rect::from_center_size(*point, egui::vec2(6.0, 6.0));
                    painter.add(egui::epaint::Shape::rect_filled(rect, 1.0, stroke_color));
                }
            }

            if response.clicked()
                && let Some(pos) = response.interact_pointer_pos()
                && rect.width() > 0.0
                && rect.height() > 0.0
            {
                let rel_x = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let rel_y = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                self.add_point_to_selected_segment(rel_x, rel_y);
            }
        } else {
            ui.label("Select an image from the left pane to begin annotating.");
        }
    }

    fn has_dataset(&self) -> bool {
        self.dataset
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    fn load_image(&mut self, ctx: &egui::Context, path: &Path) {
        match image::open(path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                let texture = ctx.load_texture(
                    format!("loaded_image_{}", path.display()),
                    color_image,
                    egui::TextureOptions::LINEAR,
                );
                self.current_image = Some((path.to_path_buf(), texture));
                let segment_count = {
                    let mut guard = self.dataset.write().expect("dataset lock poisoned");
                    if let Some(dataset) = guard.as_mut() {
                        match dataset.load_segments_for_image(path) {
                            Ok(count) => Some(count),
                            Err(err) => {
                                self.status_message = Some(err);
                                None
                            }
                        }
                    } else {
                        None
                    }
                };
                self.selected_segment_id = None;
                self.data_dirty = false;
                if let Some(count) = segment_count {
                    self.status_message = Some(format!(
                        "Loaded image {} with {} segment(s)",
                        path.display(),
                        count
                    ));
                } else {
                    self.status_message = Some(format!("Loaded image {}", path.display()));
                }
            }
            Err(err) => {
                self.status_message =
                    Some(format!("Failed to load image {}: {err}", path.display()));
            }
        }
    }

    fn navigate_image(&mut self, direction: NavDirection, ctx: &egui::Context) {
        let Some(target) = self.next_image_path(direction) else {
            self.status_message = Some("No images available".to_owned());
            return;
        };
        if self.data_dirty {
            self.pending_action = Some(PendingAction::Navigate(target));
            self.show_save_prompt = true;
        } else {
            self.load_image(ctx, &target);
        }
    }

    fn next_image_path(&self, direction: NavDirection) -> Option<PathBuf> {
        let guard = self.dataset.read().ok()?;
        let dataset = guard.as_ref()?;
        let images = dataset.flattened_image_paths();
        if images.is_empty() {
            return None;
        }
        let current_idx = self
            .current_image
            .as_ref()
            .and_then(|(path, _)| images.iter().position(|p| p == path));
        let target_idx = match current_idx {
            Some(idx) => match direction {
                NavDirection::Previous => (idx + images.len() - 1) % images.len(),
                NavDirection::Next => (idx + 1) % images.len(),
            },
            None => match direction {
                NavDirection::Previous => images.len() - 1,
                NavDirection::Next => 0,
            },
        };
        images.get(target_idx).cloned()
    }

    fn execute_pending_action(&mut self, ctx: &egui::Context) {
        if let Some(action) = self.pending_action.take() {
            match action {
                PendingAction::SaveOnly => {}
                PendingAction::Navigate(path) => {
                    self.load_image(ctx, &path);
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn segment_colors(id: usize, selected: bool) -> (egui::Color32, egui::Color32) {
        let hue = (((id as f32) * 137.5) % 360.0) / 360.0;
        let fill = hsva_to_color32(egui::epaint::Hsva::new(
            hue,
            0.7,
            0.9,
            if selected { 0.45 } else { 0.2 },
        ));
        let stroke = hsva_to_color32(egui::epaint::Hsva::new(hue, 0.8, 0.7, 1.0));
        (fill, stroke)
    }

    fn add_point_to_selected_segment(&mut self, x: f32, y: f32) {
        let Some(segment_id) = self.selected_segment_id else {
            self.status_message = Some("Select a segment before adding points".to_owned());
            return;
        };
        let mut guard = self.dataset.write().expect("dataset lock poisoned");
        if let Some(dataset) = guard.as_mut() {
            if let Some(segment) = dataset.segments.iter_mut().find(|seg| seg.id == segment_id) {
                segment.polygon.push(SegmentPoint { x, y });
                self.data_dirty = true;
                self.status_message = Some(format!(
                    "Added point ({x:.3}, {y:.3}) to segment #{segment_id}"
                ));
            } else {
                self.status_message = Some("Selected segment no longer exists".to_owned());
                self.selected_segment_id = None;
            }
        }
    }

    fn save_current_image_segments(&mut self) -> Result<(), String> {
        let current_image_path = self
            .current_image
            .as_ref()
            .map(|(path, _)| path.clone())
            .ok_or_else(|| "Load an image before saving segments".to_owned())?;

        let segments = {
            let guard = self
                .dataset
                .read()
                .map_err(|_| "Dataset lock poisoned".to_owned())?;
            let dataset = guard
                .as_ref()
                .ok_or_else(|| "No dataset loaded".to_owned())?;
            dataset
                .save_metadata()
                .map_err(|err| format!("Failed to save dataset metadata: {err}"))?;
            dataset.segments.clone()
        };

        let mut lines = Vec::new();
        for segment in segments {
            if segment.polygon.len() < 3 {
                continue;
            }
            let mut line = segment.class_index.to_string();
            for point in &segment.polygon {
                let _ = write!(&mut line, " {:.6} {:.6}", point.x, point.y);
            }
            lines.push(line);
        }

        let txt_path = current_image_path.with_extension("txt");
        if let Some(parent) = txt_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("Failed to create directory {}: {err}", parent.display()))?;
        }
        fs::write(&txt_path, lines.join("\n"))
            .map_err(|err| format!("Failed to write {}: {err}", txt_path.display()))?;

        {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            if let Some(dataset) = guard.as_mut() {
                dataset.refresh_image_lists();
            }
        }

        self.status_message = Some(format!(
            "Saved {} segment(s) to {}",
            lines.len(),
            txt_path.display()
        ));
        self.data_dirty = false;
        Ok(())
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hsva_to_color32(hsva: egui::epaint::Hsva) -> egui::Color32 {
    let [r, g, b, a] = hsva.to_rgba_unmultiplied();
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgba_unmultiplied(to_u8(r), to_u8(g), to_u8(b), to_u8(a))
}
