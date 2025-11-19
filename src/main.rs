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
        // YOLO segmentation labels require at least three normalized polygon points.
        Self {
            id,
            class_index,
            polygon: vec![
                SegmentPoint { x: 0.45, y: 0.45 },
                SegmentPoint { x: 0.55, y: 0.45 },
                SegmentPoint { x: 0.50, y: 0.55 },
            ],
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
    train: Vec<String>,
    val: Vec<String>,
    test: Vec<String>,
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

    fn gather_images(root: &Path) -> DatasetImages {
        DatasetImages {
            train: Self::collect_images(&root.join("images/train")),
            val: Self::collect_images(&root.join("images/val")),
            test: Self::collect_images(&root.join("images/test")),
        }
    }

    fn collect_images(dir: &Path) -> Vec<String> {
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
                    files.push(rel);
                }
            }
        }
        files.sort();
        files
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
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.toolbar(ui);
        });

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

                for label in ["Previous", "Next"] {
                    if ui.button(label).clicked() {
                        // Future: hook into actual dataset actions.
                    }
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
            for entry in &mut dataset.classes {
                ui.horizontal(|ui| {
                    ui.label(format!("ID {:02}", entry.id));
                    ui.text_edit_singleline(&mut entry.name);
                });
            }
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
                for entry in list {
                    let full_path = root_path.join("images").join(split_dir).join(entry);
                    if ui.small_button(entry).clicked() {
                        self.load_image(ui.ctx(), &full_path);
                    }
                }
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
        self.worker.notify_dataset_changed();
    }

    fn handle_save_dataset(&mut self) {
        let guard = self.dataset.read().expect("dataset lock poisoned");
        match guard.as_ref() {
            Some(dataset) => match dataset.save_metadata() {
                Ok(()) => {
                    self.status_message = Some(format!(
                        "Saved dataset metadata to {}",
                        dataset.root.join("dataset.yaml").display()
                    ));
                }
                Err(err) => {
                    self.status_message = Some(format!("Failed to save dataset: {err}"));
                }
            },
            None => {
                self.status_message = Some("No dataset to save".to_owned());
            }
        }
    }

    fn segment_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Segments");
            if ui.button("New Segment").clicked() {
                let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
                if let Some(dataset) = dataset_guard.as_mut() {
                    let default_class = dataset.classes.first().map_or(0, |c| c.id);
                    let next_id = dataset.next_segment_id();
                    dataset
                        .segments
                        .push(SegmentEntry::new(next_id, default_class));
                } else {
                    self.status_message = Some("Open a dataset before adding segments".to_owned());
                }
            }
        });
        ui.separator();

        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open a dataset to manage segments");
            return;
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            if dataset.segments.is_empty() {
                ui.label("No segments created yet");
            }
            let classes_snapshot = dataset.classes.clone();
            for segment in &mut dataset.segments {
                let selected_label =
                    Self::class_label_from_classes(segment.class_index, &classes_snapshot);
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("Segment {}", segment.id));
                        egui::ComboBox::from_id_salt(("segment_class", segment.id))
                            .selected_text(selected_label)
                            .show_ui(ui, |ui| {
                                for class in &classes_snapshot {
                                    ui.selectable_value(
                                        &mut segment.class_index,
                                        class.id,
                                        format!("{} - {}", class.id, class.name),
                                    );
                                }
                            });
                    });
                    ui.separator();
                    ui.label("Polygon points (normalized 0-1)");
                    let mut remove_point: Option<usize> = None;
                    let polygon_len = segment.polygon.len();
                    for (idx, point) in segment.polygon.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("P{}", idx + 1));
                            ui.add(
                                egui::DragValue::new(&mut point.x)
                                    .range(0.0..=1.0)
                                    .speed(0.01),
                            )
                            .on_hover_text("X coordinate");
                            ui.add(
                                egui::DragValue::new(&mut point.y)
                                    .range(0.0..=1.0)
                                    .speed(0.01),
                            )
                            .on_hover_text("Y coordinate");
                            if polygon_len > 3 && ui.small_button("Remove").clicked() {
                                remove_point = Some(idx);
                            }
                        });
                    }
                    if let Some(idx) = remove_point {
                        segment.polygon.remove(idx);
                    }
                    if ui.button("Add point").clicked() {
                        segment.polygon.push(SegmentPoint::default());
                    }
                });
            }
        });
    }

    fn class_label_from_classes(class_id: usize, classes: &[ClassEntry]) -> String {
        classes.iter().find(|c| c.id == class_id).map_or_else(
            || format!("{class_id} - Unknown"),
            |c| format!("{} - {}", c.id, c.name),
        )
    }

    fn center_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Image");
        ui.separator();
        if let Some((path, texture)) = &self.current_image {
            ui.label(path.display().to_string());
            egui::ScrollArea::both().show(ui, |ui| {
                ui.add(egui::widgets::Image::new(texture).fit_to_exact_size(texture.size_vec2()));
            });
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
                self.status_message = Some(format!("Loaded image {}", path.display()));
            }
            Err(err) => {
                self.status_message =
                    Some(format!("Failed to load image {}: {err}", path.display()));
            }
        }
    }
}
