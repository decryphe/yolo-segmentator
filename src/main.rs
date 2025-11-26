#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

mod dataset;
mod gui;
mod watcher;

use eframe::egui;
use rfd::FileDialog;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    dataset::{DirectoryWatcher, ImagePurpose},
    gui::{SavePromptResult, show_save_prompt},
};

enum PendingAction {
    SaveOnly,
    Navigate(dataset::ImageReference),
}

#[derive(Clone, Copy)]
enum NavDirection {
    Previous,
    Next,
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
    dataset: Arc<RwLock<Option<dataset::Dataset>>>,
    worker: DirectoryWatcher,
    status_message: Option<String>,
    current_image: Option<ActiveImage>,
    selected_segment_id: Option<usize>,
    data_dirty: bool,
    show_save_prompt: bool,
    pending_action: Option<PendingAction>,
}

impl MyApp {
    fn new(ctx: egui::Context) -> Self {
        let dataset = Arc::new(RwLock::new(None));
        let worker = DirectoryWatcher::new(dataset.clone(), ctx);
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

struct ActiveImage {
    reference: dataset::ImageReference,
    texture: egui::TextureHandle,
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

        if self.show_save_prompt
            && let Some(choice) = show_save_prompt(ctx, "Save Changes?")
        {
            match choice {
                SavePromptResult::Yes => match self.save_current_image_segments() {
                    Ok(()) => {
                        self.data_dirty = false;
                        self.show_save_prompt = false;
                        self.execute_pending_action(ctx);
                    }
                    Err(err) => {
                        self.status_message = Some(err);
                    }
                },
                SavePromptResult::No => {
                    self.data_dirty = false;
                    self.show_save_prompt = false;
                    self.execute_pending_action(ctx);
                }
            }
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
                .map(|dataset| (dataset.root.clone(), dataset.name(), dataset.images_view()))
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
                    dataset.classes.push(dataset::ClassEntry {
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
        for purpose in ImagePurpose::ALL {
            let title = purpose.display_name();
            let list = images.list(purpose);
            ui.label(egui::RichText::new(title).strong());
            if list.is_empty() {
                ui.label("  (no images)");
            } else {
                egui::Grid::new(format!("image_grid_{title}"))
                    .striped(true)
                    .num_columns(3)
                    .show(ui, |ui| {
                        for entry in list {
                            ui.label(entry.path.display().to_string());
                            ui.label(if entry.has_labels { "" } else { "(new)" });
                            if ui.small_button("Load").clicked() {
                                self.load_image(ui.ctx(), purpose, &entry.path);
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

        let dataset = dataset::Dataset::new(selected_dir);
        let dataset_root = dataset.root.clone();
        match dataset.initialize_on_disk() {
            Ok(()) => match dataset::Dataset::load(dataset_root.clone()) {
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

        match dataset::Dataset::load(selected_dir.clone()) {
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
        let Some(active) = self.current_image.as_ref() else {
            ui.label("Load an image to manage segments");
            return;
        };
        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open a dataset to manage segments");
            return;
        };
        let classes_snapshot = dataset.classes.clone();
        let default_class = classes_snapshot.first().map_or(0, |c| c.id);
        let Ok(loaded_image) =
            dataset.ensure_image_loaded(active.reference.split, &active.reference.relative_path)
        else {
            ui.label("Unable to load current image segments");
            return;
        };

        ui.horizontal(|ui| {
            ui.heading("Segments");
            if ui.button("New Segment").clicked() {
                let next_id = loaded_image.next_segment_id();
                loaded_image
                    .segments
                    .push(dataset::SegmentEntry::new(next_id, default_class));
                self.selected_segment_id = Some(next_id);
                loaded_image.mark_dirty();
                self.data_dirty = true;
                self.status_message = Some(format!("Created segment #{next_id}"));
            }
        });
        ui.separator();

        if loaded_image.segments.is_empty() {
            ui.label("No segments created yet");
            return;
        }

        let mut segment_to_remove = None;
        let mut dirty = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("segment_grid")
                .striped(true)
                .num_columns(5)
                .show(ui, |ui| {
                    for (idx, segment) in loaded_image.segments.iter_mut().enumerate() {
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
                            dirty = true;
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

        if let Some(idx) = segment_to_remove.filter(|&i| i < loaded_image.segments.len()) {
            let removed = loaded_image.segments.remove(idx);
            if self.selected_segment_id == Some(removed.id) {
                self.selected_segment_id = None;
            }
            dirty = true;
            self.data_dirty = true;
        }
        if dirty {
            loaded_image.mark_dirty();
        }
    }

    fn class_label_from_classes(class_id: usize, classes: &[dataset::ClassEntry]) -> String {
        classes.iter().find(|c| c.id == class_id).map_or_else(
            || format!("{class_id} - Unknown"),
            |c| format!("{} - {}", c.id, c.name),
        )
    }

    fn current_segments_snapshot(&self) -> Vec<dataset::SegmentEntry> {
        let Some(active) = &self.current_image else {
            return Vec::new();
        };
        self.dataset
            .read()
            .ok()
            .and_then(|guard| {
                guard.as_ref().and_then(|dataset| {
                    dataset
                        .segments_snapshot(active.reference.split, &active.reference.relative_path)
                })
            })
            .unwrap_or_default()
    }

    #[allow(clippy::cast_precision_loss)]
    fn center_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Image");
        ui.separator();
        if let Some(active) = &self.current_image {
            let segments_snapshot = self.current_segments_snapshot();
            if let Some(selected) = self.selected_segment_id {
                ui.label(format!("Editing segment #{selected}"));
            } else {
                ui.label("Select a segment to edit");
            }
            ui.label(active.reference.full_path.display().to_string());
            let texture_id = active.texture.id();
            let texture_size = active.texture.size_vec2();
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

    fn load_image(&mut self, ctx: &egui::Context, split: ImagePurpose, relative_path: &Path) {
        let (color_image, reference, segment_count, dirty) = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            let Some(dataset) = guard.as_mut() else {
                self.status_message = Some("Open a dataset before loading images".to_owned());
                return;
            };
            let (color_image, segment_count, dirty) = {
                let loaded = match dataset.ensure_image_loaded(split, relative_path) {
                    Ok(img) => img,
                    Err(err) => {
                        self.status_message = Some(err);
                        return;
                    }
                };
                (loaded.as_color_image(), loaded.segments.len(), loaded.dirty)
            };
            let reference = dataset.image_reference(split, relative_path);
            (color_image, reference, segment_count, dirty)
        };

        let texture = ctx.load_texture(
            format!("loaded_image_{}", reference.full_path.display()),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.current_image = Some(ActiveImage { reference, texture });
        self.selected_segment_id = None;
        self.data_dirty = dirty;
        if let Some(active) = &self.current_image {
            self.status_message = Some(format!(
                "Loaded image {} with {} segment(s)",
                active.reference.full_path.display(),
                segment_count
            ));
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
            self.load_image(ctx, target.split, &target.relative_path);
        }
    }

    fn next_image_path(&self, direction: NavDirection) -> Option<dataset::ImageReference> {
        let guard = self.dataset.read().ok()?;
        let dataset = guard.as_ref()?;
        let images = dataset.flattened_image_refs();
        if images.is_empty() {
            return None;
        }
        let current_idx = self.current_image.as_ref().and_then(|active| {
            images.iter().position(|entry| {
                entry.matches(active.reference.split, &active.reference.relative_path)
            })
        });
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
                PendingAction::Navigate(reference) => {
                    self.load_image(ctx, reference.split, &reference.relative_path);
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
        let Some(active) = self.current_image.as_ref() else {
            self.status_message = Some("Load an image before editing segments".to_owned());
            return;
        };
        let mut guard = self.dataset.write().expect("dataset lock poisoned");
        if let Some(dataset) = guard.as_mut() {
            let Ok(loaded_image) = dataset
                .ensure_image_loaded(active.reference.split, &active.reference.relative_path)
            else {
                self.status_message = Some("Unable to load current image for editing".to_owned());
                return;
            };
            if let Some(segment) = loaded_image
                .segments
                .iter_mut()
                .find(|seg| seg.id == segment_id)
            {
                segment.polygon.push(dataset::SegmentPoint { x, y });
                loaded_image.mark_dirty();
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
        let reference = self
            .current_image
            .as_ref()
            .map(|img| img.reference.clone())
            .ok_or_else(|| "Load an image before saving segments".to_owned())?;

        let count = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            let dataset = guard
                .as_mut()
                .ok_or_else(|| "No dataset loaded".to_owned())?;
            dataset
                .save_metadata()
                .map_err(|err| format!("Failed to save dataset metadata: {err}"))?;
            dataset.save_loaded_image(reference.split, &reference.relative_path)?
        };

        self.data_dirty = false;
        let txt_path = reference.full_path.with_extension("txt");
        self.status_message = Some(format!(
            "Saved {count} segment(s) to {}",
            txt_path.display()
        ));
        Ok(())
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hsva_to_color32(hsva: egui::epaint::Hsva) -> egui::Color32 {
    let [r, g, b, a] = hsva.to_rgba_unmultiplied();
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgba_unmultiplied(to_u8(r), to_u8(g), to_u8(b), to_u8(a))
}
