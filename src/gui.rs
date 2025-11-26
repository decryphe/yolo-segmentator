use eframe::egui;

/// Result of asking the user to save pending changes.
pub enum SavePromptResult {
    Yes,
    No,
}

/// Shows a modal save prompt window and returns whether the user chose `Yes`, `No`, or neither.
///
/// The window id must be unique per caller; `title` is the window title to display.
pub fn show_save_prompt(ctx: &egui::Context, title: &str) -> Option<SavePromptResult> {
    let mut result = None;
    egui::Window::new(title)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label("Would you like to save the changes?");
            ui.horizontal(|ui| {
                if ui.button("Yes").clicked() {
                    result = Some(SavePromptResult::Yes);
                }
                if ui.button("No").clicked() {
                    result = Some(SavePromptResult::No);
                }
            });
        });
    result
}
