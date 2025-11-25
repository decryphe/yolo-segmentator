# Project Overview
- Rust + egui desktop tool for preparing YOLO segmentation datasets.
- `MyApp` holds UI state; the active dataset is stored in `Arc<RwLock<Option<Dataset>>>` so both UI and a worker thread can mutate it.
- `DirectoryWatcher` spawns a worker thread on startup, listening to an `mpsc::UnboundedReceiver<()>` for dataset change events and watching the `images/*` folders with `notify`.

# Key Components
- `dataset.rs` defines `Dataset`, classes, segments, image discovery, YOLO txt read/write, and watcher abstractions.
- `main.rs` renders toolbar, left class/image panel, center image editor, right segment list, and status bar; handles dialogs for new/open/save and image navigation.
- Worker loop rebuilds watchers on dataset change, filters `notify` events to create/modify/remove, refreshes image lists, and requests egui repaints.
- Image editor draws scaled textures, polygons/lines/points, supports point addition via clicks, and writes YOLO polygons to `.txt` files.

# UX Decisions
- Toolbar shows `New Dataset` when closed, `Close Dataset` otherwise; buttons exist for open/save/prev/next plus left/right keybindings.
- Unsaved changes trigger a modal asking to save before navigation or explicit saves.
- Left panel order: dataset info → classes grid → image grids per split showing `(new)` indicator and `Load` button.
- Segment panel grid: per-segment row with ID, class combo, point count, Edit/Delete buttons; editing happens in center view only.

# Operational Notes
- Always run `cargo fmt` after code changes and check with `cargo clippy -D warnings`.
- Loading/saving uses `serde_yaml_ng`; image annotations serialize to YOLO polygon `.txt`.
- Watcher triggers UI refreshes and image list updates; ensure locks are short-lived (prefer reads for display, writes only when mutating).
