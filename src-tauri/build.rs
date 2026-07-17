fn main() {
    const COMMANDS: &[&str] = &[
        "request_open_ticket",
        "read_open_ticket",
        "request_save_ticket",
        "write_save_ticket",
        "release_source",
        "confirm_exit",
        "cancel_exit",
    ];

    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("failed to prepare the Tauri build");
}
