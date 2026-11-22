fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "app_info",
            "start",
            "choose_data_dir",
            "change_data_dir",
            "check_update",
            "install_update",
            "pick_folder",
            "pick_save_path",
            "save_file",
        ]),
    ))
    .expect("failed to run tauri-build");
}
