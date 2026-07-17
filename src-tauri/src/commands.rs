use crate::{
    broker::{
        Broker, OPEN_EXTENSIONS, OpenTicketInfo, SAVE_TICKET_HEADER, SaveMode, SaveTicketInfo,
        SaveTicketRequest, WriteSaveResult,
    },
    error::{HostError, HostResult},
    file_io::{
        atomic_replace, canonicalize_destination, canonicalize_existing, fingerprint,
        path_has_variant, read_file,
    },
    state::HostState,
};
use std::sync::{Arc, Mutex, MutexGuard};
use tauri::{AppHandle, State, WebviewWindow, ipc::InvokeBody};
use tauri_plugin_dialog::DialogExt;

#[tauri::command]
pub(crate) async fn request_open_ticket(
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, HostState>,
) -> HostResult<Option<OpenTicketInfo>> {
    let selected = app
        .dialog()
        .file()
        .set_parent(&window)
        .set_title("Open voxel volume")
        .add_filter("Voxel volumes", OPEN_EXTENSIONS)
        .blocking_pick_file();
    let Some(selected) = selected else {
        return Ok(None);
    };
    let selected = selected.into_path().map_err(|_| {
        HostError::invalid_request("The selected item is not a local filesystem path.")
    })?;
    let path = canonicalize_existing(&selected)?;
    let metadata = std::fs::metadata(&path)
        .map_err(|error| HostError::io("Inspecting the selected file", &error))?;
    if !metadata.is_file() {
        return Err(HostError::invalid_request(
            "The selected item is not a regular file.",
        ));
    }

    let broker = state.broker.clone();
    lock_broker(&broker)?
        .issue_open(path, metadata.len())
        .map(Some)
}

#[tauri::command]
pub(crate) async fn read_open_ticket(
    ticket: String,
    state: State<'_, HostState>,
) -> HostResult<tauri::ipc::Response> {
    let broker = state.broker.clone();
    let open_ticket = lock_broker(&broker)?.take_open(&ticket)?;
    let source_id = open_ticket.source_id;
    let path = open_ticket.path;
    let read_result = tauri::async_runtime::spawn_blocking(move || read_file(&path))
        .await
        .map_err(|_| HostError::internal("The file reader stopped unexpectedly."))?;

    match read_result {
        Ok(read) => {
            lock_broker(&broker)?.complete_open(source_id, read.fingerprint)?;
            Ok(tauri::ipc::Response::new(read.bytes))
        }
        Err(error) => {
            lock_broker(&broker)?.fail_open(source_id);
            Err(error)
        }
    }
}

#[tauri::command]
pub(crate) async fn request_save_ticket(
    request: SaveTicketRequest,
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, HostState>,
) -> HostResult<Option<SaveTicketInfo>> {
    let broker = state.broker.clone();
    let (source_id, source, allowed_extensions, suggested_name) =
        lock_broker(&broker)?.source_for_save(&request)?;

    if request.mode == SaveMode::OverwriteSource {
        let expected = source
            .fingerprint
            .clone()
            .ok_or_else(HostError::source_not_ready)?;
        return Ok(Some(lock_broker(&broker)?.issue_save(
            source_id,
            request.mode,
            source.path,
            source.source_variant,
            expected,
        )));
    }

    let extension_refs: Vec<&str> = allowed_extensions.iter().map(String::as_str).collect();
    let mut dialog = app
        .dialog()
        .file()
        .set_parent(&window)
        .set_title(match request.mode {
            SaveMode::SaveAsAdopt => "Save voxel volume as",
            SaveMode::SaveCopy => "Save a copy of the voxel volume",
            SaveMode::OverwriteSource => unreachable!("handled above"),
        })
        .set_file_name(suggested_name)
        .add_filter("Voxel volume", &extension_refs);
    if let Some(parent) = source.path.parent() {
        dialog = dialog.set_directory(parent);
    }
    let Some(selected) = dialog.blocking_save_file() else {
        return Ok(None);
    };
    let selected = selected.into_path().map_err(|_| {
        HostError::invalid_request("The selected destination is not a local filesystem path.")
    })?;
    let destination = canonicalize_destination(&selected)?;
    let destination_variant = allowed_extensions
        .iter()
        .find(|variant| path_has_variant(&destination, variant))
        .cloned()
        .ok_or_else(|| {
            HostError::invalid_request(
                "The destination extension is not valid for the source format family.",
            )
        })?;

    require_distinct_non_overwrite_destination(request.mode, &source.path, &destination)?;

    let expected = if destination == source.path {
        source
            .fingerprint
            .clone()
            .ok_or_else(HostError::source_not_ready)?
    } else {
        let fingerprint_path = destination.clone();
        tauri::async_runtime::spawn_blocking(move || fingerprint(&fingerprint_path))
            .await
            .map_err(|_| HostError::internal("The destination checker stopped unexpectedly."))??
    };

    Ok(Some(lock_broker(&broker)?.issue_save(
        source_id,
        request.mode,
        destination,
        destination_variant,
        expected,
    )))
}

#[tauri::command]
pub(crate) async fn write_save_ticket(
    request: tauri::ipc::Request<'_>,
    state: State<'_, HostState>,
) -> HostResult<WriteSaveResult> {
    let ticket = request
        .headers()
        .get(SAVE_TICKET_HEADER)
        .ok_or_else(|| {
            HostError::new(
                "missing_ticket_header",
                "The save request did not include an authorization ticket.",
            )
        })?
        .to_str()
        .map_err(|_| HostError::invalid_ticket())?
        .to_owned();

    let broker = state.broker.clone();
    let save_ticket = lock_broker(&broker)?.take_save(&ticket)?;
    let bytes = match request.body() {
        InvokeBody::Raw(bytes) if !bytes.is_empty() => bytes.clone(),
        _ => {
            return Err(HostError::new(
                "invalid_body",
                "The save request body must contain raw binary file data.",
            ));
        }
    };

    let destination = save_ticket.destination.clone();
    let expected = save_ticket.expected.clone();
    let fingerprint = tauri::async_runtime::spawn_blocking(move || {
        atomic_replace(&destination, &bytes, &expected)
    })
    .await
    .map_err(|_| HostError::internal("The file writer stopped unexpectedly."))??;

    lock_broker(&broker)?.complete_save(&save_ticket, fingerprint)
}

#[tauri::command]
pub(crate) fn release_source(source_id: String, state: State<'_, HostState>) -> HostResult<()> {
    lock_broker(&state.broker)?.release_source(&source_id)
}

#[tauri::command]
pub(crate) fn confirm_exit(app: AppHandle, state: State<'_, HostState>) -> HostResult<()> {
    state.confirm_exit();
    app.exit(0);
    Ok(())
}

#[tauri::command]
pub(crate) fn cancel_exit(state: State<'_, HostState>) -> HostResult<()> {
    state.cancel_exit();
    Ok(())
}

fn lock_broker(broker: &Arc<Mutex<Broker>>) -> HostResult<MutexGuard<'_, Broker>> {
    broker
        .lock()
        .map_err(|_| HostError::internal("The file authorization broker is unavailable."))
}

fn require_distinct_non_overwrite_destination(
    mode: SaveMode,
    source: &std::path::Path,
    destination: &std::path::Path,
) -> HostResult<()> {
    if mode == SaveMode::OverwriteSource || destination != source {
        return Ok(());
    }
    let action = if mode == SaveMode::SaveCopy {
        "Save a Copy"
    } else {
        "Save As"
    };
    Err(HostError::invalid_request(format!(
        "{action} requires a destination different from the source; use Save to overwrite it."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_io::variant_from_path;

    #[test]
    fn all_dialog_extensions_are_known_to_the_path_classifier() {
        for extension in OPEN_EXTENSIONS {
            let path = std::path::PathBuf::from(format!("fixture.{extension}"));
            assert_eq!(variant_from_path(&path).as_deref(), Some(*extension));
        }
    }

    #[test]
    fn save_as_cannot_silently_overwrite_the_authorized_source() {
        let source = std::path::Path::new("/authorized/source.nii.gz");

        let error =
            require_distinct_non_overwrite_destination(SaveMode::SaveAsAdopt, source, source)
                .expect_err("Save As must not become an implicit overwrite");

        assert_eq!(error.code, "invalid_request");
        assert!(
            require_distinct_non_overwrite_destination(SaveMode::OverwriteSource, source, source,)
                .is_ok()
        );
    }
}
