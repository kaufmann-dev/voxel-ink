mod broker;
mod commands;
mod error;
mod file_io;
mod state;

use crate::{
    commands::{
        cancel_exit, confirm_exit, read_open_ticket, release_source, request_open_ticket,
        request_save_ticket, write_save_ticket,
    },
    state::{ExitDecision, HostState},
};
use serde::Serialize;
use tauri::{Emitter, Manager};

const CLOSE_REQUESTED_EVENT: &str = "voxelink://close-requested";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
enum CloseReason {
    WindowClose,
    ApplicationExit,
}

fn emit_close_request(app: &tauri::AppHandle, reason: CloseReason) {
    if app.emit_to("main", CLOSE_REQUESTED_EVENT, reason).is_err() {
        app.state::<HostState>().event_delivery_failed();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .manage(HostState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            request_open_ticket,
            read_open_ticket,
            request_save_ticket,
            write_save_ticket,
            release_source,
            confirm_exit,
            cancel_exit,
        ]);

    #[cfg(feature = "wdio")]
    let builder = builder
        .plugin(tauri_plugin_wdio::init())
        .plugin(tauri_plugin_wdio_webdriver::init());

    let app = builder
        .build(tauri::generate_context!())
        .expect("error while building VoxelInk");

    app.run(|app, event| match event {
        tauri::RunEvent::WindowEvent {
            label,
            event: tauri::WindowEvent::CloseRequested { api, .. },
            ..
        } if label == "main" => {
            let decision = app.state::<HostState>().intercept_close();
            api.prevent_close();
            if decision == ExitDecision::PreventAndNotify {
                emit_close_request(app, CloseReason::WindowClose);
            }
        }
        tauri::RunEvent::ExitRequested { api, .. } => {
            let decision = app.state::<HostState>().intercept_exit();
            if decision != ExitDecision::Allow {
                api.prevent_exit();
            }
            if decision == ExitDecision::PreventAndNotify {
                emit_close_request(app, CloseReason::ApplicationExit);
            }
        }
        _ => {}
    });
}
