#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("VoxelInk is intentionally supported only on Linux");

use std::{path::PathBuf, sync::Arc};

use voxel_ink::VoxelInkApp;

const LOGO_BYTES: &[u8] = include_bytes!("../assets/voxel-ink.png");

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("voxel_ink=info")),
        )
        .with_target(false)
        .compact()
        .init();

    let initial_path = parse_arguments();
    let icon = eframe::icon_data::from_png_bytes(LOGO_BYTES)
        .expect("the embedded VoxelInk logo must be valid");
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    setup.native_adapter_selector = Some(Arc::new(select_linux_adapter));
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("VoxelInk")
            .with_app_id("app.voxelink.editor")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_icon(icon),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 24,
        centered: true,
        ..Default::default()
    };
    options.wgpu_options.wgpu_setup = setup.into();
    eframe::run_native(
        "VoxelInk",
        options,
        Box::new(move |creation_context| {
            Ok(Box::new(VoxelInkApp::new(
                creation_context,
                initial_path.clone(),
            )))
        }),
    )
}

fn parse_arguments() -> Option<PathBuf> {
    let mut arguments = std::env::args_os();
    let executable = arguments.next();
    let path = arguments.next();
    if arguments.next().is_some() {
        let executable = executable
            .as_deref()
            .and_then(|argument| std::path::Path::new(argument).file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("voxel-ink");
        eprintln!("usage: {executable} [FILE]");
        std::process::exit(2);
    }
    path.map(PathBuf::from)
}

fn select_linux_adapter(
    adapters: &[wgpu::Adapter],
    surface: Option<&wgpu::Surface<'_>>,
) -> std::result::Result<wgpu::Adapter, String> {
    adapters
        .iter()
        .filter(|adapter| surface.is_none_or(|surface| adapter.is_surface_supported(surface)))
        .min_by_key(|adapter| {
            let info = adapter.get_info();
            let backend = match info.backend {
                wgpu::Backend::Vulkan => 0,
                wgpu::Backend::Gl => 1,
                _ => 2,
            };
            let device = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Other => 3,
                wgpu::DeviceType::Cpu => 4,
            };
            (backend, device)
        })
        .cloned()
        .ok_or_else(|| "no Vulkan or OpenGL ES adapter can present to this display".to_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn backend_priority_is_vulkan_then_gles() {
        fn priority(backend: wgpu::Backend) -> u8 {
            match backend {
                wgpu::Backend::Vulkan => 0,
                wgpu::Backend::Gl => 1,
                _ => 2,
            }
        }
        assert!(priority(wgpu::Backend::Vulkan) < priority(wgpu::Backend::Gl));
    }
}
