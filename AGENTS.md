# Repository Instructions

## Platform and toolchain

VoxelInk is intentionally Linux-only. It supports Wayland and X11 through WGPU, preferring Vulkan and falling back to OpenGL ES. Rust 1.96.0, `rustfmt`, and `clippy` are pinned in `rust-toolchain.toml`; keep `Cargo.lock` committed and use `--locked` for reproducible commands. The crate forbids unsafe Rust.

## Code map

- `src/app.rs`: egui application state, panes, shortcuts, dialogs, Orbit/Fly interaction, and edit previews.
- `src/renderer.rs`: WGPU resources, shaders, camera state, volume rendering, and affine-aware scene rays.
- `src/engine.rs`: background commands/events, strokes, history coordination, and incremental remeshing.
- `src/volume.rs`: document geometry, slices, voxel storage, undo/redo, and cavity filling.
- `src/brush.rs`: picking, rays, brush footprints, and bridge geometry.
- `src/mesh.rs`: chunk management and greedy mesh generation.
- `src/io.rs` and `src/codecs/`: guarded saves and the NIfTI, BINVOX, and MagicaVoxel codecs.
- `tests/`: integration coverage grouped by UI, engine semantics, picking, geometry, persistence, document behavior, and performance.

Preserve NIfTI affine geometry through rendering, picking, editing, and saving. Keep UI work on the application side and long-running document operations in the engine command/event pipeline. Reuse existing stroke, history, highlight, and incremental-remesh paths instead of creating parallel edit behavior. Format families are deliberately not converted during Save As.

## Verification

Run the smallest relevant targeted test while iterating, then use the CI-equivalent checks before handing off a completed change:

```sh
cargo fmt --check
cargo check --locked --all-targets --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
```

UI tests live in `tests/ui_kittest.rs`. Renderer validation includes a headless Vulkan path; CI runs it with Mesa Lavapipe and `VOXEL_INK_REQUIRE_SOFTWARE_VULKAN=1`. Performance acceptance tests are opt-in through `VOXEL_INK_RUN_ACCEPTANCE=1`, with an optional local dataset selected by `VOXEL_INK_CRANIOFACIAL_NIFTI`.

Run `shellcheck` for changed packaging scripts. Validate changed desktop metadata with `desktop-file-validate`, AppStream metadata with `appstreamcli validate --no-net`, and XML with `xmllint --noout`.

## Packaging

Build `target/release/voxel-ink` before invoking packaging scripts. AppImage inputs live under `packaging/appimage/`; Fedora RPM inputs live under `packaging/rpm/` and are built with Podman. Keep the package version synchronized across Cargo and packaging metadata, and keep the application ID `app.voxelink.editor` consistent across the desktop entry, MIME definition, metainfo, icon, and native window.
