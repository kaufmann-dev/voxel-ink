# VoxelInk

<p align="center">
  <img src="assets/voxel-ink.png" alt="VoxelInk logo" width="180">
</p>

VoxelInk is a native, local-only Linux editor for binary voxel volumes. It combines synchronized axial, coronal, sagittal, and 3D views so you can inspect a mask, paint or erase voxels, bridge gaps, and fill enclosed cavities without uploading the volume anywhere.

## What it supports

- NIfTI-1 and NIfTI-2 single 3D scalar volumes: `.nii` and `.nii.gz`
- BINVOX versions 1 and 2: `.binvox`
- Single-model MagicaVoxel version 150 files: `.vox`
- Linux desktops using Wayland or X11, with Vulkan preferred and OpenGL ES as a fallback

VoxelInk treats every nonzero source value as occupied. Saving a non-binary source replaces its original scalar values with canonical binary values after showing a warning. Save As stays within the document's existing format family; it is not a format converter. Detached or multivolume NIfTI data and multi-model or colored MagicaVoxel editing are not supported.

## Install for your user

The following source installation places VoxelInk itself below `~/.local`, so it does not need `sudo`. When `~/.local/bin` comes before `/usr/bin` in `PATH`, this user copy takes precedence over a system package. Run the project commands from a VoxelInk checkout.

On Fedora, install the build and desktop-integration prerequisites once:

```sh
sudo dnf install \
  dbus-devel desktop-file-utils gcc gcc-c++ git-core \
  gtk-update-icon-cache libX11-devel libXcursor-devel libXi-devel \
  libXrandr-devel libxkbcommon-devel make mesa-libGL-devel \
  pkgconf-pkg-config rustup shared-mime-info vulkan-loader-devel \
  wayland-devel
rustup toolchain install 1.96.0 --profile minimal
```

Clone the project if necessary, then install or replace the current user copy:

```sh
git clone https://github.com/kaufmann-dev/voxel-ink.git
cd voxel-ink
cargo install --locked --force --path . --root "$HOME/.local"
```

For a later update, pull the desired source revision and run only the `cargo install` command again. Confirm that the user copy wins:

```sh
command -v voxel-ink
# Expected: /home/your-user/.local/bin/voxel-ink
```

If the command is not found, add `~/.local/bin` to your shell's `PATH` and open a new terminal:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

### Add the application launcher

`cargo install` installs the executable but not the desktop launcher, icon, or file associations. From the repository root, install those for the current user with:

```sh
install -Dm0644 assets/voxel-ink.png \
  "$HOME/.local/share/icons/hicolor/512x512/apps/app.voxelink.editor.png"
install -Dm0644 packaging/linux/app.voxelink.editor.xml \
  "$HOME/.local/share/mime/packages/app.voxelink.editor.xml"
install -Dm0644 packaging/linux/app.voxelink.editor.metainfo.xml \
  "$HOME/.local/share/metainfo/app.voxelink.editor.metainfo.xml"
install -d "$HOME/.local/share/applications"
desktop-file-install \
  --dir="$HOME/.local/share/applications" \
  --set-key=Exec --set-value="$HOME/.local/bin/voxel-ink %f" \
  --set-key=TryExec --set-value="$HOME/.local/bin/voxel-ink" \
  packaging/linux/app.voxelink.editor.desktop
update-desktop-database "$HOME/.local/share/applications"
update-mime-database "$HOME/.local/share/mime"
gtk-update-icon-cache --force --ignore-theme-index \
  "$HOME/.local/share/icons/hicolor"
```

Running that block again refreshes an existing user installation.

## Start VoxelInk

Open VoxelInk from the application launcher, start with an empty window, or open a volume directly:

```sh
voxel-ink
voxel-ink /path/to/mask.nii.gz
```

Use **File → Open** to load a supported volume. Common shortcuts are `Ctrl+O` to open, `Ctrl+S` to save, `Ctrl+Shift+S` to save as, `Ctrl+Z` to undo, and `Ctrl+Shift+Z` or `Ctrl+Y` to redo. Saves are staged through a temporary file, and VoxelInk refuses to overwrite a source that another program changed after it was opened.

## Editing basics

Choose a tool from the toolbar or press `I` for Inspect, `B` for Paint, `E` for Erase, or `L` for Bridge. The slice views preview the exact Paint, Erase, or Bridge voxels in every intersected pane, and support wheel-based slice navigation, `Ctrl`+wheel zoom, drag-to-pan, and click/drag editing. Arrow keys plus Page Up/Page Down move through slices; Space or Enter applies Paint or Erase.

With Paint, Erase, or Bridge selected, press `-` or `+` to change brush diameter, `R` or `Q` for Round or Square, `1` or `3` for One slice or 3D volume, and `U` to switch between voxel and millimetre sizing when physical units are available. Press `V` to toggle voxel outlines. A brush setting changed during an active stroke applies to the next stroke; the current stroke keeps the brush it started with.

Bridge adds one solid connection while you drag between two points. Very large bridges may be too expensive to preview exactly; in that case VoxelInk keeps the endpoints visible and shows a yellow notice instead of interrupting the drag. Shorten the bridge to restore the full preview, or release to let VoxelInk make a larger bounded attempt. If that final attempt is still too large, no voxels are changed and a nonmodal notice asks you to reduce the diameter or length.

The 3D pane offers two navigation styles:

- **Orbit** rotates around the volume, pans with the right mouse button, and zooms with the wheel.
- **Fly** uses a fixed center reticle. Click once to capture the mouse; that first click never edits. Look with the mouse, move with `W/A/S/D`, change world-Z height with Space/Shift, hold `Ctrl` to boost, and use the wheel to adjust speed. Left-click or hold to use Paint, Erase, or Bridge through the reticle. Tool, brush, outline, undo, redo, and Home camera-reset shortcuts remain available without releasing capture; file shortcuts stay suppressed. Press Escape to cancel the active stroke and release the mouse.

Orbit and Fly remember separate viewpoints. Reset affects only the active camera, and opening or closing a document returns navigation to Orbit. The **3D view** card in the repair panel provides a flight-speed slider and a shared voxel-outline toggle that affects both navigation modes.

## Build and test

The repository pins Rust 1.96.0. For development, run:

```sh
cargo fmt --check
cargo check --locked --all-targets --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
```

Run the repeatable CPU hot-path benchmarks after changing meshing, brushes, or codecs:

```sh
cargo bench --locked --bench hot_paths -- --noplot
```

The Linux CI workflow also builds an x86_64 AppImage, builds a Fedora RPM with Podman, and smoke-tests the native application.
