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

## Install a release

On Fedora x86_64, install or update to the latest release with dnf. The package adds the `voxel-ink` command, application launcher, icon, and desktop metadata:

```sh
sudo dnf install https://github.com/kaufmann-dev/voxel-ink/releases/latest/download/voxel-ink.x86_64.rpm
```

Run the same command again to update after a new release. To uninstall, run `sudo dnf remove voxel-ink`.

On other x86_64 Linux distributions, download and run the AppImage:

```sh
curl -fLO https://github.com/kaufmann-dev/voxel-ink/releases/latest/download/voxel-ink-x86_64.AppImage
chmod +x voxel-ink-x86_64.AppImage
./voxel-ink-x86_64.AppImage
```

To uninstall the AppImage, delete the file.

## Install from source

The installer builds VoxelInk from source and places it below `~/.local`, so it does not need `sudo`. When `~/.local/bin` comes before `/usr/bin` in `PATH`, this user copy takes precedence over the RPM package.

On Fedora, install the build and desktop-integration prerequisites once, then set up Rust and open a new terminal:

```sh
sudo dnf install \
  dbus-devel desktop-file-utils gcc gcc-c++ git-core \
  gtk-update-icon-cache libX11-devel libXcursor-devel libXi-devel \
  libXrandr-devel libxkbcommon-devel make mesa-libGL-devel \
  pkgconf-pkg-config rustup shared-mime-info vulkan-loader-devel \
  wayland-devel
rustup-init -y --profile minimal --default-toolchain 1.96.0
```

Clone the project, then install or replace the current user copy:

```sh
git clone https://github.com/kaufmann-dev/voxel-ink.git
cd voxel-ink
./install
```

The installer builds and installs the executable, application launcher, icon, AppStream metadata, and file associations. For a later update, pull the desired source revision and run `./install` again; existing files are replaced. Confirm that the user copy wins:

```sh
command -v voxel-ink
# Expected: /home/your-user/.local/bin/voxel-ink
```

If the installer warns that `~/.local/bin` is not first in `PATH`, add it to your shell profile and open a new terminal:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

To remove the executable, launcher, icon, metadata, and file associations, run:

```sh
./install --uninstall
```

## Start VoxelInk

Open VoxelInk from the application launcher, start with an empty window, or open a volume directly:

```sh
voxel-ink
voxel-ink /path/to/mask.nii.gz
```

Use **File → Open** to load a supported volume. Common shortcuts are `Ctrl+O` to open, `Ctrl+S` to save, `Ctrl+Shift+S` to save as, `Ctrl+Z` to undo, and `Ctrl+Shift+Z` or `Ctrl+Y` to redo. Saves are staged through a temporary file, and VoxelInk refuses to overwrite a source that another program changed after it was opened.

## Editing basics

Choose a tool from the toolbar or press `I` for Inspect, `B` for Paint, `E` for Erase, or `L` for Bridge. The slice views preview the exact Paint, Erase, or Bridge voxels in every intersected pane, and support wheel-based slice navigation, `Ctrl`+wheel zoom, drag-to-pan, and click/drag editing. Arrow keys plus Page Up/Page Down move through slices; Space or Enter applies Paint or Erase.

With Paint, Erase, or Bridge selected, press `-` or `+` to change brush diameter, `R` or `Q` for Round or Square, `1` or `3` for One slice or 3D volume, and `U` to switch between voxel and millimetre sizing when physical units are available. Press `V` to toggle voxel outlines or `C` to toggle leak light. A brush setting changed during an active stroke applies to the next stroke; the current stroke keeps the brush it started with.

Bridge adds one solid connection while you drag between two points. Very large bridges may be too expensive to preview exactly; in that case VoxelInk keeps the endpoints visible and shows a yellow notice instead of interrupting the drag. Shorten the bridge to restore the full preview, or release to let VoxelInk make a larger bounded attempt. If that final attempt is still too large, no voxels are changed and a nonmodal notice asks you to reduce the diameter or length.

The 3D pane offers two navigation styles:

- **Orbit** rotates around the volume, pans with the right mouse button, and zooms with the wheel.
- **Fly** uses a fixed center reticle. Click once to capture the mouse; that first click never edits. Look with the mouse, move with `W/A/S/D`, change world-Z height with Space/Shift, hold `Ctrl` to boost, and use the wheel to adjust speed. Level movement is selected by default, so W/S follow camera yaw on world X/Y regardless of pitch and only Space/Shift change height. The Free flight option instead makes W/S follow the full camera direction, including pitch; A/D remain level in both styles. The camera passes through solid voxels by default; enable **Collide with solid voxels** to stop the viewpoint at occupied faces and slide along them. Enabling collision while inside a solid voxel, or filling the camera voxel during an edit, returns the viewpoint to its safe exterior Fly home without changing its direction or speed. Left-click or hold to use Paint, Erase, or Bridge through the reticle. Tool, brush, outline, undo, redo, and Home camera-reset shortcuts remain available without releasing capture; file shortcuts stay suppressed. Press Escape to cancel the active stroke and release the mouse.

Orbit and Fly remember separate viewpoints. Reset affects only the active camera, and opening or closing a document returns navigation to Orbit. The **3D view** card in the repair panel provides session-only W/S movement and collision controls, a flight-speed slider, and shared voxel-outline and leak-light controls that affect both navigation modes.

**Leak light** is a diagnostic approximation of outside light travelling through face-connected empty voxels. Its continuous heatmap marks surfaces nearest outside in pale yellow, then flows through orange, crimson, and purple as physical path distance increases; sealed cavities and fully faded paths remain black. The Light reach slider controls the physical distance over which the light halves, immediately rescaling the entire gradient. VoxelInk begins calculating and preparing leak paths in the background as soon as a document opens, so an unchanged document can show leak light immediately when enabled and the repair panel reports that background progress. After an edit, VoxelInk keeps the previous heatmap visible, waits 150 ms for further edits, recomputes exact paths in the background even while leak light is hidden, and replaces the display only after the new field is ready; the status text identifies both background updates and a previous result while it is shown. Natural anatomical openings and accidental gaps propagate light identically, so VoxelInk leaves the decision about what to seal to you. Leak light changes only the view and is never saved into the volume.

## Build and test

The repository pins Rust 1.96.0. For development, run:

```sh
cargo fmt --check
cargo check --locked --all-targets --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
```

Run the repeatable CPU hot-path benchmarks after changing meshing, leak-light preparation, brushes, or codecs:

```sh
cargo bench --locked --bench hot_paths -- --noplot
```

The Linux CI workflow also builds an x86_64 AppImage, builds a Fedora RPM with Podman, and smoke-tests the native application.

## Publish a release

Set the new version in `Cargo.toml`, `packaging/rpm/voxel-ink.spec` (including a `%changelog` entry), and the first `<release>` in `packaging/linux/app.voxelink.editor.metainfo.xml`. After the commit is on `main`, tag and push it:

```sh
git tag v0.1.0
git push origin v0.1.0
```

The Linux workflow verifies that the tag matches all three versions, builds the RPM and AppImage, and publishes them to a GitHub release with the stable asset names used above.
