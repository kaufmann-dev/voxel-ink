use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    brush::Ray,
    mesh::{ChunkCoord, CompactQuad, MeshJobTag},
    volume::Matrix4,
};

const SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    index_to_world: mat4x4<f32>,
    world_to_view: mat4x4<f32>,
    index_normal_to_view: mat4x4<f32>,
    material_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
    @location(1) view_position: vec3<f32>,
};

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) origin: vec3<u32>,
    @location(1) packed: u32,
) -> VertexOutput {
    let u_length = f32((packed & 31u) + 1u);
    let v_length = f32(((packed >> 5u) & 31u) + 1u);
    let axis = (packed >> 10u) & 3u;
    let positive = ((packed >> 12u) & 1u) != 0u;

    var corner = 0u;
    if (vertex_index == 2u || vertex_index == 4u) {
        corner = 2u;
    } else if ((positive && vertex_index == 1u) || (!positive && vertex_index == 5u)) {
        corner = 1u;
    } else if ((positive && vertex_index == 5u) || (!positive && vertex_index == 1u)) {
        corner = 3u;
    }

    var index_position = vec3<f32>(origin) - vec3<f32>(0.5);
    var u_axis = 0u;
    var v_axis = 1u;
    if (axis == 0u) {
        u_axis = 1u;
        v_axis = 2u;
    } else if (axis == 1u) {
        u_axis = 2u;
        v_axis = 0u;
    } else {
        u_axis = 0u;
        v_axis = 1u;
    }
    if (corner == 1u || corner == 2u) {
        index_position[u_axis] += u_length;
    }
    if (corner == 2u || corner == 3u) {
        index_position[v_axis] += v_length;
    }

    var index_normal = vec3<f32>(0.0);
    index_normal[axis] = 1.0;
    if (!positive) {
        index_normal = -index_normal;
    }

    let world_position = uniforms.index_to_world * vec4<f32>(index_position, 1.0);
    let view_position = uniforms.world_to_view * world_position;
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.view_normal = normalize(
        (uniforms.index_normal_to_view * vec4<f32>(index_normal, 0.0)).xyz,
    );
    output.view_position = view_position.xyz;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_direction = normalize(-input.view_position);
    var normal = normalize(input.view_normal);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    // Lights are defined in view space, so the object remains legible as it rotates.
    let key_direction = normalize(vec3<f32>(-0.45, 0.55, 0.70));
    let fill_direction = normalize(vec3<f32>(0.65, -0.20, 0.55));
    let key = max(dot(normal, key_direction), 0.0);
    let fill = max(dot(normal, fill_direction), 0.0);
    let facing = clamp(dot(normal, view_direction), 0.0, 1.0);
    let rim = (1.0 - facing) * (1.0 - facing);
    let sky = normal.y * 0.5 + 0.5;
    let lighting = clamp(
        0.22 + 0.52 * key + 0.22 * fill + 0.10 * sky + 0.18 * rim,
        0.26,
        1.08,
    );
    return vec4<f32>(uniforms.material_color.rgb * lighting, 1.0);
}
"#;

const VOXEL_PREVIEW_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    index_to_world: mat4x4<f32>,
    world_to_view: mat4x4<f32>,
    index_normal_to_view: mat4x4<f32>,
    material_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) face_uv: vec2<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) view_position: vec3<f32>,
    @location(3) @interpolate(flat) preview_kind: u32,
};

@vertex
fn preview_vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) origin: vec3<u32>,
    @location(1) packed_face: u32,
) -> VertexOutput {
    let triangle_corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let axis = packed_face & 3u;
    let positive = ((packed_face >> 2u) & 1u) != 0u;
    let preview_kind = packed_face >> 3u;
    let face_uv = triangle_corners[vertex_index];

    // Depth bias keeps this exact voxel boundary visible over resident geometry.
    let cube_size = 1.0;
    var index_position = vec3<f32>(origin) - vec3<f32>(cube_size * 0.5);
    var u_axis = 1u;
    var v_axis = 2u;
    if (axis == 1u) {
        u_axis = 2u;
        v_axis = 0u;
    } else if (axis == 2u) {
        u_axis = 0u;
        v_axis = 1u;
    }
    if (positive) {
        index_position[axis] += cube_size;
    }
    index_position[u_axis] += face_uv.x * cube_size;
    index_position[v_axis] += face_uv.y * cube_size;

    var index_normal = vec3<f32>(0.0);
    index_normal[axis] = select(-1.0, 1.0, positive);
    let world_position = uniforms.index_to_world * vec4<f32>(index_position, 1.0);
    let view_position = uniforms.world_to_view * world_position;

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.face_uv = face_uv;
    output.view_normal = normalize(
        (uniforms.index_normal_to_view * vec4<f32>(index_normal, 0.0)).xyz,
    );
    output.view_position = view_position.xyz;
    output.preview_kind = preview_kind;
    return output;
}

@fragment
fn preview_fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = vec3<f32>(0.0, 0.82, 1.0);
    if (input.preview_kind == 1u) {
        base_color = vec3<f32>(1.0, 0.12, 0.015);
    } else if (input.preview_kind == 2u) {
        base_color = vec3<f32>(0.48, 1.0, 0.03);
    }

    let view_direction = normalize(-input.view_position);
    var normal = normalize(input.view_normal);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }
    let key_direction = normalize(vec3<f32>(-0.45, 0.55, 0.70));
    let diffuse = max(dot(normal, key_direction), 0.0);
    let lighting = 0.68 + 0.32 * diffuse;

    // A one-pixel-ish antialiased edge on every exposed voxel face keeps adjacent
    // candidates visibly discrete instead of turning the preview into a smooth blob.
    let edge_distance = min(
        min(input.face_uv.x, 1.0 - input.face_uv.x),
        min(input.face_uv.y, 1.0 - input.face_uv.y),
    );
    let edge_width = max(fwidth(edge_distance) * 1.35, 0.006);
    let edge_mix = 1.0 - smoothstep(0.0, edge_width, edge_distance);
    let face_color = base_color * lighting;
    let edge_color = mix(base_color * 0.32, vec3<f32>(1.0), 0.18);
    return vec4<f32>(mix(face_color, edge_color, edge_mix), 1.0);
}
"#;

const MAX_GPU_UPDATES_PER_FRAME: usize = 4;
pub const MSAA_SAMPLES: u32 = 4;
// These independent caps keep CPU adjacency work and GPU boundary work bounded.
// Four million 16-byte faces occupy 64 MB and invoke at most 24 million vertices;
// the sample scan's 50 mm volume is about 1.64 M voxels but only 92,160 faces.
pub const MAX_VOXEL_PREVIEW_VOXELS: usize = 2_000_000;
pub const MAX_VOXEL_PREVIEW_FACES: usize = 4_000_000;
const PREVIEW_ALLOCATION_CHUNK: usize = 65_536;
const PREVIEW_CANCELLATION_INTERVAL: usize = 1_024;

fn preview_color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn preview_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: -2,
            slope_scale: -0.5,
            clamp: 0.0,
        },
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoxelPreviewKind {
    Paint = 0,
    Erase = 1,
    Bridge = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PreviewFace {
    origin: [u32; 3],
    packed: u32,
}

impl PreviewFace {
    const fn new(origin: [u32; 3], axis: usize, positive: bool, kind: VoxelPreviewKind) -> Self {
        Self {
            origin,
            packed: (axis as u32) | ((positive as u32) << 2) | ((kind as u32) << 3),
        }
    }

    #[cfg(test)]
    const fn axis(self) -> usize {
        (self.packed & 3) as usize
    }

    #[cfg(test)]
    const fn positive(self) -> bool {
        ((self.packed >> 2) & 1) != 0
    }

    #[cfg(test)]
    const fn kind(self) -> u32 {
        self.packed >> 3
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoxelPreviewFailure {
    TooManyVoxels,
    TooManyFaces,
    Allocation,
}

impl std::fmt::Display for VoxelPreviewFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyVoxels => write!(
                formatter,
                "Brush preview exceeds the {MAX_VOXEL_PREVIEW_VOXELS}-voxel limit; reduce the brush size"
            ),
            Self::TooManyFaces => write!(
                formatter,
                "Brush preview exceeds the {MAX_VOXEL_PREVIEW_FACES}-face limit; reduce the brush size"
            ),
            Self::Allocation => write!(
                formatter,
                "There is not enough memory to display this brush preview; reduce the brush size"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreviewBuildError {
    Cancelled,
    Failed(VoxelPreviewFailure),
}

#[cfg(test)]
fn build_preview_faces(
    dimensions: [u32; 3],
    coordinates: Vec<[u32; 3]>,
    kind: VoxelPreviewKind,
    max_voxels: usize,
    max_faces: usize,
) -> Result<Vec<PreviewFace>, PreviewBuildError> {
    build_preview_faces_cancellable(dimensions, coordinates, kind, max_voxels, max_faces, || {
        true
    })
}

fn build_preview_faces_cancellable(
    dimensions: [u32; 3],
    coordinates: Vec<[u32; 3]>,
    kind: VoxelPreviewKind,
    max_voxels: usize,
    max_faces: usize,
    is_current: impl Fn() -> bool,
) -> Result<Vec<PreviewFace>, PreviewBuildError> {
    if !is_current() {
        return Err(PreviewBuildError::Cancelled);
    }
    let mut voxels = HashSet::new();
    let initial_voxels = coordinates
        .len()
        .min(max_voxels)
        .min(PREVIEW_ALLOCATION_CHUNK);
    voxels
        .try_reserve(initial_voxels)
        .map_err(|_| PreviewBuildError::Failed(VoxelPreviewFailure::Allocation))?;

    for (coordinate_index, coordinate) in coordinates.into_iter().enumerate() {
        if coordinate_index.is_multiple_of(PREVIEW_CANCELLATION_INTERVAL) && !is_current() {
            return Err(PreviewBuildError::Cancelled);
        }
        if coordinate[0] >= dimensions[0]
            || coordinate[1] >= dimensions[1]
            || coordinate[2] >= dimensions[2]
            || voxels.contains(&coordinate)
        {
            continue;
        }
        if voxels.len() >= max_voxels {
            return Err(PreviewBuildError::Failed(
                VoxelPreviewFailure::TooManyVoxels,
            ));
        }
        if voxels.len() == voxels.capacity() {
            let remaining = max_voxels - voxels.len();
            let additional = voxels.capacity().max(1_024).min(remaining.max(1));
            voxels
                .try_reserve(additional)
                .map_err(|_| PreviewBuildError::Failed(VoxelPreviewFailure::Allocation))?;
        }
        voxels.insert(coordinate);
    }

    let mut faces = Vec::new();
    faces
        .try_reserve_exact(voxels.len().min(max_faces).min(PREVIEW_ALLOCATION_CHUNK))
        .map_err(|_| PreviewBuildError::Failed(VoxelPreviewFailure::Allocation))?;
    for (coordinate_index, coordinate) in voxels.iter().copied().enumerate() {
        if coordinate_index.is_multiple_of(PREVIEW_CANCELLATION_INTERVAL) && !is_current() {
            return Err(PreviewBuildError::Cancelled);
        }
        for axis in 0..3 {
            for positive in [false, true] {
                let neighbor_is_previewed = if positive {
                    let neighbor_axis = coordinate[axis] + 1;
                    if neighbor_axis >= dimensions[axis] {
                        false
                    } else {
                        let mut neighbor = coordinate;
                        neighbor[axis] = neighbor_axis;
                        voxels.contains(&neighbor)
                    }
                } else if coordinate[axis] == 0 {
                    false
                } else {
                    let mut neighbor = coordinate;
                    neighbor[axis] -= 1;
                    voxels.contains(&neighbor)
                };
                if neighbor_is_previewed {
                    continue;
                }
                try_push_preview_face(
                    &mut faces,
                    PreviewFace::new(coordinate, axis, positive, kind),
                    max_faces,
                )?;
            }
        }
    }
    if !is_current() {
        return Err(PreviewBuildError::Cancelled);
    }
    Ok(faces)
}

fn try_push_preview_face(
    faces: &mut Vec<PreviewFace>,
    face: PreviewFace,
    max_faces: usize,
) -> Result<(), PreviewBuildError> {
    if faces.len() >= max_faces {
        return Err(PreviewBuildError::Failed(VoxelPreviewFailure::TooManyFaces));
    }
    if faces.len() == faces.capacity() {
        let remaining = max_faces - faces.len();
        let additional = faces.capacity().max(1_024).min(remaining.max(1));
        faces
            .try_reserve_exact(additional)
            .map_err(|_| PreviewBuildError::Failed(VoxelPreviewFailure::Allocation))?;
    }
    faces.push(face);
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    index_to_world: [[f32; 4]; 4],
    world_to_view: [[f32; 4]; 4],
    index_normal_to_view: [[f32; 4]; 4],
    material_color: [f32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            index_to_world: Mat4::IDENTITY.to_cols_array_2d(),
            world_to_view: Mat4::IDENTITY.to_cols_array_2d(),
            index_normal_to_view: Mat4::IDENTITY.to_cols_array_2d(),
            material_color: [0.80, 0.81, 0.82, 1.0],
        }
    }
}

enum SceneUpdate {
    Reset {
        epoch: u64,
        revision: u64,
        dimensions: [u32; 3],
        affine: Matrix4,
    },
    Invalidate {
        revision: u64,
    },
    Chunk {
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
    },
    Preview {
        epoch: u64,
        revision: u64,
        faces: Vec<PreviewFace>,
    },
    ClearPreview,
}

struct SceneState {
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan_ndc: egui::Vec2,
    aspect: f32,
    updates: VecDeque<SceneUpdate>,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            epoch: 0,
            revision: 0,
            dimensions: [1; 3],
            affine: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            yaw: 0.65,
            pitch: -0.45,
            zoom: 1.0,
            pan_ndc: egui::Vec2::ZERO,
            aspect: 1.0,
            updates: VecDeque::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct SceneBridge {
    state: Arc<Mutex<SceneState>>,
    needs_remesh: Arc<AtomicBool>,
    repaint: Arc<Mutex<Option<egui::Context>>>,
    preview_generation: Arc<AtomicU64>,
    preview_failure: Arc<Mutex<Option<VoxelPreviewFailure>>>,
}

impl SceneBridge {
    pub fn set_repaint_context(&self, context: egui::Context) {
        *self.repaint.lock().expect("3D repaint context poisoned") = Some(context);
    }

    pub fn reset(&self, epoch: u64, revision: u64, dimensions: [u32; 3], affine: Matrix4) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        self.cancel_preview_build_locked();
        if state.epoch != epoch {
            reset_navigation(&mut state);
        }
        state.epoch = epoch;
        state.revision = revision;
        state.dimensions = dimensions;
        state.affine = affine;
        state.updates.clear();
        state.updates.push_back(SceneUpdate::Reset {
            epoch,
            revision,
            dimensions,
            affine,
        });
        drop(state);
        self.request_repaint();
    }

    pub fn invalidate(&self, epoch: u64, revision: u64, _chunks: Vec<ChunkCoord>) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch != epoch {
            return;
        }
        self.cancel_preview_build_locked();
        state.revision = revision;
        remove_queued_preview_updates(&mut state.updates);
        state
            .updates
            .push_back(SceneUpdate::Invalidate { revision });
        drop(state);
        self.request_repaint();
    }

    pub fn insert_chunk(&self, tag: MeshJobTag, quads: Vec<CompactQuad>) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch == tag.document_epoch && state.revision == tag.revision {
            state.updates.push_back(SceneUpdate::Chunk { tag, quads });
            drop(state);
            self.request_repaint();
        }
    }

    /// Replaces the transient 3D preview with the exact supplied voxel footprint.
    ///
    /// This only schedules bounded background work, so even the largest supported
    /// brush never blocks the UI thread. `false` means the document tag was already
    /// stale; later cap or allocation failures are available through
    /// [`Self::take_preview_failure`]. Out-of-bounds coordinates are discarded,
    /// duplicates are collapsed, and adjacent candidates share no internal faces.
    pub fn set_voxel_preview(
        &self,
        epoch: u64,
        revision: u64,
        coordinates: Vec<[u32; 3]>,
        kind: VoxelPreviewKind,
    ) -> bool {
        let (dimensions, generation) = {
            let mut state = self.state.lock().expect("3D scene state poisoned");
            if state.epoch != epoch || state.revision != revision {
                return false;
            }
            let generation = self.next_preview_generation_locked();
            remove_queued_preview_updates(&mut state.updates);
            queue_latest_preview_update(&mut state.updates, SceneUpdate::ClearPreview);
            (state.dimensions, generation)
        };
        self.request_repaint();
        let scene = self.clone();
        rayon::spawn(move || {
            let faces = build_preview_faces_cancellable(
                dimensions,
                coordinates,
                kind,
                MAX_VOXEL_PREVIEW_VOXELS,
                MAX_VOXEL_PREVIEW_FACES,
                || scene.preview_generation.load(Ordering::Acquire) == generation,
            );
            scene.finish_preview_build(epoch, revision, generation, faces);
        });
        true
    }

    pub fn clear_voxel_preview(&self) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        self.cancel_preview_build_locked();
        remove_queued_preview_updates(&mut state.updates);
        queue_latest_preview_update(&mut state.updates, SceneUpdate::ClearPreview);
        drop(state);
        self.request_repaint();
    }

    pub fn clear(&self) {
        self.reset(0, 0, [1; 3], SceneState::default().affine);
    }

    /// Returns and clears the latest failure from a current preview request.
    /// Cancelled or stale builds never report a failure.
    pub fn take_preview_failure(&self) -> Option<VoxelPreviewFailure> {
        self.preview_failure
            .lock()
            .expect("3D preview failure state poisoned")
            .take()
    }

    pub fn orbit(&self, delta: egui::Vec2) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        state.yaw -= delta.x * 0.008;
        state.pitch = (state.pitch + delta.y * 0.008).clamp(-1.5, 1.5);
    }

    pub fn pan(&self, delta_pixels: egui::Vec2, viewport_size: egui::Vec2) {
        if !delta_pixels.x.is_finite()
            || !delta_pixels.y.is_finite()
            || !viewport_size.x.is_finite()
            || !viewport_size.y.is_finite()
            || viewport_size.x <= 0.0
            || viewport_size.y <= 0.0
        {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        state.pan_ndc += egui::vec2(
            2.0 * delta_pixels.x / viewport_size.x,
            -2.0 * delta_pixels.y / viewport_size.y,
        );
    }

    pub fn reset_view(&self) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        reset_navigation(&mut state);
    }

    pub fn zoom(&self, factor: f32) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        state.zoom = (state.zoom * factor).clamp(0.2, 8.0);
    }

    pub fn set_aspect(&self, aspect: f32) {
        self.state.lock().expect("3D scene state poisoned").aspect = aspect.max(0.1);
    }

    #[must_use]
    pub fn ray_for_pointer(&self, normalized: [f32; 2]) -> Ray {
        let state = self.state.lock().expect("3D scene state poisoned");
        let (view, projection, _) = scene_camera(&state);
        let inverse = (projection * view).inverse();
        let near = inverse.project_point3(Vec3::new(normalized[0], normalized[1], 0.0));
        let far = inverse.project_point3(Vec3::new(normalized[0], normalized[1], 1.0));
        let direction = (far - near).normalize_or_zero();
        Ray {
            origin: near.as_dvec3().to_array(),
            direction: direction.as_dvec3().to_array(),
        }
    }

    #[must_use]
    pub fn take_remesh_request(&self) -> bool {
        self.needs_remesh.swap(false, Ordering::AcqRel)
    }

    fn request_repaint(&self) {
        if let Some(context) = self
            .repaint
            .lock()
            .expect("3D repaint context poisoned")
            .as_ref()
        {
            context.request_repaint();
        }
    }

    fn next_preview_generation_locked(&self) -> u64 {
        *self
            .preview_failure
            .lock()
            .expect("3D preview failure state poisoned") = None;
        self.preview_generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn cancel_preview_build_locked(&self) {
        let _ = self.next_preview_generation_locked();
    }

    fn finish_preview_build(
        &self,
        epoch: u64,
        revision: u64,
        generation: u64,
        result: Result<Vec<PreviewFace>, PreviewBuildError>,
    ) {
        if matches!(result, Err(PreviewBuildError::Cancelled)) {
            return;
        }

        let mut state = self.state.lock().expect("3D scene state poisoned");
        if self.preview_generation.load(Ordering::Acquire) != generation
            || state.epoch != epoch
            || state.revision != revision
        {
            return;
        }
        match result {
            Ok(faces) if faces.is_empty() => {}
            Ok(faces) => {
                queue_latest_preview_update(
                    &mut state.updates,
                    SceneUpdate::Preview {
                        epoch,
                        revision,
                        faces,
                    },
                );
            }
            Err(PreviewBuildError::Failed(failure)) => {
                *self
                    .preview_failure
                    .lock()
                    .expect("3D preview failure state poisoned") = Some(failure);
            }
            Err(PreviewBuildError::Cancelled) => return,
        }
        drop(state);
        self.request_repaint();
    }
}

fn remove_queued_preview_updates(updates: &mut VecDeque<SceneUpdate>) {
    updates.retain(|update| {
        !matches!(
            update,
            SceneUpdate::Preview { .. } | SceneUpdate::ClearPreview
        )
    });
}

fn queue_latest_preview_update(updates: &mut VecDeque<SceneUpdate>, update: SceneUpdate) {
    // Preserve producer order. In particular, a replacement chunk submitted just
    // before a reduced stroke overlay must reach the GPU before those preview voxels
    // disappear, otherwise the edited surface visibly blinks out for one frame.
    updates.push_back(update);
}

fn reset_navigation(state: &mut SceneState) {
    let defaults = SceneState::default();
    state.yaw = defaults.yaw;
    state.pitch = defaults.pitch;
    state.zoom = defaults.zoom;
    state.pan_ndc = defaults.pan_ndc;
}

pub struct VolumePaintCallback {
    scene: SceneBridge,
    target_format: wgpu::TextureFormat,
}

impl VolumePaintCallback {
    #[must_use]
    pub const fn new(scene: SceneBridge, target_format: wgpu::TextureFormat) -> Self {
        Self {
            scene,
            target_format,
        }
    }
}

impl CallbackTrait for VolumePaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<GpuVolumeRenderer>().is_none() {
            resources.insert(GpuVolumeRenderer::new(device, self.target_format));
            self.scene.needs_remesh.store(true, Ordering::Release);
        }
        let renderer = resources
            .get_mut::<GpuVolumeRenderer>()
            .expect("renderer was inserted above");
        let mut state = self.scene.state.lock().expect("3D scene state poisoned");
        for _ in 0..MAX_GPU_UPDATES_PER_FRAME {
            let Some(update) = state.updates.pop_front() else {
                break;
            };
            renderer.apply_update(device, update);
        }
        let uniforms = scene_uniforms(&state);
        let has_more_updates = !state.updates.is_empty();
        drop(state);
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        if has_more_updates
            && let Some(context) = self
                .scene
                .repaint
                .lock()
                .expect("3D repaint context poisoned")
                .as_ref()
        {
            context.request_repaint();
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(renderer) = resources.get::<GpuVolumeRenderer>() else {
            return;
        };
        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(0, &renderer.bind_group, &[]);
        for chunk in renderer.chunks.values() {
            render_pass.set_vertex_buffer(0, chunk.buffer.slice(..));
            render_pass.draw(0..6, 0..chunk.instances);
        }
        if let Some(preview) = renderer.preview.as_ref() {
            render_pass.set_pipeline(&renderer.preview_pipeline);
            render_pass.set_bind_group(0, &renderer.bind_group, &[]);
            render_pass.set_vertex_buffer(0, preview.buffer.slice(..));
            render_pass.draw(0..6, 0..preview.faces);
        }
    }
}

struct GpuChunk {
    buffer: wgpu::Buffer,
    instances: u32,
    generation: u64,
}

struct GpuPreview {
    buffer: wgpu::Buffer,
    faces: u32,
}

struct GpuVolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    preview_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    chunks: HashMap<ChunkCoord, GpuChunk>,
    preview: Option<GpuPreview>,
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
}

impl GpuVolumeRenderer {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelInk compact quad shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelInk volume uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelInk volume bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VoxelInk volume bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VoxelInk volume pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelInk compact quad pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<CompactQuad>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let preview_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelInk voxel preview shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_PREVIEW_SHADER.into()),
        });
        let preview_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelInk voxel preview pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &preview_shader,
                entry_point: Some("preview_vertex_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<PreviewFace>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &preview_shader,
                entry_point: Some("preview_fragment_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(preview_color_target(target_format))],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(preview_depth_stencil_state()),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            preview_pipeline,
            uniform_buffer,
            bind_group,
            chunks: HashMap::new(),
            preview: None,
            epoch: 0,
            revision: 0,
            dimensions: [1; 3],
            affine: SceneState::default().affine,
        }
    }

    fn apply_update(&mut self, device: &wgpu::Device, update: SceneUpdate) {
        match update {
            SceneUpdate::Reset {
                epoch,
                revision,
                dimensions,
                affine,
            } => {
                self.chunks.clear();
                self.preview = None;
                self.epoch = epoch;
                self.revision = revision;
                self.dimensions = dimensions;
                self.affine = affine;
            }
            SceneUpdate::Invalidate { revision } => {
                self.revision = revision;
                self.preview = None;
            }
            SceneUpdate::Chunk { tag, quads } => {
                if tag.document_epoch != self.epoch || tag.revision != self.revision {
                    return;
                }
                if self
                    .chunks
                    .get(&tag.chunk)
                    .is_some_and(|chunk| chunk.generation > tag.chunk_generation)
                {
                    return;
                }
                if quads.is_empty() {
                    self.chunks.remove(&tag.chunk);
                    return;
                }
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("VoxelInk chunk quad instances"),
                    contents: bytemuck::cast_slice(&quads),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                self.chunks.insert(
                    tag.chunk,
                    GpuChunk {
                        buffer,
                        instances: quads.len() as u32,
                        generation: tag.chunk_generation,
                    },
                );
            }
            SceneUpdate::Preview {
                epoch,
                revision,
                faces,
            } => {
                let Ok(face_count) = u32::try_from(faces.len()) else {
                    return;
                };
                if epoch != self.epoch
                    || revision != self.revision
                    || faces.is_empty()
                    || faces.len() > MAX_VOXEL_PREVIEW_FACES
                {
                    return;
                }
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("VoxelInk preview boundary-face instances"),
                    contents: bytemuck::cast_slice(&faces),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                self.preview = Some(GpuPreview {
                    buffer,
                    faces: face_count,
                });
            }
            SceneUpdate::ClearPreview => {
                self.preview = None;
            }
        }
    }
}

fn scene_uniforms(state: &SceneState) -> Uniforms {
    let (view, projection, affine) = scene_camera(state);
    let model_view = view * affine;
    Uniforms {
        view_projection: (projection * view).to_cols_array_2d(),
        index_to_world: affine.to_cols_array_2d(),
        world_to_view: view.to_cols_array_2d(),
        index_normal_to_view: normal_matrix(model_view).to_cols_array_2d(),
        ..Uniforms::default()
    }
}

fn normal_matrix(model_view: Mat4) -> Mat4 {
    let determinant = model_view.determinant();
    if determinant.is_finite() && determinant != 0.0 {
        let normal_matrix = model_view.inverse().transpose();
        if normal_matrix.is_finite() {
            return normal_matrix;
        }
    }
    Mat4::IDENTITY
}

fn scene_camera(state: &SceneState) -> (Mat4, Mat4, Mat4) {
    let affine = affine_mat4(state.affine);
    let mut corners = Vec::with_capacity(8);
    for z in [-0.5, state.dimensions[2] as f32 - 0.5] {
        for y in [-0.5, state.dimensions[1] as f32 - 0.5] {
            for x in [-0.5, state.dimensions[0] as f32 - 0.5] {
                corners.push(affine.transform_point3(Vec3::new(x, y, z)));
            }
        }
    }
    let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
    let radius = corners
        .iter()
        .map(|corner| corner.distance(center))
        .fold(1.0_f32, f32::max)
        .max(0.5);
    let direction = Vec3::new(
        state.yaw.cos() * state.pitch.cos(),
        state.yaw.sin() * state.pitch.cos(),
        state.pitch.sin(),
    );
    let eye = center + direction * radius * 2.6 / state.zoom;
    let view = Mat4::look_at_rh(eye, center, Vec3::Z);
    let projection = Mat4::from_translation(Vec3::new(state.pan_ndc.x, state.pan_ndc.y, 0.0))
        * Mat4::perspective_rh(
            45_f32.to_radians(),
            state.aspect,
            radius * 0.01,
            radius * 12.0,
        );
    (view, projection, affine)
}

fn affine_mat4(matrix: Matrix4) -> Mat4 {
    Mat4::from_cols_array_2d(&[
        [
            matrix[0][0] as f32,
            matrix[1][0] as f32,
            matrix[2][0] as f32,
            matrix[3][0] as f32,
        ],
        [
            matrix[0][1] as f32,
            matrix[1][1] as f32,
            matrix[2][1] as f32,
            matrix[3][1] as f32,
        ],
        [
            matrix[0][2] as f32,
            matrix[1][2] as f32,
            matrix[2][2] as f32,
            matrix[3][2] as f32,
        ],
        [
            matrix[0][3] as f32,
            matrix[1][3] as f32,
            matrix[2][3] as f32,
            matrix[3][3] as f32,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_affine_is_transposed_for_wgsl_columns() {
        let matrix = [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 3.0, 0.0, -2.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            affine_mat4(matrix).transform_point3(Vec3::new(1.0, 2.0, 3.0)),
            Vec3::new(7.0, 4.0, 13.0)
        );
    }

    #[test]
    fn inverse_transpose_normal_handles_shear_scale_and_reflection() {
        let model_view = Mat4::from_cols(
            glam::Vec4::new(-2.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.75, 3.0, 0.0, 0.0),
            glam::Vec4::new(0.25, -0.5, 4.0, 0.0),
            glam::Vec4::new(8.0, -3.0, 12.0, 1.0),
        );
        let transformed_tangent_y = model_view.transform_vector3(Vec3::Y);
        let transformed_tangent_z = model_view.transform_vector3(Vec3::Z);
        let transformed_axis_x = model_view.transform_vector3(Vec3::X);
        let transformed_normal = normal_matrix(model_view)
            .transform_vector3(Vec3::X)
            .normalize();

        assert!(transformed_normal.dot(transformed_tangent_y).abs() < 1.0e-6);
        assert!(transformed_normal.dot(transformed_tangent_z).abs() < 1.0e-6);
        assert!(transformed_normal.dot(transformed_axis_x) > 0.0);
    }

    #[test]
    fn production_renderer_uses_four_sample_msaa() {
        assert_eq!(MSAA_SAMPLES, 4);
        assert!(MSAA_SAMPLES.is_power_of_two());
    }

    #[test]
    fn preview_pipeline_is_opaque_and_writes_depth() {
        assert_eq!(
            preview_color_target(wgpu::TextureFormat::Bgra8UnormSrgb).blend,
            Some(wgpu::BlendState::REPLACE)
        );
        assert_eq!(
            preview_depth_stencil_state().depth_write_enabled,
            Some(true)
        );
        assert!(VOXEL_PREVIEW_SHADER.contains("), 1.0);"));
    }

    #[test]
    fn invalidation_queues_only_a_revision_advance() {
        let bridge = SceneBridge::default();
        bridge.reset(4, 8, [32; 3], SceneState::default().affine);
        bridge.invalidate(4, 9, vec![ChunkCoord::new(0, 0, 0)]);
        let state = bridge.state.lock().unwrap();
        assert_eq!(state.revision, 9);
        assert_eq!(state.updates.len(), 2);
        assert!(matches!(
            state.updates.back(),
            Some(SceneUpdate::Invalidate { revision: 9 })
        ));
    }

    #[test]
    fn preview_replaces_pending_preview_and_removes_shared_internal_faces() {
        let bridge = SceneBridge::default();
        bridge.reset(4, 8, [4, 5, 6], SceneState::default().affine);
        assert!(bridge.set_voxel_preview(
            4,
            8,
            vec![[0, 0, 0], [0, 0, 0], [3, 4, 5], [4, 0, 0]],
            VoxelPreviewKind::Paint,
        ));
        assert!(bridge.set_voxel_preview(
            4,
            8,
            vec![[1, 2, 3], [2, 2, 3], [1, 2, 3]],
            VoxelPreviewKind::Erase,
        ));

        wait_for_preview_kind(&bridge, VoxelPreviewKind::Erase);

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 3);
        assert!(matches!(
            state.updates.front(),
            Some(SceneUpdate::Reset { .. })
        ));
        assert!(matches!(
            state.updates.get(1),
            Some(SceneUpdate::ClearPreview)
        ));
        let Some(SceneUpdate::Preview {
            epoch,
            revision,
            faces,
        }) = state.updates.back()
        else {
            panic!("latest voxel preview should follow the pending reset");
        };
        assert_eq!((*epoch, *revision), (4, 8));
        assert_eq!(faces.len(), 10);
        assert!(
            faces
                .iter()
                .all(|face| face.kind() == VoxelPreviewKind::Erase as u32)
        );
        assert!(
            !faces
                .iter()
                .any(|face| { face.origin == [1, 2, 3] && face.axis() == 0 && face.positive() })
        );
        assert!(
            !faces
                .iter()
                .any(|face| { face.origin == [2, 2, 3] && face.axis() == 0 && !face.positive() })
        );
    }

    #[test]
    fn singleton_preview_has_one_face_per_axis_and_direction() {
        let faces =
            build_preview_faces([3; 3], vec![[1, 1, 1]], VoxelPreviewKind::Bridge, 1, 6).unwrap();

        assert_eq!(faces.len(), 6);
        for axis in 0..3 {
            for positive in [false, true] {
                assert!(faces.iter().any(|face| {
                    face.origin == [1, 1, 1]
                        && face.axis() == axis
                        && face.positive() == positive
                        && face.kind() == VoxelPreviewKind::Bridge as u32
                }));
            }
        }
    }

    #[test]
    fn solid_preview_block_work_scales_with_surface_area() {
        let dimensions = [12, 10, 8];
        let coordinates = (0..dimensions[2])
            .flat_map(|z| {
                (0..dimensions[1]).flat_map(move |y| (0..dimensions[0]).map(move |x| [x, y, z]))
            })
            .collect::<Vec<_>>();
        let faces = build_preview_faces(
            dimensions,
            coordinates,
            VoxelPreviewKind::Paint,
            MAX_VOXEL_PREVIEW_VOXELS,
            MAX_VOXEL_PREVIEW_FACES,
        )
        .unwrap();
        let expected_surface_faces = 2
            * (dimensions[0] * dimensions[1]
                + dimensions[0] * dimensions[2]
                + dimensions[1] * dimensions[2]);

        assert_eq!(faces.len(), expected_surface_faces as usize);
        assert!(faces.len() < 12 * 10 * 8);
    }

    #[test]
    fn preview_limits_count_unique_voxels_and_exposed_faces() {
        assert!(matches!(
            build_preview_faces(
                [4; 3],
                vec![[1, 1, 1], [1, 1, 1], [2, 1, 1]],
                VoxelPreviewKind::Paint,
                1,
                6,
            ),
            Err(PreviewBuildError::Failed(
                VoxelPreviewFailure::TooManyVoxels
            ))
        ));
        assert_eq!(
            build_preview_faces(
                [4; 3],
                vec![[1, 1, 1], [1, 1, 1], [8, 8, 8]],
                VoxelPreviewKind::Paint,
                1,
                6,
            )
            .unwrap()
            .len(),
            6
        );
        assert!(matches!(
            build_preview_faces([4; 3], vec![[1, 1, 1]], VoxelPreviewKind::Paint, 1, 5,),
            Err(PreviewBuildError::Failed(VoxelPreviewFailure::TooManyFaces))
        ));
    }

    #[test]
    fn preview_face_build_stops_when_its_generation_is_cancelled() {
        let checks = std::sync::atomic::AtomicUsize::new(0);
        let coordinates = (0..4_096).map(|x| [x, 0, 0]).collect();
        let result = build_preview_faces_cancellable(
            [4_096, 1, 1],
            coordinates,
            VoxelPreviewKind::Paint,
            4_096,
            24_576,
            || checks.fetch_add(1, Ordering::AcqRel) == 0,
        );

        assert!(matches!(result, Err(PreviewBuildError::Cancelled)));
        assert!(checks.load(Ordering::Acquire) >= 2);
    }

    #[test]
    fn configured_caps_admit_a_50_mm_anisotropic_square_volume() {
        // At the sample scan's 0.315 x 0.315 x 0.8 mm spacing, a 50 mm square
        // volume is conservatively bounded by 160 x 160 x 64 voxels.
        let [x, y, z] = [160_usize, 160, 64];
        let voxels = x * y * z;
        let boundary_faces = 2 * (x * y + x * z + y * z);

        assert!(voxels <= MAX_VOXEL_PREVIEW_VOXELS);
        assert!(boundary_faces <= MAX_VOXEL_PREVIEW_FACES);
        assert!(boundary_faces * 6 < voxels * 36 / 100);
    }

    #[test]
    fn stale_preview_is_ignored_and_invalidation_clears_a_pending_preview() {
        let bridge = SceneBridge::default();
        bridge.reset(7, 11, [8; 3], SceneState::default().affine);
        assert!(!bridge.set_voxel_preview(6, 11, vec![[1, 1, 1]], VoxelPreviewKind::Paint));
        assert!(!bridge.set_voxel_preview(7, 10, vec![[1, 1, 1]], VoxelPreviewKind::Paint));
        assert!(bridge.set_voxel_preview(7, 11, vec![[1, 1, 1]], VoxelPreviewKind::Bridge));
        let stale_generation = bridge.preview_generation.load(Ordering::Acquire);
        bridge.invalidate(7, 12, vec![ChunkCoord::new(0, 0, 0)]);
        bridge.finish_preview_build(
            7,
            11,
            stale_generation,
            build_preview_faces(
                [8; 3],
                vec![[1, 1, 1]],
                VoxelPreviewKind::Bridge,
                MAX_VOXEL_PREVIEW_VOXELS,
                MAX_VOXEL_PREVIEW_FACES,
            ),
        );

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 2);
        assert!(matches!(
            state.updates.front(),
            Some(SceneUpdate::Reset { .. })
        ));
        assert!(matches!(
            state.updates.back(),
            Some(SceneUpdate::Invalidate { revision: 12 })
        ));
    }

    #[test]
    fn chunk_update_stays_ahead_of_reduced_preview() {
        let bridge = SceneBridge::default();
        bridge.reset(3, 6, [8; 3], SceneState::default().affine);
        bridge.insert_chunk(
            MeshJobTag {
                document_epoch: 3,
                revision: 6,
                chunk_generation: 2,
                chunk: ChunkCoord::new(0, 0, 0),
            },
            Vec::new(),
        );
        assert!(bridge.set_voxel_preview(3, 6, vec![[2, 2, 2]], VoxelPreviewKind::Paint,));
        wait_for_preview_kind(&bridge, VoxelPreviewKind::Paint);

        let state = bridge.state.lock().unwrap();
        let chunk_index = state
            .updates
            .iter()
            .position(|update| matches!(update, SceneUpdate::Chunk { .. }))
            .unwrap();
        let preview_index = state
            .updates
            .iter()
            .position(|update| matches!(update, SceneUpdate::Preview { .. }))
            .unwrap();
        assert!(chunk_index < preview_index);
    }

    #[test]
    fn only_current_preview_failures_are_reported_once() {
        let bridge = SceneBridge::default();
        bridge.reset(3, 6, [8; 3], SceneState::default().affine);
        let generation = bridge.preview_generation.load(Ordering::Acquire);
        bridge.finish_preview_build(
            3,
            6,
            generation,
            Err(PreviewBuildError::Failed(VoxelPreviewFailure::TooManyFaces)),
        );

        assert_eq!(
            bridge.take_preview_failure(),
            Some(VoxelPreviewFailure::TooManyFaces)
        );
        assert_eq!(bridge.take_preview_failure(), None);

        let stale_generation = generation;
        bridge.clear_voxel_preview();
        bridge.finish_preview_build(
            3,
            6,
            stale_generation,
            Err(PreviewBuildError::Failed(VoxelPreviewFailure::Allocation)),
        );
        assert_eq!(bridge.take_preview_failure(), None);
    }

    #[test]
    fn clearing_preview_replaces_pending_preview_after_document_transition() {
        let bridge = SceneBridge::default();
        bridge.reset(2, 3, [8; 3], SceneState::default().affine);
        assert!(bridge.set_voxel_preview(2, 3, vec![[4, 4, 4]], VoxelPreviewKind::Paint));
        bridge.clear_voxel_preview();

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 2);
        assert!(matches!(
            state.updates.front(),
            Some(SceneUpdate::Reset { .. })
        ));
        assert!(matches!(
            state.updates.back(),
            Some(SceneUpdate::ClearPreview)
        ));
    }

    fn wait_for_preview_kind(bridge: &SceneBridge, kind: VoxelPreviewKind) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let has_preview = bridge.state.lock().unwrap().updates.iter().any(|update| {
                matches!(
                    update,
                    SceneUpdate::Preview { faces, .. }
                        if faces.first().is_some_and(|face| face.kind() == kind as u32)
                )
            });
            if has_preview {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the latest background preview"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn singleton_camera_frames_affine_transformed_voxel_surfaces() {
        let state = SceneState {
            dimensions: [1; 3],
            affine: [
                [100.0, 0.0, 0.0, 0.0],
                [0.0, 100.0, 0.0, 0.0],
                [0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            ..Default::default()
        };
        let (view, _, _) = scene_camera(&state);
        let eye = view.inverse().transform_point3(Vec3::ZERO);
        assert!(eye.length() > 200.0);
    }

    #[test]
    fn rightward_orbit_drag_moves_the_volume_right_on_screen() {
        let viewport = egui::vec2(800.0, 600.0);
        let bridge = SceneBridge::default();
        bridge.set_aspect(viewport.x / viewport.y);
        let landmark = Vec3::new(0.5, 0.0, 0.0);
        let before = {
            let state = bridge.state.lock().unwrap();
            project_to_viewport(&state, landmark, viewport)
        };

        bridge.orbit(egui::vec2(25.0, 0.0));

        let after = {
            let state = bridge.state.lock().unwrap();
            project_to_viewport(&state, landmark, viewport)
        };
        assert!(
            after.x > before.x,
            "a rightward object drag must move a visible volume landmark right ({before:?} -> {after:?})"
        );
    }

    #[test]
    fn pan_moves_projected_volume_by_the_drag_delta() {
        let viewport = egui::vec2(800.0, 600.0);
        let drag = egui::vec2(80.0, -45.0);
        let bridge = SceneBridge::default();
        bridge.set_aspect(viewport.x / viewport.y);
        let before = {
            let state = bridge.state.lock().unwrap();
            project_to_viewport(&state, Vec3::ZERO, viewport)
        };

        bridge.pan(drag, viewport);

        let after = {
            let state = bridge.state.lock().unwrap();
            project_to_viewport(&state, Vec3::ZERO, viewport)
        };
        let projected_delta = after - before;
        assert!((projected_delta.x - drag.x).abs() < 0.001);
        assert!((projected_delta.y - drag.y).abs() < 0.001);
    }

    #[test]
    fn navigation_survives_remesh_reset_and_clears_for_a_new_document() {
        let bridge = SceneBridge::default();
        let affine = SceneState::default().affine;
        bridge.reset(4, 1, [32; 3], affine);
        bridge.pan(egui::vec2(40.0, -30.0), egui::vec2(800.0, 600.0));
        bridge.orbit(egui::vec2(25.0, 10.0));
        bridge.zoom(1.5);
        let navigation = {
            let state = bridge.state.lock().unwrap();
            (state.pan_ndc, state.yaw, state.pitch, state.zoom)
        };

        bridge.reset(4, 2, [32; 3], affine);
        {
            let state = bridge.state.lock().unwrap();
            assert_eq!(
                (state.pan_ndc, state.yaw, state.pitch, state.zoom),
                navigation
            );
        }

        bridge.reset(5, 1, [32; 3], affine);
        let state = bridge.state.lock().unwrap();
        let defaults = SceneState::default();
        assert_eq!(state.pan_ndc, defaults.pan_ndc);
        assert_eq!(state.yaw, defaults.yaw);
        assert_eq!(state.pitch, defaults.pitch);
        assert_eq!(state.zoom, defaults.zoom);
    }

    #[test]
    fn reset_view_restores_navigation_defaults_without_changing_the_document() {
        let bridge = SceneBridge::default();
        let affine = SceneState::default().affine;
        bridge.reset(7, 12, [32; 3], affine);
        bridge.orbit(egui::vec2(40.0, 25.0));
        bridge.pan(egui::vec2(80.0, -30.0), egui::vec2(800.0, 600.0));
        bridge.zoom(2.0);

        bridge.reset_view();

        let state = bridge.state.lock().unwrap();
        let defaults = SceneState::default();
        assert_eq!(state.epoch, 7);
        assert_eq!(state.revision, 12);
        assert_eq!(state.dimensions, [32; 3]);
        assert_eq!(state.yaw, defaults.yaw);
        assert_eq!(state.pitch, defaults.pitch);
        assert_eq!(state.zoom, defaults.zoom);
        assert_eq!(state.pan_ndc, defaults.pan_ndc);
    }

    fn project_to_viewport(
        state: &SceneState,
        world_position: Vec3,
        viewport: egui::Vec2,
    ) -> egui::Vec2 {
        let (view, projection, _) = scene_camera(state);
        let ndc = (projection * view).project_point3(world_position);
        egui::vec2(
            (ndc.x + 1.0) * 0.5 * viewport.x,
            (1.0 - ndc.y) * 0.5 * viewport.y,
        )
    }

    #[test]
    fn headless_software_vulkan_creates_the_production_pipeline() {
        if std::env::var_os("VOXEL_INK_REQUIRE_SOFTWARE_VULKAN").is_none() {
            eprintln!("set VOXEL_INK_REQUIRE_SOFTWARE_VULKAN=1 to exercise Mesa Vulkan");
            return;
        }

        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: true,
            compatible_surface: None,
        }))
        .expect("Mesa software Vulkan adapter must be available");
        let info = adapter.get_info();
        assert_eq!(info.backend, wgpu::Backend::Vulkan);
        assert_eq!(info.device_type, wgpu::DeviceType::Cpu);
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let mut renderer = GpuVolumeRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        let affine = SceneState::default().affine;
        let chunk = ChunkCoord::new(0, 0, 0);
        renderer.apply_update(
            &device,
            SceneUpdate::Reset {
                epoch: 4,
                revision: 8,
                dimensions: [32; 3],
                affine,
            },
        );
        renderer.apply_update(
            &device,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 3,
                    chunk,
                },
                quads: vec![
                    CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true)
                        .expect("unit face is representable"),
                ],
            },
        );
        assert!(renderer.chunks.contains_key(&chunk));

        renderer.apply_update(
            &device,
            SceneUpdate::Preview {
                epoch: 4,
                revision: 8,
                faces: build_preview_faces(
                    [32; 3],
                    vec![[2, 3, 4], [3, 3, 4]],
                    VoxelPreviewKind::Erase,
                    MAX_VOXEL_PREVIEW_VOXELS,
                    MAX_VOXEL_PREVIEW_FACES,
                )
                .unwrap(),
            },
        );
        assert_eq!(
            renderer.preview.as_ref().map(|preview| preview.faces),
            Some(10)
        );
        assert!(
            renderer.chunks.contains_key(&chunk),
            "transient preview replacement must not alter resident chunk meshes"
        );

        renderer.apply_update(&device, SceneUpdate::Invalidate { revision: 9 });
        assert_eq!(renderer.revision, 9);
        assert!(renderer.preview.is_none());
        assert!(
            renderer.chunks.contains_key(&chunk),
            "the previous mesh must remain resident until its replacement is ready"
        );

        renderer.apply_update(
            &device,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 9,
                    chunk_generation: 4,
                    chunk,
                },
                quads: Vec::new(),
            },
        );
        assert!(
            !renderer.chunks.contains_key(&chunk),
            "an empty replacement removes the old resident mesh atomically"
        );
    }
}
