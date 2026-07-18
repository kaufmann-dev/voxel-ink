use std::{
    collections::{HashMap, VecDeque},
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
    mesh::{ChunkCoord, CompactQuad, MeshJobTag, MeshUpdatePriority},
    volume::Matrix4,
};

const SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    index_to_world: mat4x4<f32>,
    world_to_view: mat4x4<f32>,
    index_normal_to_view: mat4x4<f32>,
    material_color: vec4<f32>,
    near_surface_depth: f32,
    inverse_scene_depth_span: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
    @location(1) view_position: vec3<f32>,
    @location(2) corner_occlusion: f32,
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
    let flipped_diagonal = ((packed >> 21u) & 1u) != 0u;

    var corner = 0u;
    if (!flipped_diagonal) {
        if (vertex_index == 2u || vertex_index == 4u) {
            corner = 2u;
        } else if ((positive && vertex_index == 1u) || (!positive && vertex_index == 5u)) {
            corner = 1u;
        } else if ((positive && vertex_index == 5u) || (!positive && vertex_index == 1u)) {
            corner = 3u;
        }
    } else if (positive) {
        let corners = array<u32, 6>(0u, 1u, 3u, 1u, 2u, 3u);
        corner = corners[vertex_index];
    } else {
        let corners = array<u32, 6>(0u, 3u, 1u, 1u, 3u, 2u);
        corner = corners[vertex_index];
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
    output.corner_occlusion = f32((packed >> (13u + corner * 2u)) & 3u);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_direction = normalize(-input.view_position);
    var normal = normalize(input.view_normal);
    let geometric_back_facing = dot(normal, view_direction) < 0.0;
    let surface_factor = select(1.0, 0.68, geometric_back_facing);
    if (geometric_back_facing) {
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
    let base_lighting = clamp(
        0.16 + 0.52 * key + 0.22 * fill + 0.10 * sky + 0.18 * rim,
        0.12,
        1.08,
    );
    let scene_depth = -input.view_position.z;
    let normalized_scene_depth = clamp(
        (scene_depth - uniforms.near_surface_depth) * uniforms.inverse_scene_depth_span,
        0.0,
        1.0,
    );
    let depth_factor = mix(1.0, 0.62, normalized_scene_depth);
    let voxel_ao_factor = mix(1.0, 0.76, clamp(input.corner_occlusion / 3.0, 0.0, 1.0));
    let lighting = base_lighting * depth_factor * surface_factor * voxel_ao_factor;
    return vec4<f32>(uniforms.material_color.rgb * lighting, 1.0);
}
"#;

const VOXEL_HIGHLIGHT_SHADER: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    index_to_world: mat4x4<f32>,
    world_to_view: mat4x4<f32>,
    index_normal_to_view: mat4x4<f32>,
    material_color: vec4<f32>,
    near_surface_depth: f32,
    inverse_scene_depth_span: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) face_uv: vec2<f32>,
    @location(1) @interpolate(flat) highlight_kind: u32,
};

@vertex
fn highlight_vertex_main(
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
    let highlight_kind = packed_face >> 3u;
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

    let world_position = uniforms.index_to_world * vec4<f32>(index_position, 1.0);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.face_uv = face_uv;
    output.highlight_kind = highlight_kind;
    return output;
}

@fragment
fn highlight_fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = vec3<f32>(66.0, 214.0, 124.0) / 255.0;
    var fill_alpha = 20.0 / 255.0;
    if (input.highlight_kind == 1u) {
        base_color = vec3<f32>(255.0, 91.0, 69.0) / 255.0;
        fill_alpha = 56.0 / 255.0;
    } else if (input.highlight_kind == 2u) {
        base_color = vec3<f32>(244.0, 166.0, 42.0) / 255.0;
    } else if (input.highlight_kind == 3u) {
        base_color = vec3<f32>(255.0, 216.0, 77.0) / 255.0;
        fill_alpha = 56.0 / 255.0;
    }

    // A one-pixel-ish antialiased edge on every exposed voxel face keeps adjacent
    // candidates visibly discrete instead of turning the highlight into a smooth blob.
    let edge_distance = min(
        min(input.face_uv.x, 1.0 - input.face_uv.x),
        min(input.face_uv.y, 1.0 - input.face_uv.y),
    );
    let edge_width = max(fwidth(edge_distance) * 1.35, 0.006);
    let edge_mix = 1.0 - smoothstep(0.0, edge_width, edge_distance);
    let alpha = mix(fill_alpha, 230.0 / 255.0, edge_mix);
    return vec4<f32>(base_color, alpha);
}
"#;

const MAX_GPU_UPDATES_PER_FRAME: usize = 4;
pub const MSAA_SAMPLES: u32 = 4;
// These independent caps keep CPU adjacency work and GPU boundary work bounded.
// Four million 16-byte faces occupy 64 MB and invoke at most 24 million vertices;
// the sample scan's 50 mm volume is about 1.64 M voxels but only 92,160 faces.
pub const MAX_VOXEL_HIGHLIGHT_VOXELS: usize = 2_000_000;
pub const MAX_VOXEL_HIGHLIGHT_FACES: usize = 4_000_000;
const HIGHLIGHT_ALLOCATION_CHUNK: usize = 65_536;
const HIGHLIGHT_CANCELLATION_INTERVAL: usize = 1_024;

fn highlight_color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn highlight_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: Some(false),
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
pub enum VoxelHighlightKind {
    Paint = 0,
    Erase = 1,
    BridgePath = 2,
    BridgeAnchor = 3,
}

impl VoxelHighlightKind {
    #[must_use]
    pub const fn color(self) -> [u8; 3] {
        match self {
            Self::Paint => [0x42, 0xd6, 0x7c],
            Self::Erase => [0xff, 0x5b, 0x45],
            Self::BridgePath => [0xf4, 0xa6, 0x2a],
            Self::BridgeAnchor => [0xff, 0xd8, 0x4d],
        }
    }

    #[must_use]
    pub const fn fill_alpha(self) -> u8 {
        match self {
            Self::Paint | Self::BridgePath => 20,
            Self::Erase | Self::BridgeAnchor => 56,
        }
    }

    #[must_use]
    pub const fn outline_alpha(self) -> u8 {
        230
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct HighlightFace {
    origin: [u32; 3],
    packed: u32,
}

impl HighlightFace {
    const fn new(origin: [u32; 3], axis: usize, positive: bool, kind: VoxelHighlightKind) -> Self {
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
pub enum VoxelHighlightFailure {
    TooManyVoxels,
    TooManyFaces,
    Allocation,
}

impl std::fmt::Display for VoxelHighlightFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyVoxels => write!(
                formatter,
                "Voxel highlight exceeds the {MAX_VOXEL_HIGHLIGHT_VOXELS}-voxel limit; reduce the brush size"
            ),
            Self::TooManyFaces => write!(
                formatter,
                "Voxel highlight exceeds the {MAX_VOXEL_HIGHLIGHT_FACES}-face limit; reduce the brush size"
            ),
            Self::Allocation => write!(
                formatter,
                "There is not enough memory to display this voxel highlight; reduce the brush size"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HighlightBuildError {
    Cancelled,
    Failed(VoxelHighlightFailure),
}

#[cfg(test)]
fn build_highlight_faces(
    dimensions: [u32; 3],
    highlights: Vec<([u32; 3], VoxelHighlightKind)>,
    max_voxels: usize,
    max_faces: usize,
) -> Result<Vec<HighlightFace>, HighlightBuildError> {
    build_highlight_faces_cancellable(dimensions, highlights, max_voxels, max_faces, || true)
}

fn build_highlight_faces_cancellable(
    dimensions: [u32; 3],
    highlights: Vec<([u32; 3], VoxelHighlightKind)>,
    max_voxels: usize,
    max_faces: usize,
    is_current: impl Fn() -> bool,
) -> Result<Vec<HighlightFace>, HighlightBuildError> {
    if !is_current() {
        return Err(HighlightBuildError::Cancelled);
    }
    let mut voxels = HashMap::new();
    let initial_voxels = highlights
        .len()
        .min(max_voxels)
        .min(HIGHLIGHT_ALLOCATION_CHUNK);
    voxels
        .try_reserve(initial_voxels)
        .map_err(|_| HighlightBuildError::Failed(VoxelHighlightFailure::Allocation))?;

    for (highlight_index, (coordinate, kind)) in highlights.into_iter().enumerate() {
        if highlight_index.is_multiple_of(HIGHLIGHT_CANCELLATION_INTERVAL) && !is_current() {
            return Err(HighlightBuildError::Cancelled);
        }
        if coordinate[0] >= dimensions[0]
            || coordinate[1] >= dimensions[1]
            || coordinate[2] >= dimensions[2]
        {
            continue;
        }
        if let Some(existing) = voxels.get_mut(&coordinate) {
            if kind == VoxelHighlightKind::BridgeAnchor {
                *existing = kind;
            }
            continue;
        }
        if voxels.len() >= max_voxels {
            return Err(HighlightBuildError::Failed(
                VoxelHighlightFailure::TooManyVoxels,
            ));
        }
        if voxels.len() == voxels.capacity() {
            let remaining = max_voxels - voxels.len();
            let additional = voxels.capacity().max(1_024).min(remaining.max(1));
            voxels
                .try_reserve(additional)
                .map_err(|_| HighlightBuildError::Failed(VoxelHighlightFailure::Allocation))?;
        }
        voxels.insert(coordinate, kind);
    }

    let mut faces = Vec::new();
    faces
        .try_reserve_exact(voxels.len().min(max_faces).min(HIGHLIGHT_ALLOCATION_CHUNK))
        .map_err(|_| HighlightBuildError::Failed(VoxelHighlightFailure::Allocation))?;
    for (highlight_index, (&coordinate, &kind)) in voxels.iter().enumerate() {
        if highlight_index.is_multiple_of(HIGHLIGHT_CANCELLATION_INTERVAL) && !is_current() {
            return Err(HighlightBuildError::Cancelled);
        }
        for axis in 0..3 {
            for positive in [false, true] {
                let neighbor_is_highlighted = if positive {
                    let neighbor_axis = coordinate[axis] + 1;
                    if neighbor_axis >= dimensions[axis] {
                        false
                    } else {
                        let mut neighbor = coordinate;
                        neighbor[axis] = neighbor_axis;
                        voxels.contains_key(&neighbor)
                    }
                } else if coordinate[axis] == 0 {
                    false
                } else {
                    let mut neighbor = coordinate;
                    neighbor[axis] -= 1;
                    voxels.contains_key(&neighbor)
                };
                if neighbor_is_highlighted {
                    continue;
                }
                try_push_highlight_face(
                    &mut faces,
                    HighlightFace::new(coordinate, axis, positive, kind),
                    max_faces,
                )?;
            }
        }
    }
    if !is_current() {
        return Err(HighlightBuildError::Cancelled);
    }
    Ok(faces)
}

fn try_push_highlight_face(
    faces: &mut Vec<HighlightFace>,
    face: HighlightFace,
    max_faces: usize,
) -> Result<(), HighlightBuildError> {
    if faces.len() >= max_faces {
        return Err(HighlightBuildError::Failed(
            VoxelHighlightFailure::TooManyFaces,
        ));
    }
    if faces.len() == faces.capacity() {
        let remaining = max_faces - faces.len();
        let additional = faces.capacity().max(1_024).min(remaining.max(1));
        faces
            .try_reserve_exact(additional)
            .map_err(|_| HighlightBuildError::Failed(VoxelHighlightFailure::Allocation))?;
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
    near_surface_depth: f32,
    inverse_scene_depth_span: f32,
    _padding: [f32; 2],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            index_to_world: Mat4::IDENTITY.to_cols_array_2d(),
            world_to_view: Mat4::IDENTITY.to_cols_array_2d(),
            index_normal_to_view: Mat4::IDENTITY.to_cols_array_2d(),
            material_color: [0.80, 0.81, 0.82, 1.0],
            near_surface_depth: 0.0,
            inverse_scene_depth_span: 1.0,
            _padding: [0.0; 2],
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
        priority: MeshUpdatePriority,
    },
    Highlight {
        epoch: u64,
        revision: u64,
        faces: Vec<HighlightFace>,
    },
    ClearHighlight,
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
    highlight_generation: Arc<AtomicU64>,
    highlight_failure: Arc<Mutex<Option<VoxelHighlightFailure>>>,
}

impl SceneBridge {
    pub fn set_repaint_context(&self, context: egui::Context) {
        *self.repaint.lock().expect("3D repaint context poisoned") = Some(context);
    }

    pub fn reset(&self, epoch: u64, revision: u64, dimensions: [u32; 3], affine: Matrix4) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        self.cancel_highlight_build_locked();
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
        self.cancel_highlight_build_locked();
        state.revision = revision;
        remove_queued_highlight_updates(&mut state.updates);
        state
            .updates
            .push_back(SceneUpdate::Invalidate { revision });
        drop(state);
        self.request_repaint();
    }

    pub fn insert_chunk(
        &self,
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
        priority: MeshUpdatePriority,
    ) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch == tag.document_epoch && state.revision == tag.revision {
            queue_chunk_update(
                &mut state.updates,
                SceneUpdate::Chunk {
                    tag,
                    quads,
                    priority,
                },
                priority,
            );
            drop(state);
            self.request_repaint();
        }
    }

    /// Replaces the transient 3D highlight with the exact supplied voxel set.
    ///
    /// This only schedules bounded background work, so even the largest supported
    /// brush never blocks the UI thread. `false` means the document tag was already
    /// stale; later cap or allocation failures are available through
    /// [`Self::take_highlight_failure`]. Out-of-bounds coordinates are discarded,
    /// duplicates are collapsed with bridge anchors taking precedence, and adjacent
    /// candidates share no internal faces even when their highlight kinds differ.
    pub fn set_voxel_highlight(
        &self,
        epoch: u64,
        revision: u64,
        highlights: Vec<([u32; 3], VoxelHighlightKind)>,
    ) -> bool {
        let (dimensions, generation) = {
            let mut state = self.state.lock().expect("3D scene state poisoned");
            if state.epoch != epoch || state.revision != revision {
                return false;
            }
            let generation = self.next_highlight_generation_locked();
            remove_queued_highlight_updates(&mut state.updates);
            (state.dimensions, generation)
        };
        let scene = self.clone();
        rayon::spawn(move || {
            let faces = build_highlight_faces_cancellable(
                dimensions,
                highlights,
                MAX_VOXEL_HIGHLIGHT_VOXELS,
                MAX_VOXEL_HIGHLIGHT_FACES,
                || scene.highlight_generation.load(Ordering::Acquire) == generation,
            );
            scene.finish_highlight_build(epoch, revision, generation, faces);
        });
        true
    }

    pub fn clear_voxel_highlight(&self) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        self.cancel_highlight_build_locked();
        remove_queued_highlight_updates(&mut state.updates);
        queue_latest_highlight_update(&mut state.updates, SceneUpdate::ClearHighlight);
        drop(state);
        self.request_repaint();
    }

    pub fn clear(&self) {
        self.reset(0, 0, [1; 3], SceneState::default().affine);
    }

    /// Returns and clears the latest failure from a current highlight request.
    /// Cancelled or stale builds never report a failure.
    pub fn take_highlight_failure(&self) -> Option<VoxelHighlightFailure> {
        self.highlight_failure
            .lock()
            .expect("3D highlight failure state poisoned")
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
        let (view, projection, _, _, _) = scene_camera(&state);
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

    fn next_highlight_generation_locked(&self) -> u64 {
        *self
            .highlight_failure
            .lock()
            .expect("3D highlight failure state poisoned") = None;
        self.highlight_generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn cancel_highlight_build_locked(&self) {
        let _ = self.next_highlight_generation_locked();
    }

    fn finish_highlight_build(
        &self,
        epoch: u64,
        revision: u64,
        generation: u64,
        result: Result<Vec<HighlightFace>, HighlightBuildError>,
    ) {
        if matches!(result, Err(HighlightBuildError::Cancelled)) {
            return;
        }

        let mut state = self.state.lock().expect("3D scene state poisoned");
        if self.highlight_generation.load(Ordering::Acquire) != generation
            || state.epoch != epoch
            || state.revision != revision
        {
            return;
        }
        match result {
            Ok(faces) if faces.is_empty() => {
                queue_latest_highlight_update(&mut state.updates, SceneUpdate::ClearHighlight);
            }
            Ok(faces) => {
                queue_latest_highlight_update(
                    &mut state.updates,
                    SceneUpdate::Highlight {
                        epoch,
                        revision,
                        faces,
                    },
                );
            }
            Err(HighlightBuildError::Failed(failure)) => {
                queue_latest_highlight_update(&mut state.updates, SceneUpdate::ClearHighlight);
                *self
                    .highlight_failure
                    .lock()
                    .expect("3D highlight failure state poisoned") = Some(failure);
            }
            Err(HighlightBuildError::Cancelled) => return,
        }
        drop(state);
        self.request_repaint();
    }
}

fn remove_queued_highlight_updates(updates: &mut VecDeque<SceneUpdate>) {
    updates.retain(|update| {
        !matches!(
            update,
            SceneUpdate::Highlight { .. } | SceneUpdate::ClearHighlight
        )
    });
}

fn queue_latest_highlight_update(updates: &mut VecDeque<SceneUpdate>, update: SceneUpdate) {
    let barrier_end = updates
        .iter()
        .rposition(|queued| {
            matches!(
                queued,
                SceneUpdate::Reset { .. } | SceneUpdate::Invalidate { .. }
            )
        })
        .map_or(0, |index| index + 1);
    let insertion_index = updates
        .iter()
        .enumerate()
        .skip(barrier_end)
        .find_map(|(index, queued)| {
            matches!(
                queued,
                SceneUpdate::Chunk {
                    priority: MeshUpdatePriority::Background,
                    ..
                }
            )
            .then_some(index)
        })
        .unwrap_or(updates.len());
    updates.insert(insertion_index, update);
}

fn queue_chunk_update(
    updates: &mut VecDeque<SceneUpdate>,
    update: SceneUpdate,
    priority: MeshUpdatePriority,
) {
    if priority == MeshUpdatePriority::Background {
        updates.push_back(update);
        return;
    }

    let barrier_end = updates
        .iter()
        .rposition(|queued| {
            matches!(
                queued,
                SceneUpdate::Reset { .. } | SceneUpdate::Invalidate { .. }
            )
        })
        .map_or(0, |index| index + 1);
    let incoming_tag = match &update {
        SceneUpdate::Chunk { tag, .. } => *tag,
        _ => unreachable!("only mesh chunks use the prioritized update queue"),
    };

    for index in (barrier_end..updates.len()).rev() {
        let remove = matches!(
            &updates[index],
            SceneUpdate::Chunk { tag, .. }
                if tag.chunk == incoming_tag.chunk
                    && tag.chunk_generation <= incoming_tag.chunk_generation
        );
        if remove {
            let _ = updates.remove(index);
        }
    }

    let insertion_index = updates
        .iter()
        .enumerate()
        .skip(barrier_end)
        .find_map(|(index, queued)| {
            matches!(
                queued,
                SceneUpdate::Chunk {
                    priority: MeshUpdatePriority::Background,
                    ..
                }
            )
            .then_some(index)
        })
        .unwrap_or(updates.len());
    updates.insert(insertion_index, update);
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
        if let Some(highlight) = renderer.highlight.as_ref() {
            render_pass.set_pipeline(&renderer.highlight_pipeline);
            render_pass.set_bind_group(0, &renderer.bind_group, &[]);
            render_pass.set_vertex_buffer(0, highlight.buffer.slice(..));
            render_pass.draw(0..6, 0..highlight.faces);
        }
    }
}

struct GpuChunk {
    buffer: wgpu::Buffer,
    instances: u32,
}

struct GpuHighlight {
    buffer: wgpu::Buffer,
    faces: u32,
}

struct GpuVolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    chunks: HashMap<ChunkCoord, GpuChunk>,
    chunk_generations: HashMap<ChunkCoord, u64>,
    highlight: Option<GpuHighlight>,
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
        let highlight_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelInk voxel highlight shader"),
            source: wgpu::ShaderSource::Wgsl(VOXEL_HIGHLIGHT_SHADER.into()),
        });
        let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelInk voxel highlight pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &highlight_shader,
                entry_point: Some("highlight_vertex_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<HighlightFace>() as u64,
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
                module: &highlight_shader,
                entry_point: Some("highlight_fragment_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(highlight_color_target(target_format))],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(highlight_depth_stencil_state()),
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
            highlight_pipeline,
            uniform_buffer,
            bind_group,
            chunks: HashMap::new(),
            chunk_generations: HashMap::new(),
            highlight: None,
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
                self.chunk_generations.clear();
                self.highlight = None;
                self.epoch = epoch;
                self.revision = revision;
                self.dimensions = dimensions;
                self.affine = affine;
            }
            SceneUpdate::Invalidate { revision } => {
                self.revision = revision;
                self.highlight = None;
            }
            SceneUpdate::Chunk { tag, quads, .. } => {
                if tag.document_epoch != self.epoch || tag.revision != self.revision {
                    return;
                }
                if self
                    .chunk_generations
                    .get(&tag.chunk)
                    .is_some_and(|generation| *generation > tag.chunk_generation)
                {
                    return;
                }
                self.chunk_generations
                    .insert(tag.chunk, tag.chunk_generation);
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
                    },
                );
            }
            SceneUpdate::Highlight {
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
                    || faces.len() > MAX_VOXEL_HIGHLIGHT_FACES
                {
                    return;
                }
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("VoxelInk highlight boundary-face instances"),
                    contents: bytemuck::cast_slice(&faces),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                self.highlight = Some(GpuHighlight {
                    buffer,
                    faces: face_count,
                });
            }
            SceneUpdate::ClearHighlight => {
                self.highlight = None;
            }
        }
    }
}

fn scene_uniforms(state: &SceneState) -> Uniforms {
    let (view, projection, affine, near_surface_depth, inverse_scene_depth_span) =
        scene_camera(state);
    let model_view = view * affine;
    Uniforms {
        view_projection: (projection * view).to_cols_array_2d(),
        index_to_world: affine.to_cols_array_2d(),
        world_to_view: view.to_cols_array_2d(),
        index_normal_to_view: normal_matrix(model_view).to_cols_array_2d(),
        near_surface_depth,
        inverse_scene_depth_span,
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

fn scene_camera(state: &SceneState) -> (Mat4, Mat4, Mat4, f32, f32) {
    let affine = affine_mat4(state.affine);
    let corners: [Vec3; 8] = std::array::from_fn(|corner| {
        affine.transform_point3(Vec3::new(
            if corner & 1 == 0 {
                -0.5
            } else {
                state.dimensions[0] as f32 - 0.5
            },
            if corner & 2 == 0 {
                -0.5
            } else {
                state.dimensions[1] as f32 - 0.5
            },
            if corner & 4 == 0 {
                -0.5
            } else {
                state.dimensions[2] as f32 - 0.5
            },
        ))
    });
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
    let (near_surface_depth, far_surface_depth) =
        corners
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(near, far), corner| {
                let depth = -view.transform_point3(*corner).z;
                (near.min(depth), far.max(depth))
            });
    let inverse_scene_depth_span =
        1.0 / (far_surface_depth - near_surface_depth).max(radius * 1.0e-6);
    let projection = Mat4::from_translation(Vec3::new(state.pan_ndc.x, state.pan_ndc.y, 0.0))
        * Mat4::perspective_rh(
            45_f32.to_radians(),
            state.aspect,
            radius * 0.01,
            radius * 12.0,
        );
    (
        view,
        projection,
        affine,
        near_surface_depth,
        inverse_scene_depth_span,
    )
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
        assert_eq!(size_of::<Uniforms>(), 288);
    }

    #[test]
    fn highlight_pipeline_alpha_blends_without_writing_depth() {
        assert_eq!(
            highlight_color_target(wgpu::TextureFormat::Bgra8UnormSrgb).blend,
            Some(wgpu::BlendState::ALPHA_BLENDING)
        );
        assert_eq!(
            highlight_depth_stencil_state().depth_write_enabled,
            Some(false)
        );
        assert!(VOXEL_HIGHLIGHT_SHADER.contains("230.0 / 255.0"));
        assert!(!VOXEL_HIGHLIGHT_SHADER.contains("0.0, 0.82, 1.0"));
    }

    #[test]
    fn highlight_palette_exposes_exact_colors_and_alphas() {
        assert_eq!(VoxelHighlightKind::Paint.color(), [0x42, 0xd6, 0x7c]);
        assert_eq!(VoxelHighlightKind::Erase.color(), [0xff, 0x5b, 0x45]);
        assert_eq!(VoxelHighlightKind::BridgePath.color(), [0xf4, 0xa6, 0x2a]);
        assert_eq!(VoxelHighlightKind::BridgeAnchor.color(), [0xff, 0xd8, 0x4d]);
        assert_eq!(VoxelHighlightKind::Paint.fill_alpha(), 20);
        assert_eq!(VoxelHighlightKind::Erase.fill_alpha(), 56);
        assert_eq!(VoxelHighlightKind::BridgePath.fill_alpha(), 20);
        assert_eq!(VoxelHighlightKind::BridgeAnchor.fill_alpha(), 56);
        for kind in [
            VoxelHighlightKind::Paint,
            VoxelHighlightKind::Erase,
            VoxelHighlightKind::BridgePath,
            VoxelHighlightKind::BridgeAnchor,
        ] {
            assert_eq!(kind.outline_alpha(), 230);
        }
    }

    #[test]
    fn equal_normal_fragments_darken_with_depth_internal_facing_and_voxel_occlusion() {
        let depth_factor = |depth: f32, near: f32, inverse_span: f32| {
            let normalized = ((depth - near) * inverse_span).clamp(0.0, 1.0);
            1.0 + (0.62 - 1.0) * normalized
        };
        let near = 10.0;
        let inverse_span = 0.05;
        let nearest = depth_factor(near, near, inverse_span);
        let middle = depth_factor(20.0, near, inverse_span);
        let farthest = depth_factor(30.0, near, inverse_span);

        assert!((nearest - 1.0).abs() < f32::EPSILON);
        assert!(nearest > middle && middle > farthest);
        assert!((farthest - 0.62).abs() < f32::EPSILON);
        assert!(0.68 * nearest < nearest);
        let ao_factor = |level: f32| 1.0 + (0.76 - 1.0) * (level / 3.0).clamp(0.0, 1.0);
        assert!((ao_factor(0.0) - 1.0).abs() < f32::EPSILON);
        assert!(ao_factor(0.0) > ao_factor(1.0));
        assert!(ao_factor(1.0) > ao_factor(2.0));
        assert!(ao_factor(2.0) > ao_factor(3.0));
        assert!((ao_factor(3.0) - 0.76).abs() < f32::EPSILON);
        assert!(SHADER.contains("mix(1.0, 0.62, normalized_scene_depth)"));
        assert!(SHADER.contains("select(1.0, 0.68, geometric_back_facing)"));
        assert!(SHADER.contains("mix(1.0, 0.76"));
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
    fn highlight_replaces_pending_highlight_and_removes_shared_internal_faces() {
        let bridge = SceneBridge::default();
        bridge.reset(4, 8, [4, 5, 6], SceneState::default().affine);
        assert!(bridge.set_voxel_highlight(
            4,
            8,
            typed_highlights(
                vec![[0, 0, 0], [0, 0, 0], [3, 4, 5], [4, 0, 0]],
                VoxelHighlightKind::Paint,
            ),
        ));
        assert!(bridge.set_voxel_highlight(
            4,
            8,
            typed_highlights(
                vec![[1, 2, 3], [2, 2, 3], [1, 2, 3]],
                VoxelHighlightKind::Erase,
            ),
        ));

        wait_for_highlight_kind(&bridge, VoxelHighlightKind::Erase);

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 2);
        assert!(matches!(
            state.updates.front(),
            Some(SceneUpdate::Reset { .. })
        ));
        let Some(SceneUpdate::Highlight {
            epoch,
            revision,
            faces,
        }) = state.updates.back()
        else {
            panic!("latest voxel highlight should follow the pending reset");
        };
        assert_eq!((*epoch, *revision), (4, 8));
        assert_eq!(faces.len(), 10);
        assert!(
            faces
                .iter()
                .all(|face| face.kind() == VoxelHighlightKind::Erase as u32)
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
    fn singleton_highlight_has_one_face_per_axis_and_direction() {
        let faces = build_highlight_faces(
            [3; 3],
            vec![([1, 1, 1], VoxelHighlightKind::BridgePath)],
            1,
            6,
        )
        .unwrap();

        assert_eq!(faces.len(), 6);
        for axis in 0..3 {
            for positive in [false, true] {
                assert!(faces.iter().any(|face| {
                    face.origin == [1, 1, 1]
                        && face.axis() == axis
                        && face.positive() == positive
                        && face.kind() == VoxelHighlightKind::BridgePath as u32
                }));
            }
        }
    }

    #[test]
    fn mixed_highlight_kinds_share_no_internal_faces_and_anchor_wins_duplicates() {
        let faces = build_highlight_faces(
            [4; 3],
            vec![
                ([1, 1, 1], VoxelHighlightKind::BridgePath),
                ([2, 1, 1], VoxelHighlightKind::BridgePath),
                ([1, 1, 1], VoxelHighlightKind::BridgeAnchor),
            ],
            2,
            12,
        )
        .unwrap();

        assert_eq!(faces.len(), 10);
        assert!(
            faces
                .iter()
                .filter(|face| face.origin == [1, 1, 1])
                .all(|face| face.kind() == VoxelHighlightKind::BridgeAnchor as u32)
        );
        assert!(
            faces
                .iter()
                .filter(|face| face.origin == [2, 1, 1])
                .all(|face| face.kind() == VoxelHighlightKind::BridgePath as u32)
        );
        assert!(
            !faces
                .iter()
                .any(|face| { face.origin == [1, 1, 1] && face.axis() == 0 && face.positive() })
        );
        assert!(
            !faces
                .iter()
                .any(|face| { face.origin == [2, 1, 1] && face.axis() == 0 && !face.positive() })
        );
    }

    #[test]
    fn solid_highlight_block_work_scales_with_surface_area() {
        let dimensions = [12, 10, 8];
        let highlights = (0..dimensions[2])
            .flat_map(|z| {
                (0..dimensions[1]).flat_map(move |y| (0..dimensions[0]).map(move |x| [x, y, z]))
            })
            .map(|coordinate| (coordinate, VoxelHighlightKind::Paint))
            .collect::<Vec<_>>();
        let faces = build_highlight_faces(
            dimensions,
            highlights,
            MAX_VOXEL_HIGHLIGHT_VOXELS,
            MAX_VOXEL_HIGHLIGHT_FACES,
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
    fn highlight_limits_count_unique_voxels_and_exposed_faces() {
        assert!(matches!(
            build_highlight_faces(
                [4; 3],
                typed_highlights(
                    vec![[1, 1, 1], [1, 1, 1], [2, 1, 1]],
                    VoxelHighlightKind::Paint,
                ),
                1,
                6,
            ),
            Err(HighlightBuildError::Failed(
                VoxelHighlightFailure::TooManyVoxels
            ))
        ));
        assert_eq!(
            build_highlight_faces(
                [4; 3],
                typed_highlights(
                    vec![[1, 1, 1], [1, 1, 1], [8, 8, 8]],
                    VoxelHighlightKind::Paint,
                ),
                1,
                6,
            )
            .unwrap()
            .len(),
            6
        );
        assert!(matches!(
            build_highlight_faces([4; 3], vec![([1, 1, 1], VoxelHighlightKind::Paint)], 1, 5,),
            Err(HighlightBuildError::Failed(
                VoxelHighlightFailure::TooManyFaces
            ))
        ));
    }

    #[test]
    fn highlight_face_build_stops_when_its_generation_is_cancelled() {
        let checks = std::sync::atomic::AtomicUsize::new(0);
        let highlights = (0..4_096)
            .map(|x| ([x, 0, 0], VoxelHighlightKind::Paint))
            .collect();
        let result =
            build_highlight_faces_cancellable([4_096, 1, 1], highlights, 4_096, 24_576, || {
                checks.fetch_add(1, Ordering::AcqRel) == 0
            });

        assert!(matches!(result, Err(HighlightBuildError::Cancelled)));
        assert!(checks.load(Ordering::Acquire) >= 2);
    }

    #[test]
    fn configured_caps_admit_a_50_mm_anisotropic_square_volume() {
        // At the sample scan's 0.315 x 0.315 x 0.8 mm spacing, a 50 mm square
        // volume is conservatively bounded by 160 x 160 x 64 voxels.
        let [x, y, z] = [160_usize, 160, 64];
        let voxels = x * y * z;
        let boundary_faces = 2 * (x * y + x * z + y * z);

        assert!(voxels <= MAX_VOXEL_HIGHLIGHT_VOXELS);
        assert!(boundary_faces <= MAX_VOXEL_HIGHLIGHT_FACES);
        assert!(boundary_faces * 6 < voxels * 36 / 100);
    }

    #[test]
    fn stale_highlight_is_ignored_and_invalidation_clears_a_pending_highlight() {
        let bridge = SceneBridge::default();
        bridge.reset(7, 11, [8; 3], SceneState::default().affine);
        let highlight = vec![([1, 1, 1], VoxelHighlightKind::Paint)];
        assert!(!bridge.set_voxel_highlight(6, 11, highlight.clone()));
        assert!(!bridge.set_voxel_highlight(7, 10, highlight.clone()));
        assert!(bridge.set_voxel_highlight(7, 11, highlight));
        let stale_generation = bridge.highlight_generation.load(Ordering::Acquire);
        bridge.invalidate(7, 12, vec![ChunkCoord::new(0, 0, 0)]);
        bridge.finish_highlight_build(
            7,
            11,
            stale_generation,
            build_highlight_faces(
                [8; 3],
                vec![([1, 1, 1], VoxelHighlightKind::BridgePath)],
                MAX_VOXEL_HIGHLIGHT_VOXELS,
                MAX_VOXEL_HIGHLIGHT_FACES,
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
    fn highlight_follows_live_chunks_and_precedes_background_backlog() {
        let bridge = SceneBridge::default();
        bridge.reset(3, 6, [64, 8, 8], SceneState::default().affine);
        let tag = |chunk, priority| MeshJobTag {
            document_epoch: 3,
            revision: 6,
            chunk_generation: 2,
            edit_generation: u64::from(priority == MeshUpdatePriority::Live),
            chunk,
        };
        bridge.insert_chunk(
            tag(ChunkCoord::new(0, 0, 0), MeshUpdatePriority::Background),
            Vec::new(),
            MeshUpdatePriority::Background,
        );
        bridge.insert_chunk(
            tag(ChunkCoord::new(1, 0, 0), MeshUpdatePriority::Live),
            Vec::new(),
            MeshUpdatePriority::Live,
        );
        assert!(bridge.set_voxel_highlight(3, 6, vec![([2, 2, 2], VoxelHighlightKind::Paint)],));
        wait_for_highlight_kind(&bridge, VoxelHighlightKind::Paint);

        let state = bridge.state.lock().unwrap();
        let live_index = state
            .updates
            .iter()
            .position(|update| {
                matches!(
                    update,
                    SceneUpdate::Chunk {
                        priority: MeshUpdatePriority::Live,
                        ..
                    }
                )
            })
            .unwrap();
        let background_index = state
            .updates
            .iter()
            .position(|update| {
                matches!(
                    update,
                    SceneUpdate::Chunk {
                        priority: MeshUpdatePriority::Background,
                        ..
                    }
                )
            })
            .unwrap();
        let highlight_index = state
            .updates
            .iter()
            .position(|update| matches!(update, SceneUpdate::Highlight { .. }))
            .unwrap();
        assert!(live_index < highlight_index);
        assert!(highlight_index < background_index);
    }

    #[test]
    fn live_chunks_follow_barriers_precede_background_and_prune_older_same_chunk() {
        let bridge = SceneBridge::default();
        let affine = SceneState::default().affine;
        let replaced = ChunkCoord::new(1, 0, 0);
        let other_live = ChunkCoord::new(2, 0, 0);
        let background = ChunkCoord::new(3, 0, 0);
        let tag = |revision, generation, chunk| MeshJobTag {
            document_epoch: 4,
            revision,
            chunk_generation: generation,
            edit_generation: 0,
            chunk,
        };
        bridge.reset(4, 8, [128, 32, 32], affine);
        bridge.insert_chunk(
            tag(8, 1, replaced),
            Vec::new(),
            MeshUpdatePriority::Background,
        );
        bridge.invalidate(4, 9, vec![replaced]);
        bridge.insert_chunk(
            tag(9, 1, background),
            Vec::new(),
            MeshUpdatePriority::Background,
        );
        bridge.insert_chunk(
            tag(9, 2, replaced),
            Vec::new(),
            MeshUpdatePriority::Background,
        );
        bridge.insert_chunk(
            tag(9, 3, replaced),
            Vec::new(),
            MeshUpdatePriority::Background,
        );
        bridge.insert_chunk(tag(9, 3, replaced), Vec::new(), MeshUpdatePriority::Live);
        bridge.insert_chunk(tag(9, 1, other_live), Vec::new(), MeshUpdatePriority::Live);

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 6);
        assert!(matches!(state.updates[0], SceneUpdate::Reset { .. }));
        assert!(matches!(
            state.updates[1],
            SceneUpdate::Chunk { tag, .. }
                if tag.revision == 8 && tag.chunk == replaced
        ));
        assert!(matches!(state.updates[2], SceneUpdate::Invalidate { .. }));
        assert!(matches!(
            state.updates[3],
            SceneUpdate::Chunk {
                tag,
                priority: MeshUpdatePriority::Live,
                ..
            } if tag.chunk == replaced && tag.chunk_generation == 3
        ));
        assert!(matches!(
            state.updates[4],
            SceneUpdate::Chunk {
                tag,
                priority: MeshUpdatePriority::Live,
                ..
            } if tag.chunk == other_live
        ));
        assert!(matches!(
            state.updates[5],
            SceneUpdate::Chunk {
                tag,
                priority: MeshUpdatePriority::Background,
                ..
            } if tag.chunk == background
        ));
    }

    #[test]
    fn only_current_highlight_failures_are_reported_once() {
        let bridge = SceneBridge::default();
        bridge.reset(3, 6, [8; 3], SceneState::default().affine);
        let generation = bridge.highlight_generation.load(Ordering::Acquire);
        bridge.finish_highlight_build(
            3,
            6,
            generation,
            Err(HighlightBuildError::Failed(
                VoxelHighlightFailure::TooManyFaces,
            )),
        );

        assert_eq!(
            bridge.take_highlight_failure(),
            Some(VoxelHighlightFailure::TooManyFaces)
        );
        assert_eq!(bridge.take_highlight_failure(), None);

        let stale_generation = generation;
        bridge.clear_voxel_highlight();
        bridge.finish_highlight_build(
            3,
            6,
            stale_generation,
            Err(HighlightBuildError::Failed(
                VoxelHighlightFailure::Allocation,
            )),
        );
        assert_eq!(bridge.take_highlight_failure(), None);
    }

    #[test]
    fn clearing_highlight_replaces_pending_highlight_after_document_transition() {
        let bridge = SceneBridge::default();
        bridge.reset(2, 3, [8; 3], SceneState::default().affine);
        assert!(bridge.set_voxel_highlight(2, 3, vec![([4, 4, 4], VoxelHighlightKind::Paint)],));
        bridge.clear_voxel_highlight();

        let state = bridge.state.lock().unwrap();
        assert_eq!(state.updates.len(), 2);
        assert!(matches!(
            state.updates.front(),
            Some(SceneUpdate::Reset { .. })
        ));
        assert!(matches!(
            state.updates.back(),
            Some(SceneUpdate::ClearHighlight)
        ));
    }

    fn typed_highlights(
        coordinates: Vec<[u32; 3]>,
        kind: VoxelHighlightKind,
    ) -> Vec<([u32; 3], VoxelHighlightKind)> {
        coordinates
            .into_iter()
            .map(|coordinate| (coordinate, kind))
            .collect()
    }

    fn wait_for_highlight_kind(bridge: &SceneBridge, kind: VoxelHighlightKind) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let has_highlight = bridge.state.lock().unwrap().updates.iter().any(|update| {
                matches!(
                    update,
                    SceneUpdate::Highlight { faces, .. }
                        if faces.first().is_some_and(|face| face.kind() == kind as u32)
                )
            });
            if has_highlight {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the latest background highlight"
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
        let (view, _, _, _, _) = scene_camera(&state);
        let eye = view.inverse().transform_point3(Vec3::ZERO);
        assert!(eye.length() > 200.0);
    }

    #[test]
    fn camera_depth_span_matches_the_projected_volume_box_at_cardinal_and_oblique_angles() {
        let mut state = SceneState {
            dimensions: [617, 560, 231],
            affine: [
                [0.315, 0.0, 0.0, 0.0],
                [0.0, 0.315, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            pitch: 0.0,
            ..Default::default()
        };
        for (yaw, expected_span) in [
            (0.0, state.dimensions[0] as f32 * 0.315),
            (
                std::f32::consts::FRAC_PI_2,
                state.dimensions[1] as f32 * 0.315,
            ),
        ] {
            state.yaw = yaw;
            let (_, _, _, _, inverse_span) = scene_camera(&state);
            assert!((inverse_span.recip() - expected_span).abs() < 1.0e-3);
        }

        state.yaw = 0.65;
        state.pitch = -0.45;
        let (view, _, affine, near, inverse_span) = scene_camera(&state);
        let corners: [Vec3; 8] = std::array::from_fn(|corner| {
            affine.transform_point3(Vec3::new(
                if corner & 1 == 0 {
                    -0.5
                } else {
                    state.dimensions[0] as f32 - 0.5
                },
                if corner & 2 == 0 {
                    -0.5
                } else {
                    state.dimensions[1] as f32 - 0.5
                },
                if corner & 4 == 0 {
                    -0.5
                } else {
                    state.dimensions[2] as f32 - 0.5
                },
            ))
        });
        let normalized = corners.map(|corner| {
            let depth = -view.transform_point3(corner).z;
            (depth - near) * inverse_span
        });
        let minimum = normalized.into_iter().fold(f32::INFINITY, f32::min);
        let maximum = normalized.into_iter().fold(f32::NEG_INFINITY, f32::max);
        assert!(minimum.abs() < 1.0e-5);
        assert!((maximum - 1.0).abs() < 1.0e-5);
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
        let (view, projection, _, _, _) = scene_camera(state);
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
                    edit_generation: 0,
                    chunk,
                },
                quads: vec![
                    CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .expect("unit face is representable"),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        assert!(renderer.chunks.contains_key(&chunk));

        renderer.apply_update(
            &device,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 8,
                faces: build_highlight_faces(
                    [32; 3],
                    typed_highlights(vec![[2, 3, 4], [3, 3, 4]], VoxelHighlightKind::Erase),
                    MAX_VOXEL_HIGHLIGHT_VOXELS,
                    MAX_VOXEL_HIGHLIGHT_FACES,
                )
                .unwrap(),
            },
        );
        assert_eq!(
            renderer.highlight.as_ref().map(|highlight| highlight.faces),
            Some(10)
        );
        assert!(
            renderer.chunks.contains_key(&chunk),
            "transient highlight must not alter resident chunk meshes"
        );

        renderer.apply_update(&device, SceneUpdate::Invalidate { revision: 9 });
        assert_eq!(renderer.revision, 9);
        assert!(renderer.highlight.is_none());
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
                    edit_generation: 0,
                    chunk,
                },
                quads: Vec::new(),
                priority: MeshUpdatePriority::Live,
            },
        );
        assert!(
            !renderer.chunks.contains_key(&chunk),
            "an empty replacement removes the old resident mesh atomically"
        );
        renderer.apply_update(
            &device,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 9,
                    chunk_generation: 3,
                    edit_generation: 0,
                    chunk,
                },
                quads: vec![
                    CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .expect("unit face is representable"),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        assert!(
            !renderer.chunks.contains_key(&chunk),
            "an older background upload must not resurrect a live empty replacement"
        );
    }
}
