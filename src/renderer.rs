use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem::size_of,
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
    brush::{Ray, SegmentSweep, segment_surface_hit_with_occupancy},
    mesh::{
        CHUNK_SIZE, ChunkCoord, CompactQuad, MeshJobTag, MeshUpdatePriority, PackedLeakDistance,
        pack_leak_distances,
    },
    volume::{
        LeakDistanceField, Matrix4, SharedOccupancy, SharedOccupancyRead, linear_index,
        transform_point, transform_vector,
    },
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
    leak_half_distance: f32,
    leak_distance_scale: f32,
    voxel_seams: u32,
    leak_light: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Snorm16x4 exposes the connected 0..32767 step range as 0.0..1.0.
// Restore the integer magnitude before converting steps to physical distance.
const LEAK_DISTANCE_SNORM_STEP_RANGE: f32 = 32767.0;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
    @location(1) view_position: vec3<f32>,
    @location(2) corner_occlusion: f32,
    @location(3) index_position: vec3<f32>,
    @location(4) @interpolate(flat) face_axis: u32,
    @location(5) leak_distance: f32,
    @location(6) @interpolate(flat) view_geometric_normal: vec3<f32>,
};

fn octahedral_sign_not_zero(value: f32) -> f32 {
    return select(1.0, -1.0, value < 0.0);
}

fn decode_shading_normal(packed: u32) -> vec3<f32> {
    let encoded = (packed >> 22u) & 0x3ffu;
    var normal = vec3<f32>(
        (f32(encoded & 0x1fu) - 15.0) / 15.0,
        (f32((encoded >> 5u) & 0x1fu) - 15.0) / 15.0,
        0.0,
    );
    normal.z = 1.0 - abs(normal.x) - abs(normal.y);
    if (normal.z < 0.0) {
        let original = normal.xy;
        normal.x = (1.0 - abs(original.y)) * octahedral_sign_not_zero(original.x);
        normal.y = (1.0 - abs(original.x)) * octahedral_sign_not_zero(original.y);
    }
    return normalize(normal);
}

fn expand_vertex(
    vertex_index: u32,
    origin: vec3<u32>,
    packed: u32,
    corner_leak_distances: vec4<f32>,
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
    let geometric_index_normal = index_normal;
    // The occupancy-derived normal restores local curvature cues that the six
    // geometric voxel normals cannot express. Retaining a geometric component
    // keeps silhouettes and step changes visibly voxel-like.
    index_normal = normalize(0.35 * index_normal + 0.65 * decode_shading_normal(packed));

    let world_position = uniforms.index_to_world * vec4<f32>(index_position, 1.0);
    let view_position = uniforms.world_to_view * world_position;
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.view_normal = normalize(
        (uniforms.index_normal_to_view * vec4<f32>(index_normal, 0.0)).xyz,
    );
    output.view_geometric_normal = normalize(
        (uniforms.index_normal_to_view * vec4<f32>(geometric_index_normal, 0.0)).xyz,
    );
    output.view_position = view_position.xyz;
    output.corner_occlusion = f32((packed >> (13u + corner * 2u)) & 3u);
    output.index_position = index_position;
    output.face_axis = axis;
    output.leak_distance = corner_leak_distances[corner];
    return output;
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) origin: vec3<u32>,
    @location(1) packed: u32,
) -> VertexOutput {
    return expand_vertex(vertex_index, origin, packed, vec4<f32>(-1.0));
}

@vertex
fn leak_vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) origin: vec3<u32>,
    @location(1) packed: u32,
    @location(2) corner_leak_distances: vec4<f32>,
) -> VertexOutput {
    return expand_vertex(vertex_index, origin, packed, corner_leak_distances);
}

fn voxel_seam(coordinate: f32) -> f32 {
    let coordinate_width = max(fwidth(coordinate), 1.0e-5);
    let unit_coordinate = fract(coordinate + 0.5);
    let boundary_distance = min(unit_coordinate, 1.0 - unit_coordinate);
    let antialiased_line = 1.0 - smoothstep(0.0, coordinate_width * 0.85, boundary_distance);
    let pixels_per_voxel = 1.0 / coordinate_width;
    let subpixel_fade = smoothstep(0.75, 2.0, pixels_per_voxel);
    return antialiased_line * subpixel_fade;
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

fn heatmap_segment(
    value: f32,
    lower_edge: f32,
    upper_edge: f32,
    lower_color: vec3<f32>,
    upper_color: vec3<f32>,
) -> vec3<f32> {
    let amount = clamp((value - lower_edge) / (upper_edge - lower_edge), 0.0, 1.0);
    return srgb_to_linear(mix(lower_color, upper_color, amount));
}

fn leak_heatmap(intensity: f32) -> vec3<f32> {
    let value = clamp(intensity, 0.0, 1.0);
    let black = vec3<f32>(0.0);
    let purple = vec3<f32>(87.0, 16.0, 110.0) / 255.0;
    let crimson = vec3<f32>(188.0, 55.0, 84.0) / 255.0;
    let orange = vec3<f32>(249.0, 142.0, 9.0) / 255.0;
    let yellow = vec3<f32>(252.0, 255.0, 164.0) / 255.0;
    if (value < 0.25) {
        return heatmap_segment(value, 0.0, 0.25, black, purple);
    }
    if (value < 0.5) {
        return heatmap_segment(value, 0.25, 0.5, purple, crimson);
    }
    if (value < 0.75) {
        return heatmap_segment(value, 0.5, 0.75, crimson, orange);
    }
    return heatmap_segment(value, 0.75, 1.0, orange, yellow);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_direction = normalize(-input.view_position);
    var normal = normalize(input.view_normal);
    let geometric_back_facing =
        dot(normalize(input.view_geometric_normal), view_direction) < 0.0;
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
    var lit_color = uniforms.material_color.rgb * lighting;
    if (uniforms.leak_light != 0u) {
        let connected = input.leak_distance >= 0.0;
        let physical_distance = max(input.leak_distance, 0.0) * LEAK_DISTANCE_SNORM_STEP_RANGE
            * uniforms.leak_distance_scale;
        let half_distance = max(uniforms.leak_half_distance, 1.0e-12);
        let leak_intensity = exp2(-physical_distance / half_distance);
        let connected_light = select(0.0, leak_intensity, connected);
        let diagnostic_shading = (0.75 + 0.25 * base_lighting)
            * mix(1.0, surface_factor, 0.35)
            * mix(1.0, voxel_ao_factor, 0.5);
        lit_color = select(
            vec3<f32>(0.0),
            leak_heatmap(connected_light) * diagnostic_shading,
            connected,
        );
    }
    if (uniforms.voxel_seams == 0u) {
        return vec4<f32>(lit_color, 1.0);
    }

    var seam_coordinates = input.index_position.xy;
    if (input.face_axis == 0u) {
        seam_coordinates = input.index_position.yz;
    } else if (input.face_axis == 1u) {
        seam_coordinates = input.index_position.zx;
    }
    let seam_strength = max(
        voxel_seam(seam_coordinates.x),
        voxel_seam(seam_coordinates.y),
    );
    let seam_color = lit_color * 0.38;
    return vec4<f32>(mix(lit_color, seam_color, seam_strength * 0.72), 1.0);
}
"#;

const LEAK_DISTANCE_COMPUTE_SHADER: &str = r#"
struct CompactQuad {
    origin: vec3<u32>,
    packed: u32,
};

struct LeakComputeParams {
    chunk_origin_and_count: vec4<u32>,
    halo_dimensions: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: LeakComputeParams;
@group(0) @binding(1) var<storage, read> quads: array<CompactQuad>;
@group(0) @binding(2) var<storage, read> halo_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words: array<vec2<u32>>;

fn halo_step(index: u32) -> u32 {
    let word = halo_words[index / 2u];
    return (word >> ((index & 1u) * 16u)) & 0xffffu;
}

fn snorm_bits(step: u32) -> u32 {
    return select(min(step, 0x7fffu), 0x8000u, step == 0xffffu);
}

fn sample_corner(
    quad: CompactQuad,
    axis: u32,
    u_axis: u32,
    v_axis: u32,
    corner: u32,
) -> u32 {
    let local_origin = quad.origin - params.chunk_origin_and_count.xyz;
    let positive = ((quad.packed >> 12u) & 1u) != 0u;
    let u_length = (quad.packed & 31u) + 1u;
    let v_length = ((quad.packed >> 5u) & 31u) + 1u;
    let maximum_u = corner == 1u || corner == 2u;
    let maximum_v = corner == 2u || corner == 3u;
    var halo = local_origin + vec3<u32>(1u);
    halo[axis] = local_origin[axis] + select(0u, 1u, positive);
    halo[u_axis] += select(0u, u_length - 1u, maximum_u);
    halo[v_axis] += select(0u, v_length - 1u, maximum_v);
    let dimensions = params.halo_dimensions.xyz;
    let index = halo.x + dimensions.x * (halo.y + dimensions.y * halo.z);
    return snorm_bits(halo_step(index));
}

@compute @workgroup_size(64)
fn prepare_leak_distance(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.chunk_origin_and_count.w) {
        return;
    }
    let quad = quads[index];
    let axis = (quad.packed >> 10u) & 3u;
    var u_axis = 0u;
    var v_axis = 1u;
    if (axis == 0u) {
        u_axis = 1u;
        v_axis = 2u;
    } else if (axis == 1u) {
        u_axis = 2u;
        v_axis = 0u;
    }
    let corner_0 = sample_corner(quad, axis, u_axis, v_axis, 0u);
    let corner_1 = sample_corner(quad, axis, u_axis, v_axis, 1u);
    let corner_2 = sample_corner(quad, axis, u_axis, v_axis, 2u);
    let corner_3 = sample_corner(quad, axis, u_axis, v_axis, 3u);
    output_words[index] = vec2<u32>(
        corner_0 | (corner_1 << 16u),
        corner_2 | (corner_3 << 16u),
    );
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
    leak_half_distance: f32,
    leak_distance_scale: f32,
    voxel_seams: u32,
    leak_light: u32,
    _padding: vec2<u32>,
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

const DEPTH_RESET_SHADER: &str = r#"
@vertex
fn depth_reset_vertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn depth_reset_fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

const MAX_GPU_UPDATES_PER_FRAME: usize = 4;
const MAX_LEAK_DISTANCE_CHUNKS_PER_FRAME: usize = 2;
const LEAK_COMPUTE_WORKGROUP_SIZE: u32 = 64;
const LEAK_COMPUTE_STORAGE_BINDINGS: u32 = 3;
const LEAK_COMPUTE_BINDINGS: u32 = 4;
const LEAK_COMPUTE_PARAMETER_BYTES: u64 = 32;
const MAX_CHUNK_QUADS: u64 = 6 * CHUNK_SIZE as u64 * CHUNK_SIZE as u64 * CHUNK_SIZE as u64;
const MAX_CHUNK_QUAD_BYTES: u64 = MAX_CHUNK_QUADS * size_of::<CompactQuad>() as u64;
const MAX_CHUNK_LEAK_WORKGROUPS: u32 =
    MAX_CHUNK_QUADS.div_ceil(LEAK_COMPUTE_WORKGROUP_SIZE as u64) as u32;
pub const MSAA_SAMPLES: u32 = 4;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const ORBIT_FIELD_OF_VIEW_RADIANS: f32 = 45_f32.to_radians();
const FLY_FIELD_OF_VIEW_RADIANS: f32 = 70_f32.to_radians();
const FLY_LOOK_SENSITIVITY: f32 = 0.0025;
const MAX_FLY_PITCH: f32 = 1.5;
const MAX_FLY_FRAME_DELTA_SECONDS: f32 = 0.1;
const FLY_BOOST_MULTIPLIER: f32 = 4.0;
const FLY_SPEED_SCROLL_SENSITIVITY: f32 = 0.002;
const FLY_MIN_SPEED_RADIUS_FACTOR: f32 = 0.002;
const FLY_INITIAL_SPEED_RADIUS_FACTOR: f32 = 0.15;
const FLY_MAX_SPEED_RADIUS_FACTOR: f32 = 2.0;
const FLY_CLIP_EPSILON_RADIUS_FACTOR: f32 = 1.0e-4;
const FLY_CLIP_MARGIN_RADIUS_FACTOR: f32 = 0.01;
const FLY_COLLISION_SKIN_VOXELS: f64 = 1.0e-4;
const MAX_FLY_COLLISION_CONTACTS: usize = 3;
const MAX_FLY_COLLISION_VOXEL_STEPS: usize = 65_536;
// These independent caps keep CPU adjacency work and GPU boundary work bounded.
// Four million 16-byte faces occupy 64 MB and invoke at most 24 million vertices;
// the sample scan's 50 mm volume is about 1.64 M voxels but only 92,160 faces.
pub const MAX_VOXEL_HIGHLIGHT_VOXELS: usize = 2_000_000;
pub const MAX_VOXEL_HIGHLIGHT_FACES: usize = 4_000_000;
// Keep small chunk buffers through normal remesh variation, but release a
// pathological peak once the replacement is substantially smaller.
const CHUNK_BUFFER_REUSE_BASE_BYTES: u64 = 64 * 1024;
const MAX_CHUNK_BUFFER_REUSE_RATIO: u64 = 4;
// Preserve enough storage for common dense previews without pinning the 64 MiB hard maximum.
const MAX_RETAINED_HIGHLIGHT_BUFFER_BYTES: u64 = 4 * 1024 * 1024;
const HIGHLIGHT_ALLOCATION_CHUNK: usize = 65_536;
const HIGHLIGHT_CANCELLATION_INTERVAL: usize = 1_024;

/// Returns whether an adapter can prepare reach-independent leak distances in compute.
///
/// No optional shader feature is required: the compute shader writes signed-normalized
/// 16-bit vertex data with integer operations. Buffers are padded to 16 bytes so the
/// downlevel unaligned-binding flag is deliberately not part of this check.
#[must_use]
pub fn leak_compute_supported(adapter: &wgpu::Adapter) -> bool {
    leak_compute_supported_for(
        adapter.get_downlevel_capabilities().flags,
        &adapter.limits(),
    )
}

fn leak_compute_supported_for(flags: wgpu::DownlevelFlags, limits: &wgpu::Limits) -> bool {
    flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        && limits.max_bind_groups >= 1
        && limits.max_bindings_per_bind_group >= LEAK_COMPUTE_BINDINGS
        && limits.max_uniform_buffers_per_shader_stage >= 1
        && limits.max_uniform_buffer_binding_size >= LEAK_COMPUTE_PARAMETER_BYTES
        && limits.max_storage_buffers_per_shader_stage >= LEAK_COMPUTE_STORAGE_BINDINGS
        && limits.max_storage_buffer_binding_size >= MAX_CHUNK_QUAD_BYTES
        && limits.max_buffer_size >= MAX_CHUNK_QUAD_BYTES
        && limits.max_compute_invocations_per_workgroup >= LEAK_COMPUTE_WORKGROUP_SIZE
        && limits.max_compute_workgroup_size_x >= LEAK_COMPUTE_WORKGROUP_SIZE
        && limits.max_compute_workgroup_size_y >= 1
        && limits.max_compute_workgroup_size_z >= 1
        && limits.max_compute_workgroups_per_dimension >= MAX_CHUNK_LEAK_WORKGROUPS
}
const CLIP_PLANE_RELATIVE_TOLERANCE: f64 = 128.0 * f32::EPSILON as f64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VolumeViewMode {
    #[default]
    Orbit,
    Fly,
}

/// Direction basis used by the forward and backward Fly controls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FlyMovementBasis {
    /// Keep forward and backward movement parallel to world X/Y.
    #[default]
    Level,
    /// Follow the camera's full yaw-and-pitch view direction.
    ViewRelative,
}

/// Whether Fly movement passes through or collides with occupied voxels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FlyCollisionMode {
    /// Preserve unrestricted flight through all volume geometry.
    #[default]
    PassThrough,
    /// Sweep the camera viewpoint against occupied voxels.
    SolidVoxels,
}

/// Signed movement axes for one Fly-mode frame.
///
/// `forward` uses the selected [`FlyMovementBasis`], `right` strafes
/// horizontally, and `world_up` moves along world Z. Each axis is clamped to
/// `-1.0..=1.0`; combined movement is normalized so diagonals are not faster.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyMovement {
    pub forward: f32,
    pub right: f32,
    pub world_up: f32,
    pub boost: bool,
}

impl FlyMovement {
    #[must_use]
    pub const fn new(forward: f32, right: f32, world_up: f32, boost: bool) -> Self {
        Self {
            forward,
            right,
            world_up,
            boost,
        }
    }
}

fn highlight_color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn highlight_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: 64,
            slope_scale: 0.0,
            clamp: 0.0,
        },
    }
}

fn volume_depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
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
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    index_to_world: [[f32; 4]; 4],
    world_to_view: [[f32; 4]; 4],
    index_normal_to_view: [[f32; 4]; 4],
    material_color: [f32; 4],
    near_surface_depth: f32,
    inverse_scene_depth_span: f32,
    leak_half_distance: f32,
    leak_distance_scale: f32,
    voxel_seams: u32,
    leak_light: u32,
    _padding: [u32; 2],
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
            leak_half_distance: 1.0,
            leak_distance_scale: 1.0,
            voxel_seams: 0,
            leak_light: 0,
            _padding: [0; 2],
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
    LeakDistanceField {
        epoch: u64,
        revision: u64,
        generation: u64,
        field: Arc<LeakDistanceField>,
    },
    Highlight {
        epoch: u64,
        revision: u64,
        faces: Vec<HighlightFace>,
    },
    ClearHighlight,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrbitCameraState {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan_ndc: egui::Vec2,
}

impl Default for OrbitCameraState {
    fn default() -> Self {
        Self {
            yaw: 0.65,
            pitch: -0.45,
            zoom: 1.0,
            pan_ndc: egui::Vec2::ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FlyHomeState {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FlyCameraState {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    home: FlyHomeState,
}

impl FlyCameraState {
    fn reset_to_home(&mut self) {
        self.position = self.home.position;
        self.yaw = self.home.yaw;
        self.pitch = self.home.pitch;
        self.speed = self.home.speed;
    }
}

struct SceneState {
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
    world_to_index: Matrix4,
    occupancy: Option<SharedOccupancy>,
    view_mode: VolumeViewMode,
    orbit: OrbitCameraState,
    fly: Option<FlyCameraState>,
    fly_movement_basis: FlyMovementBasis,
    fly_collision_mode: FlyCollisionMode,
    voxel_seams: bool,
    leak_light: bool,
    leak_half_distance: f32,
    aspect: f32,
    updates: VecDeque<SceneUpdate>,
}

struct SceneReset {
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
    world_to_index: Matrix4,
    occupancy: Option<SharedOccupancy>,
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
            world_to_index: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            occupancy: None,
            view_mode: VolumeViewMode::Orbit,
            orbit: OrbitCameraState::default(),
            fly: None,
            fly_movement_basis: FlyMovementBasis::default(),
            fly_collision_mode: FlyCollisionMode::default(),
            voxel_seams: true,
            leak_light: false,
            leak_half_distance: 1.0,
            aspect: 1.0,
            updates: VecDeque::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct SceneBridge {
    state: Arc<Mutex<SceneState>>,
    needs_remesh: Arc<AtomicBool>,
    leak_distance_ready: Arc<AtomicBool>,
    leak_distance_available: Arc<AtomicBool>,
    repaint: Arc<Mutex<Option<egui::Context>>>,
    highlight_generation: Arc<AtomicU64>,
    highlight_failure: Arc<Mutex<Option<VoxelHighlightFailure>>>,
}

impl SceneBridge {
    pub fn set_repaint_context(&self, context: egui::Context) {
        *self.repaint.lock().expect("3D repaint context poisoned") = Some(context);
    }

    pub fn reset(
        &self,
        epoch: u64,
        revision: u64,
        dimensions: [u32; 3],
        affine: Matrix4,
        world_to_index: Matrix4,
        occupancy: SharedOccupancy,
    ) {
        self.reset_scene(
            SceneReset {
                epoch,
                revision,
                dimensions,
                affine,
                world_to_index,
                occupancy: Some(occupancy),
            },
            false,
        );
    }

    fn reset_scene(&self, reset: SceneReset, force_navigation_reset: bool) {
        let SceneReset {
            epoch,
            revision,
            dimensions,
            affine,
            world_to_index,
            occupancy,
        } = reset;
        let mut state = self.state.lock().expect("3D scene state poisoned");
        self.cancel_highlight_build_locked();
        if force_navigation_reset || state.epoch != epoch {
            reset_document_navigation(&mut state);
        }
        state.epoch = epoch;
        state.revision = revision;
        state.dimensions = dimensions;
        state.affine = affine;
        state.world_to_index = world_to_index;
        state.occupancy = occupancy;
        self.leak_distance_ready.store(false, Ordering::Release);
        self.leak_distance_available.store(false, Ordering::Release);
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
        self.leak_distance_ready.store(false, Ordering::Release);
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
        let default = SceneState::default();
        self.reset_scene(
            SceneReset {
                epoch: 0,
                revision: 0,
                dimensions: [1; 3],
                affine: default.affine,
                world_to_index: default.world_to_index,
                occupancy: None,
            },
            true,
        );
    }

    /// Returns and clears the latest failure from a current highlight request.
    /// Cancelled or stale builds never report a failure.
    pub fn take_highlight_failure(&self) -> Option<VoxelHighlightFailure> {
        self.highlight_failure
            .lock()
            .expect("3D highlight failure state poisoned")
            .take()
    }

    #[must_use]
    pub fn view_mode(&self) -> VolumeViewMode {
        self.state
            .lock()
            .expect("3D scene state poisoned")
            .view_mode
    }

    pub fn set_view_mode(&self, mode: VolumeViewMode) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode == mode {
            return;
        }
        if mode == VolumeViewMode::Fly && state.fly.is_none() {
            state.fly = Some(fly_camera_from_orbit(&state));
        }
        state.view_mode = mode;
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn fly_movement_basis(&self) -> FlyMovementBasis {
        self.state
            .lock()
            .expect("3D scene state poisoned")
            .fly_movement_basis
    }

    pub fn set_fly_movement_basis(&self, basis: FlyMovementBasis) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.fly_movement_basis == basis {
            return;
        }
        state.fly_movement_basis = basis;
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn fly_collision_mode(&self) -> FlyCollisionMode {
        self.state
            .lock()
            .expect("3D scene state poisoned")
            .fly_collision_mode
    }

    pub fn set_fly_collision_mode(&self, mode: FlyCollisionMode) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.fly_collision_mode == mode {
            return;
        }
        state.fly_collision_mode = mode;
        if mode == FlyCollisionMode::SolidVoxels {
            let _ = recover_fly_if_inside(&mut state);
        }
        drop(state);
        self.request_repaint();
    }

    /// Moves an overlapping collision-enabled Fly camera back to its safe home.
    ///
    /// This is called after engine events so edits that fill the camera voxel
    /// cannot leave the next rendered frame inside solid geometry.
    pub fn reconcile_fly_collision(&self) -> bool {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        let changed = recover_fly_if_inside(&mut state);
        drop(state);
        if changed {
            self.request_repaint();
        }
        changed
    }

    #[must_use]
    pub fn voxel_seams(&self) -> bool {
        self.state
            .lock()
            .expect("3D scene state poisoned")
            .voxel_seams
    }

    pub fn set_voxel_seams(&self, enabled: bool) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.voxel_seams == enabled {
            return;
        }
        state.voxel_seams = enabled;
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn leak_light(&self) -> bool {
        self.state
            .lock()
            .expect("3D scene state poisoned")
            .leak_light
    }

    pub fn set_leak_light(&self, enabled: bool) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.leak_light == enabled {
            return;
        }
        state.leak_light = enabled;
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn leak_light_reach(&self) -> f64 {
        f64::from(
            self.state
                .lock()
                .expect("3D scene state poisoned")
                .leak_half_distance,
        )
    }

    pub fn set_leak_light_reach(&self, half_distance: f64) {
        let half_distance = half_distance as f32;
        if !half_distance.is_finite() || half_distance <= 0.0 {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.leak_half_distance == half_distance {
            return;
        }
        state.leak_half_distance = half_distance;
        drop(state);
        self.request_repaint();
    }

    pub fn set_leak_distance_field(
        &self,
        epoch: u64,
        revision: u64,
        generation: u64,
        field: Arc<LeakDistanceField>,
    ) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch != epoch || state.revision != revision {
            return;
        }
        self.leak_distance_ready.store(false, Ordering::Release);
        state.updates.retain(|update| {
            !matches!(
                update,
                SceneUpdate::LeakDistanceField {
                    epoch: queued_epoch,
                    revision: queued_revision,
                    generation: queued_generation,
                    ..
                } if *queued_epoch == epoch
                    && *queued_revision == revision
                    && *queued_generation <= generation
            )
        });
        state.updates.push_back(SceneUpdate::LeakDistanceField {
            epoch,
            revision,
            generation,
            field,
        });
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn leak_distance_ready(&self) -> bool {
        self.leak_distance_ready.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn leak_distance_available(&self) -> bool {
        self.leak_distance_available.load(Ordering::Acquire)
    }

    pub fn orbit(&self, delta: egui::Vec2) {
        if !delta.x.is_finite() || !delta.y.is_finite() {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode != VolumeViewMode::Orbit || delta == egui::Vec2::ZERO {
            return;
        }
        state.orbit.yaw -= delta.x * 0.008;
        state.orbit.pitch = (state.orbit.pitch + delta.y * 0.008).clamp(-1.5, 1.5);
        drop(state);
        self.request_repaint();
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
        if state.view_mode != VolumeViewMode::Orbit || delta_pixels == egui::Vec2::ZERO {
            return;
        }
        state.orbit.pan_ndc += egui::vec2(
            2.0 * delta_pixels.x / viewport_size.x,
            -2.0 * delta_pixels.y / viewport_size.y,
        );
        drop(state);
        self.request_repaint();
    }

    pub fn reset_view(&self) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        match state.view_mode {
            VolumeViewMode::Orbit => state.orbit = OrbitCameraState::default(),
            VolumeViewMode::Fly => {
                if let Some(fly) = state.fly.as_mut() {
                    fly.reset_to_home();
                }
            }
        }
        drop(state);
        self.request_repaint();
    }

    pub fn zoom(&self, factor: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode != VolumeViewMode::Orbit || (factor - 1.0).abs() <= f32::EPSILON {
            return;
        }
        state.orbit.zoom = (state.orbit.zoom * factor).clamp(0.2, 8.0);
        drop(state);
        self.request_repaint();
    }

    pub fn set_aspect(&self, aspect: f32) {
        if !aspect.is_finite() {
            return;
        }
        let aspect = aspect.max(0.1);
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.aspect == aspect {
            return;
        }
        state.aspect = aspect;
        drop(state);
        self.request_repaint();
    }

    pub fn fly_look(&self, delta: egui::Vec2) {
        if !delta.x.is_finite() || !delta.y.is_finite() || delta == egui::Vec2::ZERO {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode != VolumeViewMode::Fly {
            return;
        }
        let Some(fly) = state.fly.as_mut() else {
            return;
        };
        fly.yaw = wrap_angle(fly.yaw - delta.x * FLY_LOOK_SENSITIVITY);
        fly.pitch =
            (fly.pitch - delta.y * FLY_LOOK_SENSITIVITY).clamp(-MAX_FLY_PITCH, MAX_FLY_PITCH);
        drop(state);
        self.request_repaint();
    }

    pub fn fly_move(&self, movement: FlyMovement, delta_seconds: f32) -> bool {
        if !delta_seconds.is_finite()
            || delta_seconds <= 0.0
            || !movement.forward.is_finite()
            || !movement.right.is_finite()
            || !movement.world_up.is_finite()
        {
            return false;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode != VolumeViewMode::Fly {
            return false;
        }
        if state.fly_collision_mode == FlyCollisionMode::SolidVoxels
            && recover_fly_if_inside(&mut state)
        {
            drop(state);
            self.request_repaint();
            return true;
        }
        let movement_basis = state.fly_movement_basis;
        let Some(fly) = state.fly else {
            return false;
        };
        let forward_axis = movement.forward.clamp(-1.0, 1.0);
        let right_axis = movement.right.clamp(-1.0, 1.0);
        let world_up_axis = movement.world_up.clamp(-1.0, 1.0);
        let forward = match movement_basis {
            FlyMovementBasis::Level => Vec3::new(fly.yaw.cos(), fly.yaw.sin(), 0.0),
            FlyMovementBasis::ViewRelative => fly_forward(fly.yaw, fly.pitch),
        };
        let right = Vec3::new(fly.yaw.sin(), -fly.yaw.cos(), 0.0);
        let direction = (forward * forward_axis + right * right_axis + Vec3::Z * world_up_axis)
            .normalize_or_zero();
        if direction == Vec3::ZERO {
            return false;
        }
        let boost = if movement.boost {
            FLY_BOOST_MULTIPLIER
        } else {
            1.0
        };
        let delta_seconds = delta_seconds.min(MAX_FLY_FRAME_DELTA_SECONDS);
        let displacement = direction * fly.speed * boost * delta_seconds;
        let position = match state.fly_collision_mode {
            FlyCollisionMode::PassThrough => fly.position + displacement,
            FlyCollisionMode::SolidVoxels => {
                let Some(occupancy) = state.occupancy.clone() else {
                    return false;
                };
                let Some(occupancy) = occupancy.try_read() else {
                    return false;
                };
                resolve_fly_collision(
                    state.dimensions,
                    state.affine,
                    state.world_to_index,
                    fly.position,
                    displacement,
                    &occupancy,
                )
            }
        };
        if !position.is_finite() {
            return false;
        }
        if position == fly.position {
            return false;
        }
        state
            .fly
            .as_mut()
            .expect("the Fly camera remains initialized")
            .position = position;
        drop(state);
        self.request_repaint();
        true
    }

    pub fn adjust_fly_speed(&self, wheel_delta: f32) {
        if !wheel_delta.is_finite() || wheel_delta.abs() <= f32::EPSILON {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.view_mode != VolumeViewMode::Fly {
            return;
        }
        let radius = scene_bounds(&state).radius;
        let (minimum, maximum) = fly_speed_bounds(radius);
        let Some(fly) = state.fly.as_mut() else {
            return;
        };
        let factor = (wheel_delta * FLY_SPEED_SCROLL_SENSITIVITY).exp();
        if !factor.is_finite() {
            fly.speed = if wheel_delta.is_sign_positive() {
                maximum
            } else {
                minimum
            };
        } else {
            fly.speed = (fly.speed * factor).clamp(minimum, maximum);
        }
        drop(state);
        self.request_repaint();
    }

    pub fn set_fly_speed(&self, speed: f32) {
        if !speed.is_finite() || speed <= 0.0 {
            return;
        }
        let mut state = self.state.lock().expect("3D scene state poisoned");
        let radius = scene_bounds(&state).radius;
        let (minimum, maximum) = fly_speed_bounds(radius);
        let Some(fly) = state.fly.as_mut() else {
            return;
        };
        let speed = speed.clamp(minimum, maximum);
        if fly.speed == speed {
            return;
        }
        fly.speed = speed;
        drop(state);
        self.request_repaint();
    }

    #[must_use]
    pub fn fly_speed_range(&self) -> [f32; 2] {
        let state = self.state.lock().expect("3D scene state poisoned");
        let (minimum, maximum) = fly_speed_bounds(scene_bounds(&state).radius);
        [minimum, maximum]
    }

    #[must_use]
    pub fn fly_speed(&self) -> Option<f32> {
        let state = self.state.lock().expect("3D scene state poisoned");
        state.fly.map(|fly| fly.speed)
    }

    #[must_use]
    pub fn ray_for_pointer(&self, normalized: [f32; 2]) -> Ray {
        let state = self.state.lock().expect("3D scene state poisoned");
        let (view, projection, _, _, _, eye, forward) = scene_camera(&state);
        let inverse = (projection * view).inverse();
        let near = inverse.project_point3(Vec3::new(normalized[0], normalized[1], 1.0));
        let direction = (near - eye).try_normalize().unwrap_or(forward);
        Ray {
            origin: eye.as_dvec3().to_array(),
            direction: direction.as_dvec3().to_array(),
        }
    }

    #[must_use]
    pub fn center_ray(&self) -> Ray {
        self.ray_for_pointer([0.0, 0.0])
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

fn recover_fly_if_inside(state: &mut SceneState) -> bool {
    if state.fly_collision_mode != FlyCollisionMode::SolidVoxels {
        return false;
    }
    let Some(fly) = state.fly else {
        return false;
    };
    let Some(occupancy) = state.occupancy.clone() else {
        return false;
    };
    let Some(occupancy) = occupancy.try_read() else {
        return false;
    };
    if fly_position_occupied(
        state.dimensions,
        state.world_to_index,
        fly.position,
        &occupancy,
    ) != Some(true)
    {
        return false;
    }
    state
        .fly
        .as_mut()
        .expect("the checked Fly camera remains initialized")
        .position = fly.home.position;
    true
}

fn fly_position_occupied(
    dimensions: [u32; 3],
    world_to_index: Matrix4,
    position: Vec3,
    occupancy: &SharedOccupancyRead<'_>,
) -> Option<bool> {
    if occupancy.dimensions() != dimensions {
        return None;
    }
    let index = transform_point(
        world_to_index,
        [
            f64::from(position.x),
            f64::from(position.y),
            f64::from(position.z),
        ],
    );
    if !index.iter().all(|value| value.is_finite()) {
        return None;
    }
    let voxel = index.map(|value| (value + 0.5).floor());
    if voxel
        .iter()
        .zip(dimensions)
        .any(|(&coordinate, dimension)| coordinate < 0.0 || coordinate >= f64::from(dimension))
    {
        return Some(false);
    }
    let voxel = voxel.map(|coordinate| coordinate as u32);
    occupancy.occupied(linear_index(dimensions, voxel)?)
}

fn resolve_fly_collision(
    dimensions: [u32; 3],
    index_to_world: Matrix4,
    world_to_index: Matrix4,
    position: Vec3,
    displacement: Vec3,
    occupancy: &SharedOccupancyRead<'_>,
) -> Vec3 {
    if occupancy.dimensions() != dimensions {
        return position;
    }
    let mut current = transform_point(
        world_to_index,
        [
            f64::from(position.x),
            f64::from(position.y),
            f64::from(position.z),
        ],
    );
    let mut remaining = transform_vector(
        world_to_index,
        [
            f64::from(displacement.x),
            f64::from(displacement.y),
            f64::from(displacement.z),
        ],
    );
    if !current
        .iter()
        .chain(&remaining)
        .all(|value| value.is_finite())
    {
        return position;
    }

    for _ in 0..MAX_FLY_COLLISION_CONTACTS {
        if remaining
            .iter()
            .all(|component| component.abs() <= f64::EPSILON)
        {
            break;
        }
        match segment_surface_hit_with_occupancy(
            dimensions,
            current,
            remaining,
            MAX_FLY_COLLISION_VOXEL_STEPS,
            |index| occupancy.occupied(index),
        ) {
            SegmentSweep::Clear => {
                for axis in 0..3 {
                    current[axis] += remaining[axis];
                }
                break;
            }
            SegmentSweep::Indeterminate => break,
            SegmentSweep::Hit(hit) => {
                let entry_t = hit.entry_t.clamp(0.0, 1.0);
                let mut contact =
                    std::array::from_fn(|axis| current[axis] + remaining[axis] * entry_t);
                let mut blocked = hit.entry_axes;
                if !blocked.contains(&true) {
                    let tolerance = f64::EPSILON
                        * contact
                            .iter()
                            .chain(&remaining)
                            .map(|value| value.abs())
                            .fold(1.0_f64, f64::max)
                        * 32.0;
                    for axis in 0..3 {
                        let minimum = f64::from(hit.voxel[axis]) - 0.5;
                        let maximum = f64::from(hit.voxel[axis]) + 0.5;
                        blocked[axis] = (remaining[axis] > 0.0
                            && (contact[axis] - minimum).abs() <= tolerance)
                            || (remaining[axis] < 0.0
                                && (contact[axis] - maximum).abs() <= tolerance);
                    }
                }
                if !blocked.contains(&true) {
                    break;
                }
                let unconsumed = 1.0 - entry_t;
                let incoming = remaining;
                for axis in 0..3 {
                    remaining[axis] *= unconsumed;
                    if !blocked[axis] {
                        continue;
                    }
                    if incoming[axis] > 0.0 {
                        contact[axis] =
                            f64::from(hit.voxel[axis]) - 0.5 - FLY_COLLISION_SKIN_VOXELS;
                    } else if incoming[axis] < 0.0 {
                        contact[axis] =
                            f64::from(hit.voxel[axis]) + 0.5 + FLY_COLLISION_SKIN_VOXELS;
                    }
                    remaining[axis] = 0.0;
                }
                current = contact;
            }
        }
    }

    let world = transform_point(index_to_world, current);
    Vec3::new(world[0] as f32, world[1] as f32, world[2] as f32)
}

fn reset_document_navigation(state: &mut SceneState) {
    state.view_mode = VolumeViewMode::Orbit;
    state.orbit = OrbitCameraState::default();
    state.fly = None;
    state.leak_light = false;
}

pub struct VolumePaintCallback {
    scene: SceneBridge,
    target_format: wgpu::TextureFormat,
    compute_supported: bool,
}

impl VolumePaintCallback {
    #[must_use]
    pub const fn new(
        scene: SceneBridge,
        target_format: wgpu::TextureFormat,
        compute_supported: bool,
    ) -> Self {
        Self {
            scene,
            target_format,
            compute_supported,
        }
    }
}

impl CallbackTrait for VolumePaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if resources.get::<GpuVolumeRenderer>().is_none() {
            resources.insert(GpuVolumeRenderer::new(
                device,
                self.target_format,
                self.compute_supported,
            ));
            self.scene.needs_remesh.store(true, Ordering::Release);
        }
        let renderer = resources
            .get_mut::<GpuVolumeRenderer>()
            .expect("renderer was inserted above");
        let (updates, uniforms, has_more_updates, has_pending_geometry_updates) = {
            let mut state = self.scene.state.lock().expect("3D scene state poisoned");
            let updates: [Option<SceneUpdate>; MAX_GPU_UPDATES_PER_FRAME] =
                std::array::from_fn(|_| state.updates.pop_front());
            let uniforms = scene_uniforms(&state);
            let has_pending_geometry_updates = state.updates.iter().any(|update| {
                matches!(
                    update,
                    SceneUpdate::Reset { .. }
                        | SceneUpdate::Invalidate { .. }
                        | SceneUpdate::Chunk { .. }
                        | SceneUpdate::LeakDistanceField { .. }
                )
            });
            (
                updates,
                uniforms,
                !state.updates.is_empty(),
                has_pending_geometry_updates,
            )
        };
        for update in updates.into_iter().flatten() {
            renderer.apply_update(device, queue, update);
        }
        let leak_preparation_pending =
            renderer.prepare_leak_distances(device, encoder, MAX_LEAK_DISTANCE_CHUNKS_PER_FRAME);
        if !has_pending_geometry_updates {
            renderer.activate_pending_leak_distances();
        }
        let leak_distance_available = renderer.leak_distance_available();
        let leak_distance_ready = renderer.leak_distance_ready() && !has_pending_geometry_updates;
        let leak_readiness_changed = self
            .scene
            .leak_distance_ready
            .swap(leak_distance_ready, Ordering::AcqRel)
            != leak_distance_ready;
        self.scene
            .leak_distance_available
            .store(leak_distance_available, Ordering::Release);
        let mut uniforms = uniforms;
        uniforms.leak_distance_scale = renderer.leak_distance_scale();
        uniforms.leak_light = u32::from(uniforms.leak_light != 0 && leak_distance_available);
        renderer.refresh_chunk_visibility(&uniforms);
        if renderer.uniforms != uniforms {
            queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            renderer.uniforms = uniforms;
        }
        if (has_more_updates || leak_preparation_pending || leak_readiness_changed)
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
        render_pass.set_pipeline(&renderer.depth_reset_pipeline);
        render_pass.draw(0..3, 0..1);
        render_pass.set_bind_group(0, &renderer.bind_group, &[]);
        for chunk in renderer.chunks.values() {
            if !chunk.visible {
                continue;
            }
            if renderer.uniforms.leak_light != 0
                && let Some(leak_distance) = &chunk.leak_distance
            {
                render_pass.set_pipeline(&renderer.leak_pipeline);
                render_pass.set_vertex_buffer(1, leak_distance.slice(..));
            } else {
                render_pass.set_pipeline(&renderer.pipeline);
            }
            render_pass.set_vertex_buffer(0, chunk.buffer.slice(..));
            render_pass.draw(0..6, 0..chunk.instances);
        }
        if let Some(highlight) = renderer
            .highlight
            .as_ref()
            .filter(|highlight| highlight.faces != 0)
        {
            render_pass.set_pipeline(&renderer.highlight_pipeline);
            render_pass.set_bind_group(0, &renderer.bind_group, &[]);
            render_pass.set_vertex_buffer(0, highlight.buffer.slice(..));
            render_pass.draw(0..6, 0..highlight.faces);
        }
    }
}

struct GpuChunk {
    buffer: wgpu::Buffer,
    leak_distance: Option<wgpu::Buffer>,
    pending_leak_distance: Option<wgpu::Buffer>,
    retained_quads: Option<Vec<CompactQuad>>,
    instances: u32,
    visible: bool,
    visibility_dirty: bool,
}

struct GpuLeakCompute {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LeakComputeParams {
    chunk_origin_and_count: [u32; 4],
    halo_dimensions: [u32; 4],
}

struct TaggedLeakDistanceField {
    epoch: u64,
    revision: u64,
    generation: u64,
    field: Arc<LeakDistanceField>,
}

struct GpuHighlight {
    buffer: wgpu::Buffer,
    faces: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct ChunkVisibilityView {
    view_projection: [[f32; 4]; 4],
    index_to_world: [[f32; 4]; 4],
    dimensions: [u32; 3],
}

impl ChunkVisibilityView {
    fn new(uniforms: &Uniforms, dimensions: [u32; 3]) -> Self {
        Self {
            view_projection: uniforms.view_projection,
            index_to_world: uniforms.index_to_world,
            dimensions,
        }
    }

    fn index_to_clip(self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.view_projection)
            * Mat4::from_cols_array_2d(&self.index_to_world)
    }
}

struct GpuVolumeRenderer {
    depth_reset_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    leak_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniforms: Uniforms,
    bind_group: wgpu::BindGroup,
    leak_compute: Option<GpuLeakCompute>,
    chunks: HashMap<ChunkCoord, GpuChunk>,
    chunk_generations: HashMap<ChunkCoord, u64>,
    chunk_visibility_view: Option<ChunkVisibilityView>,
    has_dirty_chunk_visibility: bool,
    highlight: Option<GpuHighlight>,
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
    leak_distance_field: Option<TaggedLeakDistanceField>,
    leak_distance_pending: VecDeque<ChunkCoord>,
    leak_distance_pending_set: HashSet<ChunkCoord>,
    pending_leak_distance_field: Option<TaggedLeakDistanceField>,
    pending_leak_distance_chunks: VecDeque<ChunkCoord>,
    pending_leak_distance_set: HashSet<ChunkCoord>,
}

impl GpuVolumeRenderer {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        compute_supported: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelInk compact quad shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniforms = Uniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelInk volume uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
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
        let depth_reset_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VoxelInk reverse-Z depth reset shader"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_RESET_SHADER.into()),
        });
        let depth_reset_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelInk reverse-Z depth reset pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &depth_reset_shader,
                entry_point: Some("depth_reset_vertex"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_reset_shader,
                entry_point: Some("depth_reset_fragment"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
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
            depth_stencil: Some(volume_depth_stencil_state()),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let leak_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VoxelInk leak-light quad pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("leak_vertex_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
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
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<PackedLeakDistance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Snorm16x4,
                            offset: 0,
                            shader_location: 2,
                        }],
                    },
                ],
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
            depth_stencil: Some(volume_depth_stencil_state()),
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
        let leak_compute = compute_supported.then(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("VoxelInk leak-distance compute shader"),
                source: wgpu::ShaderSource::Wgsl(LEAK_DISTANCE_COMPUTE_SHADER.into()),
            });
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("VoxelInk leak-distance compute bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("VoxelInk leak-distance compute pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VoxelInk leak-distance preparation pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prepare_leak_distance"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            GpuLeakCompute {
                pipeline,
                bind_group_layout,
            }
        });
        Self {
            depth_reset_pipeline,
            pipeline,
            leak_pipeline,
            highlight_pipeline,
            uniform_buffer,
            uniforms,
            bind_group,
            leak_compute,
            chunks: HashMap::new(),
            chunk_generations: HashMap::new(),
            chunk_visibility_view: None,
            has_dirty_chunk_visibility: false,
            highlight: None,
            epoch: 0,
            revision: 0,
            dimensions: [1; 3],
            affine: SceneState::default().affine,
            leak_distance_field: None,
            leak_distance_pending: VecDeque::new(),
            leak_distance_pending_set: HashSet::new(),
            pending_leak_distance_field: None,
            pending_leak_distance_chunks: VecDeque::new(),
            pending_leak_distance_set: HashSet::new(),
        }
    }

    fn apply_update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, update: SceneUpdate) {
        match update {
            SceneUpdate::Reset {
                epoch,
                revision,
                dimensions,
                affine,
            } => {
                self.chunks.clear();
                self.chunk_generations.clear();
                self.chunk_visibility_view = None;
                self.has_dirty_chunk_visibility = false;
                self.highlight = None;
                self.epoch = epoch;
                self.revision = revision;
                self.dimensions = dimensions;
                self.affine = affine;
                self.clear_leak_distance_field();
            }
            SceneUpdate::Invalidate { revision } => {
                self.revision = revision;
                self.clear_highlight();
                self.clear_pending_leak_distance_field();
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
                    self.leak_distance_pending_set.remove(&tag.chunk);
                    self.pending_leak_distance_set.remove(&tag.chunk);
                    return;
                }
                let instances = quads.len() as u32;
                let contents = bytemuck::cast_slice(&quads);
                if let Some(chunk) = self.chunks.get_mut(&tag.chunk)
                    && can_reuse_chunk_buffer(chunk.buffer.size(), contents.len() as u64)
                {
                    queue.write_buffer(&chunk.buffer, 0, contents);
                    chunk.leak_distance = None;
                    chunk.pending_leak_distance = None;
                    chunk.retained_quads = self.leak_compute.is_none().then_some(quads);
                    chunk.instances = instances;
                    if self.leak_distance_field.is_some() {
                        self.enqueue_leak_distance(tag.chunk);
                    }
                    if self.pending_leak_distance_field.is_some() {
                        self.enqueue_pending_leak_distance(tag.chunk);
                    }
                    return;
                }
                let mut usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
                if self.leak_compute.is_some() {
                    usage |= wgpu::BufferUsages::STORAGE;
                }
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("VoxelInk chunk quad instances"),
                    contents,
                    usage,
                });
                let retained_quads = self.leak_compute.is_none().then_some(quads);
                let previous_visibility = self
                    .chunks
                    .get(&tag.chunk)
                    .map(|chunk| (chunk.visible, chunk.visibility_dirty));
                let (visible, visibility_dirty) = previous_visibility.unwrap_or((true, true));
                self.has_dirty_chunk_visibility |= previous_visibility.is_none();
                self.chunks.insert(
                    tag.chunk,
                    GpuChunk {
                        buffer,
                        leak_distance: None,
                        pending_leak_distance: None,
                        retained_quads,
                        instances,
                        visible,
                        visibility_dirty,
                    },
                );
                if self.leak_distance_field.is_some() {
                    if let Some(chunk) = self.chunks.get_mut(&tag.chunk) {
                        chunk.leak_distance = None;
                    }
                    self.enqueue_leak_distance(tag.chunk);
                }
                if self.pending_leak_distance_field.is_some() {
                    self.enqueue_pending_leak_distance(tag.chunk);
                }
            }
            SceneUpdate::LeakDistanceField {
                epoch,
                revision,
                generation,
                field,
            } => {
                let expected_steps = self
                    .dimensions
                    .into_iter()
                    .try_fold(1_usize, |total, dimension| {
                        total.checked_mul(dimension as usize)
                    });
                if epoch != self.epoch
                    || revision != self.revision
                    || expected_steps != Some(field.steps().len())
                {
                    return;
                }
                if let Some(current) = self.pending_leak_distance_field.as_mut()
                    && current.epoch == epoch
                    && current.revision == revision
                {
                    if current.generation >= generation {
                        return;
                    }
                    if Arc::ptr_eq(&current.field, &field) {
                        current.generation = generation;
                        return;
                    }
                }
                self.pending_leak_distance_field = Some(TaggedLeakDistanceField {
                    epoch,
                    revision,
                    generation,
                    field,
                });
                self.pending_leak_distance_chunks.clear();
                self.pending_leak_distance_set.clear();
                let chunks = self.chunks.keys().copied().collect::<Vec<_>>();
                for chunk in self.chunks.values_mut() {
                    chunk.pending_leak_distance = None;
                }
                for chunk in chunks {
                    self.enqueue_pending_leak_distance(chunk);
                }
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
                let contents = bytemuck::cast_slice(&faces);
                if let Some(highlight) = self.highlight.as_mut()
                    && highlight.buffer.size() >= contents.len() as u64
                {
                    queue.write_buffer(&highlight.buffer, 0, contents);
                    highlight.faces = face_count;
                    return;
                }
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("VoxelInk highlight boundary-face instances"),
                    contents,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
                self.highlight = Some(GpuHighlight {
                    buffer,
                    faces: face_count,
                });
            }
            SceneUpdate::ClearHighlight => {
                self.clear_highlight();
            }
        }
    }

    fn clear_leak_distance_field(&mut self) {
        self.leak_distance_field = None;
        self.leak_distance_pending.clear();
        self.leak_distance_pending_set.clear();
        for chunk in self.chunks.values_mut() {
            chunk.leak_distance = None;
            chunk.pending_leak_distance = None;
        }
        self.clear_pending_leak_distance_field();
    }

    fn clear_pending_leak_distance_field(&mut self) {
        self.pending_leak_distance_field = None;
        self.pending_leak_distance_chunks.clear();
        self.pending_leak_distance_set.clear();
        for chunk in self.chunks.values_mut() {
            chunk.pending_leak_distance = None;
        }
    }

    fn enqueue_leak_distance(&mut self, chunk: ChunkCoord) {
        if self.chunks.contains_key(&chunk) && self.leak_distance_pending_set.insert(chunk) {
            self.leak_distance_pending.push_back(chunk);
        }
    }

    fn enqueue_pending_leak_distance(&mut self, chunk: ChunkCoord) {
        if self.chunks.contains_key(&chunk) && self.pending_leak_distance_set.insert(chunk) {
            self.pending_leak_distance_chunks.push_back(chunk);
        }
    }

    fn leak_distance_scale(&self) -> f32 {
        self.leak_distance_field
            .as_ref()
            .map(|tagged| tagged.field.distance_scale() as f32)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(1.0)
    }

    fn leak_distance_ready(&self) -> bool {
        self.leak_distance_field
            .as_ref()
            .is_some_and(|field| field.revision == self.revision)
            && self.leak_distance_pending_set.is_empty()
            && self
                .chunks
                .values()
                .all(|chunk| chunk.leak_distance.is_some())
    }

    fn leak_distance_available(&self) -> bool {
        self.leak_distance_field.is_some()
            && self
                .chunks
                .values()
                .any(|chunk| chunk.leak_distance.is_some())
    }

    fn activate_pending_leak_distances(&mut self) {
        if self.pending_leak_distance_field.is_none()
            || !self.pending_leak_distance_set.is_empty()
            || self
                .chunks
                .values()
                .any(|chunk| chunk.pending_leak_distance.is_none())
        {
            return;
        }
        self.leak_distance_field = self.pending_leak_distance_field.take();
        self.leak_distance_pending.clear();
        self.leak_distance_pending_set.clear();
        for chunk in self.chunks.values_mut() {
            chunk.leak_distance = chunk.pending_leak_distance.take();
        }
    }

    /// Prepares at most `maximum_chunks` diagnostic buffers and returns whether
    /// another frame is required. The visible mode remains disabled until every
    /// current chunk has a prepared buffer, so progressive work is never mistaken
    /// for a sealed surface.
    fn prepare_leak_distances(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        maximum_chunks: usize,
    ) -> bool {
        if self.leak_distance_field.is_none() && self.pending_leak_distance_field.is_none() {
            return false;
        }
        let mut failed = false;
        let mut prepared_count = 0;
        while prepared_count < maximum_chunks {
            let coordinate = loop {
                let Some(coordinate) = self.leak_distance_pending.pop_front() else {
                    break None;
                };
                if self.leak_distance_pending_set.remove(&coordinate) {
                    break Some(coordinate);
                }
            };
            let Some(coordinate) = coordinate else {
                break;
            };
            prepared_count += 1;
            let Some(field) = self
                .leak_distance_field
                .as_ref()
                .map(|tagged| Arc::clone(&tagged.field))
            else {
                break;
            };
            let Some((chunk_origin, halo_dimensions, halo_steps)) =
                copy_leak_distance_halo(self.dimensions, coordinate, field.steps())
            else {
                failed = true;
                break;
            };
            let prepared = if let Some(compute) = &self.leak_compute {
                let Some(chunk) = self.chunks.get(&coordinate) else {
                    continue;
                };
                prepare_gpu_leak_distance(
                    device,
                    encoder,
                    compute,
                    chunk,
                    chunk_origin,
                    halo_dimensions,
                    &halo_steps,
                )
            } else {
                let Some(chunk) = self.chunks.get(&coordinate) else {
                    continue;
                };
                prepare_cpu_leak_distance(device, chunk, chunk_origin, halo_dimensions, &halo_steps)
            };
            let Some(prepared) = prepared else {
                failed = true;
                break;
            };
            if let Some(chunk) = self.chunks.get_mut(&coordinate) {
                chunk.leak_distance = Some(prepared);
            }
        }
        if failed {
            self.clear_leak_distance_field();
            return false;
        }
        while prepared_count < maximum_chunks {
            let coordinate = loop {
                let Some(coordinate) = self.pending_leak_distance_chunks.pop_front() else {
                    break None;
                };
                if self.pending_leak_distance_set.remove(&coordinate) {
                    break Some(coordinate);
                }
            };
            let Some(coordinate) = coordinate else {
                break;
            };
            prepared_count += 1;
            let Some(field) = self
                .pending_leak_distance_field
                .as_ref()
                .map(|tagged| Arc::clone(&tagged.field))
            else {
                break;
            };
            let Some((chunk_origin, halo_dimensions, halo_steps)) =
                copy_leak_distance_halo(self.dimensions, coordinate, field.steps())
            else {
                failed = true;
                break;
            };
            let prepared = if let Some(compute) = &self.leak_compute {
                self.chunks.get(&coordinate).and_then(|chunk| {
                    prepare_gpu_leak_distance(
                        device,
                        encoder,
                        compute,
                        chunk,
                        chunk_origin,
                        halo_dimensions,
                        &halo_steps,
                    )
                })
            } else {
                self.chunks.get(&coordinate).and_then(|chunk| {
                    prepare_cpu_leak_distance(
                        device,
                        chunk,
                        chunk_origin,
                        halo_dimensions,
                        &halo_steps,
                    )
                })
            };
            let Some(prepared) = prepared else {
                failed = true;
                break;
            };
            if let Some(chunk) = self.chunks.get_mut(&coordinate) {
                chunk.pending_leak_distance = Some(prepared);
            }
        }
        if failed {
            self.clear_pending_leak_distance_field();
        }
        !self.leak_distance_pending_set.is_empty() || !self.pending_leak_distance_set.is_empty()
    }

    fn clear_highlight(&mut self) {
        if let Some(highlight) = self.highlight.as_mut()
            && highlight.buffer.size() <= MAX_RETAINED_HIGHLIGHT_BUFFER_BYTES
        {
            highlight.faces = 0;
        } else {
            self.highlight = None;
        }
    }

    fn refresh_chunk_visibility(&mut self, uniforms: &Uniforms) -> usize {
        let view = ChunkVisibilityView::new(uniforms, self.dimensions);
        let refresh_all = self.chunk_visibility_view != Some(view);
        if !refresh_all && !self.has_dirty_chunk_visibility {
            return 0;
        }

        let index_to_clip = view.index_to_clip();
        let mut refreshed = 0;
        for (coordinate, chunk) in &mut self.chunks {
            if refresh_all || chunk.visibility_dirty {
                chunk.visible =
                    chunk_intersects_clip_volume(*coordinate, view.dimensions, index_to_clip);
                chunk.visibility_dirty = false;
                refreshed += 1;
            }
        }
        self.chunk_visibility_view = Some(view);
        self.has_dirty_chunk_visibility = false;
        refreshed
    }
}

fn copy_leak_distance_halo(
    dimensions: [u32; 3],
    coordinate: ChunkCoord,
    field_steps: &[u16],
) -> Option<([u32; 3], [u32; 3], Vec<u16>)> {
    let expected_steps = dimensions
        .into_iter()
        .try_fold(1_usize, |total, dimension| {
            total.checked_mul(dimension as usize)
        })?;
    if field_steps.len() != expected_steps {
        return None;
    }
    let origin = coordinate.origin()?;
    let extent = coordinate.extent(dimensions)?;
    let halo_dimensions = extent.map(|axis| axis + 2);
    let halo_len = halo_dimensions
        .into_iter()
        .try_fold(1_usize, |total, dimension| {
            total.checked_mul(dimension as usize)
        })?;
    let mut halo = Vec::new();
    halo.try_reserve_exact(halo_len).ok()?;
    let width = dimensions[0] as usize;
    let height = dimensions[1] as usize;
    for local_z in 0..halo_dimensions[2] {
        let global_z = i64::from(origin[2]) + i64::from(local_z) - 1;
        for local_y in 0..halo_dimensions[1] {
            let global_y = i64::from(origin[1]) + i64::from(local_y) - 1;
            for local_x in 0..halo_dimensions[0] {
                let global_x = i64::from(origin[0]) + i64::from(local_x) - 1;
                if !(0..i64::from(dimensions[0])).contains(&global_x)
                    || !(0..i64::from(dimensions[1])).contains(&global_y)
                    || !(0..i64::from(dimensions[2])).contains(&global_z)
                {
                    // Space beyond the stored volume is the exterior source.
                    halo.push(0);
                    continue;
                }
                let x = usize::try_from(global_x).ok()?;
                let y = usize::try_from(global_y).ok()?;
                let z = usize::try_from(global_z).ok()?;
                halo.push(field_steps[x + width * (y + height * z)]);
            }
        }
    }
    Some((origin, halo_dimensions, halo))
}

fn prepare_cpu_leak_distance(
    device: &wgpu::Device,
    chunk: &GpuChunk,
    chunk_origin: [u32; 3],
    halo_dimensions: [u32; 3],
    halo_steps: &[u16],
) -> Option<wgpu::Buffer> {
    let quads = chunk.retained_quads.as_deref()?;
    let packed = pack_leak_distances(quads, chunk_origin, halo_dimensions, halo_steps).ok()?;
    if packed.len() != chunk.instances as usize {
        return None;
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelInk CPU-prepared chunk leak distances"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::VERTEX,
        }),
    )
}

fn prepare_gpu_leak_distance(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    compute: &GpuLeakCompute,
    chunk: &GpuChunk,
    chunk_origin: [u32; 3],
    halo_dimensions: [u32; 3],
    halo_steps: &[u16],
) -> Option<wgpu::Buffer> {
    if chunk.instances == 0 {
        return None;
    }
    let mut halo_words = Vec::new();
    let word_count = halo_steps.len().div_ceil(2);
    let padded_word_count = word_count.div_ceil(4) * 4;
    halo_words.try_reserve_exact(padded_word_count).ok()?;
    halo_words.resize(padded_word_count, 0_u32);
    for (index, &step) in halo_steps.iter().enumerate() {
        halo_words[index / 2] |= u32::from(step) << ((index % 2) * 16);
    }
    let halo_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VoxelInk packed chunk leak-distance halo"),
        contents: bytemuck::cast_slice(&halo_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_bytes =
        u64::from(chunk.instances).checked_mul(size_of::<PackedLeakDistance>() as u64)?;
    let padded_output_bytes = output_bytes.div_ceil(16) * 16;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VoxelInk GPU-prepared chunk leak distances"),
        size: padded_output_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let parameters = LeakComputeParams {
        chunk_origin_and_count: [
            chunk_origin[0],
            chunk_origin[1],
            chunk_origin[2],
            chunk.instances,
        ],
        halo_dimensions: [
            halo_dimensions[0],
            halo_dimensions[1],
            halo_dimensions[2],
            0,
        ],
    };
    let parameter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VoxelInk leak-distance compute parameters"),
        contents: bytemuck::bytes_of(&parameters),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("VoxelInk leak-distance compute bind group"),
        layout: &compute.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: parameter_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chunk.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: halo_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VoxelInk prepare chunk leak distances"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(chunk.instances.div_ceil(LEAK_COMPUTE_WORKGROUP_SIZE), 1, 1);
    }
    Some(output)
}

fn can_reuse_chunk_buffer(existing_bytes: u64, replacement_bytes: u64) -> bool {
    let reuse_ceiling = replacement_bytes
        .saturating_mul(MAX_CHUNK_BUFFER_REUSE_RATIO)
        .max(CHUNK_BUFFER_REUSE_BASE_BYTES);
    existing_bytes >= replacement_bytes && existing_bytes <= reuse_ceiling
}

fn chunk_intersects_clip_volume(
    coordinate: ChunkCoord,
    dimensions: [u32; 3],
    index_to_clip: Mat4,
) -> bool {
    let Some(origin) = coordinate.origin() else {
        return true;
    };
    let Some(extent) = coordinate.extent(dimensions) else {
        return true;
    };
    if !index_to_clip.is_finite() {
        return true;
    }

    let minimum: [f64; 3] = std::array::from_fn(|axis| f64::from(origin[axis]) - 0.5);
    let maximum: [f64; 3] =
        std::array::from_fn(|axis| f64::from(origin[axis]) + f64::from(extent[axis]) - 0.5);
    let columns = index_to_clip.to_cols_array_2d();
    let rows: [[f64; 4]; 4] =
        std::array::from_fn(|row| std::array::from_fn(|column| f64::from(columns[column][row])));
    let add_rows = |left: usize, right: usize| {
        std::array::from_fn(|column| rows[left][column] + rows[right][column])
    };
    let subtract_rows = |left: usize, right: usize| {
        std::array::from_fn(|column| rows[left][column] - rows[right][column])
    };
    // WGPU's homogeneous clip volume is -w <= x,y <= w and 0 <= z <= w.
    let planes = [
        add_rows(3, 0),
        subtract_rows(3, 0),
        add_rows(3, 1),
        subtract_rows(3, 1),
        rows[2],
        subtract_rows(3, 2),
    ];

    for plane in planes {
        // A clip-plane distance is linear in index space, so its maximum over the
        // transformed chunk occurs at one AABB support corner. Only a strictly
        // negative maximum can prove the entire chunk is outside. The relative
        // margin covers f32 matrix composition and shader arithmetic differences.
        let mut maximum_distance = plane[3];
        let mut magnitude = plane[3].abs();
        for axis in 0..3 {
            let at_minimum = plane[axis] * minimum[axis];
            let at_maximum = plane[axis] * maximum[axis];
            maximum_distance += at_minimum.max(at_maximum);
            magnitude += at_minimum.abs().max(at_maximum.abs());
        }
        let tolerance = magnitude.max(1.0) * CLIP_PLANE_RELATIVE_TOLERANCE;
        if maximum_distance < -tolerance {
            return false;
        }
    }
    true
}

fn scene_uniforms(state: &SceneState) -> Uniforms {
    let (view, projection, affine, near_surface_depth, inverse_scene_depth_span, _, _) =
        scene_camera(state);
    let model_view = view * affine;
    Uniforms {
        view_projection: (projection * view).to_cols_array_2d(),
        index_to_world: affine.to_cols_array_2d(),
        world_to_view: view.to_cols_array_2d(),
        index_normal_to_view: normal_matrix(model_view).to_cols_array_2d(),
        near_surface_depth,
        inverse_scene_depth_span,
        leak_half_distance: state.leak_half_distance,
        voxel_seams: u32::from(state.voxel_seams),
        leak_light: u32::from(state.leak_light),
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

#[derive(Clone, Copy)]
struct SceneBounds {
    affine: Mat4,
    corners: [Vec3; 8],
    center: Vec3,
    radius: f32,
}

fn scene_bounds(state: &SceneState) -> SceneBounds {
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
    SceneBounds {
        affine,
        corners,
        center,
        radius,
    }
}

fn orbit_camera(state: &SceneState, bounds: SceneBounds) -> (Vec3, Vec3, Mat4, Mat4) {
    let outward = Vec3::new(
        state.orbit.yaw.cos() * state.orbit.pitch.cos(),
        state.orbit.yaw.sin() * state.orbit.pitch.cos(),
        state.orbit.pitch.sin(),
    );
    let eye = bounds.center + outward * bounds.radius * 2.6 / state.orbit.zoom;
    let forward = -outward;
    let view = Mat4::look_at_rh(eye, bounds.center, Vec3::Z);
    let projection =
        Mat4::from_translation(Vec3::new(state.orbit.pan_ndc.x, state.orbit.pan_ndc.y, 0.0))
            * Mat4::perspective_infinite_reverse_rh(
                ORBIT_FIELD_OF_VIEW_RADIANS,
                state.aspect,
                bounds.radius * 0.01,
            );
    (eye, forward, view, projection)
}

fn fly_camera_from_orbit(state: &SceneState) -> FlyCameraState {
    let bounds = scene_bounds(state);
    let (position, fallback_forward, view, projection) = orbit_camera(state, bounds);
    let inverse = (projection * view).inverse();
    let near = inverse.project_point3(Vec3::new(0.0, 0.0, 1.0));
    let forward = (near - position)
        .try_normalize()
        .unwrap_or(fallback_forward);
    let yaw = forward.y.atan2(forward.x);
    let pitch = forward
        .z
        .clamp(-1.0, 1.0)
        .asin()
        .clamp(-MAX_FLY_PITCH, MAX_FLY_PITCH);
    let (minimum_speed, maximum_speed) = fly_speed_bounds(bounds.radius);
    let speed =
        (bounds.radius * FLY_INITIAL_SPEED_RADIUS_FACTOR).clamp(minimum_speed, maximum_speed);
    let home = FlyHomeState {
        position,
        yaw,
        pitch,
        speed,
    };
    FlyCameraState {
        position,
        yaw,
        pitch,
        speed,
        home,
    }
}

fn fly_forward(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        yaw.cos() * pitch.cos(),
        yaw.sin() * pitch.cos(),
        pitch.sin(),
    )
}

fn fly_speed_bounds(radius: f32) -> (f32, f32) {
    let minimum = (radius * FLY_MIN_SPEED_RADIUS_FACTOR).max(1.0e-4);
    let maximum = (radius * FLY_MAX_SPEED_RADIUS_FACTOR).max(minimum);
    (minimum, maximum)
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn fly_clip_planes(view: Mat4, bounds: SceneBounds) -> (f32, f32) {
    let (minimum_depth, maximum_depth) = bounds.corners.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), corner| {
            let depth = -view.transform_point3(*corner).z;
            (minimum.min(depth), maximum.max(depth))
        },
    );
    let epsilon = (bounds.radius * FLY_CLIP_EPSILON_RADIUS_FACTOR).max(1.0e-4);
    let margin = (bounds.radius * FLY_CLIP_MARGIN_RADIUS_FACTOR).max(epsilon * 2.0);
    let near = if minimum_depth.is_finite() && minimum_depth > epsilon {
        (minimum_depth - margin).max(epsilon)
    } else {
        epsilon
    };
    let bounds_far = if maximum_depth.is_finite() {
        maximum_depth + margin
    } else {
        bounds.radius * 4.0
    };
    let far = bounds_far
        .max(near + epsilon)
        .max(near * 1.001)
        .max(epsilon * 2.0);
    (near, far)
}

fn scene_camera(state: &SceneState) -> (Mat4, Mat4, Mat4, f32, f32, Vec3, Vec3) {
    let bounds = scene_bounds(state);
    let (eye, forward, view, projection) = match (state.view_mode, state.fly) {
        (VolumeViewMode::Fly, Some(fly)) => {
            let forward = fly_forward(fly.yaw, fly.pitch);
            let view = Mat4::look_at_rh(fly.position, fly.position + forward, Vec3::Z);
            let (near, _) = fly_clip_planes(view, bounds);
            let projection = Mat4::perspective_infinite_reverse_rh(
                FLY_FIELD_OF_VIEW_RADIANS,
                state.aspect,
                near,
            );
            (fly.position, forward, view, projection)
        }
        _ => orbit_camera(state, bounds),
    };
    let (near_surface_depth, far_surface_depth) =
        bounds
            .corners
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(near, far), corner| {
                let depth = -view.transform_point3(*corner).z;
                (near.min(depth), far.max(depth))
            });
    let inverse_scene_depth_span =
        1.0 / (far_surface_depth - near_surface_depth).max(bounds.radius * 1.0e-6);
    (
        view,
        projection,
        bounds.affine,
        near_surface_depth,
        inverse_scene_depth_span,
        eye,
        forward,
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
    use crate::{
        codecs::{BinvoxMetadata, CodecSaveState, SourceVariant},
        volume::{SourceIdentity, SpatialUnits, VolumeDocument, VolumeGeometry},
    };
    use std::{path::PathBuf, time::SystemTime};

    fn reset_bridge(
        bridge: &SceneBridge,
        epoch: u64,
        revision: u64,
        dimensions: [u32; 3],
        affine: Matrix4,
    ) {
        let voxel_count = dimensions.into_iter().map(|axis| axis as usize).product();
        let occupancy = vec![0; voxel_count];
        let _ =
            reset_bridge_with_occupancy(bridge, epoch, revision, dimensions, affine, &occupancy);
    }

    fn reset_bridge_with_occupancy(
        bridge: &SceneBridge,
        epoch: u64,
        revision: u64,
        dimensions: [u32; 3],
        affine: Matrix4,
        occupancy: &[u8],
    ) -> SharedOccupancy {
        let geometry = VolumeGeometry::new(dimensions, affine, SpatialUnits::Unknown).unwrap();
        let (occupancy, _) = SharedOccupancy::from_binary(dimensions, occupancy).unwrap();
        bridge.reset(
            epoch,
            revision,
            dimensions,
            affine,
            geometry.world_to_index,
            occupancy.clone(),
        );
        occupancy
    }

    fn test_document(dimensions: [u32; 3], occupancy: Vec<u8>) -> VolumeDocument {
        VolumeDocument::new(
            VolumeGeometry::new(
                dimensions,
                SceneState::default().affine,
                SpatialUnits::Millimeter,
            )
            .unwrap(),
            occupancy,
            SourceIdentity {
                canonical_path: PathBuf::from("/tmp/renderer-collision.binvox"),
                size: 0,
                modified: SystemTime::UNIX_EPOCH,
                sha256: [0; 32],
            },
            SourceVariant::Binvox { version: 1 },
            CodecSaveState::Binvox(BinvoxMetadata {
                version: 1,
                dimensions,
                translation: [0.0; 3],
                scale: 1.0,
            }),
            false,
        )
        .unwrap()
    }

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
    fn chunk_culling_uses_surface_extents_tolerance_and_wgpu_depth_planes() {
        let dimensions = [64, 1, 64];
        let x_chunk = ChunkCoord::new(1, 0, 0);
        assert!(!chunk_intersects_clip_volume(
            x_chunk,
            dimensions,
            Mat4::IDENTITY,
        ));

        // Chunk 1 spans x = 31.5..63.5, including voxel surface extents. Moving
        // its maximum to x = -1 touches the left clip plane and remains visible.
        assert!(chunk_intersects_clip_volume(
            x_chunk,
            dimensions,
            Mat4::from_translation(Vec3::new(-64.5, 0.0, 0.0)),
        ));
        assert!(chunk_intersects_clip_volume(
            x_chunk,
            dimensions,
            Mat4::from_translation(Vec3::new(-64.500_1, 0.0, 0.0)),
        ));
        assert!(!chunk_intersects_clip_volume(
            x_chunk,
            dimensions,
            Mat4::from_translation(Vec3::new(-64.51, 0.0, 0.0)),
        ));

        let origin_chunk = ChunkCoord::new(0, 0, 0);
        assert!(chunk_intersects_clip_volume(
            origin_chunk,
            dimensions,
            Mat4::IDENTITY,
        ));
        assert!(!chunk_intersects_clip_volume(
            origin_chunk,
            dimensions,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -32.51)),
        ));
        assert!(!chunk_intersects_clip_volume(
            origin_chunk,
            dimensions,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.51)),
        ));
    }

    #[test]
    fn chunk_buffer_reuse_is_bounded_around_the_replacement_size() {
        assert!(can_reuse_chunk_buffer(32, 16));
        assert!(can_reuse_chunk_buffer(CHUNK_BUFFER_REUSE_BASE_BYTES, 16));
        assert!(!can_reuse_chunk_buffer(
            CHUNK_BUFFER_REUSE_BASE_BYTES + 1,
            16
        ));
        assert!(can_reuse_chunk_buffer(4 * 128 * 1024, 128 * 1024));
        assert!(!can_reuse_chunk_buffer(4 * 128 * 1024 + 1, 128 * 1024));
        assert!(!can_reuse_chunk_buffer(16, 32));
    }

    #[test]
    fn chunk_culling_keeps_non_finite_and_unrepresentable_bounds_visible() {
        let mut non_finite = Mat4::IDENTITY.to_cols_array();
        non_finite[0] = f32::NAN;
        assert!(chunk_intersects_clip_volume(
            ChunkCoord::new(1, 0, 0),
            [64, 1, 1],
            Mat4::from_cols_array(&non_finite),
        ));
        assert!(chunk_intersects_clip_volume(
            ChunkCoord::new(u32::MAX, 0, 0),
            [u32::MAX, 1, 1],
            Mat4::IDENTITY,
        ));
    }

    #[test]
    fn chunk_culling_is_conservative_for_shear_reflection_and_camera_inside() {
        let mut orbit = SceneState {
            dimensions: [73, 41, 19],
            affine: [
                [-0.315, 0.014, 0.18, 106.0],
                [-0.014, -0.315, -0.07, 289.0],
                [0.03, -0.05, 0.8, -679.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            ..SceneState::default()
        };
        let uniforms = scene_uniforms(&orbit);
        let index_to_clip = Mat4::from_cols_array_2d(&uniforms.view_projection)
            * Mat4::from_cols_array_2d(&uniforms.index_to_world);
        let grid = crate::mesh::chunk_grid(orbit.dimensions);
        for z in 0..grid[2] {
            for y in 0..grid[1] {
                for x in 0..grid[0] {
                    assert!(
                        chunk_intersects_clip_volume(
                            ChunkCoord::new(x, y, z),
                            orbit.dimensions,
                            index_to_clip,
                        ),
                        "orbit framing culled affine chunk {x},{y},{z}"
                    );
                }
            }
        }

        let position = affine_mat4(orbit.affine).transform_point3(Vec3::splat(48.0));
        let home = FlyHomeState {
            position,
            yaw: 0.0,
            pitch: 0.0,
            speed: 1.0,
        };
        orbit.dimensions = [96; 3];
        orbit.view_mode = VolumeViewMode::Fly;
        orbit.fly = Some(FlyCameraState {
            position,
            yaw: 0.0,
            pitch: 0.0,
            speed: 1.0,
            home,
        });
        let uniforms = scene_uniforms(&orbit);
        let index_to_clip = Mat4::from_cols_array_2d(&uniforms.view_projection)
            * Mat4::from_cols_array_2d(&uniforms.index_to_world);
        assert!(chunk_intersects_clip_volume(
            ChunkCoord::new(1, 1, 1),
            orbit.dimensions,
            index_to_clip,
        ));
    }

    #[test]
    fn production_renderer_uses_four_sample_msaa() {
        assert_eq!(MSAA_SAMPLES, 4);
        assert!(MSAA_SAMPLES.is_power_of_two());
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
        assert_eq!(
            volume_depth_stencil_state().depth_compare,
            Some(wgpu::CompareFunction::GreaterEqual)
        );
        assert_eq!(volume_depth_stencil_state().depth_write_enabled, Some(true));
        assert!(DEPTH_RESET_SHADER.contains("vec4<f32>(positions[vertex_index], 0.0, 1.0)"));
        assert_eq!(size_of::<Uniforms>(), 304);
    }

    #[test]
    fn leak_compute_capability_requires_compute_storage_and_workgroup_limits() {
        let limits = wgpu::Limits::downlevel_defaults();
        assert!(leak_compute_supported_for(
            wgpu::DownlevelFlags::COMPUTE_SHADERS,
            &limits,
        ));
        assert!(!leak_compute_supported_for(
            wgpu::DownlevelFlags::empty(),
            &limits,
        ));

        let mut limited = limits.clone();
        limited.max_storage_buffers_per_shader_stage = LEAK_COMPUTE_STORAGE_BINDINGS - 1;
        assert!(!leak_compute_supported_for(
            wgpu::DownlevelFlags::COMPUTE_SHADERS,
            &limited,
        ));
        limited = limits;
        limited.max_compute_workgroup_size_x = LEAK_COMPUTE_WORKGROUP_SIZE - 1;
        assert!(!leak_compute_supported_for(
            wgpu::DownlevelFlags::COMPUTE_SHADERS,
            &limited,
        ));
    }

    #[test]
    fn leak_reach_is_a_uniform_only_setting() {
        let bridge = SceneBridge::default();
        assert_eq!(bridge.leak_light_reach(), 1.0);
        assert!(bridge.state.lock().unwrap().updates.is_empty());

        bridge.set_leak_light_reach(7.25);
        assert_eq!(bridge.leak_light_reach(), 7.25);
        let state = bridge.state.lock().unwrap();
        assert_eq!(scene_uniforms(&state).leak_half_distance, 7.25);
        assert!(state.updates.is_empty());
        drop(state);

        bridge.set_leak_light_reach(f64::NAN);
        bridge.set_leak_light_reach(0.0);
        assert_eq!(bridge.leak_light_reach(), 7.25);
    }

    #[test]
    fn latest_leak_field_generation_replaces_queued_work_and_clears_readiness() {
        let bridge = SceneBridge::default();
        reset_bridge(&bridge, 7, 3, [2; 3], SceneState::default().affine);
        let field = Arc::new(
            crate::volume::compute_leak_distance(
                &[0; 8],
                [2; 3],
                &[1.0; 3],
                &crate::cancel::CancellationToken::default(),
            )
            .unwrap(),
        );
        bridge.leak_distance_ready.store(true, Ordering::Release);
        bridge.set_leak_distance_field(7, 3, 4, Arc::clone(&field));
        bridge.set_leak_distance_field(7, 3, 5, Arc::clone(&field));

        assert!(!bridge.leak_distance_ready());
        let state = bridge.state.lock().unwrap();
        let queued = state
            .updates
            .iter()
            .filter_map(|update| match update {
                SceneUpdate::LeakDistanceField {
                    generation,
                    field: queued,
                    ..
                } => Some((*generation, Arc::ptr_eq(queued, &field))),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(queued, vec![(5, true)]);
    }

    #[test]
    fn voxel_seams_are_shader_derived_and_shared_by_orbit_and_fly() {
        let mut orbit_state = SceneState::default();
        assert_eq!(scene_uniforms(&orbit_state).voxel_seams, 1);

        orbit_state.voxel_seams = false;
        assert_eq!(scene_uniforms(&orbit_state).voxel_seams, 0);

        let mut fly_state = SceneState::default();
        fly_state.fly = Some(fly_camera_from_orbit(&fly_state));
        fly_state.view_mode = VolumeViewMode::Fly;
        assert_eq!(scene_uniforms(&fly_state).voxel_seams, 1);

        fly_state.voxel_seams = false;
        assert_eq!(scene_uniforms(&fly_state).voxel_seams, 0);

        assert!(SHADER.contains("@location(3) index_position: vec3<f32>"));
        assert!(SHADER.contains("@interpolate(flat) face_axis: u32"));
        assert!(SHADER.contains("fwidth(coordinate)"));
        assert!(SHADER.contains("fract(coordinate + 0.5)"));
        assert!(SHADER.contains("subpixel_fade"));
        assert!(SHADER.contains("if (uniforms.voxel_seams == 0u)"));
        assert!(SHADER.contains("return vec4<f32>(lit_color, 1.0);"));
        assert!(SHADER.contains("if (uniforms.leak_light != 0u)"));
        assert!(SHADER.contains("exp2(-physical_distance / half_distance)"));
        assert!(SHADER.contains("input.leak_distance >= 0.0"));
        assert_eq!(size_of::<CompactQuad>(), 16);
        assert_eq!(size_of::<PackedLeakDistance>(), 8);
    }

    #[test]
    fn occupancy_surface_normals_are_decoded_and_blended_with_voxel_faces() {
        assert!(SHADER.contains("fn decode_shading_normal(packed: u32)"));
        assert!(SHADER.contains("let encoded = (packed >> 22u) & 0x3ffu;"));
        assert!(SHADER.contains("index_normal = normalize("));
        assert!(SHADER.contains("0.35 * index_normal + 0.65 * decode_shading_normal(packed)"));
        assert!(
            SHADER.contains("@location(6) @interpolate(flat) view_geometric_normal: vec3<f32>")
        );
        assert!(SHADER.contains("output.view_geometric_normal = normalize("));
        assert!(
            SHADER.contains("dot(normalize(input.view_geometric_normal), view_direction) < 0.0")
        );
        assert!(!SHADER.contains("let geometric_back_facing = dot(normal, view_direction) < 0.0;"));
    }

    #[test]
    fn leak_distance_shader_restores_the_snorm_step_range() {
        assert!(SHADER.contains("const LEAK_DISTANCE_SNORM_STEP_RANGE: f32 = 32767.0;"));
        assert!(SHADER.contains("max(input.leak_distance, 0.0) * LEAK_DISTANCE_SNORM_STEP_RANGE"));
    }

    #[test]
    fn leak_light_shader_uses_continuous_physical_distance() {
        assert!(SHADER.contains("exp2(-physical_distance / half_distance)"));
        assert!(!SHADER.contains("band_leak_distance"));
        assert!(!SHADER.contains("LEAK_BANDS_PER_HALF_DISTANCE"));
    }

    #[test]
    fn leak_light_shader_uses_a_continuous_inferno_heatmap_in_linear_color() {
        for stop in [
            "vec3<f32>(0.0)",
            "vec3<f32>(87.0, 16.0, 110.0) / 255.0",
            "vec3<f32>(188.0, 55.0, 84.0) / 255.0",
            "vec3<f32>(249.0, 142.0, 9.0) / 255.0",
            "vec3<f32>(252.0, 255.0, 164.0) / 255.0",
        ] {
            assert!(SHADER.contains(stop), "missing heatmap stop {stop}");
        }
        assert!(SHADER.contains("fn srgb_to_linear(color: vec3<f32>)"));
        assert!(SHADER.contains("return srgb_to_linear(mix(lower_color, upper_color, amount));"));
        assert!(SHADER.contains("leak_heatmap(connected_light) * diagnostic_shading"));
        assert!(SHADER.contains("0.75 + 0.25 * base_lighting"));
        assert!(SHADER.contains("mix(1.0, surface_factor, 0.35)"));
        assert!(SHADER.contains("mix(1.0, voxel_ao_factor, 0.5)"));
    }

    #[test]
    fn leak_light_shader_uses_a_black_unlit_baseline() {
        assert!(SHADER.contains("select(0.0, leak_intensity, connected)"));
        assert!(SHADER.contains("lit_color = select("));
        assert!(SHADER.contains("vec3<f32>(0.0),"));
        assert!(!SHADER.contains("0.10 + 0.90 * leak_intensity"));
    }

    #[test]
    fn orbit_and_fly_use_their_expected_fields_of_view() {
        let mut state = SceneState::default();
        let (_, orbit_projection, _, _, _, _, _) = scene_camera(&state);
        let expected_orbit_scale = 1.0 / (22.5_f32.to_radians().tan());
        assert!((orbit_projection.y_axis.y - expected_orbit_scale).abs() < 1.0e-6);

        state.fly = Some(fly_camera_from_orbit(&state));
        state.view_mode = VolumeViewMode::Fly;
        let (_, fly_projection, _, _, _, _, _) = scene_camera(&state);
        let expected_fly_scale = 1.0 / (35_f32.to_radians().tan());
        assert!((fly_projection.y_axis.y - expected_fly_scale).abs() < 1.0e-6);
    }

    #[test]
    fn orbit_and_fly_use_infinite_reverse_z_projection() {
        let mut state = SceneState::default();
        let bounds = scene_bounds(&state);
        let (orbit_view, orbit_projection, _, _, _, orbit_eye, orbit_forward) =
            scene_camera(&state);
        let orbit_near = bounds.radius * 0.01;
        let orbit_near_depth = (orbit_projection * orbit_view)
            .project_point3(orbit_eye + orbit_forward * orbit_near)
            .z;
        let orbit_far_depth = (orbit_projection * orbit_view)
            .project_point3(orbit_eye + orbit_forward * bounds.radius * 10.0)
            .z;
        assert!((orbit_near_depth - 1.0).abs() < 1.0e-5);
        assert!(orbit_far_depth > 0.0 && orbit_far_depth < orbit_near_depth);

        state.fly = Some(fly_camera_from_orbit(&state));
        state.view_mode = VolumeViewMode::Fly;
        let (fly_view, fly_projection, _, _, _, fly_eye, fly_forward) = scene_camera(&state);
        let (fly_near, _) = fly_clip_planes(fly_view, bounds);
        let fly_near_depth = (fly_projection * fly_view)
            .project_point3(fly_eye + fly_forward * fly_near)
            .z;
        let fly_far_depth = (fly_projection * fly_view)
            .project_point3(fly_eye + fly_forward * bounds.radius * 10.0)
            .z;
        assert!((fly_near_depth - 1.0).abs() < 1.0e-5);
        assert!(fly_far_depth > 0.0 && fly_far_depth < fly_near_depth);
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
        assert_eq!(
            highlight_depth_stencil_state().depth_compare,
            Some(wgpu::CompareFunction::GreaterEqual)
        );
        assert_eq!(highlight_depth_stencil_state().format, DEPTH_FORMAT);
        assert_eq!(highlight_depth_stencil_state().bias.constant, 64);
        assert_eq!(highlight_depth_stencil_state().bias.slope_scale, 0.0);
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
        reset_bridge(&bridge, 4, 8, [32; 3], SceneState::default().affine);
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
        reset_bridge(&bridge, 4, 8, [4, 5, 6], SceneState::default().affine);
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
        reset_bridge(&bridge, 7, 11, [8; 3], SceneState::default().affine);
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
        reset_bridge(&bridge, 3, 6, [64, 8, 8], SceneState::default().affine);
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
        reset_bridge(&bridge, 4, 8, [128, 32, 32], affine);
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
        reset_bridge(&bridge, 3, 6, [8; 3], SceneState::default().affine);
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
        reset_bridge(&bridge, 2, 3, [8; 3], SceneState::default().affine);
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

    fn assert_vec3_close(actual: Vec3, expected: Vec3, tolerance: f32) {
        assert!(
            actual.distance(expected) <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn ray_origin(ray: Ray) -> Vec3 {
        Vec3::new(
            ray.origin[0] as f32,
            ray.origin[1] as f32,
            ray.origin[2] as f32,
        )
    }

    fn ray_direction(ray: Ray) -> Vec3 {
        Vec3::new(
            ray.direction[0] as f32,
            ray.direction[1] as f32,
            ray.direction[2] as f32,
        )
    }

    #[test]
    fn first_fly_entry_preserves_the_pan_aware_orbit_viewpoint() {
        let bridge = SceneBridge::default();
        reset_bridge(
            &bridge,
            3,
            7,
            [31, 17, 9],
            [
                [0.0, -2.0, 0.25, 10.0],
                [3.0, 0.0, 0.5, -5.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        bridge.set_aspect(16.0 / 9.0);
        bridge.orbit(egui::vec2(31.0, -19.0));
        bridge.pan(egui::vec2(73.0, -41.0), egui::vec2(900.0, 600.0));

        let (expected_eye, expected_forward) = {
            let state = bridge.state.lock().unwrap();
            let (view, projection, _, _, _, eye, fallback_forward) = scene_camera(&state);
            let far = (projection * view)
                .inverse()
                .project_point3(Vec3::new(0.0, 0.0, 1.0));
            (eye, (far - eye).try_normalize().unwrap_or(fallback_forward))
        };
        let orbit_ray = bridge.center_ray();
        assert_vec3_close(ray_origin(orbit_ray), expected_eye, 1.0e-4);
        assert_vec3_close(ray_direction(orbit_ray), expected_forward, 1.0e-5);

        bridge.set_view_mode(VolumeViewMode::Fly);

        let fly = bridge.state.lock().unwrap().fly.unwrap();
        assert_vec3_close(fly.position, expected_eye, 1.0e-4);
        assert_vec3_close(fly_forward(fly.yaw, fly.pitch), expected_forward, 1.0e-5);
        let fly_ray = bridge.center_ray();
        assert_vec3_close(ray_origin(fly_ray), expected_eye, 1.0e-4);
        assert_vec3_close(ray_direction(fly_ray), expected_forward, 1.0e-5);
    }

    #[test]
    fn unpanned_affine_fly_center_ray_points_at_the_transformed_volume_center() {
        let bridge = SceneBridge::default();
        reset_bridge(
            &bridge,
            2,
            1,
            [13, 9, 7],
            [
                [0.0, -1.5, 0.2, 40.0],
                [2.5, 0.0, 0.4, -12.0],
                [0.1, 0.0, 3.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        let center = {
            let state = bridge.state.lock().unwrap();
            scene_bounds(&state).center
        };
        bridge.set_view_mode(VolumeViewMode::Fly);

        let ray = bridge.center_ray();
        let expected = (center - ray_origin(ray)).normalize();
        assert_vec3_close(ray_direction(ray), expected, 5.0e-5);
    }

    #[test]
    fn mode_switches_and_revision_resets_preserve_both_cameras() {
        let bridge = SceneBridge::default();
        let affine = SceneState::default().affine;
        reset_bridge(&bridge, 4, 1, [32; 3], affine);
        bridge.orbit(egui::vec2(25.0, -14.0));
        bridge.pan(egui::vec2(30.0, -20.0), egui::vec2(800.0, 600.0));
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.fly_look(egui::vec2(18.0, -11.0));
        bridge.fly_move(FlyMovement::new(1.0, 0.5, 0.25, true), 0.04);
        bridge.adjust_fly_speed(120.0);
        let remembered_fly = bridge.state.lock().unwrap().fly.unwrap();

        bridge.set_view_mode(VolumeViewMode::Orbit);
        bridge.orbit(egui::vec2(-12.0, 7.0));
        bridge.zoom(1.4);
        let remembered_orbit = bridge.state.lock().unwrap().orbit;

        bridge.set_view_mode(VolumeViewMode::Fly);
        assert_eq!(bridge.state.lock().unwrap().fly, Some(remembered_fly));
        reset_bridge(&bridge, 4, 2, [32; 3], affine);
        {
            let state = bridge.state.lock().unwrap();
            assert_eq!(state.view_mode, VolumeViewMode::Fly);
            assert_eq!(state.fly, Some(remembered_fly));
            assert_eq!(state.orbit, remembered_orbit);
        }

        reset_bridge(&bridge, 5, 1, [32; 3], affine);
        {
            let state = bridge.state.lock().unwrap();
            assert_eq!(state.view_mode, VolumeViewMode::Orbit);
            assert_eq!(state.orbit, OrbitCameraState::default());
            assert_eq!(state.fly, None);
        }

        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.clear();
        let state = bridge.state.lock().unwrap();
        assert_eq!(state.view_mode, VolumeViewMode::Orbit);
        assert_eq!(state.orbit, OrbitCameraState::default());
        assert_eq!(state.fly, None);

        drop(state);
        let epoch_zero_bridge = SceneBridge::default();
        epoch_zero_bridge.set_view_mode(VolumeViewMode::Fly);
        epoch_zero_bridge.clear();
        let state = epoch_zero_bridge.state.lock().unwrap();
        assert_eq!(state.view_mode, VolumeViewMode::Orbit);
        assert_eq!(state.fly, None);
    }

    #[test]
    fn fly_movement_basis_defaults_to_level_and_survives_document_resets() {
        let bridge = SceneBridge::default();
        assert_eq!(bridge.fly_movement_basis(), FlyMovementBasis::Level);

        bridge.set_fly_movement_basis(FlyMovementBasis::ViewRelative);
        reset_bridge(&bridge, 3, 1, [12; 3], SceneState::default().affine);
        assert_eq!(bridge.fly_movement_basis(), FlyMovementBasis::ViewRelative);

        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.clear();
        assert_eq!(bridge.fly_movement_basis(), FlyMovementBasis::ViewRelative);

        bridge.set_fly_movement_basis(FlyMovementBasis::Level);
        assert_eq!(bridge.fly_movement_basis(), FlyMovementBasis::Level);
    }

    #[test]
    fn reset_view_only_restores_the_active_camera() {
        let bridge = SceneBridge::default();
        reset_bridge(&bridge, 8, 1, [24, 18, 12], SceneState::default().affine);
        bridge.orbit(egui::vec2(42.0, -17.0));
        bridge.pan(egui::vec2(55.0, 21.0), egui::vec2(800.0, 600.0));
        let orbit_at_fly_entry = bridge.state.lock().unwrap().orbit;
        bridge.set_view_mode(VolumeViewMode::Fly);
        let fly_home = bridge.state.lock().unwrap().fly.unwrap().home;
        bridge.fly_look(egui::vec2(90.0, -35.0));
        bridge.fly_move(FlyMovement::new(1.0, 1.0, -1.0, true), 0.08);
        bridge.adjust_fly_speed(400.0);

        bridge.reset_view();
        let reset_fly = bridge.state.lock().unwrap().fly.unwrap();
        assert_eq!(reset_fly.position, fly_home.position);
        assert_eq!(reset_fly.yaw, fly_home.yaw);
        assert_eq!(reset_fly.pitch, fly_home.pitch);
        assert_eq!(reset_fly.speed, fly_home.speed);
        assert_eq!(bridge.state.lock().unwrap().orbit, orbit_at_fly_entry);

        bridge.set_view_mode(VolumeViewMode::Orbit);
        let remembered_fly = bridge.state.lock().unwrap().fly;
        bridge.orbit(egui::vec2(12.0, 8.0));
        bridge.zoom(1.5);
        bridge.reset_view();
        let state = bridge.state.lock().unwrap();
        assert_eq!(state.orbit, OrbitCameraState::default());
        assert_eq!(state.fly, remembered_fly);
    }

    #[test]
    fn fly_look_and_speed_are_bounded_and_reject_non_finite_input() {
        let bridge = SceneBridge::default();
        reset_bridge(&bridge, 6, 1, [40, 30, 20], SceneState::default().affine);
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.fly_look(egui::vec2(f32::MAX, f32::MAX));
        {
            let fly = bridge.state.lock().unwrap().fly.unwrap();
            assert!((-std::f32::consts::PI..=std::f32::consts::PI).contains(&fly.yaw));
            assert_eq!(fly.pitch, -MAX_FLY_PITCH);
        }
        bridge.fly_look(egui::vec2(0.0, -f32::MAX));
        assert_eq!(
            bridge.state.lock().unwrap().fly.unwrap().pitch,
            MAX_FLY_PITCH
        );

        let before_invalid = bridge.state.lock().unwrap().fly.unwrap();
        bridge.fly_look(egui::vec2(f32::NAN, 1.0));
        bridge.adjust_fly_speed(f32::NAN);
        assert_eq!(bridge.state.lock().unwrap().fly, Some(before_invalid));

        let radius = {
            let state = bridge.state.lock().unwrap();
            scene_bounds(&state).radius
        };
        let (minimum, maximum) = fly_speed_bounds(radius);
        let initial = bridge.fly_speed().unwrap();
        assert!((initial - radius * FLY_INITIAL_SPEED_RADIUS_FACTOR).abs() < 1.0e-5);
        assert_eq!(bridge.fly_speed_range(), [minimum, maximum]);

        bridge.set_fly_speed(minimum * 0.5);
        assert_eq!(bridge.fly_speed(), Some(minimum));
        bridge.set_fly_speed(maximum * 2.0);
        assert_eq!(bridge.fly_speed(), Some(maximum));
        bridge.set_fly_speed(f32::NAN);
        assert_eq!(bridge.fly_speed(), Some(maximum));

        bridge.set_view_mode(VolumeViewMode::Orbit);
        bridge.set_fly_speed(initial);
        assert_eq!(bridge.fly_speed(), Some(initial));
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.adjust_fly_speed(f32::MAX);
        assert_eq!(bridge.fly_speed(), Some(maximum));
        bridge.adjust_fly_speed(-f32::MAX);
        assert_eq!(bridge.fly_speed(), Some(minimum));
    }

    #[test]
    fn voxel_seam_setting_is_shared_and_survives_navigation_resets() {
        let bridge = SceneBridge::default();
        assert!(bridge.voxel_seams());

        bridge.set_voxel_seams(false);
        reset_bridge(&bridge, 12, 1, [8; 3], SceneState::default().affine);
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.reset_view();
        bridge.set_view_mode(VolumeViewMode::Orbit);
        bridge.reset_view();
        assert!(!bridge.voxel_seams());

        bridge.set_voxel_seams(true);
        assert!(bridge.voxel_seams());
    }

    #[test]
    fn leak_light_is_explicit_and_resets_for_a_new_document() {
        let bridge = SceneBridge::default();
        assert!(!bridge.leak_light());

        bridge.set_leak_light(true);
        assert!(bridge.leak_light());
        assert_eq!(scene_uniforms(&bridge.state.lock().unwrap()).leak_light, 1);
        reset_bridge(&bridge, 0, 1, [8; 3], SceneState::default().affine);
        assert!(bridge.leak_light());

        reset_bridge(&bridge, 1, 1, [8; 3], SceneState::default().affine);
        assert!(!bridge.leak_light());
    }

    #[test]
    fn fly_movement_bases_use_expected_forward_strafe_world_z_boost_and_delta() {
        let bridge = SceneBridge::default();
        reset_bridge(&bridge, 9, 1, [20; 3], SceneState::default().affine);
        bridge.set_view_mode(VolumeViewMode::Fly);
        let start = Vec3::new(11.0, -7.0, 3.0);
        let yaw = 0.4;
        let pitch = 0.3;
        let speed = 10.0;
        let set_pose = || {
            let mut state = bridge.state.lock().unwrap();
            let fly = state.fly.as_mut().unwrap();
            fly.position = start;
            fly.yaw = yaw;
            fly.pitch = pitch;
            fly.speed = speed;
        };
        let position = || bridge.state.lock().unwrap().fly.unwrap().position;

        set_pose();
        bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), 0.05);
        let level_forward = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
        assert_vec3_close(position() - start, level_forward * 0.5, 1.0e-5);

        set_pose();
        bridge.fly_move(FlyMovement::new(-1.0, 0.0, 0.0, false), 0.05);
        assert_vec3_close(position() - start, level_forward * -0.5, 1.0e-5);

        bridge.set_fly_movement_basis(FlyMovementBasis::ViewRelative);
        set_pose();
        bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), 0.05);
        assert_vec3_close(position() - start, fly_forward(yaw, pitch) * 0.5, 1.0e-5);

        let right = Vec3::new(yaw.sin(), -yaw.cos(), 0.0);
        for basis in [FlyMovementBasis::Level, FlyMovementBasis::ViewRelative] {
            bridge.set_fly_movement_basis(basis);
            set_pose();
            bridge.fly_move(FlyMovement::new(0.0, 1.0, 0.0, false), 0.05);
            assert_vec3_close(position() - start, right * 0.5, 1.0e-5);

            set_pose();
            bridge.fly_move(FlyMovement::new(0.0, 0.0, 1.0, false), 0.05);
            assert_vec3_close(position() - start, Vec3::Z * 0.5, 1.0e-5);
        }

        bridge.set_fly_movement_basis(FlyMovementBasis::ViewRelative);
        set_pose();
        bridge.fly_move(FlyMovement::new(1.0, 1.0, 1.0, false), 0.05);
        assert!(((position() - start).length() - 0.5).abs() < 1.0e-5);

        set_pose();
        bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, true), 10.0);
        assert!(
            ((position() - start).length()
                - speed * FLY_BOOST_MULTIPLIER * MAX_FLY_FRAME_DELTA_SECONDS)
                .abs()
                < 1.0e-5
        );

        let before_invalid = position();
        bridge.fly_move(FlyMovement::new(f32::NAN, 0.0, 0.0, false), 0.05);
        bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), f32::NAN);
        assert_eq!(position(), before_invalid);
    }

    #[test]
    fn fly_collision_defaults_to_pass_through_and_blocks_high_speed_tunneling_when_enabled() {
        let dimensions = [6, 5, 5];
        let mut occupancy = vec![0; dimensions.into_iter().product::<u32>() as usize];
        occupancy[linear_index(dimensions, [2, 2, 2]).unwrap() as usize] = 1;
        let bridge = SceneBridge::default();
        let _ = reset_bridge_with_occupancy(
            &bridge,
            1,
            1,
            dimensions,
            SceneState::default().affine,
            &occupancy,
        );
        bridge.set_view_mode(VolumeViewMode::Fly);
        let set_pose = || {
            let mut state = bridge.state.lock().unwrap();
            let fly = state.fly.as_mut().unwrap();
            fly.position = Vec3::new(0.0, 2.0, 2.0);
            fly.yaw = 0.0;
            fly.pitch = 0.0;
            fly.speed = 50.0;
        };

        assert_eq!(bridge.fly_collision_mode(), FlyCollisionMode::PassThrough);
        set_pose();
        assert!(bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), 0.1));
        assert_vec3_close(
            bridge.state.lock().unwrap().fly.unwrap().position,
            Vec3::new(5.0, 2.0, 2.0),
            1.0e-5,
        );

        set_pose();
        bridge.set_fly_collision_mode(FlyCollisionMode::SolidVoxels);
        assert!(bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), 0.1));
        let collided = bridge.state.lock().unwrap().fly.unwrap().position;
        assert!((collided.x - (1.5 - FLY_COLLISION_SKIN_VOXELS as f32)).abs() < 1.0e-5);
        assert!((collided.y - 2.0).abs() < 1.0e-5);
        assert!((collided.z - 2.0).abs() < 1.0e-5);

        assert!(
            !bridge.fly_move(FlyMovement::new(1.0, 0.0, 0.0, false), 0.1),
            "repeated movement into the same wall should remain at the contact skin"
        );
        assert_vec3_close(
            bridge.state.lock().unwrap().fly.unwrap().position,
            collided,
            1.0e-6,
        );
    }

    #[test]
    fn fly_collision_slides_along_voxel_walls() {
        let dimensions = [6, 6, 5];
        let mut occupancy = vec![0; dimensions.into_iter().product::<u32>() as usize];
        for y in 0..dimensions[1] {
            occupancy[linear_index(dimensions, [2, y, 2]).unwrap() as usize] = 1;
        }
        let bridge = SceneBridge::default();
        let _ = reset_bridge_with_occupancy(
            &bridge,
            2,
            1,
            dimensions,
            SceneState::default().affine,
            &occupancy,
        );
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.set_fly_collision_mode(FlyCollisionMode::SolidVoxels);
        {
            let mut state = bridge.state.lock().unwrap();
            let fly = state.fly.as_mut().unwrap();
            fly.position = Vec3::new(0.0, 1.1, 2.0);
            fly.yaw = 0.0;
            fly.pitch = 0.0;
            fly.speed = 30.0 * std::f32::consts::SQRT_2;
        }

        assert!(bridge.fly_move(FlyMovement::new(1.0, -1.0, 0.0, false), 0.1));
        let position = bridge.state.lock().unwrap().fly.unwrap().position;
        assert!((position.x - (1.5 - FLY_COLLISION_SKIN_VOXELS as f32)).abs() < 1.0e-5);
        assert!((position.y - 4.1).abs() < 1.0e-5, "{position:?}");
        assert!((position.z - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn fly_collision_uses_the_complete_affine_for_sweeping_and_sliding() {
        let dimensions = [6, 6, 5];
        let affine = [
            [-2.0, 0.5, 0.1, 10.0],
            [0.0, 3.0, 0.2, -4.0],
            [0.0, 0.0, 1.5, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let geometry = VolumeGeometry::new(dimensions, affine, SpatialUnits::Millimeter).unwrap();
        let mut bytes = vec![0; dimensions.into_iter().product::<u32>() as usize];
        for y in 0..dimensions[1] {
            bytes[linear_index(dimensions, [2, y, 2]).unwrap() as usize] = 1;
        }
        let (occupancy, _) = SharedOccupancy::from_binary(dimensions, &bytes).unwrap();
        let occupancy = occupancy.try_read().unwrap();
        let start_index = [0.0, 1.0, 2.0];
        let displacement_index = [4.0, 2.0, 0.0];
        let start_world = transform_point(geometry.index_to_world, start_index);
        let displacement_world = transform_vector(geometry.index_to_world, displacement_index);
        let resolved = resolve_fly_collision(
            dimensions,
            geometry.index_to_world,
            geometry.world_to_index,
            Vec3::new(
                start_world[0] as f32,
                start_world[1] as f32,
                start_world[2] as f32,
            ),
            Vec3::new(
                displacement_world[0] as f32,
                displacement_world[1] as f32,
                displacement_world[2] as f32,
            ),
            &occupancy,
        );
        let resolved_index = transform_point(
            geometry.world_to_index,
            [
                f64::from(resolved.x),
                f64::from(resolved.y),
                f64::from(resolved.z),
            ],
        );
        assert!(
            (resolved_index[0] - (1.5 - FLY_COLLISION_SKIN_VOXELS)).abs() < 2.0e-5,
            "{resolved_index:?}"
        );
        assert!(
            (resolved_index[1] - 3.0).abs() < 2.0e-5,
            "{resolved_index:?}"
        );
        assert!(
            (resolved_index[2] - 2.0).abs() < 2.0e-5,
            "{resolved_index:?}"
        );
    }

    #[test]
    fn fly_collision_keeps_empty_and_outside_paths_free_and_bounds_extreme_traversal() {
        let identity = SceneState::default().affine;
        let dimensions = [8, 1, 1];
        let (occupancy, _) = SharedOccupancy::from_binary(dimensions, &[0; 8]).unwrap();
        let occupancy = occupancy.try_read().unwrap();
        assert_vec3_close(
            resolve_fly_collision(
                dimensions,
                identity,
                identity,
                Vec3::new(-3.0, 0.0, 0.0),
                Vec3::new(-10.0, 0.0, 0.0),
                &occupancy,
            ),
            Vec3::new(-13.0, 0.0, 0.0),
            1.0e-6,
        );
        assert_vec3_close(
            resolve_fly_collision(
                dimensions,
                identity,
                identity,
                Vec3::new(-3.0, 0.0, 0.0),
                Vec3::new(12.0, 0.0, 0.0),
                &occupancy,
            ),
            Vec3::new(9.0, 0.0, 0.0),
            1.0e-6,
        );
        drop(occupancy);

        let long_dimensions = [70_000, 1, 1];
        let long_bytes = vec![0; long_dimensions[0] as usize];
        let (long_occupancy, _) =
            SharedOccupancy::from_binary(long_dimensions, &long_bytes).unwrap();
        let long_occupancy = long_occupancy.try_read().unwrap();
        assert_eq!(
            resolve_fly_collision(
                long_dimensions,
                identity,
                identity,
                Vec3::ZERO,
                Vec3::new(69_999.0, 0.0, 0.0),
                &long_occupancy,
            ),
            Vec3::ZERO,
            "an unproven path must stop conservatively when its traversal budget is exhausted"
        );
    }

    #[test]
    fn fly_collision_recovers_from_live_edits_and_survives_document_resets() {
        let dimensions = [5; 3];
        let mut document = test_document(
            dimensions,
            vec![0; dimensions.into_iter().product::<u32>() as usize],
        );
        let geometry = document.geometry().clone();
        let bridge = SceneBridge::default();
        bridge.reset(
            1,
            document.revision(),
            dimensions,
            geometry.index_to_world,
            geometry.world_to_index,
            document.shared_occupancy(),
        );
        bridge.set_view_mode(VolumeViewMode::Fly);
        bridge.set_fly_collision_mode(FlyCollisionMode::SolidVoxels);
        let (home, yaw, pitch, speed) = {
            let mut state = bridge.state.lock().unwrap();
            let fly = state.fly.as_mut().unwrap();
            fly.position = Vec3::new(2.0, 2.0, 2.0);
            fly.yaw = 0.7;
            fly.pitch = -0.2;
            fly.speed = 3.5;
            (fly.home.position, fly.yaw, fly.pitch, fly.speed)
        };
        document.begin_stroke(1).unwrap();
        document
            .apply_stroke_indices([linear_index(dimensions, [2; 3]).unwrap()])
            .unwrap();

        assert!(bridge.reconcile_fly_collision());
        let recovered = bridge.state.lock().unwrap().fly.unwrap();
        assert_eq!(recovered.position, home);
        assert_eq!(recovered.yaw, yaw);
        assert_eq!(recovered.pitch, pitch);
        assert_eq!(recovered.speed, speed);

        bridge.set_fly_collision_mode(FlyCollisionMode::PassThrough);
        bridge.state.lock().unwrap().fly.as_mut().unwrap().position = Vec3::new(2.0, 2.0, 2.0);
        bridge.set_fly_collision_mode(FlyCollisionMode::SolidVoxels);
        let recovered_on_enable = bridge.state.lock().unwrap().fly.unwrap();
        assert_eq!(recovered_on_enable.position, home);
        assert_eq!(recovered_on_enable.yaw, yaw);
        assert_eq!(recovered_on_enable.pitch, pitch);
        assert_eq!(recovered_on_enable.speed, speed);

        reset_bridge(&bridge, 2, 1, dimensions, SceneState::default().affine);
        assert_eq!(bridge.fly_collision_mode(), FlyCollisionMode::SolidVoxels);
        bridge.clear();
        assert_eq!(bridge.fly_collision_mode(), FlyCollisionMode::SolidVoxels);
    }

    #[test]
    fn fly_projection_is_finite_inside_outside_and_behind_the_volume() {
        let mut state = SceneState {
            dimensions: [73, 41, 19],
            affine: [
                [0.0, -1.8, 0.2, 12.0],
                [2.1, 0.0, 0.35, -9.0],
                [0.1, 0.0, 3.2, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            aspect: 1.0,
            ..Default::default()
        };
        state.fly = Some(fly_camera_from_orbit(&state));
        state.view_mode = VolumeViewMode::Fly;
        let bounds = scene_bounds(&state);
        let positions_and_directions = [
            (bounds.center, Vec3::X),
            (
                bounds.center + Vec3::new(3.0, -2.0, 1.0).normalize() * bounds.radius * 80.0,
                Vec3::new(-3.0, 2.0, -1.0).normalize(),
            ),
            (
                bounds.center + Vec3::new(-2.0, 1.0, 0.5).normalize() * bounds.radius * 20.0,
                Vec3::new(-2.0, 1.0, 0.5).normalize(),
            ),
        ];

        for (position, direction) in positions_and_directions {
            let fly = state.fly.as_mut().unwrap();
            fly.position = position;
            fly.yaw = direction.y.atan2(direction.x);
            fly.pitch = direction.z.clamp(-1.0, 1.0).asin();
            let (view, projection, _, near_depth, inverse_span, eye, forward) =
                scene_camera(&state);
            let (near, far) = fly_clip_planes(view, bounds);
            assert!(near.is_finite() && far.is_finite() && 0.0 < near && near < far);
            assert!(view.is_finite());
            assert!(projection.is_finite());
            assert!((projection * view).is_finite());
            assert!(near_depth.is_finite());
            assert!(inverse_span.is_finite() && inverse_span > 0.0);
            assert!(eye.is_finite() && forward.is_finite());
            assert!((projection.y_axis.y - 1.0 / (35_f32.to_radians().tan())).abs() < 1.0e-5);
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
        let (view, _, _, _, _, _, _) = scene_camera(&state);
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
            orbit: OrbitCameraState {
                pitch: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        for (yaw, expected_span) in [
            (0.0, state.dimensions[0] as f32 * 0.315),
            (
                std::f32::consts::FRAC_PI_2,
                state.dimensions[1] as f32 * 0.315,
            ),
        ] {
            state.orbit.yaw = yaw;
            let (_, _, _, _, inverse_span, _, _) = scene_camera(&state);
            assert!((inverse_span.recip() - expected_span).abs() < 1.0e-3);
        }

        state.orbit.yaw = 0.65;
        state.orbit.pitch = -0.45;
        let (view, _, affine, near, inverse_span, _, _) = scene_camera(&state);
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
        reset_bridge(&bridge, 4, 1, [32; 3], affine);
        bridge.pan(egui::vec2(40.0, -30.0), egui::vec2(800.0, 600.0));
        bridge.orbit(egui::vec2(25.0, 10.0));
        bridge.zoom(1.5);
        let navigation = {
            let state = bridge.state.lock().unwrap();
            state.orbit
        };

        reset_bridge(&bridge, 4, 2, [32; 3], affine);
        {
            let state = bridge.state.lock().unwrap();
            assert_eq!(state.orbit, navigation);
        }

        reset_bridge(&bridge, 5, 1, [32; 3], affine);
        let state = bridge.state.lock().unwrap();
        let defaults = SceneState::default();
        assert_eq!(state.orbit, defaults.orbit);
    }

    #[test]
    fn reset_view_restores_navigation_defaults_without_changing_the_document() {
        let bridge = SceneBridge::default();
        let affine = SceneState::default().affine;
        reset_bridge(&bridge, 7, 12, [32; 3], affine);
        bridge.orbit(egui::vec2(40.0, 25.0));
        bridge.pan(egui::vec2(80.0, -30.0), egui::vec2(800.0, 600.0));
        bridge.zoom(2.0);

        bridge.reset_view();

        let state = bridge.state.lock().unwrap();
        let defaults = SceneState::default();
        assert_eq!(state.epoch, 7);
        assert_eq!(state.revision, 12);
        assert_eq!(state.dimensions, [32; 3]);
        assert_eq!(state.orbit, defaults.orbit);
    }

    fn project_to_viewport(
        state: &SceneState,
        world_position: Vec3,
        viewport: egui::Vec2,
    ) -> egui::Vec2 {
        let (view, projection, _, _, _, _, _) = scene_camera(state);
        let ndc = (projection * view).project_point3(world_position);
        egui::vec2(
            (ndc.x + 1.0) * 0.5 * viewport.x,
            (1.0 - ndc.y) * 0.5 * viewport.y,
        )
    }

    #[test]
    fn headless_software_vulkan_keeps_the_nearer_thin_sheet_at_long_fly_range() {
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = GpuVolumeRenderer::new(&device, format, false);
        let coordinate = ChunkCoord::new(0, 0, 0);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 1,
                dimensions: [1, 1, 4_096],
                affine: SceneState::default().affine,
            },
        );
        let quads = [
            CompactQuad::new_with_shading_normal(
                [0, 0, 1],
                1,
                1,
                crate::mesh::FaceAxis::Z,
                true,
                [0; 4],
                Vec3::Z.to_array(),
            )
            .unwrap(),
            CompactQuad::new_with_shading_normal(
                [0, 0, 0],
                1,
                1,
                crate::mesh::FaceAxis::Z,
                true,
                [0; 4],
                Vec3::NEG_Z.to_array(),
            )
            .unwrap(),
        ];
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 1,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: coordinate,
                },
                // Draw the nearer bright face first and the farther dark face
                // second. Conventional 24-bit depth collapses these one-unit-
                // separated surfaces at this range and lets the later face win.
                quads: quads.to_vec(),
                priority: MeshUpdatePriority::Background,
            },
        );
        let mut leak_renderer = GpuVolumeRenderer::new(&device, format, false);
        leak_renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 1,
                dimensions: [1, 1, 4_096],
                affine: SceneState::default().affine,
            },
        );
        leak_renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 1,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: coordinate,
                },
                quads: quads.to_vec(),
                priority: MeshUpdatePriority::Background,
            },
        );
        let leak_distances = [
            PackedLeakDistance::new([0; 4]).unwrap(),
            PackedLeakDistance::new([20; 4]).unwrap(),
        ];
        leak_renderer
            .chunks
            .get_mut(&coordinate)
            .unwrap()
            .leak_distance = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("distant thin-sheet leak distances"),
                contents: bytemuck::cast_slice(&leak_distances),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );

        let index_to_world = Mat4::from_scale(Vec3::new(2_048.0, 2_048.0, 1.0))
            * Mat4::from_translation(Vec3::new(0.0, 0.0, -4_096.0));
        let projection =
            Mat4::perspective_infinite_reverse_rh(FLY_FIELD_OF_VIEW_RADIANS, 1.0, 0.01);
        let uniforms = Uniforms {
            view_projection: projection.to_cols_array_2d(),
            index_to_world: index_to_world.to_cols_array_2d(),
            index_normal_to_view: normal_matrix(index_to_world).to_cols_array_2d(),
            leak_half_distance: 1.0,
            leak_distance_scale: 0.1,
            voxel_seams: 0,
            ..Uniforms::default()
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let leak_uniforms = Uniforms {
            leak_light: 1,
            ..uniforms
        };
        queue.write_buffer(
            &leak_renderer.uniform_buffer,
            0,
            bytemuck::bytes_of(&leak_uniforms),
        );

        const PANE_SIZE: u32 = 64;
        const WIDTH: u32 = PANE_SIZE * 2;
        const HEIGHT: u32 = PANE_SIZE;
        let multisampled_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("distant thin-sheet multisampled color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolved_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("distant thin-sheet resolved color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("distant thin-sheet depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_color.create_view(&Default::default());
        let resolved_view = resolved_color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("distant thin-sheet encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("distant thin-sheet pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // Match egui's shared render pass. The renderer resets
                        // each pane to the reverse-Z far value before drawing.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for pane in 0..2 {
                let x = pane * PANE_SIZE;
                pass.set_viewport(x as f32, 0.0, PANE_SIZE as f32, PANE_SIZE as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, 0, PANE_SIZE, PANE_SIZE);
                pass.set_pipeline(&renderer.depth_reset_pipeline);
                pass.draw(0..3, 0..1);
                if pane == 0 {
                    let chunk = renderer.chunks.get(&coordinate).unwrap();
                    pass.set_bind_group(0, &renderer.bind_group, &[]);
                    pass.set_pipeline(&renderer.pipeline);
                    pass.set_vertex_buffer(0, chunk.buffer.slice(..));
                    pass.draw(0..6, 0..chunk.instances);
                } else {
                    let chunk = leak_renderer.chunks.get(&coordinate).unwrap();
                    pass.set_bind_group(0, &leak_renderer.bind_group, &[]);
                    pass.set_pipeline(&leak_renderer.leak_pipeline);
                    pass.set_vertex_buffer(1, chunk.leak_distance.as_ref().unwrap().slice(..));
                    pass.set_vertex_buffer(0, chunk.buffer.slice(..));
                    pass.draw(0..6, 0..chunk.instances);
                }
            }
        }

        let bytes_per_row = WIDTH * size_of::<u32>() as u32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distant thin-sheet readback"),
            size: u64::from(bytes_per_row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved_color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let slice = output.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan distant thin-sheet render must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("distant thin-sheet readback buffer must map");
        let pixels = slice.get_mapped_range();
        let sample = |x: u32| {
            let offset = ((HEIGHT / 2 * WIDTH + x) * 4) as usize;
            [pixels[offset], pixels[offset + 1], pixels[offset + 2]]
        };
        let normal = sample(PANE_SIZE / 2);
        let leak = sample(PANE_SIZE + PANE_SIZE / 2);
        assert!(
            normal.into_iter().map(u16::from).sum::<u16>() > 230,
            "the farther dark face replaced the nearer surface: {normal:?}"
        );
        assert!(
            leak[0] > 180 && leak[1] > 180 && leak[2] > 50,
            "the farther leak-light face replaced the nearer zero-distance surface: {leak:?}"
        );
    }

    #[test]
    fn headless_software_vulkan_keeps_close_coplanar_highlights_rectangular() {
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = GpuVolumeRenderer::new(&device, format, false);
        // This is the sform from the craniofacial NIfTI that exposed the bug. Keeping
        // the large index coordinates and anisotropic affine is important: the old,
        // weak depth bias looked stable on a small identity-space test plane.
        let affine = [
            [-0.314_702_45, 0.013_688_04, 0.0, 106.237_23],
            [-0.013_688_04, -0.314_702_45, 0.0, 289.382_54],
            [0.0, 0.0, 0.8, -678.939],
            [0.0, 0.0, 0.0, 1.0],
        ];
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 1,
                dimensions: [617, 560, 231],
                affine,
            },
        );
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 1,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: ChunkCoord::new(0, 0, 0),
                },
                quads: vec![
                    CompactQuad::new(
                        [288, 280, 116],
                        32,
                        32,
                        crate::mesh::FaceAxis::Z,
                        true,
                        [3, 0, 3, 0],
                    )
                    .expect("the test surface fits one greedy quad"),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 1,
                revision: 1,
                faces: (0..32)
                    .flat_map(|y| {
                        (0..32).map(move |x| {
                            HighlightFace::new(
                                [288 + x, 280 + y, 115],
                                2,
                                true,
                                VoxelHighlightKind::Erase,
                            )
                        })
                    })
                    .collect(),
            },
        );

        let affine = affine_mat4(affine);
        let target = affine.transform_point3(Vec3::new(304.0, 296.0, 115.5));
        let tangent = affine.transform_vector3(Vec3::Y).normalize();
        let normal = affine.transform_vector3(Vec3::Z).normalize();
        let eye = target - tangent * 5.0 + normal * 2.0;
        let view = Mat4::look_at_rh(eye, target, normal);
        let projection =
            Mat4::perspective_infinite_reverse_rh(FLY_FIELD_OF_VIEW_RADIANS, 1.0, 0.01);
        let uniforms = Uniforms {
            view_projection: (projection * view).to_cols_array_2d(),
            index_to_world: affine.to_cols_array_2d(),
            world_to_view: view.to_cols_array_2d(),
            index_normal_to_view: normal_matrix(view * affine).to_cols_array_2d(),
            ..Uniforms::default()
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        const WIDTH: u32 = 512;
        const HEIGHT: u32 = 512;
        let texture = |label, sample_count, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let multisampled_color = texture(
            "close highlight multisampled color",
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let resolved_color = texture(
            "close highlight resolved color",
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("close highlight depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_color.create_view(&Default::default());
        let resolved_view = resolved_color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("close highlight encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("close highlight pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&renderer.depth_reset_pipeline);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&renderer.pipeline);
            pass.set_bind_group(0, &renderer.bind_group, &[]);
            for chunk in renderer.chunks.values() {
                pass.set_vertex_buffer(0, chunk.buffer.slice(..));
                pass.draw(0..6, 0..chunk.instances);
            }
            let highlight = renderer
                .highlight
                .as_ref()
                .expect("the test highlight was uploaded");
            pass.set_pipeline(&renderer.highlight_pipeline);
            pass.set_vertex_buffer(0, highlight.buffer.slice(..));
            pass.draw(0..6, 0..highlight.faces);
        }

        let bytes_per_row = WIDTH * size_of::<u32>() as u32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("close highlight readback"),
            size: u64::from(bytes_per_row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved_color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let slice = output.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan render must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("readback buffer must map");
        let pixels = slice.get_mapped_range();
        let mut surface_mask = vec![false; (WIDTH * HEIGHT) as usize];
        let mut highlight_mask = vec![false; (WIDTH * HEIGHT) as usize];
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            let [red, green, blue, _] = <[u8; 4]>::try_from(pixel).unwrap();
            surface_mask[index] = red.max(green).max(blue) >= 16;
            highlight_mask[index] = red > green.saturating_add(8) && red > blue.saturating_add(8);
        }
        let mut surface_pixels = 0_usize;
        let mut highlighted_pixels = 0_usize;
        // Ignore the outer MSAA resolve fringe. Depth-fighting holes remain surface
        // pixels and are still counted; only pixels near the black background drop out.
        const EDGE_MARGIN: u32 = 3;
        for y in EDGE_MARGIN..HEIGHT - EDGE_MARGIN {
            for x in EDGE_MARGIN..WIDTH - EDGE_MARGIN {
                let index = (y * WIDTH + x) as usize;
                let inside_surface = [
                    index,
                    (y * WIDTH + x - EDGE_MARGIN) as usize,
                    (y * WIDTH + x + EDGE_MARGIN) as usize,
                    ((y - EDGE_MARGIN) * WIDTH + x) as usize,
                    ((y + EDGE_MARGIN) * WIDTH + x) as usize,
                ]
                .into_iter()
                .all(|sample| surface_mask[sample]);
                if inside_surface {
                    surface_pixels += 1;
                    highlighted_pixels += usize::from(highlight_mask[index]);
                }
            }
        }
        assert!(
            surface_pixels > 20_000,
            "the close surface must cover the frame"
        );
        assert!(
            highlighted_pixels * 1_000 >= surface_pixels * 985,
            "coplanar highlight coverage was only {highlighted_pixels}/{surface_pixels} pixels"
        );
    }

    #[test]
    fn headless_software_vulkan_shades_equal_faces_from_occupancy_normals() {
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = GpuVolumeRenderer::new(&device, format, false);
        let coordinate = ChunkCoord::new(0, 0, 0);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 1,
                dimensions: [2, 1, 1],
                affine: SceneState::default().affine,
            },
        );
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 1,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: coordinate,
                },
                quads: vec![
                    CompactQuad::new_with_shading_normal(
                        [0, 0, 0],
                        1,
                        1,
                        crate::mesh::FaceAxis::Z,
                        true,
                        [0; 4],
                        [0.0, 0.707_106_77, 0.707_106_77],
                    )
                    .unwrap(),
                    CompactQuad::new_with_shading_normal(
                        [1, 0, 0],
                        1,
                        1,
                        crate::mesh::FaceAxis::Z,
                        true,
                        [0; 4],
                        [0.0, -0.707_106_77, 0.707_106_77],
                    )
                    .unwrap(),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );

        let index_to_world = Mat4::from_scale_rotation_translation(
            Vec3::new(0.9, 1.6, 1.0),
            glam::Quat::IDENTITY,
            Vec3::new(-0.45, 0.0, 0.5),
        );
        let world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0));
        let uniforms = Uniforms {
            index_to_world: index_to_world.to_cols_array_2d(),
            world_to_view: world_to_view.to_cols_array_2d(),
            index_normal_to_view: normal_matrix(world_to_view * index_to_world).to_cols_array_2d(),
            ..Uniforms::default()
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let texture = |label, sample_count, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let multisampled_color = texture(
            "occupancy-normal multisampled color",
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let resolved_color = texture(
            "occupancy-normal resolved color",
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("occupancy-normal depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_color.create_view(&Default::default());
        let resolved_view = resolved_color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("occupancy-normal encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("occupancy-normal pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&renderer.depth_reset_pipeline);
            pass.draw(0..3, 0..1);
            let chunk = renderer.chunks.get(&coordinate).unwrap();
            pass.set_pipeline(&renderer.pipeline);
            pass.set_bind_group(0, &renderer.bind_group, &[]);
            pass.set_vertex_buffer(0, chunk.buffer.slice(..));
            pass.draw(0..6, 0..chunk.instances);
        }

        let bytes_per_row = WIDTH * size_of::<u32>() as u32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occupancy-normal readback"),
            size: u64::from(bytes_per_row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved_color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let slice = output.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan normal render must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("occupancy-normal readback buffer must map");
        let pixels = slice.get_mapped_range();
        let luminance = |x: u32| {
            let pixel = ((HEIGHT / 2 * WIDTH + x) * 4) as usize;
            u16::from(pixels[pixel]) + u16::from(pixels[pixel + 1]) + u16::from(pixels[pixel + 2])
        };
        let upper_bent = luminance(WIDTH / 4);
        let lower_bent = luminance(WIDTH * 3 / 4);
        assert!(
            upper_bent > lower_bent + 80,
            "occupancy normals did not create enough relief contrast: {upper_bent} vs {lower_bent}"
        );
    }

    #[test]
    fn headless_software_vulkan_keeps_grazing_front_faces_exterior_shaded() {
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let coordinate = ChunkCoord::new(0, 0, 0);
        let quad = CompactQuad::new_with_shading_normal(
            [0, 0, 0],
            1,
            1,
            crate::mesh::FaceAxis::Z,
            true,
            [0; 4],
            [0.0, 0.707_106_77, 0.707_106_77],
        )
        .unwrap();
        let mut renderer = GpuVolumeRenderer::new(&device, format, false);
        let mut leak_renderer = GpuVolumeRenderer::new(&device, format, false);
        for target in [&mut renderer, &mut leak_renderer] {
            target.apply_update(
                &device,
                &queue,
                SceneUpdate::Reset {
                    epoch: 1,
                    revision: 1,
                    dimensions: [1; 3],
                    affine: SceneState::default().affine,
                },
            );
            target.apply_update(
                &device,
                &queue,
                SceneUpdate::Chunk {
                    tag: MeshJobTag {
                        document_epoch: 1,
                        revision: 1,
                        chunk_generation: 1,
                        edit_generation: 0,
                        chunk: coordinate,
                    },
                    quads: vec![quad],
                    priority: MeshUpdatePriority::Background,
                },
            );
        }
        let leak_distance = [PackedLeakDistance::new([0; 4]).unwrap()];
        let leak_distance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grazing front-face leak distance"),
            contents: bytemuck::cast_slice(&leak_distance),
            usage: wgpu::BufferUsages::VERTEX,
        });
        leak_renderer
            .chunks
            .get_mut(&coordinate)
            .unwrap()
            .leak_distance = Some(leak_distance_buffer);

        let index_to_world = Mat4::from_scale_rotation_translation(
            Vec3::splat(1.6),
            glam::Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.8),
        );
        // At this Fly angle, the true +Z face still points toward the camera
        // while its occupancy-smoothed normal has crossed slightly behind it.
        let world_to_view = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0))
            * Mat4::from_rotation_x((-63.0_f32).to_radians());
        let index_normal_to_view = normal_matrix(world_to_view * index_to_world);
        let view_geometric_normal = index_normal_to_view.transform_vector3(Vec3::Z).normalize();
        let decoded_shading_normal = Vec3::from_array(quad.shading_normal());
        let view_shading_normal = index_normal_to_view
            .transform_vector3((0.35 * Vec3::Z + 0.65 * decoded_shading_normal).normalize())
            .normalize();
        assert!(
            view_geometric_normal.z > 0.4,
            "the test face must remain geometrically front-facing"
        );
        assert!(
            view_shading_normal.z < -0.02,
            "the smooth normal must point slightly away from the camera"
        );

        let projection =
            Mat4::perspective_infinite_reverse_rh(FLY_FIELD_OF_VIEW_RADIANS, 1.0, 0.01);
        let uniforms = Uniforms {
            view_projection: (projection * world_to_view).to_cols_array_2d(),
            index_to_world: index_to_world.to_cols_array_2d(),
            world_to_view: world_to_view.to_cols_array_2d(),
            index_normal_to_view: index_normal_to_view.to_cols_array_2d(),
            inverse_scene_depth_span: 0.0,
            ..Uniforms::default()
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let leak_uniforms = Uniforms {
            leak_light: 1,
            ..uniforms
        };
        queue.write_buffer(
            &leak_renderer.uniform_buffer,
            0,
            bytemuck::bytes_of(&leak_uniforms),
        );

        const PANE_SIZE: u32 = 64;
        const WIDTH: u32 = PANE_SIZE * 2;
        const HEIGHT: u32 = PANE_SIZE;
        let texture = |label, sample_count, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let multisampled_color = texture(
            "grazing front-face multisampled color",
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let resolved_color = texture(
            "grazing front-face resolved color",
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grazing front-face depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_color.create_view(&Default::default());
        let resolved_view = resolved_color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grazing front-face encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("grazing front-face pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for pane in 0..2 {
                let x = pane * PANE_SIZE;
                pass.set_viewport(x as f32, 0.0, PANE_SIZE as f32, PANE_SIZE as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, 0, PANE_SIZE, PANE_SIZE);
                pass.set_pipeline(&renderer.depth_reset_pipeline);
                pass.draw(0..3, 0..1);
                if pane == 0 {
                    let chunk = renderer.chunks.get(&coordinate).unwrap();
                    pass.set_bind_group(0, &renderer.bind_group, &[]);
                    pass.set_pipeline(&renderer.pipeline);
                    pass.set_vertex_buffer(0, chunk.buffer.slice(..));
                    pass.draw(0..6, 0..chunk.instances);
                } else {
                    let chunk = leak_renderer.chunks.get(&coordinate).unwrap();
                    pass.set_bind_group(0, &leak_renderer.bind_group, &[]);
                    pass.set_pipeline(&leak_renderer.leak_pipeline);
                    pass.set_vertex_buffer(1, chunk.leak_distance.as_ref().unwrap().slice(..));
                    pass.set_vertex_buffer(0, chunk.buffer.slice(..));
                    pass.draw(0..6, 0..chunk.instances);
                }
            }
        }

        let bytes_per_row = WIDTH * size_of::<u32>() as u32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grazing front-face readback"),
            size: u64::from(bytes_per_row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved_color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let slice = output.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan grazing front-face render must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("grazing front-face readback buffer must map");
        let pixels = slice.get_mapped_range();
        let sample = |x: u32| {
            let offset = ((HEIGHT / 2 * WIDTH + x) * 4) as usize;
            [pixels[offset], pixels[offset + 1], pixels[offset + 2]]
        };
        let normal = sample(PANE_SIZE / 2);
        let leak = sample(PANE_SIZE + PANE_SIZE / 2);
        assert!(
            normal.into_iter().map(u16::from).sum::<u16>() > 350,
            "the grazing front face received interior material shading: {normal:?}"
        );
        assert!(
            leak[0] > 215 && leak[1] > 220 && leak[2] > 75,
            "the grazing front face received interior leak-light shading: {leak:?}"
        );
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let mut renderer =
            GpuVolumeRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, false);
        let affine = SceneState::default().affine;
        let dimensions = [96, 32, 32];
        let visibility_uniforms = scene_uniforms(&SceneState {
            dimensions,
            affine,
            ..SceneState::default()
        });
        let chunk = ChunkCoord::new(0, 0, 0);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 4,
                revision: 8,
                dimensions,
                affine,
            },
        );
        renderer.apply_update(
            &device,
            &queue,
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
                    CompactQuad::new([1, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .expect("unit face is representable"),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        assert!(renderer.chunks.contains_key(&chunk));
        assert!(renderer.chunks.get(&chunk).unwrap().leak_distance.is_none());
        let initial_chunk_buffer = renderer
            .chunks
            .get(&chunk)
            .expect("the test chunk was uploaded")
            .buffer
            .clone();
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 1);
        assert!(renderer.chunks.get(&chunk).unwrap().visible);
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 0);

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 4,
                    edit_generation: 1,
                    chunk,
                },
                quads: vec![
                    CompactQuad::new([2, 0, 0], 1, 1, crate::mesh::FaceAxis::Y, true, [0; 4])
                        .expect("unit face is representable"),
                ],
                priority: MeshUpdatePriority::Live,
            },
        );
        let reused_chunk = renderer
            .chunks
            .get(&chunk)
            .expect("the smaller replacement remains resident");
        assert_eq!(reused_chunk.buffer, initial_chunk_buffer);
        assert_eq!(reused_chunk.instances, 1);
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 0);

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 5,
                    edit_generation: 2,
                    chunk,
                },
                quads: (0..3)
                    .map(|z| {
                        CompactQuad::new([0, 0, z], 1, 1, crate::mesh::FaceAxis::Z, true, [0; 4])
                            .expect("unit face is representable")
                    })
                    .collect(),
                priority: MeshUpdatePriority::Live,
            },
        );
        let grown_chunk = renderer
            .chunks
            .get(&chunk)
            .expect("the larger replacement remains resident");
        assert_ne!(grown_chunk.buffer, initial_chunk_buffer);
        assert_eq!(grown_chunk.instances, 3);
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 0);

        let representative_quad =
            CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::Z, true, [0; 4])
                .expect("unit face is representable");
        let oversized_quad_count =
            CHUNK_BUFFER_REUSE_BASE_BYTES as usize / size_of::<CompactQuad>() + 1;
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 6,
                    edit_generation: 3,
                    chunk,
                },
                quads: vec![representative_quad; oversized_quad_count],
                priority: MeshUpdatePriority::Live,
            },
        );
        let oversized_chunk_buffer = renderer
            .chunks
            .get(&chunk)
            .expect("the oversized replacement remains resident")
            .buffer
            .clone();
        assert!(oversized_chunk_buffer.size() > CHUNK_BUFFER_REUSE_BASE_BYTES);

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 7,
                    edit_generation: 4,
                    chunk,
                },
                quads: vec![representative_quad],
                priority: MeshUpdatePriority::Live,
            },
        );
        let shrunk_chunk = renderer
            .chunks
            .get(&chunk)
            .expect("the tiny replacement remains resident");
        assert_ne!(shrunk_chunk.buffer, oversized_chunk_buffer);
        assert_eq!(shrunk_chunk.instances, 1);

        let far_chunk = ChunkCoord::new(2, 0, 0);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 8,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: far_chunk,
                },
                quads: vec![
                    CompactQuad::new([64, 0, 0], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .expect("unit face is representable"),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 1);
        assert!(renderer.chunks.get(&far_chunk).unwrap().visible);

        let identity_clip_uniforms = Uniforms::default();
        assert_eq!(
            renderer.refresh_chunk_visibility(&identity_clip_uniforms),
            2
        );
        assert!(renderer.chunks.get(&chunk).unwrap().visible);
        assert!(!renderer.chunks.get(&far_chunk).unwrap().visible);
        assert_eq!(
            renderer.refresh_chunk_visibility(&identity_clip_uniforms),
            0
        );
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 2);

        renderer.apply_update(
            &device,
            &queue,
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
        let initial_highlight_buffer = renderer
            .highlight
            .as_ref()
            .expect("the test highlight was uploaded")
            .buffer
            .clone();
        assert!(
            renderer.chunks.contains_key(&chunk),
            "transient highlight must not alter resident chunk meshes"
        );

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 8,
                faces: (0..5)
                    .map(|x| HighlightFace::new([x, 1, 1], 0, true, VoxelHighlightKind::Paint))
                    .collect(),
            },
        );
        let reused_highlight = renderer
            .highlight
            .as_ref()
            .expect("the smaller highlight remains resident");
        assert_eq!(reused_highlight.buffer, initial_highlight_buffer);
        assert_eq!(reused_highlight.faces, 5);

        renderer.apply_update(&device, &queue, SceneUpdate::Invalidate { revision: 9 });
        assert_eq!(renderer.revision, 9);
        assert_eq!(renderer.refresh_chunk_visibility(&visibility_uniforms), 0);
        let cleared_highlight = renderer
            .highlight
            .as_ref()
            .expect("a small cleared highlight allocation is retained");
        assert_eq!(cleared_highlight.buffer, initial_highlight_buffer);
        assert_eq!(cleared_highlight.faces, 0);
        assert!(
            renderer.chunks.contains_key(&chunk),
            "the previous mesh must remain resident until its replacement is ready"
        );

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 9,
                    chunk_generation: 8,
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
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 4,
                    revision: 9,
                    chunk_generation: 5,
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

        let current_faces = (0..3)
            .map(|y| HighlightFace::new([1, y, 1], 1, true, VoxelHighlightKind::BridgePath))
            .collect::<Vec<_>>();
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 9,
                faces: current_faces.clone(),
            },
        );
        let post_invalidation_highlight = renderer
            .highlight
            .as_ref()
            .expect("the current highlight was uploaded");
        assert_eq!(post_invalidation_highlight.buffer, initial_highlight_buffer);
        assert_eq!(post_invalidation_highlight.faces, 3);

        renderer.apply_update(&device, &queue, SceneUpdate::ClearHighlight);
        let explicitly_cleared_highlight = renderer
            .highlight
            .as_ref()
            .expect("a small explicitly cleared allocation is retained");
        assert_eq!(
            explicitly_cleared_highlight.buffer,
            initial_highlight_buffer
        );
        assert_eq!(explicitly_cleared_highlight.faces, 0);

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 9,
                faces: current_faces,
            },
        );
        assert_eq!(
            renderer
                .highlight
                .as_ref()
                .expect("the retained allocation was reused after clear")
                .buffer,
            initial_highlight_buffer
        );

        let oversized_face_count =
            MAX_RETAINED_HIGHLIGHT_BUFFER_BYTES as usize / size_of::<HighlightFace>() + 1;
        assert!(oversized_face_count <= MAX_VOXEL_HIGHLIGHT_FACES);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 9,
                faces: vec![
                    HighlightFace::new([1, 1, 1], 2, true, VoxelHighlightKind::Erase,);
                    oversized_face_count
                ],
            },
        );
        let oversized_highlight_buffer = renderer
            .highlight
            .as_ref()
            .expect("the oversized visible highlight was uploaded")
            .buffer
            .clone();
        assert_ne!(oversized_highlight_buffer, initial_highlight_buffer);
        assert!(oversized_highlight_buffer.size() > MAX_RETAINED_HIGHLIGHT_BUFFER_BYTES);

        renderer.apply_update(&device, &queue, SceneUpdate::ClearHighlight);
        assert!(
            renderer.highlight.is_none(),
            "an oversized cleared highlight allocation must be released"
        );

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Highlight {
                epoch: 4,
                revision: 9,
                faces: vec![HighlightFace::new(
                    [1, 1, 1],
                    2,
                    true,
                    VoxelHighlightKind::BridgeAnchor,
                )],
            },
        );
        assert!(renderer.highlight.is_some());
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 5,
                revision: 0,
                dimensions: [16; 3],
                affine,
            },
        );
        assert!(
            renderer.highlight.is_none(),
            "a document reset must release retained highlight storage"
        );
        assert!(renderer.chunk_visibility_view.is_none());

        let _ = queue.submit(std::iter::empty());
    }

    #[test]
    fn headless_software_vulkan_prepares_packed_leak_distances_in_compute() {
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
        assert!(leak_compute_supported(&adapter));
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("software Vulkan device must initialize");
        let mut renderer =
            GpuVolumeRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, true);
        let dimensions = [3; 3];
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 2,
                dimensions,
                affine: SceneState::default().affine,
            },
        );
        let coordinate = ChunkCoord::new(0, 0, 0);
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 2,
                    chunk_generation: 1,
                    edit_generation: 0,
                    chunk: coordinate,
                },
                quads: vec![
                    CompactQuad::new([0, 0, 0], 1, 1, crate::mesh::FaceAxis::X, false, [0; 4])
                        .unwrap(),
                    CompactQuad::new([1, 1, 1], 1, 1, crate::mesh::FaceAxis::X, true, [0; 4])
                        .unwrap(),
                ],
                priority: MeshUpdatePriority::Background,
            },
        );
        let mut occupancy = vec![1_u8; 27];
        occupancy[1 + 3 * (1 + 3)] = 0;
        let field = Arc::new(
            crate::volume::compute_leak_distance(
                &occupancy,
                dimensions,
                &[1.0; 3],
                &crate::cancel::CancellationToken::default(),
            )
            .unwrap(),
        );
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::LeakDistanceField {
                epoch: 1,
                revision: 2,
                generation: 3,
                field,
            },
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("leak-distance compute test encoder"),
        });
        assert!(!renderer.prepare_leak_distances(&device, &mut encoder, 1));
        assert!(
            !renderer.leak_distance_available(),
            "a pending field must not become partially visible"
        );
        renderer.activate_pending_leak_distances();
        assert!(renderer.leak_distance_ready());
        assert!(renderer.leak_distance_available());
        let prepared = renderer
            .chunks
            .get(&coordinate)
            .and_then(|chunk| chunk.leak_distance.as_ref())
            .expect("compute must prepare the diagnostic buffer");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("leak-distance compute test readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(prepared, 0, &readback, 0, 16);
        let submission = queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan compute must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("readback buffer must map");
        let bytes = slice.get_mapped_range();
        let words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(words, vec![0, 0, 0x8000_8000, 0x8000_8000]);
        drop(bytes);
        readback.unmap();

        renderer.apply_update(&device, &queue, SceneUpdate::Invalidate { revision: 3 });
        assert!(
            renderer.leak_distance_available(),
            "revision invalidation must retain the previous completed field"
        );
        assert!(
            !renderer.leak_distance_ready(),
            "the retained field must remain distinguishable from the current revision"
        );

        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Reset {
                epoch: 1,
                revision: 3,
                dimensions: [10, 3, 3],
                affine: SceneState::default().affine,
            },
        );
        renderer.apply_update(
            &device,
            &queue,
            SceneUpdate::Chunk {
                tag: MeshJobTag {
                    document_epoch: 1,
                    revision: 3,
                    chunk_generation: 2,
                    edit_generation: 0,
                    chunk: coordinate,
                },
                quads: (1..=8)
                    .map(|x| {
                        CompactQuad::new([x, 1, 1], 1, 1, crate::mesh::FaceAxis::Z, true, [0; 4])
                            .unwrap()
                    })
                    .collect(),
                priority: MeshUpdatePriority::Background,
            },
        );
        // At a 0.1 physical-distance scale and a half-distance of 1.0, every
        // successive step must produce a distinct continuously varying color.
        // Steps 10 and 20 land exactly on the crimson half-brightness and purple
        // quarter-brightness heatmap stops.
        let diagnostic_distances = [
            PackedLeakDistance::new([i16::MAX as u16; 4]).unwrap(),
            PackedLeakDistance::new([0; 4]).unwrap(),
            PackedLeakDistance::new([3; 4]).unwrap(),
            PackedLeakDistance::new([1; 4]).unwrap(),
            PackedLeakDistance::new([2; 4]).unwrap(),
            PackedLeakDistance::new([10; 4]).unwrap(),
            PackedLeakDistance::new([20; 4]).unwrap(),
            PackedLeakDistance::new([u16::MAX; 4]).unwrap(),
        ];
        let diagnostic_distance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("continuous leak-distance test buffer"),
                contents: bytemuck::cast_slice(&diagnostic_distances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        renderer
            .chunks
            .get_mut(&coordinate)
            .expect("the test faces must remain resident")
            .leak_distance = Some(diagnostic_distance_buffer);

        let index_to_world = Mat4::from_scale_rotation_translation(
            Vec3::new(0.22, 1.5, 1.0),
            glam::Quat::IDENTITY,
            Vec3::new(-0.99, -1.5, 0.0),
        );
        let uniforms = Uniforms {
            index_to_world: index_to_world.to_cols_array_2d(),
            index_normal_to_view: normal_matrix(index_to_world).to_cols_array_2d(),
            leak_half_distance: 1.0,
            leak_distance_scale: 0.1,
            leak_light: 1,
            ..Uniforms::default()
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        const WIDTH: u32 = 128;
        const HEIGHT: u32 = 64;
        let multisampled_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("far connected leak-light multisampled color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolved_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("far connected leak-light resolved color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("far connected leak-light depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let multisampled_view = multisampled_color.create_view(&Default::default());
        let resolved_view = resolved_color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("far connected leak-light encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("far connected leak-light pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&renderer.depth_reset_pipeline);
            pass.draw(0..3, 0..1);
            let chunk = renderer.chunks.get(&coordinate).unwrap();
            pass.set_pipeline(&renderer.leak_pipeline);
            pass.set_bind_group(0, &renderer.bind_group, &[]);
            pass.set_vertex_buffer(0, chunk.buffer.slice(..));
            pass.set_vertex_buffer(1, chunk.leak_distance.as_ref().unwrap().slice(..));
            pass.draw(0..6, 0..chunk.instances);
        }

        let bytes_per_row = WIDTH * size_of::<u32>() as u32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far connected leak-light readback"),
            size: u64::from(bytes_per_row * HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            resolved_color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let slice = output.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .expect("software Vulkan leak-light render must complete");
        receiver
            .recv()
            .expect("readback callback must run")
            .expect("leak-light readback buffer must map");
        let pixels = slice.get_mapped_range();
        let sample_rgb = |x: u32| {
            let pixel = ((HEIGHT / 2 * WIDTH + x) * 4) as usize;
            [pixels[pixel + 2], pixels[pixel + 1], pixels[pixel]]
        };
        let sample_center = |x: u32| {
            let world_x = 0.22 * x as f32 - 0.99;
            ((world_x + 1.0) * WIDTH as f32 * 0.5).round() as u32
        };
        let color_distance = |left: [u8; 3], right: [u8; 3]| {
            left.into_iter()
                .zip(right)
                .map(|(left, right)| u16::from(left.abs_diff(right)))
                .sum::<u16>()
        };
        let far = sample_rgb(sample_center(1));
        let near = sample_rgb(sample_center(2));
        let distance_03 = sample_rgb(sample_center(3));
        let distance_01 = sample_rgb(sample_center(4));
        let distance_02 = sample_rgb(sample_center(5));
        let half_brightness = sample_rgb(sample_center(6));
        let quarter_brightness = sample_rgb(sample_center(7));
        let sealed = sample_rgb(sample_center(8));
        assert_eq!(
            far, [0; 3],
            "a fully faded connected surface must be #000000"
        );
        assert!(
            near[0] > 160 && near[1] > 160 && near[2] > 80,
            "a zero-distance surface must be pale yellow, got {near:?}"
        );
        assert!(
            color_distance(distance_01, distance_02) >= 8,
            "successive distances must not share a color plateau, got {distance_01:?} and {distance_02:?}"
        );
        assert!(
            color_distance(distance_02, distance_03) >= 8,
            "the gradient must continue changing at every step, got {distance_02:?} then {distance_03:?}"
        );
        assert!(
            half_brightness[0] > half_brightness[2] && half_brightness[2] > half_brightness[1],
            "half-brightness must be crimson, got {half_brightness:?}"
        );
        assert!(
            quarter_brightness[2] > quarter_brightness[0]
                && quarter_brightness[0] > quarter_brightness[1],
            "quarter-brightness must be purple, got {quarter_brightness:?}"
        );
        assert_eq!(sealed, [0; 3], "a sealed surface must remain #000000");
    }
}
