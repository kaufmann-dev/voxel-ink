use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
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
    color: vec4<f32>,
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
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

    let world_position = uniforms.index_to_world * vec4<f32>(index_position, 1.0);
    let basis_x = uniforms.index_to_world[0].xyz;
    let basis_y = uniforms.index_to_world[1].xyz;
    let basis_z = uniforms.index_to_world[2].xyz;
    var normal = cross(basis_y, basis_z);
    if (axis == 1u) {
        normal = cross(basis_z, basis_x);
    } else if (axis == 2u) {
        normal = cross(basis_x, basis_y);
    }
    if (!positive) {
        normal = -normal;
    }

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.normal = normalize(normal);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuse = max(dot(normalize(input.normal), normalize(uniforms.light.xyz)), 0.0);
    let lighting = 0.28 + 0.72 * diffuse;
    return vec4<f32>(uniforms.color.rgb * lighting, uniforms.color.a);
}
"#;

const MAX_GPU_UPDATES_PER_FRAME: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_projection: [[f32; 4]; 4],
    index_to_world: [[f32; 4]; 4],
    color: [f32; 4],
    light: [f32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            index_to_world: Mat4::IDENTITY.to_cols_array_2d(),
            color: [0.141, 0.784, 0.859, 1.0],
            light: [0.35, -0.45, 0.82, 0.0],
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
        chunks: Vec<ChunkCoord>,
    },
    Chunk {
        tag: MeshJobTag,
        quads: Vec<CompactQuad>,
    },
}

struct SceneState {
    epoch: u64,
    revision: u64,
    dimensions: [u32; 3],
    affine: Matrix4,
    yaw: f32,
    pitch: f32,
    zoom: f32,
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
}

impl SceneBridge {
    pub fn set_repaint_context(&self, context: egui::Context) {
        *self.repaint.lock().expect("3D repaint context poisoned") = Some(context);
    }

    pub fn reset(&self, epoch: u64, revision: u64, dimensions: [u32; 3], affine: Matrix4) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
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
    }

    pub fn invalidate(&self, epoch: u64, revision: u64, chunks: Vec<ChunkCoord>) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch != epoch {
            return;
        }
        state.revision = revision;
        state
            .updates
            .push_back(SceneUpdate::Invalidate { revision, chunks });
    }

    pub fn insert_chunk(&self, tag: MeshJobTag, quads: Vec<CompactQuad>) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        if state.epoch == tag.document_epoch && state.revision == tag.revision {
            state.updates.push_back(SceneUpdate::Chunk { tag, quads });
        }
    }

    pub fn clear(&self) {
        self.reset(0, 0, [1; 3], SceneState::default().affine);
    }

    pub fn orbit(&self, delta: egui::Vec2) {
        let mut state = self.state.lock().expect("3D scene state poisoned");
        state.yaw += delta.x * 0.008;
        state.pitch = (state.pitch + delta.y * 0.008).clamp(-1.5, 1.5);
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
    }
}

struct GpuChunk {
    buffer: wgpu::Buffer,
    instances: u32,
    generation: u64,
}

struct GpuVolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    chunks: HashMap<ChunkCoord, GpuChunk>,
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            chunks: HashMap::new(),
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
                self.epoch = epoch;
                self.revision = revision;
                self.dimensions = dimensions;
                self.affine = affine;
            }
            SceneUpdate::Invalidate { revision, chunks } => {
                self.revision = revision;
                for chunk in chunks {
                    self.chunks.remove(&chunk);
                }
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
        }
    }
}

fn scene_uniforms(state: &SceneState) -> Uniforms {
    let (view, projection, affine) = scene_camera(state);
    Uniforms {
        view_projection: (projection * view).to_cols_array_2d(),
        index_to_world: affine.to_cols_array_2d(),
        ..Uniforms::default()
    }
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
    let projection = Mat4::perspective_rh(
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
    fn invalidation_updates_revision_without_dropping_other_chunks() {
        let bridge = SceneBridge::default();
        bridge.reset(4, 8, [32; 3], SceneState::default().affine);
        bridge.invalidate(4, 9, vec![ChunkCoord::new(0, 0, 0)]);
        let state = bridge.state.lock().unwrap();
        assert_eq!(state.revision, 9);
        assert_eq!(state.updates.len(), 2);
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
        let _renderer = GpuVolumeRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}
