use std::{
    collections::{BTreeSet, HashMap},
    mem::size_of,
    num::NonZeroUsize,
};

use bytemuck::{Pod, Zeroable};
use crossbeam_channel::{Receiver, Sender, bounded};

use crate::{AppError, Result, cancel::CancellationToken, error::checked_voxel_count};

pub const CHUNK_SIZE: u32 = 32;
pub const CHUNK_VOXELS: usize = CHUNK_SIZE as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize;
pub const MAX_MESH_BYTES: usize = 256 * 1024 * 1024;

const HALO_WIDTH: u32 = 1;
const MAX_FACE_CELLS: usize = CHUNK_SIZE as usize * CHUNK_SIZE as usize;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ChunkCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ChunkCoord {
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub const fn as_array(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    #[must_use]
    pub const fn from_voxel(voxel: [u32; 3]) -> Self {
        Self {
            x: voxel[0] / CHUNK_SIZE,
            y: voxel[1] / CHUNK_SIZE,
            z: voxel[2] / CHUNK_SIZE,
        }
    }

    #[must_use]
    pub fn origin(self) -> Option<[u32; 3]> {
        let [x, y, z] = self.as_array();
        Some([
            x.checked_mul(CHUNK_SIZE)?,
            y.checked_mul(CHUNK_SIZE)?,
            z.checked_mul(CHUNK_SIZE)?,
        ])
    }

    #[must_use]
    pub fn extent(self, dimensions: [u32; 3]) -> Option<[u32; 3]> {
        let origin = self.origin()?;
        if origin
            .iter()
            .zip(dimensions)
            .any(|(&coordinate, dimension)| coordinate >= dimension)
        {
            return None;
        }
        Some(std::array::from_fn(|axis| {
            (dimensions[axis] - origin[axis]).min(CHUNK_SIZE)
        }))
    }
}

#[must_use]
pub const fn chunk_grid(dimensions: [u32; 3]) -> [u32; 3] {
    let mut grid = [0; 3];
    let mut axis = 0;
    while axis < 3 {
        grid[axis] = if dimensions[axis] == 0 {
            0
        } else {
            (dimensions[axis] - 1) / CHUNK_SIZE + 1
        };
        axis += 1;
    }
    grid
}

/// An isolated chunk plus a one-voxel empty-padded halo on all six sides.
///
/// The buffer remains x-major and never contains more than 34^3 bytes. It is
/// the only occupancy data a background meshing job needs to own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkInput {
    coord: ChunkCoord,
    origin: [u32; 3],
    extent: [u32; 3],
    halo_dimensions: [u32; 3],
    occupancy: Vec<u8>,
    has_occupied_core: bool,
}

impl ChunkInput {
    /// Copies one chunk and its one-voxel halo from an x-major occupancy volume.
    ///
    /// # Errors
    ///
    /// Returns an error for cancellation, invalid dimensions or chunk coordinates,
    /// an occupancy-length mismatch, or a failed bounded allocation.
    pub fn copy_from_volume(
        dimensions: [u32; 3],
        occupancy: &[u8],
        coord: ChunkCoord,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        cancellation.check()?;
        let voxel_count = checked_voxel_count(dimensions)?;
        if occupancy.len() != voxel_count {
            return Err(AppError::InvalidDimensions(format!(
                "expected {voxel_count} occupancy bytes, got {}",
                occupancy.len()
            )));
        }
        let origin = coord.origin().ok_or_else(|| {
            AppError::InvalidDimensions("chunk origin overflows u32 coordinates".into())
        })?;
        let extent = coord.extent(dimensions).ok_or_else(|| {
            AppError::InvalidDimensions(format!(
                "chunk {:?} lies outside volume dimensions {dimensions:?}",
                coord.as_array()
            ))
        })?;
        let halo_dimensions = extent.map(|axis| axis + 2 * HALO_WIDTH);
        let halo_len = halo_dimensions
            .iter()
            .try_fold(1_usize, |total, &axis| total.checked_mul(axis as usize))
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(halo_len)
            .map_err(|_| AppError::Allocation { bytes: halo_len })?;
        copied.resize(halo_len, 0);
        let volume_width = dimensions[0] as usize;
        let volume_height = dimensions[1] as usize;
        let source_x_start = origin[0].saturating_sub(HALO_WIDTH) as usize;
        let source_x_end =
            (origin[0] as usize + extent[0] as usize + HALO_WIDTH as usize).min(volume_width);
        let copied_row_width = source_x_end - source_x_start;
        let destination_x = usize::from(origin[0] == 0);
        let halo_width = halo_dimensions[0] as usize;
        let halo_plane = halo_width * halo_dimensions[1] as usize;
        let mut has_occupied_core = false;

        for local_z in 0..halo_dimensions[2] {
            cancellation.check()?;
            let global_z = i64::from(origin[2]) + i64::from(local_z) - i64::from(HALO_WIDTH);
            if !(0..i64::from(dimensions[2])).contains(&global_z) {
                continue;
            }
            let global_z = usize::try_from(global_z).map_err(|_| {
                AppError::InvalidDimensions("negative z coordinate escaped halo clipping".into())
            })?;
            for local_y in 0..halo_dimensions[1] {
                let global_y = i64::from(origin[1]) + i64::from(local_y) - i64::from(HALO_WIDTH);
                if !(0..i64::from(dimensions[1])).contains(&global_y) {
                    continue;
                }
                let global_y = usize::try_from(global_y).map_err(|_| {
                    AppError::InvalidDimensions(
                        "negative y coordinate escaped halo clipping".into(),
                    )
                })?;
                let source = source_x_start + volume_width * (global_y + volume_height * global_z);
                let destination =
                    destination_x + halo_width * local_y as usize + halo_plane * local_z as usize;
                copied[destination..destination + copied_row_width]
                    .copy_from_slice(&occupancy[source..source + copied_row_width]);
                if !has_occupied_core
                    && (origin[1]..origin[1] + extent[1]).contains(&(global_y as u32))
                    && (origin[2]..origin[2] + extent[2]).contains(&(global_z as u32))
                {
                    let core_source =
                        origin[0] as usize + volume_width * (global_y + volume_height * global_z);
                    has_occupied_core = occupancy[core_source..core_source + extent[0] as usize]
                        .iter()
                        .any(|&value| value != 0);
                }
            }
        }

        Ok(Self {
            coord,
            origin,
            extent,
            halo_dimensions,
            occupancy: copied,
            has_occupied_core,
        })
    }

    #[must_use]
    pub const fn coord(&self) -> ChunkCoord {
        self.coord
    }

    #[must_use]
    pub const fn origin(&self) -> [u32; 3] {
        self.origin
    }

    #[must_use]
    pub const fn extent(&self) -> [u32; 3] {
        self.extent
    }

    #[must_use]
    pub const fn halo_dimensions(&self) -> [u32; 3] {
        self.halo_dimensions
    }

    #[must_use]
    pub const fn copied_bytes(&self) -> usize {
        self.occupancy.len()
    }

    #[cfg(test)]
    fn occupied(&self, local: [i32; 3]) -> bool {
        debug_assert!(local[0] >= -1 && local[0] <= self.extent[0] as i32);
        debug_assert!(local[1] >= -1 && local[1] <= self.extent[1] as i32);
        debug_assert!(local[2] >= -1 && local[2] <= self.extent[2] as i32);
        let x = (local[0] + HALO_WIDTH as i32) as usize;
        let y = (local[1] + HALO_WIDTH as i32) as usize;
        let z = (local[2] + HALO_WIDTH as i32) as usize;
        let width = self.halo_dimensions[0] as usize;
        let plane = width * self.halo_dimensions[1] as usize;
        self.occupancy[x + width * y + plane * z] != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceAxis {
    X,
    Y,
    Z,
}

impl FaceAxis {
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    const fn from_index(axis: usize) -> Option<Self> {
        match axis {
            0 => Some(Self::X),
            1 => Some(Self::Y),
            2 => Some(Self::Z),
            _ => None,
        }
    }

    const fn tangent_axes(self) -> [usize; 2] {
        match self {
            Self::X => [1, 2],
            Self::Y => [2, 0],
            Self::Z => [0, 1],
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// A 16-byte GPU instance expanded into two triangles by the vertex shader.
///
/// `origin` is an integer grid boundary. Expansion subtracts 0.5 on every axis
/// so voxel index coordinates remain voxel centers under the document affine.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct CompactQuad {
    pub origin: [u32; 3],
    pub packed: u32,
}

/// Four leak-distance samples packed for a `Snorm16x4` vertex attribute.
///
/// Connected distances occupy the non-negative signed-normalized range. The
/// `u16::MAX` input sentinel is encoded as `i16::MIN`, which the GPU exposes as
/// `-1.0` and therefore keeps distinct from every connected distance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
pub struct PackedLeakDistance {
    words: [u32; 2],
}

impl PackedLeakDistance {
    /// Packs four bounded distance steps into two little-lane-order words.
    ///
    /// # Errors
    ///
    /// Returns an error when a value is neither a connected distance in
    /// `0..=i16::MAX` nor the `u16::MAX` disconnected sentinel.
    pub fn new(corners: [u16; 4]) -> Result<Self> {
        let encoded = [
            encode_leak_distance(corners[0])?,
            encode_leak_distance(corners[1])?,
            encode_leak_distance(corners[2])?,
            encode_leak_distance(corners[3])?,
        ];
        Ok(Self {
            words: [
                u32::from(encoded[0]) | (u32::from(encoded[1]) << 16),
                u32::from(encoded[2]) | (u32::from(encoded[3]) << 16),
            ],
        })
    }

    #[must_use]
    pub const fn words(self) -> [u32; 2] {
        self.words
    }

    #[must_use]
    pub const fn corner_bits(self, corner: usize) -> Option<u16> {
        if corner >= 4 {
            return None;
        }
        let word = self.words[corner / 2];
        Some(((word >> ((corner % 2) * 16)) & 0xffff) as u16)
    }
}

fn encode_leak_distance(distance: u16) -> Result<u16> {
    if distance == u16::MAX {
        return Ok(i16::MIN as u16);
    }
    if distance <= i16::MAX as u16 {
        return Ok(distance);
    }
    Err(AppError::InvalidDimensions(format!(
        "leak distance {distance} exceeds the signed 16-bit connected range"
    )))
}

impl CompactQuad {
    const LENGTH_MASK: u32 = 0x1f;
    const V_LENGTH_SHIFT: u32 = 5;
    const AXIS_SHIFT: u32 = 10;
    const POSITIVE_SHIFT: u32 = 12;
    const OCCLUSION_SHIFT: u32 = 13;
    const OCCLUSION_MASK: u32 = 0x3;
    const FLIPPED_DIAGONAL_SHIFT: u32 = 21;

    #[must_use]
    pub fn new(
        origin: [u32; 3],
        u_length: u32,
        v_length: u32,
        normal_axis: FaceAxis,
        positive: bool,
        corner_occlusion: [u8; 4],
    ) -> Option<Self> {
        if !(1..=CHUNK_SIZE).contains(&u_length)
            || !(1..=CHUNK_SIZE).contains(&v_length)
            || corner_occlusion.into_iter().any(|level| level > 3)
        {
            return None;
        }
        let mut packed = (u_length - 1)
            | ((v_length - 1) << Self::V_LENGTH_SHIFT)
            | (normal_axis.code() << Self::AXIS_SHIFT)
            | (u32::from(positive) << Self::POSITIVE_SHIFT);
        for (corner, level) in corner_occlusion.into_iter().enumerate() {
            packed |= u32::from(level) << (Self::OCCLUSION_SHIFT + corner as u32 * 2);
        }
        let diagonal_02 = u16::from(corner_occlusion[0]) + u16::from(corner_occlusion[2]);
        let diagonal_13 = u16::from(corner_occlusion[1]) + u16::from(corner_occlusion[3]);
        packed |= u32::from(diagonal_02 > diagonal_13) << Self::FLIPPED_DIAGONAL_SHIFT;
        Some(Self { origin, packed })
    }

    #[must_use]
    pub const fn u_length(self) -> u32 {
        (self.packed & Self::LENGTH_MASK) + 1
    }

    #[must_use]
    pub const fn v_length(self) -> u32 {
        ((self.packed >> Self::V_LENGTH_SHIFT) & Self::LENGTH_MASK) + 1
    }

    #[must_use]
    pub const fn normal_axis(self) -> FaceAxis {
        match FaceAxis::from_index(((self.packed >> Self::AXIS_SHIFT) & 0x3) as usize) {
            Some(axis) => axis,
            None => FaceAxis::X,
        }
    }

    #[must_use]
    pub const fn positive(self) -> bool {
        (self.packed & (1 << Self::POSITIVE_SHIFT)) != 0
    }

    #[must_use]
    pub const fn corner_occlusion(self, corner: usize) -> Option<u8> {
        if corner >= 4 {
            return None;
        }
        Some(
            ((self.packed >> (Self::OCCLUSION_SHIFT + corner as u32 * 2)) & Self::OCCLUSION_MASK)
                as u8,
        )
    }

    #[must_use]
    pub const fn flipped_diagonal(self) -> bool {
        (self.packed & (1 << Self::FLIPPED_DIAGONAL_SHIFT)) != 0
    }

    #[must_use]
    pub const fn area(self) -> u32 {
        self.u_length() * self.v_length()
    }

    #[must_use]
    pub const fn normal(self) -> [f32; 3] {
        let mut normal = [0.0; 3];
        normal[self.normal_axis().index()] = if self.positive() { 1.0 } else { -1.0 };
        normal
    }

    #[must_use]
    pub fn expanded_vertex(self, vertex_index: u32) -> Option<[f64; 3]> {
        let corner = face_vertex_corner(self.positive(), self.flipped_diagonal(), vertex_index)?;
        let [u_axis, v_axis] = self.normal_axis().tangent_axes();
        let mut vertex = self.origin.map(|coordinate| f64::from(coordinate) - 0.5);
        if corner == 1 || corner == 2 {
            vertex[u_axis] += f64::from(self.u_length());
        }
        if corner == 2 || corner == 3 {
            vertex[v_axis] += f64::from(self.v_length());
        }
        Some(vertex)
    }
}

/// Packs the outside-voxel distance at every corner of each greedy quad.
///
/// `halo_steps` is an x-major, one-voxel-padded chunk halo. Quad origins are
/// global grid boundaries, while `chunk_origin` is the global first voxel of
/// the unpadded chunk core.
///
/// # Errors
///
/// Returns an error for an invalid halo length, a quad outside the supplied
/// chunk halo, an unsupported distance encoding, arithmetic overflow, or a
/// failed bounded allocation.
pub fn pack_leak_distances(
    quads: &[CompactQuad],
    chunk_origin: [u32; 3],
    halo_dimensions: [u32; 3],
    halo_steps: &[u16],
) -> Result<Vec<PackedLeakDistance>> {
    let halo_dimensions = halo_dimensions.map(|dimension| dimension as usize);
    let halo_plane = halo_dimensions[0]
        .checked_mul(halo_dimensions[1])
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let halo_len = halo_plane
        .checked_mul(halo_dimensions[2])
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    if halo_steps.len() != halo_len {
        return Err(AppError::InvalidDimensions(format!(
            "expected {halo_len} leak-distance halo steps, got {}",
            halo_steps.len()
        )));
    }
    let output_bytes = quads
        .len()
        .checked_mul(size_of::<PackedLeakDistance>())
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(quads.len())
        .map_err(|_| AppError::Allocation {
            bytes: output_bytes,
        })?;
    let strides = [1, halo_dimensions[0], halo_plane];

    for &quad in quads {
        let local_origin = [
            quad.origin[0].checked_sub(chunk_origin[0]),
            quad.origin[1].checked_sub(chunk_origin[1]),
            quad.origin[2].checked_sub(chunk_origin[2]),
        ];
        let local_origin = [
            local_origin[0].ok_or_else(|| quad_outside_halo_error(quad, chunk_origin))? as usize,
            local_origin[1].ok_or_else(|| quad_outside_halo_error(quad, chunk_origin))? as usize,
            local_origin[2].ok_or_else(|| quad_outside_halo_error(quad, chunk_origin))? as usize,
        ];
        let normal_axis = quad.normal_axis().index();
        let [u_axis, v_axis] = quad.normal_axis().tangent_axes();
        let normal = local_origin[normal_axis]
            .checked_add(usize::from(quad.positive()))
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let u_min = local_origin[u_axis]
            .checked_add(HALO_WIDTH as usize)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let v_min = local_origin[v_axis]
            .checked_add(HALO_WIDTH as usize)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let u_max = u_min
            .checked_add(quad.u_length() as usize - 1)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let v_max = v_min
            .checked_add(quad.v_length() as usize - 1)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        if normal >= halo_dimensions[normal_axis]
            || u_max >= halo_dimensions[u_axis]
            || v_max >= halo_dimensions[v_axis]
        {
            return Err(quad_outside_halo_error(quad, chunk_origin));
        }

        let corners = [
            [u_min, v_min],
            [u_max, v_min],
            [u_max, v_max],
            [u_min, v_max],
        ];
        let distances = corners.map(|[u, v]| {
            normal * strides[normal_axis] + u * strides[u_axis] + v * strides[v_axis]
        });
        packed.push(PackedLeakDistance::new(
            distances.map(|index| halo_steps[index]),
        )?);
    }
    Ok(packed)
}

fn quad_outside_halo_error(quad: CompactQuad, chunk_origin: [u32; 3]) -> AppError {
    AppError::InvalidDimensions(format!(
        "quad at {:?} lies outside the leak-distance halo for chunk origin {chunk_origin:?}",
        quad.origin
    ))
}

const fn face_vertex_corner(
    positive: bool,
    flipped_diagonal: bool,
    vertex_index: u32,
) -> Option<u32> {
    let corners = match (positive, flipped_diagonal) {
        (true, false) => [0, 1, 2, 0, 2, 3],
        (false, false) => [0, 3, 2, 0, 2, 1],
        (true, true) => [0, 1, 3, 1, 2, 3],
        (false, true) => [0, 3, 1, 1, 3, 2],
    };
    if vertex_index < 6 {
        Some(corners[vertex_index as usize])
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshUpdatePriority {
    Background,
    Live,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshJobTag {
    pub document_epoch: u64,
    pub revision: u64,
    pub chunk_generation: u64,
    pub edit_generation: u64,
    pub chunk: ChunkCoord,
}

impl MeshJobTag {
    #[must_use]
    pub const fn is_stale(
        self,
        current_epoch: u64,
        current_revision: u64,
        current_chunk_generation: u64,
    ) -> bool {
        self.document_epoch != current_epoch
            || self.revision != current_revision
            || self.chunk_generation != current_chunk_generation
    }
}

#[must_use]
pub const fn mesh_result_is_stale(
    tag: MeshJobTag,
    current_epoch: u64,
    current_revision: u64,
    current_chunk_generation: u64,
) -> bool {
    tag.is_stale(current_epoch, current_revision, current_chunk_generation)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkMesh {
    pub quads: Vec<CompactQuad>,
    pub exposed_voxel_faces: u32,
}

impl ChunkMesh {
    #[must_use]
    pub const fn gpu_bytes(&self) -> usize {
        self.quads.len() * size_of::<CompactQuad>()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaggedChunkMesh {
    pub tag: MeshJobTag,
    pub mesh: ChunkMesh,
}

/// Generates greedy boundary quads for one copied chunk.
///
/// # Errors
///
/// Returns an error when cancelled, when the tag addresses a different chunk,
/// when derived counts overflow, or when quad allocation fails.
pub fn mesh_chunk(
    input: &ChunkInput,
    tag: MeshJobTag,
    cancellation: &CancellationToken,
) -> Result<TaggedChunkMesh> {
    cancellation.check()?;
    if input.coord != tag.chunk {
        return Err(AppError::InvalidDimensions(format!(
            "mesh tag chunk {:?} does not match copied chunk {:?}",
            tag.chunk.as_array(),
            input.coord.as_array()
        )));
    }
    if !input.has_occupied_core {
        return Ok(TaggedChunkMesh {
            tag,
            mesh: ChunkMesh::default(),
        });
    }

    let mut quads = Vec::new();
    quads
        .try_reserve_exact(6)
        .map_err(|_| AppError::Allocation {
            bytes: 6 * size_of::<CompactQuad>(),
        })?;
    let mut exposed_voxel_faces = 0_u32;
    let mut mask = [0_i8; MAX_FACE_CELLS];

    for normal_axis in FaceAxis::ALL {
        mesh_axis(
            input,
            normal_axis,
            &mut mask,
            &mut quads,
            &mut exposed_voxel_faces,
            cancellation,
        )?;
    }

    cancellation.check()?;
    quads.shrink_to_fit();
    Ok(TaggedChunkMesh {
        tag,
        mesh: ChunkMesh {
            quads,
            exposed_voxel_faces,
        },
    })
}

#[derive(Clone, Copy)]
struct PlaneLayout {
    normal_axis: FaceAxis,
    normal_axis_index: usize,
    u_axis: usize,
    v_axis: usize,
    u_extent: usize,
    v_extent: usize,
    normal_extent: usize,
    plane: usize,
    halo_strides: [usize; 3],
}

fn mesh_axis(
    input: &ChunkInput,
    normal_axis: FaceAxis,
    mask: &mut [i8; MAX_FACE_CELLS],
    quads: &mut Vec<CompactQuad>,
    exposed_voxel_faces: &mut u32,
    cancellation: &CancellationToken,
) -> Result<()> {
    let normal_axis_index = normal_axis.index();
    let [u_axis, v_axis] = normal_axis.tangent_axes();
    let u_extent = input.extent[u_axis] as usize;
    let v_extent = input.extent[v_axis] as usize;
    let normal_extent = input.extent[normal_axis_index] as usize;
    let halo_width = input.halo_dimensions[0] as usize;
    let halo_strides = [
        1,
        halo_width,
        halo_width * input.halo_dimensions[1] as usize,
    ];

    for plane in 0..=normal_extent {
        cancellation.check()?;
        let layout = PlaneLayout {
            normal_axis,
            normal_axis_index,
            u_axis,
            v_axis,
            u_extent,
            v_extent,
            normal_extent,
            plane,
            halo_strides,
        };
        let active_mask = &mut mask[..u_extent * v_extent];
        fill_face_mask(input, layout, active_mask);
        emit_greedy_quads(input, layout, active_mask, quads, exposed_voxel_faces)?;
    }
    Ok(())
}

fn fill_face_mask(input: &ChunkInput, layout: PlaneLayout, mask: &mut [i8]) {
    let normal_stride = layout.halo_strides[layout.normal_axis_index];
    let u_stride = layout.halo_strides[layout.u_axis];
    let v_stride = layout.halo_strides[layout.v_axis];
    for v in 0..layout.v_extent {
        for u in 0..layout.u_extent {
            let negative_index = layout.plane * normal_stride
                + (u + HALO_WIDTH as usize) * u_stride
                + (v + HALO_WIDTH as usize) * v_stride;
            let occupied_negative = input.occupancy[negative_index] != 0;
            let occupied_positive = input.occupancy[negative_index + normal_stride] != 0;

            mask[v * layout.u_extent + u] = match (occupied_negative, occupied_positive) {
                // A neighboring chunk owns a face when the occupied voxel is
                // in that chunk, even though it appears in this chunk's halo.
                (true, false) if layout.plane > 0 => 1,
                (false, true) if layout.plane < layout.normal_extent => -1,
                _ => 0,
            };
        }
    }
}

fn emit_greedy_quads(
    input: &ChunkInput,
    layout: PlaneLayout,
    mask: &mut [i8],
    quads: &mut Vec<CompactQuad>,
    exposed_voxel_faces: &mut u32,
) -> Result<()> {
    let mut v = 0;
    while v < layout.v_extent {
        let mut u = 0;
        while u < layout.u_extent {
            let face = mask[v * layout.u_extent + u];
            if face == 0 {
                u += 1;
                continue;
            }

            let mut width = 1;
            while u + width < layout.u_extent && mask[v * layout.u_extent + u + width] == face {
                width += 1;
            }
            let mut height = 1;
            'height: while v + height < layout.v_extent {
                for offset in 0..width {
                    if mask[(v + height) * layout.u_extent + u + offset] != face {
                        break 'height;
                    }
                }
                height += 1;
            }
            for row in v..v + height {
                mask[row * layout.u_extent + u..row * layout.u_extent + u + width].fill(0);
            }

            let mut origin = input.origin;
            origin[layout.normal_axis_index] += chunk_u32(layout.plane)?;
            origin[layout.u_axis] += chunk_u32(u)?;
            origin[layout.v_axis] += chunk_u32(v)?;
            let corner_occlusion =
                quad_corner_occlusion(input, layout, u, v, width, height, face > 0);
            let quad = CompactQuad::new(
                origin,
                chunk_u32(width)?,
                chunk_u32(height)?,
                layout.normal_axis,
                face > 0,
                corner_occlusion,
            )
            .ok_or_else(|| {
                AppError::InvalidDimensions("greedy quad exceeded chunk bounds".into())
            })?;
            *exposed_voxel_faces =
                exposed_voxel_faces
                    .checked_add(quad.area())
                    .ok_or_else(|| {
                        AppError::InvalidDimensions(
                            "chunk exposed-face count overflowed u32".into(),
                        )
                    })?;
            try_push_quad(quads, quad)?;
            u += width;
        }
        v += 1;
    }
    Ok(())
}

fn quad_corner_occlusion(
    input: &ChunkInput,
    layout: PlaneLayout,
    u: usize,
    v: usize,
    width: usize,
    height: usize,
    positive: bool,
) -> [u8; 4] {
    let mut levels = [0; 4];
    let normal_offset =
        (layout.plane + usize::from(positive)) * layout.halo_strides[layout.normal_axis_index];
    let u_stride = layout.halo_strides[layout.u_axis];
    let v_stride = layout.halo_strides[layout.v_axis];
    for (corner, [maximum_u, maximum_v]) in
        [[false, false], [true, false], [true, true], [false, true]]
            .into_iter()
            .enumerate()
    {
        let outside = normal_offset
            + (u + 1 + usize::from(maximum_u) * (width - 1)) * u_stride
            + (v + 1 + usize::from(maximum_v) * (height - 1)) * v_stride;
        let side_u = if maximum_u {
            outside + u_stride
        } else {
            outside - u_stride
        };
        let side_v = if maximum_v {
            outside + v_stride
        } else {
            outside - v_stride
        };
        let diagonal = match (maximum_u, maximum_v) {
            (true, true) => outside + u_stride + v_stride,
            (true, false) => outside + u_stride - v_stride,
            (false, true) => outside - u_stride + v_stride,
            (false, false) => outside - u_stride - v_stride,
        };
        let occupied_u = input.occupancy[side_u] != 0;
        let occupied_v = input.occupancy[side_v] != 0;
        levels[corner] = if occupied_u && occupied_v {
            3
        } else {
            u8::from(occupied_u) + u8::from(occupied_v) + u8::from(input.occupancy[diagonal] != 0)
        };
    }
    levels
}

#[cfg(test)]
fn corner_occlusion(
    input: &ChunkInput,
    side_u: [i32; 3],
    side_v: [i32; 3],
    diagonal: [i32; 3],
) -> u8 {
    let occupied_u = input.occupied(side_u);
    let occupied_v = input.occupied(side_v);
    if occupied_u && occupied_v {
        3
    } else {
        u8::from(occupied_u) + u8::from(occupied_v) + u8::from(input.occupied(diagonal))
    }
}

fn chunk_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| AppError::InvalidDimensions("chunk coordinate exceeds u32".into()))
}

fn try_push_quad(quads: &mut Vec<CompactQuad>, quad: CompactQuad) -> Result<()> {
    if quads.len() == quads.capacity() {
        let desired_capacity = quads
            .len()
            .checked_add(1)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let bytes = desired_capacity
            .checked_mul(size_of::<CompactQuad>())
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        quads
            .try_reserve(1)
            .map_err(|_| AppError::Allocation { bytes })?;
    }
    quads.push(quad);
    Ok(())
}

#[must_use]
pub fn invalidated_chunks_for_voxel(dimensions: [u32; 3], voxel: [u32; 3]) -> Vec<ChunkCoord> {
    invalidated_chunks_for_voxels(dimensions, std::iter::once(voxel))
}

#[must_use]
pub fn invalidated_chunks_for_voxels(
    dimensions: [u32; 3],
    voxels: impl IntoIterator<Item = [u32; 3]>,
) -> Vec<ChunkCoord> {
    let mut invalidated = BTreeSet::new();
    for voxel in voxels {
        add_invalidated_chunks(&mut invalidated, dimensions, voxel);
    }
    invalidated.into_iter().collect()
}

#[must_use]
pub fn invalidated_chunks_for_indices(
    dimensions: [u32; 3],
    indices: impl IntoIterator<Item = u32>,
) -> Vec<ChunkCoord> {
    let plane = u64::from(dimensions[0]) * u64::from(dimensions[1]);
    let volume = plane.saturating_mul(u64::from(dimensions[2]));
    invalidated_chunks_for_voxels(
        dimensions,
        indices.into_iter().filter_map(|index| {
            let index = u64::from(index);
            if index >= volume || dimensions[0] == 0 || dimensions[1] == 0 {
                return None;
            }
            let z = index / plane;
            let remainder = index % plane;
            let y = remainder / u64::from(dimensions[0]);
            let x = remainder % u64::from(dimensions[0]);
            Some([
                u32::try_from(x).ok()?,
                u32::try_from(y).ok()?,
                u32::try_from(z).ok()?,
            ])
        }),
    )
}

/// Derives dirty mesh chunks directly from compact linear index spans.
///
/// Each `(start, length)` pair addresses x-major voxels. The implementation
/// walks scanline segments rather than expanding every voxel index, while
/// preserving the one-voxel halo invalidation used by per-index edits.
#[must_use]
pub fn invalidated_chunks_for_index_spans(
    dimensions: [u32; 3],
    spans: impl IntoIterator<Item = (u32, u32)>,
) -> Vec<ChunkCoord> {
    let width = u64::from(dimensions[0]);
    let plane = width.saturating_mul(u64::from(dimensions[1]));
    let volume = plane.saturating_mul(u64::from(dimensions[2]));
    if width == 0 || plane == 0 || volume == 0 {
        return Vec::new();
    }

    let mut invalidated = BTreeSet::new();
    for (start, length) in spans {
        let mut index = u64::from(start).min(volume);
        let end = u64::from(start)
            .saturating_add(u64::from(length))
            .min(volume);
        while index < end {
            let z = index / plane;
            let remainder = index % plane;
            let y = remainder / width;
            let x = remainder % width;
            let row_end = end.min(index + (width - x));
            let x_end = x + (row_end - index);
            add_invalidated_chunks_for_row_range(
                &mut invalidated,
                dimensions,
                u32::try_from(z).expect("clamped volume z coordinate fits u32"),
                u32::try_from(y).expect("clamped volume y coordinate fits u32"),
                u32::try_from(x).expect("clamped volume x coordinate fits u32"),
                u32::try_from(x_end).expect("clamped volume x boundary fits u32"),
            );
            index = row_end;
        }
    }
    invalidated.into_iter().collect()
}

fn add_invalidated_chunks_for_row_range(
    invalidated: &mut BTreeSet<ChunkCoord>,
    dimensions: [u32; 3],
    z: u32,
    y: u32,
    x_start: u32,
    x_end: u32,
) {
    if x_start >= x_end || x_end > dimensions[0] || y >= dimensions[1] || z >= dimensions[2] {
        return;
    }
    let grid = chunk_grid(dimensions);
    let owner_y = y / CHUNK_SIZE;
    let owner_z = z / CHUNK_SIZE;
    let first_x = x_start / CHUNK_SIZE;
    let last_x = (x_end - 1) / CHUNK_SIZE;

    let first_consumer_x = if x_start.is_multiple_of(CHUNK_SIZE) && first_x > 0 {
        first_x - 1
    } else {
        first_x
    };
    let last_consumer_x = if x_end.is_multiple_of(CHUNK_SIZE) && last_x + 1 < grid[0] {
        last_x + 1
    } else {
        last_x
    };
    let first_consumer_y = if y.is_multiple_of(CHUNK_SIZE) && owner_y > 0 {
        owner_y - 1
    } else {
        owner_y
    };
    let last_consumer_y = if y % CHUNK_SIZE == CHUNK_SIZE - 1 && owner_y + 1 < grid[1] {
        owner_y + 1
    } else {
        owner_y
    };
    let first_consumer_z = if z.is_multiple_of(CHUNK_SIZE) && owner_z > 0 {
        owner_z - 1
    } else {
        owner_z
    };
    let last_consumer_z = if z % CHUNK_SIZE == CHUNK_SIZE - 1 && owner_z + 1 < grid[2] {
        owner_z + 1
    } else {
        owner_z
    };

    for consumer_z in first_consumer_z..=last_consumer_z {
        for consumer_y in first_consumer_y..=last_consumer_y {
            for consumer_x in first_consumer_x..=last_consumer_x {
                invalidated.insert(ChunkCoord::new(consumer_x, consumer_y, consumer_z));
            }
        }
    }
}

#[must_use]
pub fn invalidated_chunks_for_indices_bounded(
    dimensions: [u32; 3],
    indices: impl IntoIterator<Item = u32>,
    maximum_chunks: usize,
) -> Option<Vec<ChunkCoord>> {
    let plane = u64::from(dimensions[0]).checked_mul(u64::from(dimensions[1]))?;
    let volume = plane.checked_mul(u64::from(dimensions[2]))?;
    if plane == 0 {
        return Some(Vec::new());
    }
    let mut invalidated = BTreeSet::new();
    for index in indices {
        let index = u64::from(index);
        if index >= volume {
            continue;
        }
        let z = index / plane;
        let remainder = index % plane;
        let y = remainder / u64::from(dimensions[0]);
        let x = remainder % u64::from(dimensions[0]);
        let voxel = [
            u32::try_from(x).ok()?,
            u32::try_from(y).ok()?,
            u32::try_from(z).ok()?,
        ];
        add_invalidated_chunks(&mut invalidated, dimensions, voxel);
        if invalidated.len() > maximum_chunks {
            return None;
        }
    }
    Some(invalidated.into_iter().collect())
}

fn add_invalidated_chunks(
    invalidated: &mut BTreeSet<ChunkCoord>,
    dimensions: [u32; 3],
    voxel: [u32; 3],
) {
    if voxel
        .iter()
        .zip(dimensions)
        .any(|(&coordinate, dimension)| coordinate >= dimension)
    {
        return;
    }
    add_invalidated_chunks_for_row_range(
        invalidated,
        dimensions,
        voxel[2],
        voxel[1],
        voxel[0],
        voxel[0] + 1,
    );
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MeshBudgetError {
    #[error(
        "3D preview too complex: retaining {retained_bytes} bytes and adding {replacement_bytes} bytes exceeds the {cap_bytes}-byte mesh limit"
    )]
    Limit {
        cap_bytes: usize,
        retained_bytes: usize,
        replacement_bytes: usize,
    },
    #[error("not enough memory to account for another chunk mesh")]
    Allocation,
}

#[derive(Debug)]
pub struct MeshMemoryBudget {
    cap: usize,
    used: usize,
    chunks: HashMap<ChunkCoord, usize>,
    reservations: HashMap<ChunkCoord, usize>,
}

impl Default for MeshMemoryBudget {
    fn default() -> Self {
        Self::new(MAX_MESH_BYTES)
    }
}

impl MeshMemoryBudget {
    #[must_use]
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            cap: cap_bytes,
            used: 0,
            chunks: HashMap::new(),
            reservations: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn cap_bytes(&self) -> usize {
        self.cap
    }

    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.used
    }

    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.cap - self.used
    }

    /// Atomically updates the accounted resident bytes for one chunk.
    ///
    /// # Errors
    ///
    /// Returns [`MeshBudgetError::Limit`] without changing the accounting when
    /// the replacement would exceed the cap, or `Allocation` if a new map entry
    /// cannot be allocated.
    pub fn try_replace(
        &mut self,
        chunk: ChunkCoord,
        replacement_bytes: usize,
    ) -> std::result::Result<(), MeshBudgetError> {
        let previous_bytes = self.chunks.get(&chunk).copied().unwrap_or(0);
        let reserved_bytes = self.reservations.get(&chunk).copied().unwrap_or(0);
        let previous_accounted = previous_bytes.max(reserved_bytes);
        let replacement_accounted = replacement_bytes.max(reserved_bytes);
        let retained_bytes = self.used - previous_accounted;
        let next_bytes =
            retained_bytes
                .checked_add(replacement_accounted)
                .ok_or(MeshBudgetError::Limit {
                    cap_bytes: self.cap,
                    retained_bytes,
                    replacement_bytes: replacement_accounted,
                })?;
        if next_bytes > self.cap {
            return Err(MeshBudgetError::Limit {
                cap_bytes: self.cap,
                retained_bytes,
                replacement_bytes: replacement_accounted,
            });
        }
        if replacement_bytes == 0 {
            self.chunks.remove(&chunk);
        } else {
            if previous_bytes == 0 {
                self.chunks
                    .try_reserve(1)
                    .map_err(|_| MeshBudgetError::Allocation)?;
            }
            self.chunks.insert(chunk, replacement_bytes);
        }
        self.used = next_bytes;
        Ok(())
    }

    /// Reserves the currently resident byte count as a rollback floor.
    /// Repeated reservations preserve the first pre-edit value.
    pub fn reserve_current_if_absent(
        &mut self,
        chunk: ChunkCoord,
    ) -> std::result::Result<(), MeshBudgetError> {
        if self.reservations.contains_key(&chunk) {
            return Ok(());
        }
        self.reservations
            .try_reserve(1)
            .map_err(|_| MeshBudgetError::Allocation)?;
        let current = self.chunks.get(&chunk).copied().unwrap_or(0);
        self.reservations.insert(chunk, current);
        Ok(())
    }

    #[must_use]
    pub fn reserved_bytes(&self, chunk: ChunkCoord) -> Option<usize> {
        self.reservations.get(&chunk).copied()
    }

    pub fn release_reservation(&mut self, chunk: ChunkCoord) -> Option<usize> {
        let reservation = self.reservations.remove(&chunk)?;
        let actual = self.chunks.get(&chunk).copied().unwrap_or(0);
        self.used -= reservation.saturating_sub(actual);
        Some(reservation)
    }

    pub fn clear_reservations(&mut self) {
        self.reservations.clear();
        self.used = self.chunks.values().copied().sum();
    }

    /// Accounts for the compact GPU instances in a replacement chunk mesh.
    ///
    /// # Errors
    ///
    /// Returns a mesh budget limit or accounting-allocation error under the same
    /// conditions as [`Self::try_replace`].
    pub fn try_replace_mesh(
        &mut self,
        chunk: ChunkCoord,
        mesh: &ChunkMesh,
    ) -> std::result::Result<(), MeshBudgetError> {
        self.try_replace(chunk, mesh.gpu_bytes())
    }

    /// Accounts for a mesh and cancels the shared batch if the cap is exceeded.
    ///
    /// # Errors
    ///
    /// Returns a mesh budget limit or accounting-allocation error under the same
    /// conditions as [`Self::try_replace`]. Allocation failures do not cancel
    /// the batch because retry policy belongs to the caller.
    pub fn try_replace_mesh_or_cancel(
        &mut self,
        chunk: ChunkCoord,
        mesh: &ChunkMesh,
        batch_cancellation: &CancellationToken,
    ) -> std::result::Result<(), MeshBudgetError> {
        let result = self.try_replace_mesh(chunk, mesh);
        if matches!(result, Err(MeshBudgetError::Limit { .. })) {
            batch_cancellation.cancel();
        }
        result
    }

    pub fn remove(&mut self, chunk: ChunkCoord) -> usize {
        let removed = self.chunks.remove(&chunk).unwrap_or(0);
        let reserved = self.reservations.get(&chunk).copied().unwrap_or(0);
        self.used -= removed.saturating_sub(reserved);
        removed
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.reservations.clear();
        self.used = 0;
    }
}

#[must_use]
pub const fn mesh_queue_capacity(worker_count: NonZeroUsize) -> usize {
    worker_count.get().saturating_mul(2)
}

#[must_use]
pub fn bounded_mesh_queue<T>(worker_count: NonZeroUsize) -> (Sender<T>, Receiver<T>) {
    bounded(mesh_queue_capacity(worker_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(chunk: ChunkCoord) -> MeshJobTag {
        MeshJobTag {
            document_epoch: 7,
            revision: 11,
            chunk_generation: 3,
            edit_generation: 0,
            chunk,
        }
    }

    fn input(dimensions: [u32; 3], occupancy: &[u8], coord: ChunkCoord) -> ChunkInput {
        ChunkInput::copy_from_volume(dimensions, occupancy, coord, &CancellationToken::default())
            .unwrap()
    }

    fn mesh(dimensions: [u32; 3], occupancy: &[u8], coord: ChunkCoord) -> ChunkMesh {
        let input = input(dimensions, occupancy, coord);
        mesh_chunk(&input, tag(coord), &CancellationToken::default())
            .unwrap()
            .mesh
    }

    fn naive_exposed_faces(dimensions: [u32; 3], occupancy: &[u8]) -> u32 {
        let mut count = 0;
        for z in 0..dimensions[2] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let index = x as usize
                        + dimensions[0] as usize
                            * (y as usize + dimensions[1] as usize * z as usize);
                    if occupancy[index] == 0 {
                        continue;
                    }
                    for (axis, direction) in [(0, -1_i64), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]
                    {
                        let mut neighbor = [i64::from(x), i64::from(y), i64::from(z)];
                        neighbor[axis] += direction;
                        let in_bounds =
                            neighbor
                                .iter()
                                .zip(dimensions)
                                .all(|(&coordinate, extent)| {
                                    (0..i64::from(extent)).contains(&coordinate)
                                });
                        let occupied = if in_bounds {
                            let neighbor = neighbor.map(|coordinate| {
                                usize::try_from(coordinate).expect("bounds exclude negatives")
                            });
                            occupancy[neighbor[0]
                                + dimensions[0] as usize
                                    * (neighbor[1] + dimensions[1] as usize * neighbor[2])]
                                != 0
                        } else {
                            false
                        };
                        if !occupied {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    #[test]
    fn copies_only_chunk_and_one_voxel_halo() {
        let dimensions = [65, 2, 1];
        let mut occupancy = vec![0; dimensions.into_iter().product::<u32>() as usize];
        occupancy[31] = 1;
        occupancy[32] = 1;
        occupancy[64] = 1;
        let copied = input(dimensions, &occupancy, ChunkCoord::new(1, 0, 0));
        assert_eq!(copied.origin(), [32, 0, 0]);
        assert_eq!(copied.extent(), [32, 2, 1]);
        assert_eq!(copied.halo_dimensions(), [34, 4, 3]);
        assert_eq!(copied.copied_bytes(), 34 * 4 * 3);
        assert!(copied.occupied([-1, 0, 0]));
        assert!(copied.occupied([0, 0, 0]));
        assert!(copied.occupied([32, 0, 0]));
        assert!(!copied.occupied([0, -1, 0]));
    }

    #[test]
    fn empty_chunk_has_no_surface() {
        let mesh = mesh(
            [32, 32, 32],
            &vec![0; CHUNK_VOXELS],
            ChunkCoord::new(0, 0, 0),
        );
        assert!(mesh.quads.is_empty());
        assert_eq!(mesh.exposed_voxel_faces, 0);
    }

    #[test]
    fn empty_core_skips_neighbor_owned_halo_faces() {
        let mut occupancy = vec![0; 64];
        occupancy[31] = 1;
        let right = mesh([64, 1, 1], &occupancy, ChunkCoord::new(1, 0, 0));
        assert!(right.quads.is_empty());
        assert_eq!(right.exposed_voxel_faces, 0);

        let left = mesh([64, 1, 1], &occupancy, ChunkCoord::new(0, 0, 0));
        assert_eq!(left.exposed_voxel_faces, 6);
    }

    #[test]
    fn greedy_mesher_collapses_solid_chunk_to_six_quads() {
        let mesh = mesh(
            [32, 32, 32],
            &vec![1; CHUNK_VOXELS],
            ChunkCoord::new(0, 0, 0),
        );
        assert_eq!(mesh.quads.len(), 6);
        assert_eq!(mesh.exposed_voxel_faces, 6 * 32 * 32);
        assert!(mesh.quads.iter().all(|quad| quad.u_length() == 32
            && quad.v_length() == 32
            && (0..4).all(|corner| quad.corner_occlusion(corner) == Some(0))));
    }

    #[test]
    fn halo_suppresses_internal_chunk_faces() {
        let dimensions = [64, 1, 1];
        let occupancy = vec![1; 64];
        let left = mesh(dimensions, &occupancy, ChunkCoord::new(0, 0, 0));
        let right = mesh(dimensions, &occupancy, ChunkCoord::new(1, 0, 0));
        assert_eq!(left.exposed_voxel_faces + right.exposed_voxel_faces, 258);
        assert!(
            !left
                .quads
                .iter()
                .any(|quad| quad.normal_axis() == FaceAxis::X && quad.positive())
        );
        assert!(
            !right
                .quads
                .iter()
                .any(|quad| quad.normal_axis() == FaceAxis::X && !quad.positive())
        );
    }

    #[test]
    fn compact_quad_expansion_has_outward_winding_and_half_voxel_offset() {
        assert_eq!(size_of::<CompactQuad>(), 16);
        let positive = CompactQuad::new([4, 7, 9], 3, 2, FaceAxis::Y, true, [0; 4]).unwrap();
        let negative = CompactQuad::new([4, 7, 9], 3, 2, FaceAxis::Y, false, [0; 4]).unwrap();
        assert_eq!(positive.expanded_vertex(0), Some([3.5, 6.5, 8.5]));
        assert_eq!(positive.expanded_vertex(1), Some([3.5, 6.5, 11.5]));
        assert_eq!(positive.expanded_vertex(2), Some([5.5, 6.5, 11.5]));

        for quad in [positive, negative] {
            let a = quad.expanded_vertex(0).unwrap();
            let b = quad.expanded_vertex(1).unwrap();
            let c = quad.expanded_vertex(2).unwrap();
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[2].mul_add(-ac[1], ab[1] * ac[2]),
                ab[0].mul_add(-ac[2], ab[2] * ac[0]),
                ab[1].mul_add(-ac[0], ab[0] * ac[1]),
            ];
            assert!(
                cross
                    .into_iter()
                    .zip(quad.normal().map(f64::from))
                    .all(|(actual, expected)| expected.mul_add(-6.0, actual).abs() < f64::EPSILON)
            );
        }
    }

    #[test]
    fn compact_quad_packs_corner_occlusion_and_selects_the_lower_error_diagonal() {
        let quad = CompactQuad::new([4, 7, 9], 3, 2, FaceAxis::Y, true, [3, 0, 1, 2]).unwrap();
        assert_eq!(
            std::array::from_fn(|corner| quad.corner_occlusion(corner).unwrap()),
            [3, 0, 1, 2]
        );
        assert!(quad.flipped_diagonal());
        assert_eq!(quad.expanded_vertex(2), Some([5.5, 6.5, 8.5]));
        assert!(CompactQuad::new([0; 3], 1, 1, FaceAxis::Z, true, [0, 1, 2, 4]).is_none());
    }

    #[test]
    fn packed_leak_distance_matches_snorm16x4_lane_encoding() {
        let packed = PackedLeakDistance::new([0, 7, i16::MAX as u16, u16::MAX]).unwrap();

        assert_eq!(size_of::<PackedLeakDistance>(), 8);
        assert_eq!(packed.words(), [0x0007_0000, 0x8000_7fff]);
        assert_eq!(
            std::array::from_fn(|corner| packed.corner_bits(corner).unwrap()),
            [0, 7, i16::MAX as u16, i16::MIN as u16]
        );
        assert_eq!(packed.corner_bits(4), None);
        assert!(
            PackedLeakDistance::new([i16::MAX as u16 + 1, 0, 0, 0]).is_err(),
            "values between the connected range and sentinel are not representable"
        );
    }

    #[test]
    fn leak_distances_are_packed_from_each_quad_outside_corner() {
        let chunk_origin = [10, 20, 30];
        let halo_dimensions = [5, 5, 5];
        let mut halo_steps = vec![u16::MAX; 125];
        let halo_index = |[x, y, z]: [usize; 3]| x + 5 * (y + 5 * z);
        for (coordinate, distance) in [
            ([2, 2, 2], 0),
            ([3, 2, 2], 7),
            ([3, 3, 2], i16::MAX as u16),
            ([2, 3, 2], u16::MAX),
        ] {
            halo_steps[halo_index(coordinate)] = distance;
        }
        let quads = [
            CompactQuad::new([11, 21, 31], 2, 2, FaceAxis::Z, true, [0; 4]).unwrap(),
            CompactQuad::new([10, 20, 30], 1, 1, FaceAxis::Z, false, [0; 4]).unwrap(),
        ];

        let packed =
            pack_leak_distances(&quads, chunk_origin, halo_dimensions, &halo_steps).unwrap();

        assert_eq!(packed.len(), quads.len());
        assert_eq!(packed[0].words(), [0x0007_0000, 0x8000_7fff]);
        assert_eq!(packed[1].words(), [0x8000_8000; 2]);
        assert!(
            pack_leak_distances(
                &quads,
                chunk_origin,
                halo_dimensions,
                &halo_steps[..halo_steps.len() - 1],
            )
            .is_err()
        );
    }

    #[test]
    fn corner_occlusion_counts_outward_side_and_diagonal_voxels() {
        let level = |samples: [bool; 3]| {
            let dimensions = [3, 3, 3];
            let mut occupancy = vec![0; 27];
            for (occupied, coordinate) in
                samples
                    .into_iter()
                    .zip([[2_u32, 1, 1], [1, 2, 1], [2, 2, 1]])
            {
                if occupied {
                    occupancy[(coordinate[0]
                        + dimensions[0] * (coordinate[1] + dimensions[1] * coordinate[2]))
                        as usize] = 1;
                }
            }
            let input = input(dimensions, &occupancy, ChunkCoord::new(0, 0, 0));
            corner_occlusion(&input, [2, 1, 1], [1, 2, 1], [2, 2, 1])
        };
        assert_eq!(level([false, false, false]), 0);
        assert_eq!(level([false, false, true]), 1);
        assert_eq!(level([true, false, true]), 2);
        assert_eq!(level([true, true, false]), 3);
    }

    #[test]
    fn direct_corner_occlusion_addressing_matches_coordinate_reference() {
        let dimensions = [7, 6, 5];
        let mut occupancy = vec![0; 7 * 6 * 5];
        let mut state = 0x3141_5926_u32;
        for voxel in &mut occupancy {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *voxel = u8::from(state & 3 == 0);
        }
        let input = input(dimensions, &occupancy, ChunkCoord::new(0, 0, 0));
        let halo_width = input.halo_dimensions[0] as usize;
        let halo_strides = [
            1,
            halo_width,
            halo_width * input.halo_dimensions[1] as usize,
        ];

        for normal_axis in [FaceAxis::X, FaceAxis::Y, FaceAxis::Z] {
            let normal_axis_index = normal_axis.index();
            let [u_axis, v_axis] = normal_axis.tangent_axes();
            let layout_base = PlaneLayout {
                normal_axis,
                normal_axis_index,
                u_axis,
                v_axis,
                u_extent: input.extent[u_axis] as usize,
                v_extent: input.extent[v_axis] as usize,
                normal_extent: input.extent[normal_axis_index] as usize,
                plane: 0,
                halo_strides,
            };
            for plane in 0..=layout_base.normal_extent {
                let layout = PlaneLayout {
                    plane,
                    ..layout_base
                };
                for v in 0..layout.v_extent {
                    for u in 0..layout.u_extent {
                        for height in 1..=layout.v_extent - v {
                            for width in 1..=layout.u_extent - u {
                                for positive in [false, true] {
                                    let actual = quad_corner_occlusion(
                                        &input, layout, u, v, width, height, positive,
                                    );
                                    let mut expected = [0; 4];
                                    for (corner, [maximum_u, maximum_v]) in
                                        [[false, false], [true, false], [true, true], [false, true]]
                                            .into_iter()
                                            .enumerate()
                                    {
                                        let mut outside = [0_i32; 3];
                                        outside[normal_axis_index] =
                                            plane as i32 - i32::from(!positive);
                                        outside[u_axis] =
                                            (u + usize::from(maximum_u) * (width - 1)) as i32;
                                        outside[v_axis] =
                                            (v + usize::from(maximum_v) * (height - 1)) as i32;
                                        let u_direction = if maximum_u { 1 } else { -1 };
                                        let v_direction = if maximum_v { 1 } else { -1 };
                                        let mut side_u = outside;
                                        side_u[u_axis] += u_direction;
                                        let mut side_v = outside;
                                        side_v[v_axis] += v_direction;
                                        let mut diagonal = side_u;
                                        diagonal[v_axis] += v_direction;
                                        expected[corner] =
                                            corner_occlusion(&input, side_u, side_v, diagonal);
                                    }
                                    assert_eq!(actual, expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_edit_invalidates_every_chunk_consuming_its_halo() {
        let invalidated = invalidated_chunks_for_voxel([96, 96, 96], [32, 63, 31]);
        assert_eq!(invalidated.len(), 8);
        for x in [0, 1] {
            for y in [1, 2] {
                for z in [0, 1] {
                    assert!(invalidated.contains(&ChunkCoord::new(x, y, z)));
                }
            }
        }

        let partial_edge = invalidated_chunks_for_voxel([33, 33, 33], [32, 32, 32]);
        assert_eq!(partial_edge.len(), 8);
        for x in [0, 1] {
            for y in [0, 1] {
                for z in [0, 1] {
                    assert!(partial_edge.contains(&ChunkCoord::new(x, y, z)));
                }
            }
        }
    }

    #[test]
    fn linear_indices_use_x_major_volume_order_for_invalidation() {
        let dimensions = [64, 64, 2];
        let index = 31 + 64 * (32 + 64);
        let invalidated = invalidated_chunks_for_indices(dimensions, [index]);
        assert_eq!(invalidated.len(), 4);
        assert!(invalidated.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(1, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(0, 0, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(1, 0, 0)));
    }

    #[test]
    fn compact_spans_match_expanded_index_invalidation() {
        let dimensions = [67, 65, 34];
        let volume = dimensions.into_iter().product::<u32>();
        let spans = [
            (30, 5),
            (67 * 31 + 30, 8),
            (67 * 65 * 31 + 67 * 31 + 30, 140),
            (volume - 70, 100),
        ];
        let expanded = spans.into_iter().flat_map(|(start, length)| {
            let end = start.saturating_add(length).min(volume);
            start.min(volume)..end
        });
        assert_eq!(
            invalidated_chunks_for_index_spans(dimensions, spans),
            invalidated_chunks_for_indices(dimensions, expanded)
        );
    }

    #[test]
    fn compact_span_invalidation_does_not_expand_voxel_runs() {
        let invalidated = invalidated_chunks_for_index_spans([1_000_000, 1, 1], [(0, 1_000_000)]);
        assert_eq!(invalidated.len(), 31_250);
        assert_eq!(invalidated.first(), Some(&ChunkCoord::new(0, 0, 0)));
        assert_eq!(invalidated.last(), Some(&ChunkCoord::new(31_249, 0, 0)));
    }

    #[test]
    fn cancellation_stops_copy_and_meshing() {
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            ChunkInput::copy_from_volume([1, 1, 1], &[1], ChunkCoord::new(0, 0, 0), &cancellation),
            Err(AppError::Cancelled)
        ));

        let copied = input([1, 1, 1], &[1], ChunkCoord::new(0, 0, 0));
        assert!(matches!(
            mesh_chunk(&copied, tag(copied.coord()), &cancellation),
            Err(AppError::Cancelled)
        ));
    }

    #[test]
    fn stale_results_check_epoch_revision_and_chunk_generation() {
        let tag = tag(ChunkCoord::new(2, 1, 0));
        assert!(!tag.is_stale(7, 11, 3));
        assert!(tag.is_stale(8, 11, 3));
        assert!(tag.is_stale(7, 12, 3));
        assert!(tag.is_stale(7, 11, 4));
    }

    #[test]
    fn queue_is_bounded_to_twice_the_worker_count() {
        let workers = NonZeroUsize::new(3).unwrap();
        let (sender, receiver) = bounded_mesh_queue(workers);
        for value in 0..6 {
            sender.try_send(value).unwrap();
        }
        assert!(sender.try_send(6).is_err());
        assert_eq!(receiver.len(), mesh_queue_capacity(workers));
    }

    #[test]
    fn checkerboard_is_pathological_but_bounded_per_chunk() {
        let mut occupancy = vec![0; CHUNK_VOXELS];
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    if (x + y + z) % 2 == 0 {
                        occupancy[x + 32 * (y + 32 * z)] = 1;
                    }
                }
            }
        }
        let mesh = mesh([32, 32, 32], &occupancy, ChunkCoord::new(0, 0, 0));
        assert_eq!(mesh.exposed_voxel_faces, 98_304);
        assert_eq!(mesh.quads.len(), 98_304);
        assert_eq!(mesh.gpu_bytes(), 1_572_864);

        let cancellation = CancellationToken::default();
        let mut budget = MeshMemoryBudget::default();
        let accepted = MAX_MESH_BYTES / mesh.gpu_bytes();
        for index in 0..accepted {
            budget
                .try_replace_mesh(ChunkCoord::new(u32::try_from(index).unwrap(), 0, 0), &mesh)
                .unwrap();
        }
        let before = budget.used_bytes();
        assert!(matches!(
            budget.try_replace_mesh_or_cancel(
                ChunkCoord::new(u32::try_from(accepted).unwrap(), 0, 0),
                &mesh,
                &cancellation
            ),
            Err(MeshBudgetError::Limit { .. })
        ));
        assert_eq!(budget.used_bytes(), before);
        assert!(cancellation.is_cancelled());
    }

    #[test]
    fn noisy_volume_surface_area_matches_naive_reference() {
        let dimensions = [17, 13, 11];
        let mut occupancy = vec![0; 17 * 13 * 11];
        let mut state = 0x1234_5678_u32;
        for voxel in &mut occupancy {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *voxel = u8::from(state.trailing_zeros() >= 2);
        }
        let expected = naive_exposed_faces(dimensions, &occupancy);
        let mesh = mesh(dimensions, &occupancy, ChunkCoord::new(0, 0, 0));
        assert_eq!(mesh.exposed_voxel_faces, expected);
        assert_eq!(
            mesh.quads.iter().map(|quad| quad.area()).sum::<u32>(),
            expected
        );
        assert!(mesh.quads.len() < expected as usize);
    }

    #[test]
    fn budget_replacement_and_removal_are_accounted_atomically() {
        let chunk = ChunkCoord::new(1, 2, 3);
        let mut budget = MeshMemoryBudget::new(100);
        budget.try_replace(chunk, 80).unwrap();
        assert_eq!(budget.used_bytes(), 80);
        assert!(budget.try_replace(chunk, 101).is_err());
        assert_eq!(budget.used_bytes(), 80);
        budget.try_replace(chunk, 40).unwrap();
        assert_eq!(budget.remaining_bytes(), 60);
        assert_eq!(budget.remove(chunk), 40);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn mesh_budget_reservation_preserves_rollback_capacity() {
        let edited = ChunkCoord::new(0, 0, 0);
        let background = ChunkCoord::new(1, 0, 0);
        let mut budget = MeshMemoryBudget::new(150);
        budget.try_replace(edited, 100).unwrap();
        budget.reserve_current_if_absent(edited).unwrap();

        budget.try_replace(edited, 40).unwrap();
        assert_eq!(budget.used_bytes(), 100);
        assert!(matches!(
            budget.try_replace(background, 60),
            Err(MeshBudgetError::Limit { .. })
        ));

        assert_eq!(budget.release_reservation(edited), Some(100));
        assert_eq!(budget.used_bytes(), 40);
        budget.try_replace(background, 60).unwrap();
        assert_eq!(budget.used_bytes(), 100);
    }
}
