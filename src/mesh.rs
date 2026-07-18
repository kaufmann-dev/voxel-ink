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
const QUAD_RESERVE_BLOCK: usize = 1_024;

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
                for local_x in 0..halo_dimensions[0] {
                    let global_x =
                        i64::from(origin[0]) + i64::from(local_x) - i64::from(HALO_WIDTH);
                    if !(0..i64::from(dimensions[0])).contains(&global_x) {
                        continue;
                    }
                    let global_x = usize::try_from(global_x).map_err(|_| {
                        AppError::InvalidDimensions(
                            "negative x coordinate escaped halo clipping".into(),
                        )
                    })?;
                    let source = global_x
                        + dimensions[0] as usize * (global_y + dimensions[1] as usize * global_z);
                    let destination = halo_index(halo_dimensions, [local_x, local_y, local_z]);
                    copied[destination] = occupancy[source];
                }
            }
        }

        Ok(Self {
            coord,
            origin,
            extent,
            halo_dimensions,
            occupancy: copied,
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

    fn occupied(&self, local: [i32; 3]) -> bool {
        debug_assert!(
            local
                .iter()
                .zip(self.extent)
                .all(|(&coordinate, extent)| (-1..=extent.cast_signed()).contains(&coordinate))
        );
        let halo = local.map(|coordinate| (coordinate + HALO_WIDTH.cast_signed()).cast_unsigned());
        self.occupancy[halo_index(self.halo_dimensions, halo)] != 0
    }
}

const fn halo_index(dimensions: [u32; 3], coordinate: [u32; 3]) -> usize {
    coordinate[0] as usize
        + dimensions[0] as usize
            * (coordinate[1] as usize + dimensions[1] as usize * coordinate[2] as usize)
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

impl CompactQuad {
    const LENGTH_MASK: u32 = 0x1f;
    const V_LENGTH_SHIFT: u32 = 5;
    const AXIS_SHIFT: u32 = 10;
    const POSITIVE_SHIFT: u32 = 12;

    #[must_use]
    pub fn new(
        origin: [u32; 3],
        u_length: u32,
        v_length: u32,
        normal_axis: FaceAxis,
        positive: bool,
    ) -> Option<Self> {
        if !(1..=CHUNK_SIZE).contains(&u_length) || !(1..=CHUNK_SIZE).contains(&v_length) {
            return None;
        }
        let packed = (u_length - 1)
            | ((v_length - 1) << Self::V_LENGTH_SHIFT)
            | (normal_axis.code() << Self::AXIS_SHIFT)
            | (u32::from(positive) << Self::POSITIVE_SHIFT);
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
        let corner = match (self.positive(), vertex_index) {
            (true | false, 0 | 3) => 0,
            (true, 1) | (false, 5) => 1,
            (true | false, 2 | 4) => 2,
            (true, 5) | (false, 1) => 3,
            _ => return None,
        };
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshJobTag {
    pub document_epoch: u64,
    pub revision: u64,
    pub chunk_generation: u64,
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
        };
        let active_mask = &mut mask[..u_extent * v_extent];
        fill_face_mask(input, layout, active_mask)?;
        emit_greedy_quads(input, layout, active_mask, quads, exposed_voxel_faces)?;
    }
    Ok(())
}

fn fill_face_mask(input: &ChunkInput, layout: PlaneLayout, mask: &mut [i8]) -> Result<()> {
    mask.fill(0);
    for v in 0..layout.v_extent {
        for u in 0..layout.u_extent {
            let mut positive_voxel = [0_i32; 3];
            positive_voxel[layout.normal_axis_index] = chunk_i32(layout.plane)?;
            positive_voxel[layout.u_axis] = chunk_i32(u)?;
            positive_voxel[layout.v_axis] = chunk_i32(v)?;
            let mut negative_voxel = positive_voxel;
            negative_voxel[layout.normal_axis_index] -= 1;
            let occupied_negative = input.occupied(negative_voxel);
            let occupied_positive = input.occupied(positive_voxel);

            mask[v * layout.u_extent + u] = match (occupied_negative, occupied_positive) {
                // A neighboring chunk owns a face when the occupied voxel is
                // in that chunk, even though it appears in this chunk's halo.
                (true, false) if layout.plane > 0 => 1,
                (false, true) if layout.plane < layout.normal_extent => -1,
                _ => 0,
            };
        }
    }
    Ok(())
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
            let quad = CompactQuad::new(
                origin,
                chunk_u32(width)?,
                chunk_u32(height)?,
                layout.normal_axis,
                face > 0,
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

fn chunk_i32(value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| AppError::InvalidDimensions("chunk coordinate exceeds i32".into()))
}

fn chunk_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| AppError::InvalidDimensions("chunk coordinate exceeds u32".into()))
}

fn try_push_quad(quads: &mut Vec<CompactQuad>, quad: CompactQuad) -> Result<()> {
    if quads.len() == quads.capacity() {
        let desired_capacity = quads
            .len()
            .checked_add(QUAD_RESERVE_BLOCK)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let bytes = desired_capacity
            .checked_mul(size_of::<CompactQuad>())
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        quads
            .try_reserve_exact(QUAD_RESERVE_BLOCK)
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

    for owner_x in first_x..=last_x {
        invalidated.insert(ChunkCoord::new(owner_x, owner_y, owner_z));
    }
    if x_start.is_multiple_of(CHUNK_SIZE) && first_x > 0 {
        invalidated.insert(ChunkCoord::new(first_x - 1, owner_y, owner_z));
    }
    if x_end.is_multiple_of(CHUNK_SIZE) && last_x + 1 < grid[0] {
        invalidated.insert(ChunkCoord::new(last_x + 1, owner_y, owner_z));
    }
    if y.is_multiple_of(CHUNK_SIZE) && owner_y > 0 {
        for owner_x in first_x..=last_x {
            invalidated.insert(ChunkCoord::new(owner_x, owner_y - 1, owner_z));
        }
    }
    if y % CHUNK_SIZE == CHUNK_SIZE - 1 && owner_y + 1 < grid[1] {
        for owner_x in first_x..=last_x {
            invalidated.insert(ChunkCoord::new(owner_x, owner_y + 1, owner_z));
        }
    }
    if z.is_multiple_of(CHUNK_SIZE) && owner_z > 0 {
        for owner_x in first_x..=last_x {
            invalidated.insert(ChunkCoord::new(owner_x, owner_y, owner_z - 1));
        }
    }
    if z % CHUNK_SIZE == CHUNK_SIZE - 1 && owner_z + 1 < grid[2] {
        for owner_x in first_x..=last_x {
            invalidated.insert(ChunkCoord::new(owner_x, owner_y, owner_z + 1));
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
    let grid = chunk_grid(dimensions);
    let owner = ChunkCoord::from_voxel(voxel).as_array();
    invalidated.insert(ChunkCoord::new(owner[0], owner[1], owner[2]));
    for axis in 0..3 {
        if voxel[axis].is_multiple_of(CHUNK_SIZE) && owner[axis] > 0 {
            let mut neighbor = owner;
            neighbor[axis] -= 1;
            invalidated.insert(ChunkCoord::new(neighbor[0], neighbor[1], neighbor[2]));
        }
        if voxel[axis] % CHUNK_SIZE == CHUNK_SIZE - 1 && owner[axis] + 1 < grid[axis] {
            let mut neighbor = owner;
            neighbor[axis] += 1;
            invalidated.insert(ChunkCoord::new(neighbor[0], neighbor[1], neighbor[2]));
        }
    }
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
        let retained_bytes = self.used - previous_bytes;
        let next_bytes =
            retained_bytes
                .checked_add(replacement_bytes)
                .ok_or(MeshBudgetError::Limit {
                    cap_bytes: self.cap,
                    retained_bytes,
                    replacement_bytes,
                })?;
        if next_bytes > self.cap {
            return Err(MeshBudgetError::Limit {
                cap_bytes: self.cap,
                retained_bytes,
                replacement_bytes,
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
        self.used -= removed;
        removed
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
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
    fn greedy_mesher_collapses_solid_chunk_to_six_quads() {
        let mesh = mesh(
            [32, 32, 32],
            &vec![1; CHUNK_VOXELS],
            ChunkCoord::new(0, 0, 0),
        );
        assert_eq!(mesh.quads.len(), 6);
        assert_eq!(mesh.exposed_voxel_faces, 6 * 32 * 32);
        assert!(
            mesh.quads
                .iter()
                .all(|quad| quad.u_length() == 32 && quad.v_length() == 32)
        );
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
        let positive = CompactQuad::new([4, 7, 9], 3, 2, FaceAxis::Y, true).unwrap();
        let negative = CompactQuad::new([4, 7, 9], 3, 2, FaceAxis::Y, false).unwrap();
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
    fn boundary_edit_invalidates_each_face_neighbor() {
        let invalidated = invalidated_chunks_for_voxel([96, 96, 96], [32, 63, 31]);
        assert_eq!(invalidated.len(), 4);
        assert!(invalidated.contains(&ChunkCoord::new(1, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(1, 2, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(1, 1, 1)));

        let partial_edge = invalidated_chunks_for_voxel([33, 33, 33], [32, 32, 32]);
        assert_eq!(partial_edge.len(), 4);
        assert!(partial_edge.contains(&ChunkCoord::new(0, 1, 1)));
        assert!(partial_edge.contains(&ChunkCoord::new(1, 0, 1)));
        assert!(partial_edge.contains(&ChunkCoord::new(1, 1, 0)));
    }

    #[test]
    fn linear_indices_use_x_major_volume_order_for_invalidation() {
        let dimensions = [64, 64, 2];
        let index = 31 + 64 * (32 + 64);
        let invalidated = invalidated_chunks_for_indices(dimensions, [index]);
        assert_eq!(invalidated.len(), 3);
        assert!(invalidated.contains(&ChunkCoord::new(0, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(1, 1, 0)));
        assert!(invalidated.contains(&ChunkCoord::new(0, 0, 0)));
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
}
