use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    time::SystemTime,
};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{CodecSaveState, SourceVariant},
    error::checked_voxel_count,
};

pub const HISTORY_TRANSACTIONS: usize = 256;
pub const HISTORY_BYTES: usize = 128 * 1024 * 1024;
const EXTERIOR_STACK_BYTES: usize = 64 * 1024 * 1024;
const LEAK_DISTANCE_QUEUE_BYTES: usize = 64 * 1024 * 1024;
const CANCELLATION_CHECK_INTERVAL: usize = 64 * 1024;
const MAX_LEAK_DISTANCE_STEPS: u16 = i16::MAX as u16;
const SEALED_LEAK_DISTANCE: u16 = u16::MAX;
const LEAK_DISTANCE_DIAGONALS: f64 = 8.0;

pub type Matrix4 = [[f64; 4]; 4];

/// A compact, read-only view of the document's live binary occupancy.
///
/// The document keeps this one-bit-per-voxel mask synchronized with its byte
/// storage under a write lock. Consumers use [`Self::try_read`] so latency
/// sensitive work can conservatively defer rather than block during an edit.
#[derive(Clone)]
pub struct SharedOccupancy {
    dimensions: [u32; 3],
    mask: Arc<RwLock<VoxelBitSet>>,
}

impl std::fmt::Debug for SharedOccupancy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedOccupancy")
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl SharedOccupancy {
    pub(crate) fn from_binary(dimensions: [u32; 3], occupancy: &[u8]) -> Result<(Self, u64)> {
        let mut mask = VoxelBitSet::zeroed(occupancy.len())?;
        let mut occupied_count = 0_u64;
        for (index, &value) in occupancy.iter().enumerate() {
            if value > 1 {
                return Err(AppError::InvalidDimensions(
                    "occupancy must contain only zero or one".into(),
                ));
            }
            if value == 1 {
                mask.insert(index);
                occupied_count += 1;
            }
        }
        Ok((
            Self {
                dimensions,
                mask: Arc::new(RwLock::new(mask)),
            },
            occupied_count,
        ))
    }

    #[must_use]
    pub const fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    /// Acquires a coherent occupancy snapshot without waiting for an edit.
    #[must_use]
    pub fn try_read(&self) -> Option<SharedOccupancyRead<'_>> {
        self.mask.try_read().ok().map(|mask| SharedOccupancyRead {
            dimensions: self.dimensions,
            mask,
        })
    }

    fn write(&self) -> RwLockWriteGuard<'_, VoxelBitSet> {
        self.mask.write().expect("shared occupancy mask poisoned")
    }
}

pub struct SharedOccupancyRead<'a> {
    dimensions: [u32; 3],
    mask: RwLockReadGuard<'a, VoxelBitSet>,
}

impl SharedOccupancyRead<'_> {
    #[must_use]
    pub const fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    #[must_use]
    pub fn occupied(&self, index: u32) -> Option<bool> {
        let index = usize::try_from(index).ok()?;
        (index < self.mask.len).then(|| self.mask.contains(index))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpatialUnits {
    #[default]
    Unknown,
    Meter,
    Millimeter,
    Micron,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeGeometry {
    pub dimensions: [u32; 3],
    pub index_to_world: Matrix4,
    pub world_to_index: Matrix4,
    pub spacing: [f64; 3],
    pub units: SpatialUnits,
}

impl VolumeGeometry {
    pub fn new(dimensions: [u32; 3], index_to_world: Matrix4, units: SpatialUnits) -> Result<Self> {
        checked_voxel_count(dimensions)?;
        if !index_to_world
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        {
            return Err(AppError::InvalidAffine);
        }
        let world_to_index = invert_matrix(index_to_world).ok_or(AppError::InvalidAffine)?;
        let spacing = [
            vector_length([
                index_to_world[0][0],
                index_to_world[1][0],
                index_to_world[2][0],
            ]),
            vector_length([
                index_to_world[0][1],
                index_to_world[1][1],
                index_to_world[2][1],
            ]),
            vector_length([
                index_to_world[0][2],
                index_to_world[1][2],
                index_to_world[2][2],
            ]),
        ];
        if spacing
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AppError::InvalidAffine);
        }
        Ok(Self {
            dimensions,
            index_to_world,
            world_to_index,
            spacing,
            units,
        })
    }

    #[must_use]
    pub fn index_point_to_world(&self, point: [f64; 3]) -> [f64; 3] {
        transform_point(self.index_to_world, point)
    }

    #[must_use]
    pub fn world_point_to_index(&self, point: [f64; 3]) -> [f64; 3] {
        transform_point(self.world_to_index, point)
    }

    #[must_use]
    pub fn world_vector_to_index(&self, vector: [f64; 3]) -> [f64; 3] {
        transform_vector(self.world_to_index, vector)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceIdentity {
    pub canonical_path: PathBuf,
    pub size: u64,
    pub modified: SystemTime,
    pub sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Tool {
    #[default]
    Inspect,
    Paint,
    Erase,
    Bridge,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SliceAxis {
    Axial,
    Coronal,
    Sagittal,
}

impl SliceAxis {
    #[must_use]
    pub const fn normal_axis(self) -> usize {
        match self {
            Self::Sagittal => 0,
            Self::Coronal => 1,
            Self::Axial => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DocumentSummary {
    pub dimensions: [u32; 3],
    pub spacing: [f64; 3],
    pub units: SpatialUnits,
    pub source_path: PathBuf,
    pub source_variant: SourceVariant,
    pub cursor: [u32; 3],
    pub occupied_count: u64,
    pub normalized_non_binary: bool,
    pub revision: u64,
    pub saved_revision: u64,
    pub can_undo: bool,
    pub can_redo: bool,
}

impl DocumentSummary {
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.revision != self.saved_revision
    }
}

#[derive(Clone, Copy)]
pub struct VolumeView<'a> {
    pub dimensions: [u32; 3],
    pub occupancy: &'a [u8],
    pub geometry: &'a VolumeGeometry,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeakDistanceField {
    steps: Vec<u16>,
    distance_scale: f64,
    degraded: bool,
}

impl LeakDistanceField {
    #[must_use]
    pub fn steps(&self) -> &[u16] {
        &self.steps
    }

    #[must_use]
    pub const fn distance_scale(&self) -> f64 {
        self.distance_scale
    }

    #[must_use]
    pub const fn degraded(&self) -> bool {
        self.degraded
    }
}

#[derive(Clone, Debug)]
struct Transaction {
    changes: TransactionChanges,
    value: u8,
    before_revision: u64,
    after_revision: u64,
}

impl Transaction {
    const fn bytes(&self) -> usize {
        self.changes.bytes()
    }

    fn voxel_count(&self) -> u64 {
        self.changes.voxel_count()
    }

    fn changed_voxels(&self) -> ChangedVoxels<'_> {
        match &self.changes {
            TransactionChanges::Indices(indices) => ChangedVoxels::Indices(indices),
            TransactionChanges::Spans(spans) => ChangedVoxels::Spans(spans),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ChangedVoxels<'a> {
    Indices(&'a [u32]),
    Spans(&'a [IndexSpan]),
}

#[derive(Clone, Debug)]
enum TransactionChanges {
    Indices(Vec<u32>),
    Spans(Vec<IndexSpan>),
}

impl TransactionChanges {
    const fn bytes(&self) -> usize {
        match self {
            Self::Indices(indices) => indices.len() * size_of::<u32>(),
            Self::Spans(spans) => spans.len() * size_of::<IndexSpan>(),
        }
    }

    fn voxel_count(&self) -> u64 {
        match self {
            Self::Indices(indices) => indices.len() as u64,
            Self::Spans(spans) => spans.iter().map(|span| u64::from(span.len)).sum(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexSpan {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

#[derive(Debug)]
struct ActiveStroke {
    indices: Vec<u32>,
    seen: HashSet<u32>,
    value: u8,
    before_revision: u64,
}

#[derive(Debug)]
pub struct VolumeDocument {
    geometry: VolumeGeometry,
    occupancy: Vec<u8>,
    shared_occupancy: SharedOccupancy,
    source_identity: SourceIdentity,
    source_variant: SourceVariant,
    save_state: CodecSaveState,
    cursor: [u32; 3],
    occupied_count: u64,
    normalized_non_binary: bool,
    revision: u64,
    saved_revision: u64,
    next_revision: u64,
    undo: Vec<Transaction>,
    redo: Vec<Transaction>,
    history_bytes: usize,
    active_stroke: Option<ActiveStroke>,
}

impl VolumeDocument {
    pub fn new(
        geometry: VolumeGeometry,
        occupancy: Vec<u8>,
        source_identity: SourceIdentity,
        source_variant: SourceVariant,
        save_state: CodecSaveState,
        normalized_non_binary: bool,
    ) -> Result<Self> {
        let count = checked_voxel_count(geometry.dimensions)?;
        if occupancy.len() != count {
            return Err(AppError::InvalidDimensions(format!(
                "expected {count} occupancy bytes, got {}",
                occupancy.len()
            )));
        }
        let (shared_occupancy, occupied_count) =
            SharedOccupancy::from_binary(geometry.dimensions, &occupancy)?;
        let revision = 1;
        Ok(Self {
            cursor: geometry.dimensions.map(|axis| axis / 2),
            geometry,
            occupancy,
            shared_occupancy,
            source_identity,
            source_variant,
            save_state,
            occupied_count,
            normalized_non_binary,
            revision,
            saved_revision: if normalized_non_binary { 0 } else { revision },
            next_revision: revision + 1,
            undo: Vec::new(),
            redo: Vec::new(),
            history_bytes: 0,
            active_stroke: None,
        })
    }

    #[must_use]
    pub fn summary(&self) -> DocumentSummary {
        DocumentSummary {
            dimensions: self.geometry.dimensions,
            spacing: self.geometry.spacing,
            units: self.geometry.units,
            source_path: self.source_identity.canonical_path.clone(),
            source_variant: self.source_variant,
            cursor: self.cursor,
            occupied_count: self.occupied_count,
            normalized_non_binary: self.normalized_non_binary,
            revision: self.revision,
            saved_revision: self.saved_revision,
            can_undo: !self.undo.is_empty(),
            can_redo: !self.redo.is_empty(),
        }
    }

    #[must_use]
    pub const fn geometry(&self) -> &VolumeGeometry {
        &self.geometry
    }

    #[must_use]
    pub const fn source_identity(&self) -> &SourceIdentity {
        &self.source_identity
    }

    #[must_use]
    pub const fn source_variant(&self) -> SourceVariant {
        self.source_variant
    }

    #[must_use]
    pub const fn save_state(&self) -> &CodecSaveState {
        &self.save_state
    }

    #[must_use]
    pub fn view(&self) -> VolumeView<'_> {
        VolumeView {
            dimensions: self.geometry.dimensions,
            occupancy: &self.occupancy,
            geometry: &self.geometry,
        }
    }

    #[must_use]
    pub fn shared_occupancy(&self) -> SharedOccupancy {
        self.shared_occupancy.clone()
    }

    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    #[must_use]
    pub const fn saved_revision(&self) -> u64 {
        self.saved_revision
    }

    #[must_use]
    pub const fn next_revision(&self) -> u64 {
        self.next_revision
    }

    #[must_use]
    pub const fn occupied_count(&self) -> u64 {
        self.occupied_count
    }

    #[must_use]
    pub const fn cursor(&self) -> [u32; 3] {
        self.cursor
    }

    pub fn set_cursor(&mut self, cursor: [u32; 3]) {
        self.cursor = [
            cursor[0].min(self.geometry.dimensions[0] - 1),
            cursor[1].min(self.geometry.dimensions[1] - 1),
            cursor[2].min(self.geometry.dimensions[2] - 1),
        ];
    }

    pub fn begin_stroke(&mut self, value: u8) -> Result<()> {
        if value > 1 {
            return Err(AppError::InvalidDimensions(
                "stroke value must be zero or one".into(),
            ));
        }
        if self.active_stroke.is_some() {
            self.cancel_stroke();
        }
        self.active_stroke = Some(ActiveStroke {
            indices: Vec::new(),
            seen: HashSet::new(),
            value,
            before_revision: self.revision,
        });
        Ok(())
    }

    pub fn apply_stroke_indices(
        &mut self,
        indices: impl IntoIterator<Item = u32>,
    ) -> Result<usize> {
        let stroke = self.active_stroke.as_mut().ok_or_else(|| {
            AppError::InvalidDimensions("a stroke must begin before samples are applied".into())
        })?;
        let mut shared_occupancy = self.shared_occupancy.write();
        let mut changed = 0;
        for index in indices {
            let offset = usize::try_from(index).map_err(|_| {
                AppError::InvalidDimensions("stroke index does not fit in memory".into())
            })?;
            let Some(cell) = self.occupancy.get_mut(offset) else {
                return Err(AppError::InvalidDimensions(format!(
                    "stroke index {index} is outside the volume"
                )));
            };
            // Every index changed by this stroke already contains `stroke.value`,
            // so the occupancy check also rejects repeated indices without a
            // second hash-table lookup.
            if *cell == stroke.value {
                continue;
            }
            if stroke.indices.len() >= HISTORY_BYTES / size_of::<u32>() {
                return Err(AppError::Allocation {
                    bytes: HISTORY_BYTES + size_of::<u32>(),
                });
            }
            if stroke.indices.len() == stroke.indices.capacity() {
                stroke
                    .indices
                    .try_reserve(1024)
                    .map_err(|_| AppError::Allocation {
                        bytes: (stroke.indices.len() + 1024) * size_of::<u32>(),
                    })?;
            }
            if stroke.seen.len() == stroke.seen.capacity() {
                stroke
                    .seen
                    .try_reserve(1024)
                    .map_err(|_| AppError::Allocation {
                        bytes: (stroke.seen.len() + 1024) * (size_of::<u32>() + size_of::<usize>()),
                    })?;
            }
            stroke.seen.insert(index);
            if stroke.value == 1 {
                self.occupied_count += 1;
            } else {
                self.occupied_count -= 1;
            }
            *cell = stroke.value;
            shared_occupancy.set(offset, stroke.value == 1);
            stroke.indices.push(index);
            changed += 1;
        }
        Ok(changed)
    }

    #[must_use]
    pub(crate) fn active_stroke_indices(&self) -> Option<&[u32]> {
        self.active_stroke
            .as_ref()
            .map(|stroke| stroke.indices.as_slice())
    }

    pub(crate) fn stroke_start_occupied(&self, index: u32) -> Option<bool> {
        let current = self
            .occupancy
            .get(usize::try_from(index).ok()?)
            .map(|&value| value == 1)?;
        Some(self.active_stroke.as_ref().map_or(current, |stroke| {
            if stroke.seen.contains(&index) {
                stroke.value == 0
            } else {
                current
            }
        }))
    }

    pub fn finish_stroke(&mut self) -> Option<u64> {
        let stroke = self.active_stroke.take()?;
        if stroke.indices.is_empty() {
            return None;
        }
        let after_revision = self.allocate_revision();
        let transaction = Transaction {
            changes: TransactionChanges::Indices(stroke.indices),
            value: stroke.value,
            before_revision: stroke.before_revision,
            after_revision,
        };
        self.history_bytes += transaction.bytes();
        self.undo.push(transaction);
        self.history_bytes -= self.redo.iter().map(Transaction::bytes).sum::<usize>();
        self.redo.clear();
        self.revision = after_revision;
        self.trim_history();
        Some(after_revision)
    }

    pub fn cancel_stroke(&mut self) {
        let Some(stroke) = self.active_stroke.take() else {
            return;
        };
        let old_value = 1 - stroke.value;
        let mut shared_occupancy = self.shared_occupancy.write();
        for index in stroke.indices {
            self.occupancy[index as usize] = old_value;
            shared_occupancy.set(index as usize, old_value == 1);
            if old_value == 1 {
                self.occupied_count += 1;
            } else {
                self.occupied_count -= 1;
            }
        }
    }

    /// Fill every empty voxel that cannot reach a volume boundary through a
    /// six-connected path. The volume is not changed until classification and
    /// undo history construction have both completed successfully.
    pub fn fill_enclosed_cavities(&mut self, cancel: &CancellationToken) -> Result<u64> {
        self.cancel_stroke();
        cancel.check()?;

        let dimensions = self.geometry.dimensions.map(|axis| axis as usize);
        let voxel_count = self.occupancy.len() as u64;
        // Empty and full masks are common import states, and a volume thinner
        // than three voxels has no voxel that is interior on every axis.
        if self.occupied_count == 0
            || self.occupied_count == voxel_count
            || dimensions.contains(&1)
            || dimensions.contains(&2)
        {
            return Ok(0);
        }
        let exterior = classify_exterior_empty(&self.occupancy, dimensions, cancel)?;

        let max_spans = HISTORY_BYTES / size_of::<IndexSpan>();
        let mut spans: Vec<IndexSpan> = Vec::new();
        let initial_capacity = 64 * 1024;
        spans
            .try_reserve(initial_capacity)
            .map_err(|_| AppError::Allocation {
                bytes: initial_capacity * size_of::<IndexSpan>(),
            })?;
        for (index, &value) in self.occupancy.iter().enumerate() {
            if index.is_multiple_of(64 * 1024) {
                cancel.check()?;
            }
            if value != 0 || exterior.contains(index) {
                continue;
            }
            let index = index as u32;
            if let Some(last) = spans.last_mut()
                && last.start.checked_add(last.len) == Some(index)
            {
                last.len += 1;
                continue;
            }
            if spans.len() == max_spans {
                return Err(AppError::UndoHistoryLimit {
                    bytes: HISTORY_BYTES + size_of::<IndexSpan>(),
                    limit: HISTORY_BYTES,
                });
            }
            if spans.len() == spans.capacity() {
                let additional = (64 * 1024).min(max_spans - spans.len());
                spans
                    .try_reserve(additional)
                    .map_err(|_| AppError::Allocation {
                        bytes: (spans.len() + additional) * size_of::<IndexSpan>(),
                    })?;
            }
            spans.push(IndexSpan {
                start: index,
                len: 1,
            });
        }

        cancel.check()?;
        if spans.is_empty() {
            return Ok(0);
        }
        let added = spans.iter().map(|span| u64::from(span.len)).sum();
        let mut shared_occupancy = self.shared_occupancy.write();
        for span in &spans {
            let start = span.start as usize;
            let end = start + span.len as usize;
            self.occupancy[start..end].fill(1);
            shared_occupancy.fill(start, end, true);
        }
        drop(shared_occupancy);
        self.occupied_count += added;
        let before_revision = self.revision;
        let after_revision = self.allocate_revision();
        let transaction = Transaction {
            changes: TransactionChanges::Spans(spans),
            value: 1,
            before_revision,
            after_revision,
        };
        self.history_bytes += transaction.bytes();
        self.undo.push(transaction);
        self.history_bytes -= self.redo.iter().map(Transaction::bytes).sum::<usize>();
        self.redo.clear();
        self.revision = after_revision;
        self.trim_history();
        Ok(added)
    }

    /// Computes reach-independent physical path distances from the volume boundary
    /// through six-connected empty voxels. `u16::MAX` denotes occupied or enclosed
    /// space; finite values are multiplied by the returned field's distance scale.
    pub fn leak_distance_field(&self, cancel: &CancellationToken) -> Result<LeakDistanceField> {
        cancel.check()?;
        compute_leak_distance(
            &self.occupancy,
            self.geometry.dimensions,
            &self.geometry.spacing,
            cancel,
        )
    }

    pub fn undo(&mut self) -> Option<u64> {
        self.cancel_stroke();
        let transaction = self.undo.pop()?;
        self.apply_transaction(&transaction, 1 - transaction.value);
        self.revision = transaction.before_revision;
        let revision = self.revision;
        self.redo.push(transaction);
        Some(revision)
    }

    pub fn redo(&mut self) -> Option<u64> {
        self.cancel_stroke();
        let transaction = self.redo.pop()?;
        self.apply_transaction(&transaction, transaction.value);
        self.revision = transaction.after_revision;
        let revision = self.revision;
        self.undo.push(transaction);
        Some(revision)
    }

    #[must_use]
    pub(crate) fn latest_undo_changes(&self) -> Option<ChangedVoxels<'_>> {
        self.undo.last().map(Transaction::changed_voxels)
    }

    #[must_use]
    pub(crate) fn latest_redo_changes(&self) -> Option<ChangedVoxels<'_>> {
        self.redo.last().map(Transaction::changed_voxels)
    }

    pub fn mark_saved(&mut self, identity: SourceIdentity, variant: SourceVariant, adopt: bool) {
        if adopt {
            self.source_identity = identity;
            self.source_variant = variant;
        }
        self.saved_revision = self.revision;
        self.normalized_non_binary = false;
    }

    #[must_use]
    pub fn linear_index(&self, voxel: [u32; 3]) -> Option<u32> {
        linear_index(self.geometry.dimensions, voxel)
    }

    #[must_use]
    pub fn value(&self, voxel: [u32; 3]) -> Option<u8> {
        self.linear_index(voxel)
            .map(|index| self.occupancy[index as usize])
    }

    #[must_use]
    pub fn extract_slice(&self, axis: SliceAxis, coordinate: u32) -> Option<SliceImage> {
        self.try_extract_slice(axis, coordinate).ok()
    }

    pub fn try_extract_slice(&self, axis: SliceAxis, coordinate: u32) -> Result<SliceImage> {
        let dimensions = self.geometry.dimensions;
        if coordinate >= dimensions[axis.normal_axis()] {
            return Err(AppError::InvalidDimensions(format!(
                "slice coordinate {coordinate} is outside the {axis:?} axis"
            )));
        }
        let (width, height) = match axis {
            SliceAxis::Axial => (dimensions[0], dimensions[1]),
            SliceAxis::Coronal => (dimensions[0], dimensions[2]),
            SliceAxis::Sagittal => (dimensions[1], dimensions[2]),
        };
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(pixel_count)
            .map_err(|_| AppError::Allocation { bytes: pixel_count })?;
        let [nx, ny, nz] = dimensions.map(|value| value as usize);
        let coordinate = coordinate as usize;
        let plane = nx * ny;
        match axis {
            SliceAxis::Axial => {
                let start = coordinate * plane;
                pixels.extend_from_slice(&self.occupancy[start..start + plane]);
            }
            SliceAxis::Coronal => {
                for z in 0..nz {
                    let start = z * plane + coordinate * nx;
                    pixels.extend_from_slice(&self.occupancy[start..start + nx]);
                }
            }
            SliceAxis::Sagittal => {
                for z in 0..nz {
                    let plane_start = z * plane;
                    for y in 0..ny {
                        pixels.push(self.occupancy[plane_start + y * nx + coordinate]);
                    }
                }
            }
        }
        Ok(SliceImage {
            axis,
            coordinate: coordinate as u32,
            width,
            height,
            pixels,
        })
    }

    fn allocate_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }

    fn apply_transaction(&mut self, transaction: &Transaction, value: u8) {
        let mut shared_occupancy = self.shared_occupancy.write();
        match &transaction.changes {
            TransactionChanges::Indices(indices) => {
                for &index in indices {
                    self.occupancy[index as usize] = value;
                    shared_occupancy.set(index as usize, value == 1);
                }
            }
            TransactionChanges::Spans(spans) => {
                for span in spans {
                    let start = span.start as usize;
                    let end = start + span.len as usize;
                    self.occupancy[start..end].fill(value);
                    shared_occupancy.fill(start, end, value == 1);
                }
            }
        }
        let count = transaction.voxel_count();
        if value == 1 {
            self.occupied_count += count;
        } else {
            self.occupied_count -= count;
        }
    }

    fn trim_history(&mut self) {
        while self.undo.len() > HISTORY_TRANSACTIONS || self.history_bytes > HISTORY_BYTES {
            let removed = self.undo.remove(0);
            self.history_bytes -= removed.bytes();
        }
    }
}

pub(crate) fn compute_leak_distance(
    occupancy: &[u8],
    dimensions: [u32; 3],
    spacing: &[f64; 3],
    cancel: &CancellationToken,
) -> Result<LeakDistanceField> {
    let dimensions = dimensions.map(|axis| axis as usize);
    propagate_leak_distance(occupancy, dimensions, spacing, cancel)
}

#[derive(Debug)]
struct VoxelBitSet {
    len: usize,
    words: Vec<u64>,
}

impl VoxelBitSet {
    fn zeroed(len: usize) -> Result<Self> {
        let word_count = len.div_ceil(u64::BITS as usize);
        let bytes = word_count
            .checked_mul(size_of::<u64>())
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| AppError::Allocation { bytes })?;
        words.resize(word_count, 0);
        Ok(Self { len, words })
    }

    fn contains(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let word = index / u64::BITS as usize;
        let bit = index % u64::BITS as usize;
        self.words[word] & (1_u64 << bit) != 0
    }

    fn insert(&mut self, index: usize) {
        self.set(index, true);
    }

    fn set(&mut self, index: usize, occupied: bool) {
        debug_assert!(index < self.len);
        let word = index / u64::BITS as usize;
        let bit = index % u64::BITS as usize;
        let bit = 1_u64 << bit;
        if occupied {
            self.words[word] |= bit;
        } else {
            self.words[word] &= !bit;
        }
    }

    fn fill(&mut self, start: usize, end: usize, occupied: bool) {
        debug_assert!(start <= end && end <= self.len);
        if start == end {
            return;
        }
        let first_word = start / u64::BITS as usize;
        let last_word = (end - 1) / u64::BITS as usize;
        let first_bit = start % u64::BITS as usize;
        let last_bit = end % u64::BITS as usize;
        if first_word == last_word {
            let end_mask = if last_bit == 0 {
                u64::MAX
            } else {
                (1_u64 << last_bit) - 1
            };
            let mask = end_mask & (u64::MAX << first_bit);
            if occupied {
                self.words[first_word] |= mask;
            } else {
                self.words[first_word] &= !mask;
            }
            return;
        }

        let first_mask = u64::MAX << first_bit;
        if occupied {
            self.words[first_word] |= first_mask;
        } else {
            self.words[first_word] &= !first_mask;
        }
        if first_word + 1 < last_word {
            self.words[first_word + 1..last_word].fill(if occupied { u64::MAX } else { 0 });
        }
        let last_mask = if last_bit == 0 {
            u64::MAX
        } else {
            (1_u64 << last_bit) - 1
        };
        if occupied {
            self.words[last_word] |= last_mask;
        } else {
            self.words[last_word] &= !last_mask;
        }
    }
}

fn classify_exterior_empty(
    occupancy: &[u8],
    dimensions: [usize; 3],
    cancel: &CancellationToken,
) -> Result<VoxelBitSet> {
    let [nx, ny, nz] = dimensions;
    let plane = nx
        .checked_mul(ny)
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let mut exterior = VoxelBitSet::zeroed(occupancy.len())?;
    let mut stack = Vec::new();

    for z in 0..nz {
        cancel.check()?;
        for y in 0..ny {
            if y.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            let row = z * plane + y * nx;
            flood_exterior_seed(
                row,
                occupancy,
                dimensions,
                &mut exterior,
                &mut stack,
                cancel,
            )?;
            if nx > 1 {
                flood_exterior_seed(
                    row + nx - 1,
                    occupancy,
                    dimensions,
                    &mut exterior,
                    &mut stack,
                    cancel,
                )?;
            }
        }
    }
    for z in 0..nz {
        cancel.check()?;
        let plane_start = z * plane;
        for x in 0..nx {
            if x.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            flood_exterior_seed(
                plane_start + x,
                occupancy,
                dimensions,
                &mut exterior,
                &mut stack,
                cancel,
            )?;
            if ny > 1 {
                flood_exterior_seed(
                    plane_start + (ny - 1) * nx + x,
                    occupancy,
                    dimensions,
                    &mut exterior,
                    &mut stack,
                    cancel,
                )?;
            }
        }
    }
    for y in 0..ny {
        cancel.check()?;
        let row = y * nx;
        for x in 0..nx {
            if x.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            flood_exterior_seed(
                row + x,
                occupancy,
                dimensions,
                &mut exterior,
                &mut stack,
                cancel,
            )?;
            if nz > 1 {
                flood_exterior_seed(
                    (nz - 1) * plane + row + x,
                    occupancy,
                    dimensions,
                    &mut exterior,
                    &mut stack,
                    cancel,
                )?;
            }
        }
    }

    Ok(exterior)
}

fn propagate_leak_distance(
    occupancy: &[u8],
    dimensions: [usize; 3],
    spacing: &[f64; 3],
    cancel: &CancellationToken,
) -> Result<LeakDistanceField> {
    propagate_leak_distance_with_queue_limit(
        occupancy,
        dimensions,
        spacing,
        LEAK_DISTANCE_QUEUE_BYTES / size_of::<u32>(),
        cancel,
    )
}

fn propagate_leak_distance_with_queue_limit(
    occupancy: &[u8],
    dimensions: [usize; 3],
    spacing: &[f64; 3],
    maximum_queued: usize,
    cancel: &CancellationToken,
) -> Result<LeakDistanceField> {
    let distance_scale = leak_distance_scale(dimensions, spacing);
    let axis_steps = spacing.map(|step| {
        (step / distance_scale)
            .round()
            .clamp(1.0, f64::from(MAX_LEAK_DISTANCE_STEPS)) as u16
    });

    let distance_bytes = occupancy
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(occupancy.len())
        .map_err(|_| AppError::Allocation {
            bytes: distance_bytes,
        })?;
    steps.resize(occupancy.len(), SEALED_LEAK_DISTANCE);

    let bucket_count = usize::from(MAX_LEAK_DISTANCE_STEPS) + 1;
    let bucket_bytes = bucket_count
        .checked_mul(size_of::<Vec<u32>>())
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let mut buckets = Vec::new();
    buckets
        .try_reserve_exact(bucket_count)
        .map_err(|_| AppError::Allocation {
            bytes: bucket_bytes,
        })?;
    buckets.resize_with(bucket_count, Vec::new);

    let mut queued = 0_usize;
    let mut degraded = false;
    let [nx, ny, nz] = dimensions;
    let plane = nx
        .checked_mul(ny)
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;

    {
        let mut seed = |index: usize| -> Result<()> {
            if occupancy[index] != 0 || steps[index] != SEALED_LEAK_DISTANCE {
                return Ok(());
            }
            steps[index] = 0;
            if queued == maximum_queued {
                degraded = true;
                return Ok(());
            }
            buckets[0]
                .try_reserve(1)
                .map_err(|_| AppError::Allocation {
                    bytes: (queued + 1) * size_of::<u32>(),
                })?;
            buckets[0].push(index as u32);
            queued += 1;
            Ok(())
        };

        for z in 0..nz {
            cancel.check()?;
            for y in 0..ny {
                let row = z * plane + y * nx;
                seed(row)?;
                if nx > 1 {
                    seed(row + nx - 1)?;
                }
            }
        }
        for z in 0..nz {
            let plane_start = z * plane;
            for x in 0..nx {
                seed(plane_start + x)?;
                if ny > 1 {
                    seed(plane_start + (ny - 1) * nx + x)?;
                }
            }
        }
        for y in 0..ny {
            let row = y * nx;
            for x in 0..nx {
                seed(row + x)?;
                if nz > 1 {
                    seed((nz - 1) * plane + row + x)?;
                }
            }
        }
    }

    if !degraded {
        for level in 0..=MAX_LEAK_DISTANCE_STEPS {
            while let Some(index) = buckets[usize::from(level)].pop() {
                queued -= 1;
                let index = index as usize;
                if steps[index] != level {
                    continue;
                }
                if index.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                    cancel.check()?;
                }
                let z = index / plane;
                let within_plane = index % plane;
                let y = within_plane / nx;
                let x = within_plane % nx;
                for (axis, neighbor) in [
                    (0, (x > 0).then_some(index.wrapping_sub(1))),
                    (0, (x + 1 < nx).then_some(index + 1)),
                    (1, (y > 0).then_some(index.wrapping_sub(nx))),
                    (1, (y + 1 < ny).then_some(index + nx)),
                    (2, (z > 0).then_some(index.wrapping_sub(plane))),
                    (2, (z + 1 < nz).then_some(index + plane)),
                ] {
                    let Some(neighbor) = neighbor else {
                        continue;
                    };
                    if occupancy[neighbor] != 0 {
                        continue;
                    }
                    let candidate = level
                        .saturating_add(axis_steps[axis])
                        .min(MAX_LEAK_DISTANCE_STEPS);
                    if candidate >= steps[neighbor] {
                        continue;
                    }
                    steps[neighbor] = candidate;
                    if queued == maximum_queued {
                        degraded = true;
                        break;
                    }
                    let bucket = &mut buckets[usize::from(candidate)];
                    if bucket.len() == bucket.capacity() {
                        bucket.try_reserve(1).map_err(|_| AppError::Allocation {
                            bytes: (queued + 1) * size_of::<u32>(),
                        })?;
                    }
                    bucket.push(neighbor as u32);
                    queued += 1;
                }
                if degraded {
                    break;
                }
            }
            // Distances only move to higher buckets, so this level can never
            // receive another entry. Release its peak frontier allocation now
            // instead of retaining every processed level for the whole volume.
            buckets[usize::from(level)] = Vec::new();
            if degraded {
                break;
            }
        }
    }

    if degraded {
        drop(buckets);
        let exterior = classify_exterior_empty(occupancy, dimensions, cancel)?;
        for (index, step) in steps.iter_mut().enumerate() {
            if index.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            if *step == SEALED_LEAK_DISTANCE && exterior.contains(index) {
                *step = MAX_LEAK_DISTANCE_STEPS;
            }
        }
    }

    Ok(LeakDistanceField {
        steps,
        distance_scale,
        degraded,
    })
}

fn leak_distance_scale(dimensions: [usize; 3], spacing: &[f64; 3]) -> f64 {
    let diagonal = dimensions
        .into_iter()
        .zip(*spacing)
        .fold(0.0_f64, |length, (dimension, step)| {
            length.hypot(dimension as f64 * step)
        });
    (LEAK_DISTANCE_DIAGONALS * diagonal / f64::from(MAX_LEAK_DISTANCE_STEPS)).max(f64::MIN_POSITIVE)
}

fn flood_exterior_seed(
    seed: usize,
    occupancy: &[u8],
    dimensions: [usize; 3],
    exterior: &mut VoxelBitSet,
    stack: &mut Vec<u32>,
    cancel: &CancellationToken,
) -> Result<()> {
    if occupancy[seed] != 0 || exterior.contains(seed) {
        return Ok(());
    }
    push_flood_seed(stack, seed)?;
    let [nx, ny, _] = dimensions;
    let plane = nx * ny;

    let mut processed_spans = 0_usize;
    while let Some(seed) = stack.pop() {
        if processed_spans.is_multiple_of(1024) {
            cancel.check()?;
        }
        processed_spans += 1;
        let seed = seed as usize;
        if occupancy[seed] != 0 || exterior.contains(seed) {
            continue;
        }
        let z = seed / plane;
        let within_plane = seed % plane;
        let y = within_plane / nx;
        let x = within_plane % nx;
        let row = seed - x;

        let mut first_x = x;
        while first_x > 0 {
            if first_x.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            let candidate = row + first_x - 1;
            if occupancy[candidate] != 0 || exterior.contains(candidate) {
                break;
            }
            first_x -= 1;
        }
        let mut last_x = x;
        while last_x + 1 < nx {
            if last_x.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            let candidate = row + last_x + 1;
            if occupancy[candidate] != 0 || exterior.contains(candidate) {
                break;
            }
            last_x += 1;
        }
        for index in row + first_x..=row + last_x {
            if index.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
                cancel.check()?;
            }
            exterior.insert(index);
        }

        if y > 0 {
            enqueue_projected_segments(
                row - nx,
                first_x,
                last_x,
                occupancy,
                exterior,
                stack,
                cancel,
            )?;
        }
        if y + 1 < ny {
            enqueue_projected_segments(
                row + nx,
                first_x,
                last_x,
                occupancy,
                exterior,
                stack,
                cancel,
            )?;
        }
        if z > 0 {
            enqueue_projected_segments(
                row - plane,
                first_x,
                last_x,
                occupancy,
                exterior,
                stack,
                cancel,
            )?;
        }
        if z + 1 < dimensions[2] {
            enqueue_projected_segments(
                row + plane,
                first_x,
                last_x,
                occupancy,
                exterior,
                stack,
                cancel,
            )?;
        }
    }

    Ok(())
}

fn enqueue_projected_segments(
    row: usize,
    first_x: usize,
    last_x: usize,
    occupancy: &[u8],
    exterior: &VoxelBitSet,
    stack: &mut Vec<u32>,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut x = first_x;
    while x <= last_x {
        if x.is_multiple_of(CANCELLATION_CHECK_INTERVAL) {
            cancel.check()?;
        }
        let index = row + x;
        if occupancy[index] != 0 || exterior.contains(index) {
            x += 1;
            continue;
        }
        push_flood_seed(stack, index)?;
        x += 1;
        while x <= last_x {
            let index = row + x;
            if occupancy[index] != 0 || exterior.contains(index) {
                break;
            }
            x += 1;
        }
    }
    Ok(())
}

fn push_flood_seed(stack: &mut Vec<u32>, index: usize) -> Result<()> {
    let maximum_seeds = EXTERIOR_STACK_BYTES / size_of::<u32>();
    if stack.len() == maximum_seeds {
        return Err(AppError::Allocation {
            bytes: EXTERIOR_STACK_BYTES + size_of::<u32>(),
        });
    }
    if stack.len() == stack.capacity() {
        const RESERVE_BLOCK: usize = 64 * 1024;
        let additional = RESERVE_BLOCK.min(maximum_seeds - stack.len());
        stack
            .try_reserve(additional)
            .map_err(|_| AppError::Allocation {
                bytes: (stack.len() + additional) * size_of::<u32>(),
            })?;
    }
    stack.push(index as u32);
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SliceImage {
    pub axis: SliceAxis,
    pub coordinate: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[must_use]
pub fn linear_index(dimensions: [u32; 3], voxel: [u32; 3]) -> Option<u32> {
    if voxel
        .iter()
        .zip(dimensions)
        .any(|(&coordinate, dimension)| coordinate >= dimension)
    {
        return None;
    }
    let index = u64::from(voxel[0])
        + u64::from(dimensions[0])
            * (u64::from(voxel[1]) + u64::from(dimensions[1]) * u64::from(voxel[2]));
    u32::try_from(index).ok()
}

#[must_use]
pub fn transform_point(matrix: Matrix4, point: [f64; 3]) -> [f64; 3] {
    let input = [point[0], point[1], point[2], 1.0];
    let mut output = [0.0; 4];
    for row in 0..4 {
        output[row] = (0..4)
            .map(|column| matrix[row][column] * input[column])
            .sum();
    }
    let divisor = if output[3].abs() > f64::EPSILON {
        output[3]
    } else {
        1.0
    };
    [
        output[0] / divisor,
        output[1] / divisor,
        output[2] / divisor,
    ]
}

#[must_use]
pub fn transform_vector(matrix: Matrix4, vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn vector_length(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn invert_matrix(matrix: Matrix4) -> Option<Matrix4> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][row + 4] = 1.0;
    }
    for pivot in 0..4 {
        let best = (pivot..4).max_by(|&a, &b| {
            augmented[a][pivot]
                .abs()
                .total_cmp(&augmented[b][pivot].abs())
        })?;
        if augmented[best][pivot].abs() <= f64::EPSILON {
            return None;
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for value in &mut augmented[pivot] {
            *value /= divisor;
        }
        let pivot_row = augmented[pivot];
        for (row_index, row_values) in augmented.iter_mut().enumerate() {
            if row_index == pivot {
                continue;
            }
            let factor = row_values[pivot];
            for (value, pivot_value) in row_values.iter_mut().zip(pivot_row) {
                *value -= factor * pivot_value;
            }
        }
    }
    let mut inverse = [[0.0; 4]; 4];
    for row in 0..4 {
        inverse[row].copy_from_slice(&augmented[row][4..]);
    }
    inverse
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_layout_is_x_then_y_then_z() {
        assert_eq!(linear_index([4, 3, 2], [0, 0, 0]), Some(0));
        assert_eq!(linear_index([4, 3, 2], [3, 0, 0]), Some(3));
        assert_eq!(linear_index([4, 3, 2], [0, 1, 0]), Some(4));
        assert_eq!(linear_index([4, 3, 2], [0, 0, 1]), Some(12));
        assert_eq!(linear_index([4, 3, 2], [4, 0, 0]), None);
    }

    #[test]
    fn packed_voxel_fill_handles_partial_and_complete_words() {
        let mut mask = VoxelBitSet::zeroed(130).unwrap();
        mask.fill(3, 129, true);
        for index in 0..130 {
            assert_eq!(
                mask.contains(index),
                (3..129).contains(&index),
                "unexpected bit at {index}"
            );
        }

        mask.fill(63, 66, false);
        for index in 0..130 {
            assert_eq!(
                mask.contains(index),
                (3..129).contains(&index) && !(63..66).contains(&index),
                "unexpected bit after clearing {index}"
            );
        }
    }

    #[test]
    fn affine_round_trip_preserves_oblique_points() {
        let geometry = VolumeGeometry::new(
            [2, 2, 2],
            [
                [0.0, -2.0, 0.2, 11.0],
                [1.5, 0.0, 0.1, -7.0],
                [0.0, 0.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Millimeter,
        )
        .unwrap();
        let point = [0.25, 1.5, -0.5];
        let recovered = geometry.world_point_to_index(geometry.index_point_to_world(point));
        for axis in 0..3 {
            assert!((recovered[axis] - point[axis]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cavity_fill_history_cost_tracks_runs_instead_of_voxels() {
        let changes = TransactionChanges::Spans(vec![IndexSpan {
            start: 7,
            len: 50_000_000,
        }]);

        assert_eq!(changes.voxel_count(), 50_000_000);
        assert_eq!(changes.bytes(), size_of::<IndexSpan>());
        assert!(changes.bytes() < HISTORY_BYTES);
    }

    #[test]
    fn scanline_exterior_matches_naive_six_connected_floods() {
        let dimensions = [5, 4, 3];
        let voxel_count = dimensions.into_iter().product();
        let mut random = 0x5eed_u64;

        for _ in 0..64 {
            let mut occupancy = Vec::with_capacity(voxel_count);
            for _ in 0..voxel_count {
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                occupancy.push(u8::from(random >> 61 < 3));
            }

            let actual =
                classify_exterior_empty(&occupancy, dimensions, &CancellationToken::default())
                    .unwrap();
            let expected = naive_exterior(&occupancy, dimensions);
            for (index, &is_exterior) in expected.iter().enumerate() {
                assert_eq!(
                    actual.contains(index),
                    is_exterior,
                    "classification differed at index {index}"
                );
            }
        }
    }

    #[test]
    fn leak_distance_reaches_around_bends_but_not_into_sealed_space() {
        let dimensions = [7, 7, 7];
        let mut occupancy = vec![1; dimensions.into_iter().product()];
        let index =
            |voxel: [usize; 3]| voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2]);
        for voxel in [[0, 3, 3], [1, 3, 3], [2, 3, 3], [2, 2, 3], [2, 2, 2]] {
            occupancy[index(voxel)] = 0;
        }
        occupancy[index([5, 5, 5])] = 0;
        let field = propagate_leak_distance(
            &occupancy,
            dimensions,
            &[1.0; 3],
            &CancellationToken::default(),
        )
        .unwrap();

        assert_eq!(field.steps()[index([0, 3, 3])], 0);
        assert!(field.steps()[index([2, 2, 2])] > 0);
        assert!(field.steps()[index([2, 2, 2])] < MAX_LEAK_DISTANCE_STEPS);
        assert_eq!(field.steps()[index([5, 5, 5])], SEALED_LEAK_DISTANCE);
        assert!(field.distance_scale().is_finite());
        assert!(field.distance_scale() > 0.0);
        assert!(!field.degraded());
    }

    #[test]
    fn leak_distance_respects_affine_axis_spacing() {
        let dimensions = [5, 5, 5];
        let mut occupancy = vec![1; dimensions.into_iter().product()];
        let index =
            |voxel: [usize; 3]| voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2]);
        for voxel in [
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [3, 0, 2],
            [3, 1, 2],
            [3, 2, 2],
        ] {
            occupancy[index(voxel)] = 0;
        }
        let field = propagate_leak_distance(
            &occupancy,
            dimensions,
            &[4.0, 1.0, 1.0],
            &CancellationToken::default(),
        )
        .unwrap();

        assert!(field.steps()[index([3, 2, 2])] < field.steps()[index([2, 1, 2])]);
    }

    #[test]
    fn leak_distance_connectivity_matches_naive_six_connected_floods() {
        let dimensions = [5, 4, 3];
        let voxel_count = dimensions.into_iter().product();
        let mut random = 0x1ea5_d157_u64;

        for _ in 0..64 {
            let mut occupancy = Vec::with_capacity(voxel_count);
            for _ in 0..voxel_count {
                random = random
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                occupancy.push(u8::from(random >> 61 < 3));
            }
            let field = propagate_leak_distance(
                &occupancy,
                dimensions,
                &[1.0, 2.0, 3.0],
                &CancellationToken::default(),
            )
            .unwrap();
            let expected = naive_exterior(&occupancy, dimensions);

            for (index, &is_exterior) in expected.iter().enumerate() {
                assert_eq!(
                    field.steps()[index] != SEALED_LEAK_DISTANCE,
                    is_exterior,
                    "connectivity differed at index {index}"
                );
            }
        }
    }

    #[test]
    fn degraded_leak_distance_keeps_exterior_connectivity_exact() {
        let dimensions = [7, 7, 7];
        let mut occupancy = vec![1; dimensions.into_iter().product()];
        let index =
            |voxel: [usize; 3]| voxel[0] + dimensions[0] * (voxel[1] + dimensions[1] * voxel[2]);
        for voxel in [[0, 3, 3], [1, 3, 3], [2, 3, 3], [2, 2, 3], [2, 2, 2]] {
            occupancy[index(voxel)] = 0;
        }
        occupancy[index([5, 5, 5])] = 0;

        let field = propagate_leak_distance_with_queue_limit(
            &occupancy,
            dimensions,
            &[1.0; 3],
            0,
            &CancellationToken::default(),
        )
        .unwrap();

        assert!(field.degraded());
        assert_eq!(field.steps()[index([0, 3, 3])], 0);
        assert_eq!(field.steps()[index([2, 2, 2])], MAX_LEAK_DISTANCE_STEPS);
        assert_eq!(field.steps()[index([5, 5, 5])], SEALED_LEAK_DISTANCE);
    }

    fn naive_exterior(occupancy: &[u8], dimensions: [usize; 3]) -> Vec<bool> {
        let [nx, ny, nz] = dimensions;
        let plane = nx * ny;
        let mut exterior = vec![false; occupancy.len()];
        let mut queue = std::collections::VecDeque::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if x != 0 && x + 1 != nx && y != 0 && y + 1 != ny && z != 0 && z + 1 != nz {
                        continue;
                    }
                    let index = z * plane + y * nx + x;
                    if occupancy[index] == 0 && !exterior[index] {
                        exterior[index] = true;
                        queue.push_back(index);
                    }
                }
            }
        }
        while let Some(index) = queue.pop_front() {
            let z = index / plane;
            let within_plane = index % plane;
            let y = within_plane / nx;
            let x = within_plane % nx;
            for neighbor in [
                x.checked_sub(1).map(|_| index - 1),
                (x + 1 < nx).then_some(index + 1),
                y.checked_sub(1).map(|_| index - nx),
                (y + 1 < ny).then_some(index + nx),
                z.checked_sub(1).map(|_| index - plane),
                (z + 1 < nz).then_some(index + plane),
            ]
            .into_iter()
            .flatten()
            {
                if occupancy[neighbor] == 0 && !exterior[neighbor] {
                    exterior[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        exterior
    }
}
