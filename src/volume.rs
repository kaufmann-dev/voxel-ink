use std::{collections::HashSet, path::PathBuf, time::SystemTime};

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    codecs::{CodecSaveState, SourceVariant},
    error::checked_voxel_count,
};

pub const HISTORY_TRANSACTIONS: usize = 256;
pub const HISTORY_BYTES: usize = 128 * 1024 * 1024;

pub type Matrix4 = [[f64; 4]; 4];

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
    Brush,
    Eraser,
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

    fn indices(&self) -> Option<&[u32]> {
        match &self.changes {
            TransactionChanges::Indices(indices) => Some(indices),
            TransactionChanges::Spans(_) => None,
        }
    }

    const fn requires_full_mesh_reset(&self) -> bool {
        matches!(&self.changes, TransactionChanges::Spans(_))
    }
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
struct IndexSpan {
    start: u32,
    len: u32,
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
        if occupancy.iter().any(|value| *value > 1) {
            return Err(AppError::InvalidDimensions(
                "occupancy must contain only zero or one".into(),
            ));
        }
        let occupied_count = occupancy.iter().map(|&value| u64::from(value)).sum();
        let revision = 1;
        Ok(Self {
            cursor: geometry.dimensions.map(|axis| axis / 2),
            geometry,
            occupancy,
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
            if *cell == stroke.value || stroke.seen.contains(&index) {
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
            stroke.indices.push(index);
            changed += 1;
        }
        Ok(changed)
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
        for index in stroke.indices {
            self.occupancy[index as usize] = old_value;
            if old_value == 1 {
                self.occupied_count += 1;
            } else {
                self.occupied_count -= 1;
            }
        }
    }

    /// Fill gaps between the first and last occupied voxel on each left-to-right
    /// scanline. This solidifies open anatomy without relying on a watertight shell.
    pub fn make_solid(&mut self, cancel: &CancellationToken) -> Result<u64> {
        self.cancel_stroke();
        cancel.check()?;

        let [nx, ny, nz] = self.geometry.dimensions.map(|axis| axis as usize);
        let fill_axis = left_right_axis(&self.geometry);
        let line_count = match fill_axis {
            0 => ny.checked_mul(nz),
            1 => nx.checked_mul(nz),
            2 => nx.checked_mul(ny),
            _ => unreachable!("a volume has exactly three axes"),
        }
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
        let mut line_bounds = allocate_line_bounds(line_count)?;
        let plane = nx
            .checked_mul(ny)
            .ok_or(AppError::Allocation { bytes: usize::MAX })?;

        for z in 0..nz {
            cancel.check()?;
            for y in 0..ny {
                let row = z * plane + y * nx;
                for x in 0..nx {
                    if self.occupancy[row + x] == 0 {
                        continue;
                    }
                    let (line, coordinate) = match fill_axis {
                        0 => (z * ny + y, x as u32),
                        1 => (z * nx + x, y as u32),
                        2 => (y * nx + x, z as u32),
                        _ => unreachable!("a volume has exactly three axes"),
                    };
                    update_line_bounds(&mut line_bounds[line], coordinate);
                }
            }
        }

        let max_spans = HISTORY_BYTES / size_of::<IndexSpan>();
        let mut spans: Vec<IndexSpan> = Vec::new();
        let initial_capacity = line_count.min(64 * 1024);
        spans
            .try_reserve(initial_capacity)
            .map_err(|_| AppError::Allocation {
                bytes: initial_capacity * size_of::<IndexSpan>(),
            })?;
        let mut added = 0_u64;
        for z in 0..nz {
            cancel.check()?;
            for y in 0..ny {
                let row = z * plane + y * nx;
                for x in 0..nx {
                    let index = row + x;
                    if self.occupancy[index] != 0 {
                        continue;
                    }
                    let (line, coordinate) = match fill_axis {
                        0 => (z * ny + y, x as u32),
                        1 => (z * nx + x, y as u32),
                        2 => (y * nx + x, z as u32),
                        _ => unreachable!("a volume has exactly three axes"),
                    };
                    if !is_between(coordinate, line_bounds[line]) {
                        continue;
                    }
                    added += 1;
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
            }
        }

        cancel.check()?;
        if spans.is_empty() {
            return Ok(0);
        }
        for span in &spans {
            let start = span.start as usize;
            let end = start + span.len as usize;
            self.occupancy[start..end].fill(1);
        }
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
    pub(crate) fn latest_undo_indices(&self) -> Option<&[u32]> {
        self.undo.last().and_then(Transaction::indices)
    }

    #[must_use]
    pub(crate) fn latest_redo_indices(&self) -> Option<&[u32]> {
        self.redo.last().and_then(Transaction::indices)
    }

    #[must_use]
    pub(crate) fn latest_undo_requires_full_mesh_reset(&self) -> bool {
        self.undo
            .last()
            .is_some_and(Transaction::requires_full_mesh_reset)
    }

    #[must_use]
    pub(crate) fn latest_redo_requires_full_mesh_reset(&self) -> bool {
        self.redo
            .last()
            .is_some_and(Transaction::requires_full_mesh_reset)
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
        for row in 0..height {
            for column in 0..width {
                let voxel = match axis {
                    SliceAxis::Axial => [column, row, coordinate],
                    SliceAxis::Coronal => [column, coordinate, row],
                    SliceAxis::Sagittal => [coordinate, column, row],
                };
                pixels.push(self.value(voxel).ok_or_else(|| {
                    AppError::InvalidDimensions(
                        "generated slice voxel is outside the volume".into(),
                    )
                })?);
            }
        }
        Ok(SliceImage {
            axis,
            coordinate,
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
        match &transaction.changes {
            TransactionChanges::Indices(indices) => {
                for &index in indices {
                    self.occupancy[index as usize] = value;
                }
            }
            TransactionChanges::Spans(spans) => {
                for span in spans {
                    let start = span.start as usize;
                    let end = start + span.len as usize;
                    self.occupancy[start..end].fill(value);
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

fn allocate_line_bounds(len: usize) -> Result<Vec<[u32; 2]>> {
    let bytes = len
        .checked_mul(size_of::<[u32; 2]>())
        .ok_or(AppError::Allocation { bytes: usize::MAX })?;
    let mut bounds = Vec::new();
    bounds
        .try_reserve_exact(len)
        .map_err(|_| AppError::Allocation { bytes })?;
    bounds.resize(len, [u32::MAX, 0]);
    Ok(bounds)
}

fn update_line_bounds(bounds: &mut [u32; 2], coordinate: u32) {
    bounds[0] = bounds[0].min(coordinate);
    bounds[1] = bounds[1].max(coordinate);
}

const fn is_between(coordinate: u32, bounds: [u32; 2]) -> bool {
    bounds[0] < coordinate && coordinate < bounds[1]
}

fn left_right_axis(geometry: &VolumeGeometry) -> usize {
    (0..3)
        .max_by(|&left, &right| {
            (geometry.index_to_world[0][left].abs() / geometry.spacing[left])
                .total_cmp(&(geometry.index_to_world[0][right].abs() / geometry.spacing[right]))
        })
        .unwrap_or(0)
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
    fn left_right_fill_axis_uses_affine_orientation_not_voxel_scale() {
        let geometry = VolumeGeometry::new(
            [3, 3, 3],
            [
                [0.0, -4.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Millimeter,
        )
        .unwrap();

        assert_eq!(left_right_axis(&geometry), 1);
    }

    #[test]
    fn solidification_history_cost_tracks_runs_instead_of_voxels() {
        let changes = TransactionChanges::Spans(vec![IndexSpan {
            start: 7,
            len: 50_000_000,
        }]);

        assert_eq!(changes.voxel_count(), 50_000_000);
        assert_eq!(changes.bytes(), size_of::<IndexSpan>());
        assert!(changes.bytes() < HISTORY_BYTES);
    }
}
