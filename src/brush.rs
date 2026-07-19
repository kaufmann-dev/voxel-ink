use std::mem::size_of;

use crate::{
    AppError, Result,
    cancel::CancellationToken,
    volume::{SpatialUnits, Tool, VolumeGeometry, linear_index, transform_vector},
};

const INITIAL_BRUSH_INDEX_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BrushShape {
    #[default]
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BrushDepth {
    #[default]
    Slice,
    Volume,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrushSize {
    Voxels(u8),
    Millimeters(f32),
}

impl BrushSize {
    pub const MIN_VOXELS: u8 = 1;
    pub const MAX_VOXELS: u8 = 101;
    pub const MIN_MILLIMETERS: f32 = 0.5;
    pub const MAX_MILLIMETERS: f32 = 50.0;

    pub fn voxels(diameter: u8) -> Result<Self> {
        if !(Self::MIN_VOXELS..=Self::MAX_VOXELS).contains(&diameter) || diameter.is_multiple_of(2)
        {
            return Err(AppError::InvalidDimensions(
                "voxel brush diameter must be odd and between 1 and 101".into(),
            ));
        }
        Ok(Self::Voxels(diameter))
    }

    pub fn millimeters(diameter: f32) -> Result<Self> {
        if !diameter.is_finite()
            || !(Self::MIN_MILLIMETERS..=Self::MAX_MILLIMETERS).contains(&diameter)
        {
            return Err(AppError::InvalidDimensions(
                "physical brush diameter must be between 0.5 and 50 millimetres".into(),
            ));
        }
        Ok(Self::Millimeters(diameter))
    }
}

impl Default for BrushSize {
    fn default() -> Self {
        Self::Voxels(5)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BrushSpec {
    pub shape: BrushShape,
    pub depth: BrushDepth,
    pub size: BrushSize,
}

impl BrushSpec {
    #[must_use]
    pub const fn with_depth(self, depth: BrushDepth) -> Self {
        Self { depth, ..self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceHit {
    pub voxel: [u32; 3],
    pub exterior: Option<[u32; 3]>,
    pub face_normal: [i8; 3],
}

pub fn walk_supercover(
    start: [u32; 3],
    end: [u32; 3],
    mut visit: impl FnMut([u32; 3]) -> Result<()>,
) -> Result<()> {
    visit(start)?;
    if start == end {
        return Ok(());
    }
    let mut current = start.map(i64::from);
    let target = end.map(i64::from);
    let delta: [i64; 3] = std::array::from_fn(|axis| target[axis] - current[axis]);
    let step = delta.map(i64::signum);
    let distance = delta.map(i64::unsigned_abs);
    let mut crossed = [0_u64; 3];
    while current != target {
        let next_axis = (0..3)
            .filter(|&axis| crossed[axis] < distance[axis])
            .min_by(|&left, &right| {
                crossing_numerator(crossed[left])
                    .saturating_mul(u128::from(distance[right]))
                    .cmp(
                        &crossing_numerator(crossed[right])
                            .saturating_mul(u128::from(distance[left])),
                    )
            })
            .ok_or_else(|| {
                AppError::InvalidDimensions("integer supercover traversal stalled".into())
            })?;
        let next_numerator = crossing_numerator(crossed[next_axis]);
        let next_denominator = u128::from(distance[next_axis]);
        let tied: [bool; 3] = std::array::from_fn(|axis| {
            crossed[axis] < distance[axis]
                && crossing_numerator(crossed[axis]).saturating_mul(next_denominator)
                    == next_numerator.saturating_mul(u128::from(distance[axis]))
        });
        for mask in 1_u8..8 {
            if (0..3).any(|axis| mask & (1 << axis) != 0 && !tied[axis]) {
                continue;
            }
            let mut candidate = current;
            for axis in 0..3 {
                if mask & (1 << axis) != 0 {
                    candidate[axis] += step[axis];
                }
            }
            visit(candidate.map(|value| {
                u32::try_from(value).expect("supercover stays between non-negative endpoints")
            }))?;
        }
        for axis in 0..3 {
            if tied[axis] {
                current[axis] += step[axis];
                crossed[axis] += 1;
            }
        }
    }
    Ok(())
}

const fn crossing_numerator(crossed: u64) -> u128 {
    crossed as u128 * 2 + 1
}

pub fn supercover_line(start: [u32; 3], end: [u32; 3]) -> Result<Vec<[u32; 3]>> {
    let mut output = Vec::new();
    walk_supercover(start, end, |voxel| {
        if output.len() == output.capacity() {
            output
                .try_reserve(1_024)
                .map_err(|_| AppError::Allocation {
                    bytes: output
                        .len()
                        .saturating_add(1_024)
                        .saturating_mul(size_of::<[u32; 3]>()),
                })?;
        }
        output.push(voxel);
        Ok(())
    })?;
    Ok(output)
}

/// Returns the exact voxel footprint for a brush centered on `center`.
///
/// Voxel-sized brushes are evaluated in index space. Millimetre-sized round
/// brushes use the complete affine metric, while square brushes remain aligned
/// to the volume's index axes with physical extents on each axis. `normal_axis`
/// is ignored for a volume brush and fixes the one-voxel depth of a slice brush.
pub fn brush_indices(
    geometry: &VolumeGeometry,
    center: [u32; 3],
    normal_axis: usize,
    spec: BrushSpec,
) -> Result<Vec<u32>> {
    brush_indices_cancellable(
        geometry,
        center,
        normal_axis,
        spec,
        &CancellationToken::default(),
        None,
    )
}

/// Cancellable brush-footprint evaluation used by latency-sensitive previews.
///
/// `maximum_indices` limits the number of voxels in the completed footprint,
/// not the (usually larger) index-space bounding box examined while building
/// it. The public [`brush_indices`] API intentionally remains convenient for
/// edit paths that already have their own lifetime and memory bounds.
pub(crate) fn brush_indices_cancellable(
    geometry: &VolumeGeometry,
    center: [u32; 3],
    normal_axis: usize,
    spec: BrushSpec,
    cancel: &CancellationToken,
    maximum_indices: Option<usize>,
) -> Result<Vec<u32>> {
    let mut examined = 0;
    let mut output = Vec::new();
    let mut traversal = BrushTraversal {
        cancel,
        examined: &mut examined,
        maximum_examined: usize::MAX,
    };
    visit_brush_indices_cancellable(
        geometry,
        center,
        normal_axis,
        spec,
        &mut traversal,
        |index| {
            if maximum_indices.is_some_and(|maximum| output.len() >= maximum) {
                return Err(AppError::InvalidDimensions(format!(
                    "brush footprint exceeds the {}-voxel preview safety limit",
                    maximum_indices.expect("the limit was just matched")
                )));
            }
            if output.len() == output.capacity() {
                let additional = output.capacity().max(INITIAL_BRUSH_INDEX_CAPACITY);
                output
                    .try_reserve_exact(additional)
                    .map_err(|_| AppError::Allocation {
                        bytes: output
                            .len()
                            .saturating_add(additional)
                            .saturating_mul(size_of::<u32>()),
                    })?;
            }
            output.push(index);
            Ok(())
        },
    )?;
    Ok(output)
}

/// Streams a brush footprint without allocating an intermediate vector.
///
/// `examined` is shared by callers that walk several brush centers, so one
/// work budget covers an entire path rather than restarting at every center.
pub(crate) struct BrushTraversal<'a> {
    pub cancel: &'a CancellationToken,
    pub examined: &'a mut usize,
    pub maximum_examined: usize,
}

pub(crate) fn visit_brush_indices_cancellable(
    geometry: &VolumeGeometry,
    center: [u32; 3],
    normal_axis: usize,
    spec: BrushSpec,
    traversal: &mut BrushTraversal<'_>,
    mut visit: impl FnMut(u32) -> Result<()>,
) -> Result<()> {
    traversal.cancel.check()?;
    if normal_axis >= 3 || linear_index(geometry.dimensions, center).is_none() {
        return Err(AppError::InvalidDimensions(
            "brush center or slice axis lies outside the volume".into(),
        ));
    }
    let metric = BrushMetric::new(geometry, spec.size)?;
    let mut bounds = brush_index_bounds(geometry, spec.shape, metric);
    if spec.depth == BrushDepth::Slice {
        bounds[normal_axis] = 0;
    }
    let ranges: [(i64, i64); 3] = std::array::from_fn(|axis| {
        let center = i64::from(center[axis]);
        let last = i64::from(geometry.dimensions[axis] - 1);
        (-bounds[axis].min(center), bounds[axis].min(last - center))
    });
    let [width, height, _] = geometry.dimensions.map(u64::from);
    let first_x = u64::try_from(i64::from(center[0]) + ranges[0].0)
        .expect("brush ranges are clipped to the volume");
    for z in ranges[2].0..=ranges[2].1 {
        let voxel_z = u64::try_from(i64::from(center[2]) + z)
            .expect("brush ranges are clipped to the volume");
        for y in ranges[1].0..=ranges[1].1 {
            let voxel_y = u64::try_from(i64::from(center[1]) + y)
                .expect("brush ranges are clipped to the volume");
            let row_start = first_x + width * (voxel_y + height * voxel_z);
            for x in ranges[0].0..=ranges[0].1 {
                if *traversal.examined >= traversal.maximum_examined {
                    return Err(AppError::BridgeWorkLimit {
                        limit: traversal.maximum_examined,
                    });
                }
                *traversal.examined = traversal.examined.saturating_add(1);
                if *traversal.examined & 0x0fff == 0 {
                    traversal.cancel.check()?;
                }
                if !matches!(
                    (spec.shape, metric),
                    (BrushShape::Square, BrushMetric::Voxels { .. })
                ) && !brush_contains_delta(geometry, [x, y, z], spec.shape, metric)
                {
                    continue;
                }
                let x_offset =
                    u64::try_from(x - ranges[0].0).expect("brush x offset is non-negative");
                let index = u32::try_from(row_start + x_offset)
                    .expect("the volume voxel limit fits linear indices in u32");
                visit(index)?;
            }
        }
    }
    traversal.cancel.check()?;
    Ok(())
}

#[derive(Clone, Copy)]
enum BrushMetric {
    Voxels { radius: i64 },
    Physical { radius: f64 },
}

impl BrushMetric {
    fn new(geometry: &VolumeGeometry, size: BrushSize) -> Result<Self> {
        match size {
            BrushSize::Voxels(diameter) => {
                BrushSize::voxels(diameter)?;
                Ok(Self::Voxels {
                    radius: i64::from(diameter / 2),
                })
            }
            BrushSize::Millimeters(diameter) => {
                BrushSize::millimeters(diameter)?;
                Ok(Self::Physical {
                    radius: f64::from(diameter) * 0.5
                        / millimeters_per_spatial_unit(geometry.units)?,
                })
            }
        }
    }
}

fn brush_index_bounds(
    geometry: &VolumeGeometry,
    shape: BrushShape,
    metric: BrushMetric,
) -> [i64; 3] {
    match metric {
        BrushMetric::Voxels { radius } => [radius; 3],
        BrushMetric::Physical { radius } => std::array::from_fn(|axis| {
            let unbounded = match shape {
                BrushShape::Round => {
                    let inverse_row_length = (0..3)
                        .map(|component| geometry.world_to_index[axis][component].powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (radius * inverse_row_length).ceil()
                }
                BrushShape::Square => (radius / geometry.spacing[axis]).floor(),
            };
            unbounded.clamp(0.0, f64::from(geometry.dimensions[axis] - 1)) as i64
        }),
    }
}

fn brush_contains_delta(
    geometry: &VolumeGeometry,
    delta: [i64; 3],
    shape: BrushShape,
    metric: BrushMetric,
) -> bool {
    match metric {
        BrushMetric::Voxels { radius } => match shape {
            BrushShape::Round => {
                delta
                    .into_iter()
                    .map(|component| component * component)
                    .sum::<i64>()
                    <= radius * radius
            }
            BrushShape::Square => delta.into_iter().all(|component| component.abs() <= radius),
        },
        BrushMetric::Physical { radius } => match shape {
            BrushShape::Round => {
                let world = transform_vector(geometry.index_to_world, delta.map(|v| v as f64));
                let squared_distance = world.into_iter().map(|value| value * value).sum::<f64>();
                let squared_radius = radius * radius;
                squared_distance <= squared_radius + squared_radius.max(1.0) * 1.0e-12
            }
            BrushShape::Square => delta.into_iter().enumerate().all(|(axis, component)| {
                (component as f64).abs() * geometry.spacing[axis]
                    <= radius + radius.max(1.0) * 1.0e-12
            }),
        },
    }
}

fn millimeters_per_spatial_unit(units: SpatialUnits) -> Result<f64> {
    match units {
        SpatialUnits::Meter => Ok(1_000.0),
        SpatialUnits::Millimeter => Ok(1.0),
        SpatialUnits::Micron => Ok(0.001),
        SpatialUnits::Unknown => Err(AppError::InvalidDimensions(
            "this volume has no physical units; use a voxel-sized brush".into(),
        )),
    }
}

/// Finds the nearest occupied surface crossed by `world_ray`.
///
/// A ray starting in empty space hits the next occupied voxel. A ray starting
/// inside occupied data hits that run's forward exit surface, which keeps
/// no-clip Fly picking on the visible boundary instead of the camera voxel.
/// A miss stays a miss: this helper never projects onto an arbitrary slice or
/// creates a seed in empty space. `exterior` is the face-adjacent empty voxel
/// beside the hit when that voxel lies inside the volume.
#[must_use]
pub fn ray_surface_hit(
    occupancy: &[u8],
    geometry: &VolumeGeometry,
    world_ray: Ray,
) -> Option<SurfaceHit> {
    let dimensions = geometry.dimensions;
    let expected_len = dimensions
        .into_iter()
        .try_fold(1_usize, |total, axis| total.checked_mul(axis as usize))?;
    if occupancy.len() != expected_len {
        return None;
    }
    ray_surface_hit_with_occupancy(geometry, world_ray, |index| {
        occupancy
            .get(usize::try_from(index).ok()?)
            .map(|&value| value == 1)
    })
}

fn ray_surface_hit_with_occupancy(
    geometry: &VolumeGeometry,
    world_ray: Ray,
    mut occupied: impl FnMut(u32) -> Option<bool>,
) -> Option<SurfaceHit> {
    let dimensions = geometry.dimensions;
    let origin = geometry.world_point_to_index(world_ray.origin);
    let direction = geometry.world_vector_to_index(world_ray.direction);
    if direction
        .iter()
        .all(|component| component.abs() <= f64::EPSILON)
        || !origin
            .iter()
            .chain(&direction)
            .all(|value| value.is_finite())
    {
        return None;
    }
    let bounds_min = [-0.5; 3];
    let bounds_max = dimensions.map(|axis| f64::from(axis) - 0.5);
    let (entry, exit, entry_axis) = ray_box(origin, direction, bounds_min, bounds_max)?;
    let start_t = entry.max(0.0);
    if start_t > exit {
        return None;
    }
    let epsilon = 1.0e-9;
    let position = [
        origin[0] + direction[0] * (start_t + epsilon),
        origin[1] + direction[1] * (start_t + epsilon),
        origin[2] + direction[2] * (start_t + epsilon),
    ];
    let mut voxel = position.map(|value| (value + 0.5).floor() as i64);
    for axis in 0..3 {
        voxel[axis] = voxel[axis].clamp(0, i64::from(dimensions[axis]) - 1);
    }
    let step = direction.map(|component| {
        if component > 0.0 {
            1_i64
        } else if component < 0.0 {
            -1_i64
        } else {
            0_i64
        }
    });
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for axis in 0..3 {
        if step[axis] != 0 {
            let boundary = voxel[axis] as f64 + if step[axis] > 0 { 0.5 } else { -0.5 };
            t_max[axis] = (boundary - origin[axis]) / direction[axis];
            t_delta[axis] = 1.0 / direction[axis].abs();
        }
    }
    let initial = checked_voxel(voxel, dimensions)?;
    let initial_index = linear_index(dimensions, initial)?;
    let seeking_exit = entry < 0.0 && occupied(initial_index)?;
    let mut previous = None;
    let mut face_normal = [0_i8; 3];
    if entry >= 0.0
        && let Some(axis) = entry_axis
    {
        face_normal[axis] = if direction[axis] > 0.0 { -1 } else { 1 };
    }
    loop {
        let current = checked_voxel(voxel, dimensions)?;
        let index = linear_index(dimensions, current)?;
        let current_occupied = occupied(index)?;
        if current_occupied && !seeking_exit {
            return Some(SurfaceHit {
                voxel: current,
                exterior: previous,
                face_normal,
            });
        }
        let next_t = t_max.into_iter().fold(f64::INFINITY, f64::min);
        if next_t > exit {
            return None;
        }
        let tolerance = f64::EPSILON * next_t.abs().max(1.0) * 8.0;
        let tied = t_max.map(|value| (value - next_t).abs() <= tolerance);
        for axis in 0..3 {
            if tied[axis] {
                voxel[axis] += step[axis];
                t_max[axis] += t_delta[axis];
            }
        }
        if seeking_exit && current_occupied {
            let next = checked_voxel(voxel, dimensions);
            let next_is_occupied = if let Some(next) = next {
                occupied(linear_index(dimensions, next)?)?
            } else {
                false
            };
            if !next_is_occupied {
                let mut exterior = None;
                let mut outward_axis = tied
                    .iter()
                    .enumerate()
                    .find_map(|(axis, &is_tied)| is_tied.then_some(axis))?;
                for axis in 0..3 {
                    if !tied[axis] {
                        continue;
                    }
                    let mut candidate = current.map(i64::from);
                    candidate[axis] += step[axis];
                    let Some(candidate) = checked_voxel(candidate, dimensions) else {
                        outward_axis = axis;
                        continue;
                    };
                    let candidate_index = linear_index(dimensions, candidate)?;
                    if !occupied(candidate_index)? {
                        exterior = Some(candidate);
                        outward_axis = axis;
                        break;
                    }
                }
                let mut face_normal = [0; 3];
                face_normal[outward_axis] = i8::try_from(step[outward_axis]).ok()?;
                return Some(SurfaceHit {
                    voxel: current,
                    exterior,
                    face_normal,
                });
            }
            continue;
        }
        face_normal = [0; 3];
        previous = None;
        for axis in 0..3 {
            if !tied[axis] {
                continue;
            }
            let mut neighbor = voxel;
            neighbor[axis] -= step[axis];
            let Some(neighbor) = checked_voxel(neighbor, dimensions) else {
                continue;
            };
            let neighbor_index = linear_index(dimensions, neighbor)?;
            if !occupied(neighbor_index)? {
                previous = Some(neighbor);
                face_normal[axis] = i8::try_from(-step[axis]).ok()?;
                break;
            }
        }
    }
}

/// Picks the voxel that a surface editing tool should modify.
///
/// Paint targets the empty voxel immediately outside the hit surface. Erase,
/// inspect, and bridge target the occupied surface voxel itself. Nothing is
/// returned when the ray misses the occupied surface.
#[must_use]
pub fn surface_edit_target(
    dimensions: [u32; 3],
    occupancy: &[u8],
    geometry: &VolumeGeometry,
    world_ray: Ray,
    tool: Tool,
) -> Option<[u32; 3]> {
    if dimensions != geometry.dimensions {
        return None;
    }
    let hit = ray_surface_hit(occupancy, geometry, world_ray)?;
    match tool {
        Tool::Paint => hit.exterior,
        Tool::Inspect | Tool::Erase | Tool::Bridge => Some(hit.voxel),
    }
}

pub(crate) fn surface_edit_target_with_occupancy(
    geometry: &VolumeGeometry,
    world_ray: Ray,
    tool: Tool,
    occupied: impl FnMut(u32) -> Option<bool>,
) -> Option<[u32; 3]> {
    let hit = ray_surface_hit_with_occupancy(geometry, world_ray, occupied)?;
    match tool {
        Tool::Paint => hit.exterior,
        Tool::Inspect | Tool::Erase | Tool::Bridge => Some(hit.voxel),
    }
}

/// Projects a ray onto a world-space plane anchored at a voxel center.
///
/// This is intended for continuing a bridge after its first endpoint has been
/// picked from an occupied surface. It does not search for or synthesize a
/// surface hit. Parallel, backward, non-finite, and out-of-volume intersections
/// return `None`.
#[must_use]
pub fn ray_plane_voxel(
    dimensions: [u32; 3],
    geometry: &VolumeGeometry,
    world_ray: Ray,
    plane_voxel: [u32; 3],
    plane_normal: [f64; 3],
) -> Option<[u32; 3]> {
    if dimensions != geometry.dimensions || linear_index(dimensions, plane_voxel).is_none() {
        return None;
    }
    if !world_ray
        .origin
        .iter()
        .chain(&world_ray.direction)
        .chain(&plane_normal)
        .all(|value| value.is_finite())
    {
        return None;
    }
    let direction_length_squared = dot(world_ray.direction, world_ray.direction);
    let normal_length_squared = dot(plane_normal, plane_normal);
    if direction_length_squared == 0.0 || normal_length_squared == 0.0 {
        return None;
    }
    let denominator = dot(world_ray.direction, plane_normal);
    let parallel_tolerance =
        f64::EPSILON * (direction_length_squared * normal_length_squared).sqrt() * 16.0;
    if denominator.abs() <= parallel_tolerance {
        return None;
    }
    let plane_point = geometry.index_point_to_world(plane_voxel.map(f64::from));
    let from_origin = std::array::from_fn(|axis| plane_point[axis] - world_ray.origin[axis]);
    let distance = dot(from_origin, plane_normal) / denominator;
    if !distance.is_finite() || distance < 0.0 {
        return None;
    }
    let world_point =
        std::array::from_fn(|axis| world_ray.origin[axis] + world_ray.direction[axis] * distance);
    let index_point = geometry.world_point_to_index(world_point);
    if !index_point.iter().all(|value| value.is_finite()) {
        return None;
    }
    checked_voxel(index_point.map(|value| value.round() as i64), dimensions)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn checked_voxel(voxel: [i64; 3], dimensions: [u32; 3]) -> Option<[u32; 3]> {
    if voxel
        .iter()
        .zip(dimensions)
        .any(|(&coordinate, dimension)| coordinate < 0 || coordinate >= i64::from(dimension))
    {
        return None;
    }
    Some(voxel.map(|coordinate| u32::try_from(coordinate).expect("non-negative in-bounds voxel")))
}

fn ray_box(
    origin: [f64; 3],
    direction: [f64; 3],
    minimum: [f64; 3],
    maximum: [f64; 3],
) -> Option<(f64, f64, Option<usize>)> {
    let mut entry = f64::NEG_INFINITY;
    let mut exit = f64::INFINITY;
    let mut entry_axis = None;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return None;
            }
            continue;
        }
        let first = (minimum[axis] - origin[axis]) / direction[axis];
        let second = (maximum[axis] - origin[axis]) / direction[axis];
        let near = first.min(second);
        let far = first.max(second);
        if near > entry {
            entry = near;
            entry_axis = Some(axis);
        }
        exit = exit.min(far);
        if entry > exit {
            return None;
        }
    }
    Some((entry, exit, entry_axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_geometry(dimensions: [u32; 3]) -> VolumeGeometry {
        VolumeGeometry::new(
            dimensions,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Millimeter,
        )
        .unwrap()
    }

    #[test]
    fn cancellable_footprint_stops_before_allocating_when_cancelled() {
        let cancel = CancellationToken::default();
        cancel.cancel();
        let result = brush_indices_cancellable(
            &identity_geometry([101; 3]),
            [50; 3],
            0,
            BrushSpec {
                shape: BrushShape::Square,
                depth: BrushDepth::Volume,
                size: BrushSize::voxels(101).unwrap(),
            },
            &cancel,
            None,
        );
        assert!(matches!(result, Err(AppError::Cancelled)));
    }

    #[test]
    fn footprint_limit_allows_exact_limit_and_rejects_one_more_voxel() {
        let geometry = identity_geometry([5; 3]);
        let spec = BrushSpec {
            shape: BrushShape::Square,
            depth: BrushDepth::Volume,
            size: BrushSize::voxels(5).unwrap(),
        };
        let cancel = CancellationToken::default();
        assert_eq!(
            brush_indices_cancellable(&geometry, [2; 3], 0, spec, &cancel, Some(125))
                .unwrap()
                .len(),
            125
        );
        assert!(matches!(
            brush_indices_cancellable(&geometry, [2; 3], 0, spec, &cancel, Some(124)),
            Err(AppError::InvalidDimensions(message)) if message.contains("124-voxel")
        ));
    }

    #[test]
    fn small_footprint_uses_a_small_initial_allocation() {
        let footprint = brush_indices_cancellable(
            &identity_geometry([1; 3]),
            [0; 3],
            0,
            BrushSpec {
                shape: BrushShape::Round,
                depth: BrushDepth::Slice,
                size: BrushSize::voxels(1).unwrap(),
            },
            &CancellationToken::default(),
            None,
        )
        .unwrap();

        assert_eq!(footprint, vec![0]);
        assert!(footprint.capacity() <= INITIAL_BRUSH_INDEX_CAPACITY * 2);
    }
}
