use crate::{
    AppError, Result,
    volume::{SliceAxis, Tool, VolumeGeometry, linear_index},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrushDiameter(u8);

impl BrushDiameter {
    pub const MIN: u8 = 1;
    pub const MAX: u8 = 31;

    pub fn new(value: u8) -> Result<Self> {
        if !(Self::MIN..=Self::MAX).contains(&value) || value.is_multiple_of(2) {
            return Err(AppError::InvalidDimensions(
                "brush diameter must be odd and between 1 and 31".into(),
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn radius(self) -> i32 {
        (self.0 / 2) as i32
    }
}

impl Default for BrushDiameter {
    fn default() -> Self {
        Self(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DdaHit {
    pub voxel: [u32; 3],
    pub previous: Option<[u32; 3]>,
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

#[must_use]
pub fn disk_indices(
    dimensions: [u32; 3],
    axis: SliceAxis,
    center: [u32; 3],
    diameter: BrushDiameter,
) -> Vec<u32> {
    let radius = diameter.radius();
    let varying_axes = match axis {
        SliceAxis::Axial => [0, 1],
        SliceAxis::Coronal => [0, 2],
        SliceAxis::Sagittal => [1, 2],
    };
    let mut output = Vec::with_capacity((diameter.get() as usize).pow(2));
    for second in -radius..=radius {
        for first in -radius..=radius {
            if first * first + second * second > radius * radius {
                continue;
            }
            let mut voxel = center.map(i64::from);
            voxel[varying_axes[0]] += i64::from(first);
            voxel[varying_axes[1]] += i64::from(second);
            if let Some(voxel) = checked_voxel(voxel, dimensions)
                && let Some(index) = linear_index(dimensions, voxel)
            {
                output.push(index);
            }
        }
    }
    output
}

#[must_use]
pub fn sphere_indices(dimensions: [u32; 3], center: [u32; 3], diameter: BrushDiameter) -> Vec<u32> {
    let radius = diameter.radius();
    let mut output = Vec::with_capacity((diameter.get() as usize).pow(3));
    for z in -radius..=radius {
        for y in -radius..=radius {
            for x in -radius..=radius {
                if x * x + y * y + z * z > radius * radius {
                    continue;
                }
                let voxel = [
                    i64::from(center[0]) + i64::from(x),
                    i64::from(center[1]) + i64::from(y),
                    i64::from(center[2]) + i64::from(z),
                ];
                if let Some(voxel) = checked_voxel(voxel, dimensions)
                    && let Some(index) = linear_index(dimensions, voxel)
                {
                    output.push(index);
                }
            }
        }
    }
    output
}

#[must_use]
pub fn dda_first_occupied(
    dimensions: [u32; 3],
    occupancy: &[u8],
    geometry: &VolumeGeometry,
    world_ray: Ray,
) -> Option<DdaHit> {
    let expected_len = dimensions
        .into_iter()
        .try_fold(1_usize, |total, axis| total.checked_mul(axis as usize))?;
    if occupancy.len() != expected_len {
        return None;
    }
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
    let mut previous = None;
    let mut face_normal = [0_i8; 3];
    if entry >= 0.0
        && let Some(axis) = entry_axis
    {
        face_normal[axis] = if direction[axis] > 0.0 { -1 } else { 1 };
    }
    loop {
        let current = checked_voxel(voxel, dimensions)?;
        let index = linear_index(dimensions, current)? as usize;
        if occupancy.get(index).copied() == Some(1) {
            return Some(DdaHit {
                voxel: current,
                previous,
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
            let neighbor_index = linear_index(dimensions, neighbor)? as usize;
            if occupancy[neighbor_index] == 0 {
                previous = Some(neighbor);
                face_normal[axis] = i8::try_from(-step[axis]).ok()?;
                break;
            }
        }
    }
}

#[must_use]
pub fn ray_edit_target(
    dimensions: [u32; 3],
    occupancy: &[u8],
    geometry: &VolumeGeometry,
    world_ray: Ray,
    tool: Tool,
    active_slice: SliceAxis,
    slice_coordinate: u32,
) -> Option<[u32; 3]> {
    if let Some(hit) = dda_first_occupied(dimensions, occupancy, geometry, world_ray) {
        return match tool {
            Tool::Inspect | Tool::Eraser => Some(hit.voxel),
            Tool::Brush => hit.previous.or_else(|| {
                let candidate = [
                    i64::from(hit.voxel[0]) + i64::from(hit.face_normal[0]),
                    i64::from(hit.voxel[1]) + i64::from(hit.face_normal[1]),
                    i64::from(hit.voxel[2]) + i64::from(hit.face_normal[2]),
                ];
                checked_voxel(candidate, dimensions)
            }),
        };
    }
    if tool != Tool::Brush {
        return None;
    }
    let origin = geometry.world_point_to_index(world_ray.origin);
    let direction = geometry.world_vector_to_index(world_ray.direction);
    let axis = active_slice.normal_axis();
    if direction[axis].abs() <= f64::EPSILON {
        return None;
    }
    let t = (f64::from(slice_coordinate) - origin[axis]) / direction[axis];
    if t < 0.0 || !t.is_finite() {
        return None;
    }
    let point = [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ];
    let mut voxel = point.map(|value| value.round() as i64);
    voxel[axis] = i64::from(slice_coordinate);
    checked_voxel(voxel, dimensions)
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
    use crate::volume::SpatialUnits;

    fn identity_geometry(dimensions: [u32; 3]) -> VolumeGeometry {
        VolumeGeometry::new(
            dimensions,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            SpatialUnits::Unknown,
        )
        .unwrap()
    }

    #[test]
    fn disk_and_sphere_are_clipped_and_symmetric() {
        let diameter = BrushDiameter::new(3).unwrap();
        assert_eq!(
            disk_indices([5, 5, 5], SliceAxis::Axial, [2, 2, 2], diameter).len(),
            5
        );
        assert_eq!(sphere_indices([5, 5, 5], [2, 2, 2], diameter).len(), 7);
        assert_eq!(sphere_indices([5, 5, 5], [0, 0, 0], diameter).len(), 4);
    }

    #[test]
    fn supercover_includes_corner_touching_cells() {
        let cells = supercover_line([0, 0, 0], [2, 2, 0]).unwrap();
        assert!(cells.contains(&[1, 0, 0]));
        assert!(cells.contains(&[0, 1, 0]));
        assert!(cells.contains(&[1, 1, 0]));
        assert_eq!(cells.last(), Some(&[2, 2, 0]));
    }

    #[test]
    fn dda_hits_first_cell_and_brush_uses_outside_neighbor() {
        let dimensions = [4, 3, 2];
        let mut occupancy = vec![0; 24];
        occupancy[linear_index(dimensions, [2, 1, 0]).unwrap() as usize] = 1;
        let geometry = identity_geometry(dimensions);
        let ray = Ray {
            origin: [-2.0, 1.0, 0.0],
            direction: [1.0, 0.0, 0.0],
        };
        let hit = dda_first_occupied(dimensions, &occupancy, &geometry, ray).unwrap();
        assert_eq!(hit.voxel, [2, 1, 0]);
        assert_eq!(hit.previous, Some([1, 1, 0]));
        assert_eq!(
            ray_edit_target(
                dimensions,
                &occupancy,
                &geometry,
                ray,
                Tool::Brush,
                SliceAxis::Axial,
                0,
            ),
            Some([1, 1, 0])
        );
    }

    #[test]
    fn dda_steps_tied_axes_without_false_edge_hits() {
        let dimensions = [2, 2, 1];
        let geometry = identity_geometry(dimensions);
        let ray = Ray {
            origin: [-1.0, -1.0, 0.0],
            direction: [1.0, 1.0, 0.0],
        };
        let mut occupancy = [0; 4];
        occupancy[linear_index(dimensions, [1, 0, 0]).unwrap() as usize] = 1;
        assert_eq!(
            dda_first_occupied(dimensions, &occupancy, &geometry, ray),
            None
        );
        occupancy.fill(0);
        occupancy[linear_index(dimensions, [1, 1, 0]).unwrap() as usize] = 1;
        let hit = dda_first_occupied(dimensions, &occupancy, &geometry, ray).unwrap();
        assert_eq!(hit.voxel, [1, 1, 0]);
        assert!(matches!(hit.previous, Some([0, 1, 0] | [1, 0, 0])));
    }

    #[test]
    fn miss_seeds_active_slice() {
        let dimensions = [5, 5, 5];
        let geometry = identity_geometry(dimensions);
        let target = ray_edit_target(
            dimensions,
            &[0; 125],
            &geometry,
            Ray {
                origin: [2.0, 2.0, -5.0],
                direction: [0.0, 0.0, 1.0],
            },
            Tool::Brush,
            SliceAxis::Axial,
            3,
        );
        assert_eq!(target, Some([2, 2, 3]));
    }
}
