mod common;

use std::collections::HashSet;

use voxel_ink::{
    brush::{
        BrushDiameter, Ray, dda_first_occupied, disk_indices, ray_edit_target, sphere_indices,
        supercover_line,
    },
    volume::{SliceAxis, SpatialUnits, Tool, VolumeGeometry, linear_index, transform_vector},
};

use common::identity_geometry;

#[test]
fn brush_diameters_accept_only_odd_values_in_the_fixed_range() {
    for accepted in (1..=31).step_by(2) {
        assert_eq!(BrushDiameter::new(accepted).unwrap().get(), accepted);
    }
    for rejected in [0, 2, 16, 30, 32, u8::MAX] {
        assert!(BrushDiameter::new(rejected).is_err());
    }
}

#[test]
fn disk_and_sphere_footprints_are_integer_symmetric_and_clipped() {
    let diameter = BrushDiameter::new(5).unwrap();
    let dimensions = [7, 7, 7];
    let disk = disk_indices(dimensions, SliceAxis::Coronal, [3, 4, 3], diameter);
    assert_eq!(disk.len(), 13);
    assert_eq!(
        disk.iter().copied().collect::<HashSet<_>>().len(),
        disk.len()
    );
    for index in disk {
        let voxel = voxel_for_index(dimensions, index);
        assert_eq!(voxel[1], 4);
        let dx = i64::from(voxel[0]) - 3;
        let dz = i64::from(voxel[2]) - 3;
        assert!(dx * dx + dz * dz <= 4);
    }

    let sphere = sphere_indices(dimensions, [3, 3, 3], diameter);
    assert_eq!(sphere.len(), 33);
    assert_eq!(
        sphere.iter().copied().collect::<HashSet<_>>().len(),
        sphere.len()
    );
    assert_eq!(
        disk_indices(dimensions, SliceAxis::Axial, [0, 0, 0], diameter).len(),
        6
    );
    assert_eq!(sphere_indices(dimensions, [0, 0, 0], diameter).len(), 11);
}

#[test]
fn supercover_includes_every_tied_neighbor_without_duplicates() {
    let diagonal = supercover_line([0, 0, 0], [1, 1, 1]).unwrap();
    let expected = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
    assert_eq!(
        diagonal.iter().copied().collect::<HashSet<_>>(),
        HashSet::from(expected)
    );
    assert_eq!(diagonal.first(), Some(&[0, 0, 0]));
    assert_eq!(diagonal.last(), Some(&[1, 1, 1]));

    let reverse = supercover_line([5, 3, 1], [0, 0, 0]).unwrap();
    assert_eq!(reverse.first(), Some(&[5, 3, 1]));
    assert_eq!(reverse.last(), Some(&[0, 0, 0]));
    assert_eq!(
        reverse.iter().copied().collect::<HashSet<_>>().len(),
        reverse.len()
    );
    for pair in reverse.windows(2) {
        assert!(
            (0..3).all(|axis| pair[0][axis].abs_diff(pair[1][axis]) <= 1),
            "supercover emitted a gap: {:?}",
            pair
        );
    }
}

#[test]
fn dda_uses_the_complete_affine_and_returns_the_first_surface() {
    let dimensions = [5, 3, 2];
    let affine = [
        [0.0, -2.0, 0.25, 10.0],
        [3.0, 0.0, 0.5, -5.0],
        [0.0, 0.0, 4.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let geometry = VolumeGeometry::new(dimensions, affine, SpatialUnits::Millimeter).unwrap();
    let mut occupancy = vec![0; 30];
    occupancy[linear_index(dimensions, [2, 1, 0]).unwrap() as usize] = 1;
    occupancy[linear_index(dimensions, [4, 1, 0]).unwrap() as usize] = 1;
    let ray = world_ray_from_index(&geometry, [-2.0, 1.0, 0.0], [1.0, 0.0, 0.0]);

    let hit = dda_first_occupied(dimensions, &occupancy, &geometry, ray).unwrap();
    assert_eq!(hit.voxel, [2, 1, 0]);
    assert_eq!(hit.previous, Some([1, 1, 0]));
    assert_eq!(hit.face_normal, [-1, 0, 0]);
    assert_eq!(
        ray_edit_target(
            dimensions,
            &occupancy,
            &geometry,
            ray,
            Tool::Eraser,
            SliceAxis::Axial,
            0,
        ),
        Some([2, 1, 0])
    );
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

    let reverse = world_ray_from_index(&geometry, [7.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
    let reverse_hit = dda_first_occupied(dimensions, &occupancy, &geometry, reverse).unwrap();
    assert_eq!(reverse_hit.voxel, [4, 1, 0]);
    assert_eq!(reverse_hit.previous, None);
    assert_eq!(reverse_hit.face_normal, [1, 0, 0]);
}

#[test]
fn an_empty_volume_is_seeded_only_by_a_forward_brush_slice_intersection() {
    let dimensions = [5, 5, 5];
    let geometry = identity_geometry(dimensions);
    let occupancy = vec![0; 125];
    let forward = Ray {
        origin: [2.2, 1.7, -5.0],
        direction: [0.0, 0.0, 1.0],
    };
    assert_eq!(
        ray_edit_target(
            dimensions,
            &occupancy,
            &geometry,
            forward,
            Tool::Brush,
            SliceAxis::Axial,
            3,
        ),
        Some([2, 2, 3])
    );
    assert_eq!(
        ray_edit_target(
            dimensions,
            &occupancy,
            &geometry,
            forward,
            Tool::Eraser,
            SliceAxis::Axial,
            3,
        ),
        None
    );

    let parallel = Ray {
        origin: [2.0, 2.0, 1.0],
        direction: [1.0, 0.0, 0.0],
    };
    assert_eq!(
        ray_edit_target(
            dimensions,
            &occupancy,
            &geometry,
            parallel,
            Tool::Brush,
            SliceAxis::Axial,
            3,
        ),
        None
    );
    let away = Ray {
        origin: [2.0, 2.0, -5.0],
        direction: [0.0, 0.0, -1.0],
    };
    assert_eq!(
        ray_edit_target(
            dimensions,
            &occupancy,
            &geometry,
            away,
            Tool::Brush,
            SliceAxis::Axial,
            3,
        ),
        None
    );
}

#[test]
fn dda_rejects_degenerate_and_non_finite_rays() {
    let geometry = identity_geometry([2, 2, 2]);
    let occupancy = vec![1; 8];
    assert!(
        dda_first_occupied(
            [2, 2, 2],
            &occupancy,
            &geometry,
            Ray {
                origin: [0.0; 3],
                direction: [0.0; 3],
            },
        )
        .is_none()
    );
    assert!(
        dda_first_occupied(
            [2, 2, 2],
            &occupancy,
            &geometry,
            Ray {
                origin: [f64::NAN, 0.0, 0.0],
                direction: [1.0, 0.0, 0.0],
            },
        )
        .is_none()
    );
}

fn voxel_for_index(dimensions: [u32; 3], index: u32) -> [u32; 3] {
    let plane = dimensions[0] * dimensions[1];
    let z = index / plane;
    let remainder = index % plane;
    [remainder % dimensions[0], remainder / dimensions[0], z]
}

fn world_ray_from_index(geometry: &VolumeGeometry, origin: [f64; 3], direction: [f64; 3]) -> Ray {
    Ray {
        origin: geometry.index_point_to_world(origin),
        direction: transform_vector(geometry.index_to_world, direction),
    }
}
