mod common;

use std::collections::HashSet;

use voxel_ink::{
    AppError,
    brush::{
        BrushDepth, BrushShape, BrushSize, BrushSpec, Ray, brush_indices, ray_plane_voxel,
        ray_surface_hit, supercover_line, surface_edit_target,
    },
    volume::{SpatialUnits, Tool, VolumeGeometry, linear_index, transform_vector},
};

use common::identity_geometry;

#[test]
fn brush_sizes_validate_the_full_supported_ranges() {
    for diameter in (BrushSize::MIN_VOXELS..=BrushSize::MAX_VOXELS).step_by(2) {
        assert_eq!(
            BrushSize::voxels(diameter).unwrap(),
            BrushSize::Voxels(diameter)
        );
    }
    for diameter in [0, 2, 50, 100, 102, u8::MAX] {
        assert!(BrushSize::voxels(diameter).is_err());
    }

    for diameter in [0.5, 3.25, 50.0] {
        assert_eq!(
            BrushSize::millimeters(diameter).unwrap(),
            BrushSize::Millimeters(diameter)
        );
    }
    for diameter in [0.0, 0.49, 50.01, f32::INFINITY, f32::NAN] {
        assert!(BrushSize::millimeters(diameter).is_err());
    }
}

#[test]
fn voxel_brushes_have_exact_round_square_slice_and_volume_footprints() {
    let geometry = identity_geometry([9; 3]);
    let center = [4; 3];
    let round = BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Slice,
        size: BrushSize::voxels(5).unwrap(),
    };
    let square = BrushSpec {
        shape: BrushShape::Square,
        ..round
    };

    for normal_axis in 0..3 {
        let round_slice = brush_indices(&geometry, center, normal_axis, round).unwrap();
        assert_eq!(round_slice.len(), 13);
        assert_unique(&round_slice);
        for index in round_slice {
            let voxel = voxel_for_index(geometry.dimensions, index);
            assert_eq!(voxel[normal_axis], center[normal_axis]);
            let squared_distance = (0..3)
                .filter(|&axis| axis != normal_axis)
                .map(|axis| {
                    i64::from(voxel[axis])
                        .abs_diff(i64::from(center[axis]))
                        .pow(2)
                })
                .sum::<u64>();
            assert!(squared_distance <= 4);
        }

        let square_slice = brush_indices(&geometry, center, normal_axis, square).unwrap();
        assert_eq!(square_slice.len(), 25);
        assert_unique(&square_slice);
        assert!(square_slice.into_iter().all(|index| {
            let voxel = voxel_for_index(geometry.dimensions, index);
            voxel[normal_axis] == center[normal_axis]
                && (0..3).all(|axis| voxel[axis].abs_diff(center[axis]) <= 2)
        }));
    }

    let round_volume =
        brush_indices(&geometry, center, 0, round.with_depth(BrushDepth::Volume)).unwrap();
    assert_eq!(round_volume.len(), 33);
    assert_unique(&round_volume);
    assert!(round_volume.into_iter().all(|index| {
        voxel_for_index(geometry.dimensions, index)
            .into_iter()
            .zip(center)
            .map(|(coordinate, center)| i64::from(coordinate).abs_diff(i64::from(center)).pow(2))
            .sum::<u64>()
            <= 4
    }));

    let square_volume =
        brush_indices(&geometry, center, 2, square.with_depth(BrushDepth::Volume)).unwrap();
    assert_eq!(square_volume.len(), 125);
    assert_unique(&square_volume);
}

#[test]
fn brush_footprints_clip_exactly_at_volume_boundaries() {
    let geometry = identity_geometry([7; 3]);
    let center = [0; 3];
    let round = BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Slice,
        size: BrushSize::voxels(5).unwrap(),
    };
    let square = BrushSpec {
        shape: BrushShape::Square,
        ..round
    };

    assert_eq!(brush_indices(&geometry, center, 2, round).unwrap().len(), 6);
    assert_eq!(
        brush_indices(&geometry, center, 2, square).unwrap().len(),
        9
    );
    assert_eq!(
        brush_indices(&geometry, center, 2, round.with_depth(BrushDepth::Volume))
            .unwrap()
            .len(),
        11
    );
    assert_eq!(
        brush_indices(&geometry, center, 2, square.with_depth(BrushDepth::Volume))
            .unwrap()
            .len(),
        27
    );
}

#[test]
fn millimeter_brushes_respect_anisotropic_spacing_and_depth() {
    let geometry = VolumeGeometry::new(
        [15; 3],
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Millimeter,
    )
    .unwrap();
    let center = [7; 3];
    let round = BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Volume,
        size: BrushSize::millimeters(3.0).unwrap(),
    };
    let footprint = brush_indices(&geometry, center, 2, round).unwrap();
    for voxel in [[4, 7, 7], [10, 7, 7], [7, 6, 7], [7, 8, 7]] {
        assert!(footprint.contains(&linear_index(geometry.dimensions, voxel).unwrap()));
    }
    for voxel in [[3, 7, 7], [11, 7, 7], [7, 7, 6], [7, 7, 8]] {
        assert!(!footprint.contains(&linear_index(geometry.dimensions, voxel).unwrap()));
    }

    let square = BrushSpec {
        shape: BrushShape::Square,
        ..round
    };
    assert_eq!(
        brush_indices(&geometry, center, 2, square).unwrap().len(),
        21
    );
    assert_eq!(
        brush_indices(&geometry, center, 0, square.with_depth(BrushDepth::Slice))
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn round_millimeter_brush_uses_the_complete_affine_metric() {
    let geometry = VolumeGeometry::new(
        [9; 3],
        [
            [1.0, 1.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Millimeter,
    )
    .unwrap();
    let footprint = brush_indices(
        &geometry,
        [4; 3],
        2,
        BrushSpec {
            shape: BrushShape::Round,
            depth: BrushDepth::Slice,
            size: BrushSize::millimeters(2.0).unwrap(),
        },
    )
    .unwrap();

    assert!(footprint.contains(&linear_index(geometry.dimensions, [5, 3, 4]).unwrap()));
    assert!(!footprint.contains(&linear_index(geometry.dimensions, [5, 5, 4]).unwrap()));
}

#[test]
fn millimeter_brushes_convert_meter_and_micron_affines() {
    for (spacing, units) in [
        (0.001, SpatialUnits::Meter),
        (1.0, SpatialUnits::Millimeter),
        (1_000.0, SpatialUnits::Micron),
    ] {
        let geometry = VolumeGeometry::new(
            [5; 3],
            [
                [spacing, 0.0, 0.0, 0.0],
                [0.0, spacing, 0.0, 0.0],
                [0.0, 0.0, spacing, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            units,
        )
        .unwrap();
        let footprint = brush_indices(
            &geometry,
            [2; 3],
            2,
            BrushSpec {
                shape: BrushShape::Round,
                depth: BrushDepth::Volume,
                size: BrushSize::millimeters(2.0).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(footprint.len(), 7, "wrong conversion for {units:?}");
    }
}

#[test]
fn invalid_brush_specs_and_unknown_physical_units_are_rejected() {
    let unknown_geometry = VolumeGeometry::new(
        [5; 3],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        SpatialUnits::Unknown,
    )
    .unwrap();
    let physical = BrushSpec {
        shape: BrushShape::Round,
        depth: BrushDepth::Slice,
        size: BrushSize::millimeters(3.0).unwrap(),
    };
    assert!(matches!(
        brush_indices(&unknown_geometry, [2; 3], 2, physical),
        Err(AppError::InvalidDimensions(_))
    ));

    for size in [
        BrushSize::Voxels(2),
        BrushSize::Voxels(0),
        BrushSize::Millimeters(f32::NAN),
    ] {
        assert!(
            brush_indices(
                &unknown_geometry,
                [2; 3],
                2,
                BrushSpec {
                    size,
                    ..BrushSpec::default()
                }
            )
            .is_err()
        );
    }
    assert!(brush_indices(&unknown_geometry, [5, 2, 2], 2, BrushSpec::default()).is_err());
    assert!(brush_indices(&unknown_geometry, [2; 3], 3, BrushSpec::default()).is_err());
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
    assert_unique_voxels(&reverse);
    for pair in reverse.windows(2) {
        assert!(
            (0..3).all(|axis| pair[0][axis].abs_diff(pair[1][axis]) <= 1),
            "supercover emitted a gap: {pair:?}"
        );
    }
}

#[test]
fn surface_picking_uses_the_complete_affine_and_returns_the_first_surface() {
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

    let hit = ray_surface_hit(&occupancy, &geometry, ray).unwrap();
    assert_eq!(hit.voxel, [2, 1, 0]);
    assert_eq!(hit.exterior, Some([1, 1, 0]));
    assert_eq!(hit.face_normal, [-1, 0, 0]);
    assert_eq!(
        surface_edit_target(dimensions, &occupancy, &geometry, ray, Tool::Paint),
        Some([1, 1, 0])
    );
    for tool in [Tool::Inspect, Tool::Erase, Tool::Bridge] {
        assert_eq!(
            surface_edit_target(dimensions, &occupancy, &geometry, ray, tool),
            Some([2, 1, 0])
        );
    }

    let reverse = world_ray_from_index(&geometry, [7.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
    let reverse_hit = ray_surface_hit(&occupancy, &geometry, reverse).unwrap();
    assert_eq!(reverse_hit.voxel, [4, 1, 0]);
    assert_eq!(reverse_hit.exterior, None);
    assert_eq!(reverse_hit.face_normal, [1, 0, 0]);
    assert_eq!(
        surface_edit_target(dimensions, &occupancy, &geometry, reverse, Tool::Paint),
        None
    );
}

#[test]
fn surface_picking_never_falls_back_to_an_active_slice_or_empty_space() {
    let dimensions = [5; 3];
    let geometry = identity_geometry(dimensions);
    let occupancy = vec![0; 125];
    let ray = Ray {
        origin: [2.0, 2.0, -5.0],
        direction: [0.0, 0.0, 1.0],
    };

    assert_eq!(ray_surface_hit(&occupancy, &geometry, ray), None);
    for tool in [Tool::Paint, Tool::Erase, Tool::Bridge, Tool::Inspect] {
        assert_eq!(
            surface_edit_target(dimensions, &occupancy, &geometry, ray, tool),
            None
        );
    }
    assert_eq!(
        surface_edit_target([4, 5, 5], &occupancy, &geometry, ray, Tool::Paint),
        None
    );
}

#[test]
fn surface_traversal_steps_tied_axes_without_false_edge_hits() {
    let dimensions = [2, 2, 1];
    let geometry = identity_geometry(dimensions);
    let ray = Ray {
        origin: [-1.0, -1.0, 0.0],
        direction: [1.0, 1.0, 0.0],
    };
    let mut occupancy = [0; 4];
    occupancy[linear_index(dimensions, [1, 0, 0]).unwrap() as usize] = 1;
    assert_eq!(ray_surface_hit(&occupancy, &geometry, ray), None);
    occupancy.fill(0);
    occupancy[linear_index(dimensions, [1, 1, 0]).unwrap() as usize] = 1;
    let hit = ray_surface_hit(&occupancy, &geometry, ray).unwrap();
    assert_eq!(hit.voxel, [1, 1, 0]);
    assert!(matches!(hit.exterior, Some([0, 1, 0] | [1, 0, 0])));
}

#[test]
fn surface_picking_rejects_bad_buffers_and_degenerate_rays() {
    let geometry = identity_geometry([2; 3]);
    let occupancy = vec![1; 8];
    for ray in [
        Ray {
            origin: [0.0; 3],
            direction: [0.0; 3],
        },
        Ray {
            origin: [f64::NAN, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
        },
    ] {
        assert_eq!(ray_surface_hit(&occupancy, &geometry, ray), None);
    }
    assert_eq!(
        ray_surface_hit(
            &occupancy[..7],
            &geometry,
            Ray {
                origin: [-1.0, 0.0, 0.0],
                direction: [1.0, 0.0, 0.0]
            }
        ),
        None
    );
}

#[test]
fn bridge_plane_projection_uses_world_space_and_rounds_to_a_voxel() {
    let dimensions = [7; 3];
    let affine = [
        [0.0, -2.0, 0.25, 10.0],
        [3.0, 0.0, 0.5, -5.0],
        [0.0, 0.0, 4.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let geometry = VolumeGeometry::new(dimensions, affine, SpatialUnits::Millimeter).unwrap();
    let ray = world_ray_from_index(&geometry, [5.2, 1.8, -2.0], [0.0, 0.0, 1.0]);

    assert_eq!(
        ray_plane_voxel(dimensions, &geometry, ray, [3, 3, 3], [0.0, 0.0, 8.0]),
        Some([5, 2, 3])
    );
}

#[test]
fn bridge_plane_projection_rejects_unsafe_intersections() {
    let dimensions = [7; 3];
    let geometry = identity_geometry(dimensions);
    let anchor = [3; 3];
    let normal = [0.0, 0.0, 1.0];

    for ray in [
        Ray {
            origin: [2.0, 2.0, 0.0],
            direction: [1.0, 0.0, 0.0],
        },
        Ray {
            origin: [2.0, 2.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        },
        Ray {
            origin: [20.0, 2.0, 0.0],
            direction: [0.0, 0.0, 1.0],
        },
    ] {
        assert_eq!(
            ray_plane_voxel(dimensions, &geometry, ray, anchor, normal),
            None
        );
    }
    let forward = Ray {
        origin: [2.0, 2.0, 0.0],
        direction: [0.0, 0.0, 1.0],
    };
    assert_eq!(
        ray_plane_voxel([6, 7, 7], &geometry, forward, anchor, normal),
        None
    );
    assert_eq!(
        ray_plane_voxel(dimensions, &geometry, forward, anchor, [0.0; 3]),
        None
    );
    assert_eq!(
        ray_plane_voxel(dimensions, &geometry, forward, anchor, [f64::NAN, 0.0, 1.0]),
        None
    );
}

fn assert_unique(indices: &[u32]) {
    assert_eq!(
        indices.iter().copied().collect::<HashSet<_>>().len(),
        indices.len()
    );
}

fn assert_unique_voxels(voxels: &[[u32; 3]]) {
    assert_eq!(
        voxels.iter().copied().collect::<HashSet<_>>().len(),
        voxels.len()
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
