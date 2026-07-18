mod common;

use voxel_ink::{
    AppError,
    volume::{SliceAxis, SpatialUnits, VolumeGeometry, linear_index, transform_vector},
};

use common::make_document;

#[test]
fn linear_index_is_x_major_and_checks_each_bound() {
    let dimensions = [4, 3, 2];
    assert_eq!(linear_index(dimensions, [0, 0, 0]), Some(0));
    assert_eq!(linear_index(dimensions, [3, 0, 0]), Some(3));
    assert_eq!(linear_index(dimensions, [0, 1, 0]), Some(4));
    assert_eq!(linear_index(dimensions, [0, 0, 1]), Some(12));
    assert_eq!(linear_index(dimensions, [3, 2, 1]), Some(23));
    assert_eq!(linear_index(dimensions, [4, 0, 0]), None);
    assert_eq!(linear_index(dimensions, [0, 3, 0]), None);
    assert_eq!(linear_index(dimensions, [0, 0, 2]), None);

    assert_eq!(
        linear_index([65_536, 32_768, 1], [65_535, 32_767, 0]),
        Some(u32::MAX / 2)
    );
}

#[test]
fn oblique_affine_round_trips_points_vectors_and_spacing() {
    let affine = [
        [0.0, -2.0, 0.3, 10.0],
        [3.0, 0.0, 0.4, -5.0],
        [0.0, 0.0, 4.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let geometry = VolumeGeometry::new([7, 8, 9], affine, SpatialUnits::Millimeter).unwrap();
    assert_eq!(geometry.spacing[0], 3.0);
    assert_eq!(geometry.spacing[1], 2.0);
    assert!((geometry.spacing[2] - 4.031_128_874_149_275).abs() < 1.0e-12);

    let index_point = [1.25, -2.5, 4.75];
    let recovered = geometry.world_point_to_index(geometry.index_point_to_world(index_point));
    for axis in 0..3 {
        assert!((recovered[axis] - index_point[axis]).abs() < 1.0e-10);
    }

    let index_vector = [1.0, 0.0, 0.0];
    let world_vector = transform_vector(affine, index_vector);
    assert_eq!(world_vector, [0.0, 3.0, 0.0]);
    let recovered_vector = geometry.world_vector_to_index(world_vector);
    for axis in 0..3 {
        assert!((recovered_vector[axis] - index_vector[axis]).abs() < 1.0e-12);
    }
}

#[test]
fn geometry_rejects_empty_singular_and_non_finite_inputs() {
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert!(matches!(
        VolumeGeometry::new([0, 1, 1], identity, SpatialUnits::Unknown),
        Err(AppError::InvalidDimensions(_))
    ));

    let mut singular = identity;
    singular[2] = singular[1];
    assert!(matches!(
        VolumeGeometry::new([1; 3], singular, SpatialUnits::Unknown),
        Err(AppError::InvalidAffine)
    ));

    let mut non_finite = identity;
    non_finite[0][0] = f64::NAN;
    assert!(matches!(
        VolumeGeometry::new([1; 3], non_finite, SpatialUnits::Unknown),
        Err(AppError::InvalidAffine)
    ));
}

#[test]
fn all_slice_axes_use_canonical_pixel_order_and_reject_bad_coordinates() {
    let occupancy = vec![
        0, 1, 0, 1, 0, 1, // z = 0
        1, 1, 0, 0, 1, 1, // z = 1
    ];
    let document = make_document([3, 2, 2], occupancy, false);

    let axial = document.extract_slice(SliceAxis::Axial, 1).unwrap();
    assert_eq!((axial.width, axial.height), (3, 2));
    assert_eq!(axial.pixels, vec![1, 1, 0, 0, 1, 1]);

    let coronal = document.extract_slice(SliceAxis::Coronal, 1).unwrap();
    assert_eq!((coronal.width, coronal.height), (3, 2));
    assert_eq!(coronal.pixels, vec![1, 0, 1, 0, 1, 1]);

    let sagittal = document.extract_slice(SliceAxis::Sagittal, 1).unwrap();
    assert_eq!((sagittal.width, sagittal.height), (2, 2));
    assert_eq!(sagittal.pixels, vec![1, 0, 1, 1]);

    assert!(document.extract_slice(SliceAxis::Axial, 2).is_none());
    assert!(document.extract_slice(SliceAxis::Coronal, 2).is_none());
    assert!(document.extract_slice(SliceAxis::Sagittal, 3).is_none());
}

#[test]
fn cursor_coordinates_are_clamped_to_the_document() {
    let mut document = make_document([3, 2, 4], vec![0; 24], false);
    assert_eq!(document.cursor(), [1, 1, 2]);
    document.set_cursor([u32::MAX, 0, 99]);
    assert_eq!(document.cursor(), [2, 0, 3]);
    assert_eq!(document.summary().cursor, [2, 0, 3]);
}
