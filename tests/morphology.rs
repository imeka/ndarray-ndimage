use ndarray::s;

use ndarray_image::{binary_erosion, dim_minus_1, Kernel3d, Mask};

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.7.0)
fn test_binary_erosion() {
    let mut mask = Mask::from_elem((4, 5, 6), true);

    let mut gt = Mask::from_elem((4, 5, 6), false);
    gt.slice_mut(s![1..3, 1..4, 1..5]).fill(true);
    assert_eq!(binary_erosion(&mask, Kernel3d::Star), gt);

    mask[(0, 2, 2)] = false;
    gt[(1, 2, 2)] = false;
    assert_eq!(binary_erosion(&mask, Kernel3d::Star), gt);
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.7.0)
fn test_binary_erosion_hole() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    let mut gt = Mask::from_elem((11, 11, 11), false);
    let (width, height, depth) = dim_minus_1(&mask);
    gt.slice_mut(s![1..width, 1..height, 1..depth]).fill(true);
    // Remove the star shape in the image center.
    gt.slice_mut(s![4..7, 5, 5]).fill(false);
    gt.slice_mut(s![5, 4..7, 5]).fill(false);
    gt.slice_mut(s![5, 5, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, Kernel3d::Star));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.7.0)
fn test_binary_erosion_cube_kernel() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    let mut gt = Mask::from_elem((11, 11, 11), false);
    let (width, height, depth) = dim_minus_1(&mask);
    gt.slice_mut(s![1..width, 1..height, 1..depth]).fill(true);
    // Remove the cube shape in the image center.
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, Kernel3d::Full));
}
