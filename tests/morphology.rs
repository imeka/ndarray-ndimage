use ndarray::s;

use ndarray_ndimage::{binary_dilation, binary_erosion, dim_minus_1, Kernel3d, Mask};

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

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.7.0)
fn test_binary_dilation_plain() {
    let w = 7;
    let h = 7;
    let d = 7;

    let mut mask = Mask::from_elem((w, h, d), false);
    mask.slice_mut(s![2..w - 1, 2..h - 1, 2..d - 1]).fill(true);

    let mut gt = Mask::from_elem((w, h, d), false);

    // [0, 0, 0, 0, 0, 0, 0]
    // [0, 0, 0, 0, 0, 0, 0]
    // [0, 0, 1, 1, 1, 1, 0]
    // [0, 0, 1, 1, 1, 1, 0]
    // [0, 0, 1, 1, 1, 1, 0]
    // [0, 0, 1, 1, 1, 1, 0]
    // [0, 0, 0, 0, 0, 0, 0]
    gt.slice_mut(s![1, 2..h - 1, 2..d - 1]).fill(true);
    gt.slice_mut(s![6, 2..h - 1, 2..d - 1]).fill(true);

    // [0, 0, 0, 0, 0, 0, 0]
    // [0, 0, 1, 1, 1, 1, 0]
    // [0, 1, 1, 1, 1, 1, 1]
    // [0, 1, 1, 1, 1, 1, 1]
    // [0, 1, 1, 1, 1, 1, 1]
    // [0, 1, 1, 1, 1, 1, 1]
    // [0, 0, 1, 1, 1, 1, 0]
    gt.slice_mut(s![2..w - 1, 1, 2..d - 1]).fill(true);
    gt.slice_mut(s![2..w - 1, 2..h - 1, 1..d]).fill(true);
    gt.slice_mut(s![2..w - 1, h - 1, 2..d - 1]).fill(true);

    assert_eq!(gt, binary_dilation(&mask, Kernel3d::Star));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.7.0)
fn test_binary_dilation_corner() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask.slice_mut(s![7.., 7.., 7..]).fill(false);

    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![8.., 8.., 8..]).fill(false);

    assert_eq!(gt, binary_dilation(&mask, Kernel3d::Full));
}
