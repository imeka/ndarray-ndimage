use ndarray::{s, Array3};

use ndarray_ndimage::{
    binary_closing, binary_dilation, binary_erosion, binary_opening, dim_minus, Kernel3d, Mask,
};

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion() {
    let mut mask = Mask::from_elem((4, 5, 6), true);

    let mut gt = Mask::from_elem((4, 5, 6), false);
    gt.slice_mut(s![1..3, 1..4, 1..5]).fill(true);
    assert_eq!(binary_erosion(&mask, &Kernel3d::Star, 1), gt);

    mask[(0, 2, 2)] = false;
    gt[(1, 2, 2)] = false;
    assert_eq!(binary_erosion(&mask.view(), &Kernel3d::Star, 1), gt);

    let mut mask = Mask::from_elem((6, 7, 8), false);
    mask.slice_mut(s![1..5, 1..6, 1..7]).fill(true);
    let mut gt = Mask::from_elem((6, 7, 8), false);
    gt.slice_mut(s![2..4, 2..5, 2..6]).fill(true);
    assert_eq!(binary_erosion(&mask, &Kernel3d::Star, 1), gt);

    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask.slice_mut(s![2.., 1.., 1..]).fill(true);
    let mut gt = Mask::from_elem((7, 7, 7), false);
    gt.slice_mut(s![4, 3..5, 3..5]).fill(true);
    assert_eq!(gt, binary_erosion(&mask.view(), &Kernel3d::Star, 2));

    let mut mask = Mask::from_elem((9, 9, 9), false);
    mask.slice_mut(s![2.., 1.., ..]).fill(true);
    let mut gt = Mask::from_elem((9, 9, 9), false);
    gt.slice_mut(s![5, 4..6, 3..6]).fill(true);
    assert_eq!(gt, binary_erosion(&mask.view(), &Kernel3d::Star, 3));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_hole() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    let mut gt = Mask::from_elem((11, 11, 11), false);
    let (width, height, depth) = dim_minus(&mask, 1);
    gt.slice_mut(s![1..width, 1..height, 1..depth]).fill(true);
    // Remove the star shape in the image center.
    gt.slice_mut(s![4..7, 5, 5]).fill(false);
    gt.slice_mut(s![5, 4..7, 5]).fill(false);
    gt.slice_mut(s![5, 5, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Star, 1));
    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::GenericOwned(Kernel3d::Star.array()), 1));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_ball_kernel() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    let mut gt = Mask::from_elem((11, 11, 11), false);
    let (width, height, depth) = dim_minus(&mask, 1);
    gt.slice_mut(s![1..width, 1..height, 1..depth]).fill(true);
    // Remove the ball shape in the image center.
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(false);
    gt.slice_mut(s![4..7; 2, 4..7; 2, 4..7; 2]).fill(true);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Ball, 1));
    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::GenericOwned(Kernel3d::Ball.array()), 1));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_full_kernel() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    let mut gt = Mask::from_elem((11, 11, 11), false);
    let (width, height, depth) = dim_minus(&mask, 1);
    gt.slice_mut(s![1..width, 1..height, 1..depth]).fill(true);
    // Remove the cube shape in the image center.
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Full, 1));
    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::GenericOwned(Kernel3d::Full.array()), 1));

    let mut gt = Mask::from_elem((11, 11, 11), false);
    gt.slice_mut(s![2..-2, 2..-2, 2..-2]).fill(true);
    gt.slice_mut(s![3..8, 3..8, 3..8]).fill(false);
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_erosion(&mask, &kernel, 1));

    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(10, 10, 10)] = false;
    let mut gt = Mask::from_elem((11, 11, 11), false);
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(true);
    gt[(6, 6, 6)] = false;
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_erosion(&mask, &kernel, 2));

    let mask = Mask::from_elem((13, 13, 13), true);
    let mut gt = Mask::from_elem((13, 13, 13), false);
    gt[(6, 6, 6)] = true;
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_erosion(&mask, &kernel, 3));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
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

    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Star, 1));
    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::GenericOwned(Kernel3d::Star.array()), 1));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask.slice_mut(s![4, 4, 4..]).fill(true);
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![3..6, 3..6, 3..]).fill(true);
    gt.slice_mut(s![3..6; 2, 3..6; 2, 3]).fill(false);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Ball, 1));
    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::GenericOwned(Kernel3d::Ball.array()), 1));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(4, 4, 4)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![2.., 2.., 2..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Full, 2));
    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::GenericOwned(Kernel3d::Full.array()), 2));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(4, 5, 5)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![1.., 2.., 2..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Full, 3));
    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::GenericOwned(Kernel3d::Full.array()), 3));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![1..6, 2.., 3..]).fill(true);
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_dilation(&mask, &kernel, 1));

    let mut mask = Mask::from_elem((9, 9, 9), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((9, 9, 9), false);
    gt.slice_mut(s![..8, .., 1..]).fill(true);
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_dilation(&mask, &kernel, 2));

    let mut mask = Mask::from_elem((11, 11, 11), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((11, 11, 11), false);
    gt.slice_mut(s![..10, .., ..]).fill(true);
    let kernel = Kernel3d::GenericOwned(Array3::from_elem((5, 5, 5), true));
    assert_eq!(gt, binary_dilation(&mask, &kernel, 3));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_dilation_corner() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask.slice_mut(s![7.., 7.., 7..]).fill(false);

    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![8.., 8.., 8..]).fill(false);

    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::Full, 1));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_opening() {
    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask.slice_mut(s![2..6, 2..6, 2..6]).fill(true);
    let mut gt = Mask::from_elem(mask.dim(), false);
    assert_eq!(gt, binary_opening(&mask.view(), &Kernel3d::Star, 2));

    mask.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
    gt.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
    assert_eq!(gt, binary_opening(&mask.view(), &Kernel3d::Full, 2));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_closing() {
    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask[(3, 3, 3)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt[(3, 3, 3)] = true;
    assert_eq!(gt, binary_closing(&mask.view(), &Kernel3d::Star, 2));

    mask.slice_mut(s![2..5, 2..5, 2..5]).fill(true);
    gt.slice_mut(s![2..5, 2..5, 2..5]).fill(true);
    assert_eq!(gt, binary_closing(&mask.view(), &Kernel3d::Star, 2));

    mask.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
    mask.slice_mut(s![2..5, 2..5, 2..5]).fill(false);
    assert_eq!(gt, binary_closing(&mask.view(), &Kernel3d::Star, 2));
}
