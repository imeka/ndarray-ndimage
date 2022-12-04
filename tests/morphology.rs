use ndarray::{s, Array3};

use ndarray_ndimage::{
    binary_closing, binary_dilation, binary_erosion, binary_opening, Kernel3d, Mask,
};

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion() {
    let star = Kernel3d::Star.generate();

    let mut mask = Mask::from_elem((4, 5, 6), true);
    mask[(0, 2, 2)] = false;
    let mut gt = Mask::from_elem((4, 5, 6), true);
    gt[(0, 1, 2)] = false;
    gt[(0, 2, 1)] = false;
    gt[(0, 2, 2)] = false;
    gt[(0, 2, 3)] = false;
    gt[(0, 3, 2)] = false;
    gt[(1, 2, 2)] = false;
    assert_eq!(binary_erosion(&mask.view(), &star, 1), gt);

    let mut mask = Mask::from_elem((6, 7, 8), false);
    mask.slice_mut(s![1..5, 1..6, 1..7]).fill(true);
    let mut gt = Mask::from_elem((6, 7, 8), false);
    gt.slice_mut(s![2..4, 2..5, 2..6]).fill(true);
    assert_eq!(binary_erosion(&mask, &star, 1), gt);

    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask.slice_mut(s![2.., 1.., 1..]).fill(true);
    let mut gt = Mask::from_elem((7, 7, 7), false);
    gt.slice_mut(s![4.., 3.., 3..]).fill(true);
    assert_eq!(gt, binary_erosion(&mask.view(), &star, 2));

    let mut mask = Mask::from_elem((9, 9, 9), false);
    mask.slice_mut(s![2.., 1.., ..]).fill(true);
    let mut gt = Mask::from_elem((9, 9, 9), false);
    gt.slice_mut(s![5.., 4.., ..]).fill(true);
    assert_eq!(gt, binary_erosion(&mask.view(), &star, 3));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_hole() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    // Remove the star shape in the image center.
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![4..7, 5, 5]).fill(false);
    gt.slice_mut(s![5, 4..7, 5]).fill(false);
    gt.slice_mut(s![5, 5, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Star.generate(), 1));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_ball_kernel() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    // Remove the ball shape in the image center.
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(false);
    gt.slice_mut(s![4..7; 2, 4..7; 2, 4..7; 2]).fill(true);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Ball.generate(), 1));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_full_kernel() {
    let kernel5 = Array3::from_elem((5, 5, 5), true);

    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    // Remove the cube shape in the image center.
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![4..7, 4..7, 4..7]).fill(false);

    assert_eq!(gt, binary_erosion(&mask, &Kernel3d::Full.generate(), 1));

    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![3..8, 3..8, 3..8]).fill(false);
    assert_eq!(gt, binary_erosion(&mask, &kernel5, 1));

    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(10, 10, 10)] = false;
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![6.., 6.., 6..]).fill(false);
    assert_eq!(gt, binary_erosion(&mask, &kernel5, 2));

    let mask = Mask::from_elem((13, 13, 13), true);
    let gt = Mask::from_elem((13, 13, 13), true);
    assert_eq!(gt, binary_erosion(&mask, &kernel5, 3));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_dilation_plain() {
    let kernel5 = Array3::from_elem((5, 5, 5), true);
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

    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Star.generate(), 1));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask.slice_mut(s![4, 4, 4..]).fill(true);
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![3..6, 3..6, 3..]).fill(true);
    gt.slice_mut(s![3..6; 2, 3..6; 2, 3]).fill(false);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Ball.generate(), 1));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(4, 4, 4)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![2.., 2.., 2..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Full.generate(), 2));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(4, 5, 5)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![1.., 2.., 2..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask.view(), &Kernel3d::Full.generate(), 3));

    let mut mask = Mask::from_elem((w, h, d), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((w, h, d), false);
    gt.slice_mut(s![1..6, 2.., 3..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask, &kernel5, 1));

    let mut mask = Mask::from_elem((9, 9, 9), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((9, 9, 9), false);
    gt.slice_mut(s![..8, .., 1..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask, &kernel5, 2));

    let mut mask = Mask::from_elem((11, 11, 11), false);
    mask[(3, 4, 5)] = true;
    let mut gt = Mask::from_elem((11, 11, 11), false);
    gt.slice_mut(s![..10, .., ..]).fill(true);
    assert_eq!(gt, binary_dilation(&mask, &kernel5, 3));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_dilation_corner() {
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask.slice_mut(s![7.., 7.., 7..]).fill(false);

    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![8.., 8.., 8..]).fill(false);

    assert_eq!(gt, binary_dilation(&mask, &Kernel3d::Full.generate(), 1));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_opening() {
    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask.slice_mut(s![2..6, 2..6, 2..6]).fill(true);
    let mut gt = Mask::from_elem(mask.dim(), false);
    assert_eq!(gt, binary_opening(&mask, &Kernel3d::Star.generate(), 2));

    mask.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
    gt.slice_mut(s![1..6, 1..6, 1..6]).fill(true);
    assert_eq!(gt, binary_opening(&mask.view(), &Kernel3d::Full.generate(), 2));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_binary_closing() {
    let star = Kernel3d::Star.generate();

    let mut mask = Mask::from_elem((7, 7, 7), false);
    mask[(3, 3, 3)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt[(3, 3, 3)] = true;
    assert_eq!(gt, binary_closing(&mask.view(), &star, 2));

    // SciPy use a single border_value of 0 or 1, while we use 0 (dilation) and 1 (erosion)
    mask.slice_mut(s![2..5, 2..5, 2..5]).fill(true);
    gt.slice_mut(s![2..5, 2..5, 2..5]).fill(true);
    gt[(1, 3, 3)] = true;
    gt[(3, 1, 3)] = true;
    gt[(3, 3, 1)] = true;
    gt[(3, 3, 5)] = true;
    gt[(3, 5, 3)] = true;
    gt[(5, 3, 3)] = true;
    assert_eq!(gt, binary_closing(&mask, &star, 2));

    mask.slice_mut(s![1.., 1.., 1..]).fill(true);
    mask.slice_mut(s![2..5, 2..5, 2..5]).fill(false);
    let mut gt = Mask::from_elem(mask.dim(), true);
    gt[(0, 0, 0)] = false;
    gt[(0, 0, 1)] = false;
    gt[(0, 0, 2)] = false;
    gt[(0, 1, 0)] = false;
    gt[(0, 1, 1)] = false;
    gt[(0, 2, 0)] = false;
    gt[(1, 0, 0)] = false;
    gt[(1, 1, 0)] = false;
    gt[(1, 0, 1)] = false;
    gt[(2, 0, 0)] = false;
    assert_eq!(gt, binary_closing(&mask.view(), &star, 2));
}

#[test] // Results verified with the `binary_dilation` function from SciPy. (v1.9)
fn test_asymmetric_kernel() {
    let mut mask = Mask::from_elem((4, 5, 6), false);
    mask[(1, 2, 2)] = true;

    let mut star = Kernel3d::Star.generate();
    star[(0, 1, 0)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt[(0, 2, 1)] = true;
    gt[(0, 2, 2)] = true;
    gt[(1, 1, 2)] = true;
    gt[(1, 2, 1)] = true;
    gt[(1, 2, 2)] = true;
    gt[(1, 2, 3)] = true;
    gt[(1, 3, 2)] = true;
    gt[(2, 2, 2)] = true;
    assert_eq!(binary_dilation(&mask.view(), &star, 1), gt);

    let mut star = Kernel3d::Star.generate();
    star[(1, 0, 2)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt[(0, 2, 2)] = true;
    gt[(1, 1, 2)] = true;
    gt[(1, 1, 3)] = true;
    gt[(1, 2, 1)] = true;
    gt[(1, 2, 2)] = true;
    gt[(1, 2, 3)] = true;
    gt[(1, 3, 2)] = true;
    gt[(2, 2, 2)] = true;
    assert_eq!(binary_dilation(&mask.view(), &star, 1), gt);

    let mut mask = Mask::from_elem((4, 5, 6), true);
    mask[(2, 2, 1)] = false;

    let mut star = Kernel3d::Star.generate();
    star[(0, 1, 0)] = true;
    let mut gt = Mask::from_elem(mask.dim(), true);
    gt[(1, 2, 1)] = false;
    gt[(2, 1, 1)] = false;
    gt[(2, 2, 0)] = false;
    gt[(2, 2, 1)] = false;
    gt[(2, 2, 2)] = false;
    gt[(2, 3, 1)] = false;
    gt[(3, 2, 1)] = false;
    gt[(3, 2, 2)] = false;
    assert_eq!(binary_erosion(&mask.view(), &star, 1), gt);

    let mut star = Kernel3d::Star.generate();
    star[(1, 0, 2)] = true;
    let mut gt = Mask::from_elem(mask.dim(), true);
    gt[(1, 2, 1)] = false;
    gt[(2, 1, 1)] = false;
    gt[(2, 2, 0)] = false;
    gt[(2, 2, 1)] = false;
    gt[(2, 2, 2)] = false;
    gt[(2, 3, 0)] = false;
    gt[(2, 3, 1)] = false;
    gt[(3, 2, 1)] = false;
    assert_eq!(binary_erosion(&mask.view(), &star, 1), gt);
}
