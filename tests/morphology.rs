use ndarray::{s, Array3, ShapeBuilder};

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
    let star = Kernel3d::Star.generate();
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    // Remove the star shape in the image center.
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![4..7, 4..7, 4..7]).assign(&!&star);

    assert_eq!(gt, binary_erosion(&mask, &star, 1));
}

#[test] // Results verified with the `binary_erosion` function from SciPy. (v1.9)
fn test_binary_erosion_ball_kernel() {
    let ball = Kernel3d::Ball.generate();
    let mut mask = Mask::from_elem((11, 11, 11), true);
    mask[(5, 5, 5)] = false;

    // Remove the ball shape in the image center.
    let mut gt = Mask::from_elem((11, 11, 11), true);
    gt.slice_mut(s![4..7, 4..7, 4..7]).assign(&!&ball);

    assert_eq!(gt, binary_erosion(&mask, &ball, 1));
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
    gt.slice_mut(s![0..3, 1..4, 1..4]).assign(&star);
    gt[(0, 2, 1)] = true;
    assert_eq!(binary_dilation(&mask.view(), &star, 1), gt);

    let mut star = Kernel3d::Star.generate();
    star[(1, 0, 2)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt.slice_mut(s![0..3, 1..4, 1..4]).assign(&star);
    gt[(1, 1, 3)] = true;
    assert_eq!(binary_dilation(&mask.view(), &star, 1), gt);

    let mut mask = Mask::from_elem((4, 5, 6), true);
    mask[(2, 2, 1)] = false;

    let mut star = Kernel3d::Star.generate();
    star[(0, 1, 0)] = true;
    let mut gt = Mask::from_elem(mask.dim(), true);
    gt.slice_mut(s![1..4, 1..4, 0..3]).assign(&!Kernel3d::Star.generate());
    gt[(3, 2, 2)] = false;
    assert_eq!(binary_erosion(&mask.view(), &star, 1), gt);

    let mut star = Kernel3d::Star.generate();
    star[(1, 0, 2)] = true;
    let mut gt = Mask::from_elem(mask.dim(), true);
    gt.slice_mut(s![1..4, 1..4, 0..3]).assign(&!Kernel3d::Star.generate());
    gt[(2, 3, 0)] = false;
    assert_eq!(binary_erosion(&mask.view(), &star, 1), gt);
}

#[test] // Results are logical. Both orders should always give the same results.
fn test_memory_order() {
    let mut star = Kernel3d::Star.generate();
    let test_owned = |dim: (usize, usize, usize), kernel: &Array3<bool>| {
        let test = Array3::from_elem(dim, true);
        let c = binary_erosion(&test, &kernel, 1);
        let mut test_f = Array3::from_elem(test.dim().f(), true);
        test_f.assign(&test);
        let f = binary_erosion(&test_f, &kernel, 1);
        assert_eq!(c, f);
    };
    test_owned((4, 5, 6), &star);
    test_owned((5, 5, 5), &star);
    test_owned((6, 5, 4), &star);

    star[(0, 1, 0)] = true;
    test_owned((4, 5, 6), &star);
    test_owned((5, 5, 5), &star);
    test_owned((6, 5, 4), &star);

    star[(0, 1, 0)] = false;
    star[(1, 0, 2)] = true;
    test_owned((4, 5, 6), &star);
    test_owned((5, 5, 5), &star);
    test_owned((6, 5, 4), &star);

    let mut star_f = Array3::from_elem(star.dim().f(), false);
    star_f.assign(&star);
    test_owned((4, 5, 6), &star_f);
    test_owned((5, 5, 5), &star_f);
    test_owned((6, 5, 4), &star_f);

    star_f[(0, 1, 0)] = true;
    test_owned((4, 5, 6), &star_f);
    test_owned((5, 5, 5), &star_f);
    test_owned((6, 5, 4), &star_f);

    star_f[(0, 1, 0)] = false;
    star_f[(1, 0, 2)] = true;
    test_owned((4, 5, 6), &star_f);
    test_owned((5, 5, 5), &star_f);
    test_owned((6, 5, 4), &star_f);

    let kernel = Array3::from_elem((5, 5, 5), true);
    let kernel_view = kernel.slice(s![..;2, ..;2, ..;2]);
    let test_view = |dim: (usize, usize, usize)| {
        let test = Array3::from_elem(dim, true);
        let c = binary_erosion(&test, &kernel_view, 1);
        let mut test_f = Array3::from_elem(test.dim().f(), true);
        test_f.assign(&test);
        let f = binary_erosion(&test_f, &kernel_view, 1);
        assert_eq!(c, f);
    };
    test_view((4, 5, 6));
    test_view((5, 5, 5));
    test_view((6, 5, 4));
}
