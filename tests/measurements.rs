use ndarray::{arr3, s, Array3};

use ndarray_ndimage::{
    label, label_histogram, largest_connected_components, most_frequent_label, Kernel3d, Mask,
};

#[test] // Results verified with the `label` function from SciPy. (v1.7.0)
fn test_label_0() {
    let star = Kernel3d::Star.generate();
    let data = Array3::zeros((3, 3, 3));
    let (labels, nb_features) = label(&data.mapv(|v| v > 0), &star);
    assert_eq!(labels, data);
    assert_eq!(nb_features, 0);
    assert_eq!(label_histogram(&labels, nb_features), vec![27]);
    assert_eq!(most_frequent_label(&labels, nb_features), None);
}

#[test] // Results verified with the `label` function from SciPy. (v1.7.0)
fn test_label_2() {
    let star = Kernel3d::Star.generate();
    let data = arr3(&[
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ]);
    let (labels, nb_features) = label(&data.mapv(|v| v > 0), &star);
    assert_eq!(labels, data);
    assert_eq!(nb_features, 1);
    assert_eq!(label_histogram(&labels, nb_features), vec![18, 9]);
    assert_eq!(most_frequent_label(&labels, nb_features), Some((1, 9)));
}

#[test] // Results verified with the `label` function from SciPy. (v1.7.0)
fn test_label_3() {
    let star = Kernel3d::Star.generate();
    let data = arr3(&[
        [[2, 2, 2], [2, 2, 2], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ]);
    let gt = arr3(&[
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]);
    let (labels, nb_features) = label(&data.mapv(|v| v > 0), &star);
    assert_eq!(labels, gt);
    assert_eq!(nb_features, 2);
    assert_eq!(label_histogram(&labels, nb_features), vec![12, 6, 9]);
    assert_eq!(most_frequent_label(&labels, nb_features), Some((2, 9)));
}

#[test] // Results verified with the `label` function from SciPy. (v1.7.0)
fn test_label_4() {
    let star = Kernel3d::Star.generate();
    let data = arr3(&[
        [[0.9, 0.9, 0.9, 0.9], [0.0, 0.7, 0.8, 0.7], [0.0, 0.0, 0.0, 0.0]],
        [[0.9, 0.9, 0.9, 0.8], [0.9, 0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 0.7]],
        [[0.9, 0.9, 0.9, 0.9], [0.9, 0.8, 0.9, 0.8], [0.0, 0.0, 0.9, 0.8]],
        [[0.8, 0.7, 0.7, 0.8], [0.9, 0.0, 0.8, 0.9], [0.9, 0.9, 0.8, 0.7]],
        [[0.7, 0.7, 0.8, 0.8], [0.9, 0.8, 0.9, 0.9], [0.9, 0.8, 0.7, 0.7]],
        [[0.0, 0.9, 0.7, 0.7], [0.7, 0.9, 0.9, 0.8], [0.7, 0.9, 0.9, 0.8]],
        [[0.0, 0.8, 0.7, 0.7], [0.7, 0.7, 0.9, 0.9], [0.8, 0.8, 0.8, 0.9]],
        [[0.0, 0.7, 0.8, 0.8], [0.0, 0.7, 0.9, 0.9], [0.7, 0.8, 0.7, 0.9]],
        [[0.0, 0.7, 0.7, 0.7], [0.0, 0.8, 0.8, 0.7], [0.7, 0.9, 0.7, 0.7]],
        [[0.0, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.0, 0.9, 0.8, 0.8]],
        [[0.0, 0.8, 0.9, 0.8], [0.7, 0.8, 0.8, 0.9], [0.0, 0.0, 0.8, 0.8]],
        [[0.0, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.9, 0.8, 0.9], [0.7, 0.9, 0.8, 0.7], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.7, 0.7], [0.0, 0.0, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.7, 0.7, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ]);
    let gt = arr3(&[
        [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]],
        [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 2, 2], [0, 0, 2, 0], [0, 0, 0, 0]],
        [[3, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
    ]);
    let (labels, nb_features) = label(&data.mapv(|v| v >= 0.7), &star);
    assert_eq!(labels, gt);
    assert_eq!(nb_features, 3);
    assert_eq!(label_histogram(&gt, nb_features), vec![71, 127, 3, 3]);
    assert_eq!(most_frequent_label(&gt, nb_features), Some((1, 127)));
}

#[test] // Results verified with the `label` function from SciPy. (v1.7.0)
fn test_label_5() {
    let star = Kernel3d::Star.generate();
    let data = arr3(&[
        [
            [0.9, 0.8, 0.7, 0.0, 0.7],
            [0.0, 1.0, 0.8, 0.0, 0.8],
            [0.0, 0.9, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.7],
            [0.0, 0.0, 0.0, 0.8, 1.0],
            [0.0, 0.0, 0.0, 0.8, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.0, 1.0, 0.8, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.9, 0.8, 0.0],
            [0.0, 0.0, 1.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ]);
    let gt = arr3(&[
        [
            [1, 1, 1, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
        ],
    ]);
    let (labels, nb_features) = label(&data.mapv(|v| v >= 0.7).view(), &star.view());
    assert_eq!(labels, gt);
    assert_eq!(nb_features, 3);
    assert_eq!(label_histogram(&gt, nb_features), vec![113, 33, 2, 2]);
    assert_eq!(most_frequent_label(&gt, nb_features), Some((1, 33)));
}

#[test] // Results verified manually.
fn test_largest_connected_components() {
    let star = Kernel3d::Star.generate();
    let mut mask = Mask::from_elem((10, 10, 10), false);
    mask.slice_mut(s![2..4, 2..4, 2..4]).fill(true);
    mask.slice_mut(s![6..8, 6..8, 6..8]).fill(true);
    mask[(7, 7, 8)] = true;

    let mut gt = Mask::from_elem(mask.dim(), false);
    gt.slice_mut(s![6..8, 6..8, 6..8]).fill(true);
    gt[(7, 7, 8)] = true;
    assert_eq!(largest_connected_components(&mask, &star).unwrap(), gt);

    mask[(3, 3, 4)] = true;
    mask[(3, 4, 4)] = true;
    let mut gt = Mask::from_elem(mask.dim(), false);
    gt.slice_mut(s![2..4, 2..4, 2..4]).fill(true);
    gt[(3, 3, 4)] = true;
    gt[(3, 4, 4)] = true;
    assert_eq!(largest_connected_components(&mask.view(), &star.view()).unwrap(), gt);
}

#[test] // Results verified with the `label` function from SciPy. (v1.9.1)
fn test_label_different_kernels() {
    let data = arr3(&[
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    ]);
    let star_result = arr3(&[
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2]],
        [[0, 0, 0, 0], [1, 0, 3, 0], [0, 0, 0, 0]],
        [[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 6, 0]],
    ]);
    let ball_result = arr3(&[
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    ]);
    let full_result = arr3(&[
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    ]);
    let odd1_kernel = arr3(&[
        [[true, false, true], [true, false, false], [false, true, true]],
        [[false, true, false], [false, true, false], [false, true, false]],
        [[true, true, false], [false, false, true], [true, false, true]],
    ]);
    let odd1_result = arr3(&[
        [[0, 1, 0, 0], [2, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    ]);
    let odd2_kernel = arr3(&[
        [[true, false, true], [false, false, false], [true, false, true]],
        [[false, false, false], [false, true, false], [false, false, false]],
        [[true, false, true], [false, false, false], [true, false, true]],
    ]);
    let odd2_result = arr3(&[
        [[0, 1, 0, 0], [2, 3, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        [[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 5, 0]],
    ]);
    {
        let (labels, nb_features) = label(&data.mapv(|v| v > 0), &Kernel3d::Star.generate());
        assert_eq!(labels, star_result);
        assert_eq!(nb_features, 6);
    }
    {
        let (labels, nb_features) = label(&data.mapv(|v| v > 0), &Kernel3d::Ball.generate());
        assert_eq!(labels, ball_result);
        assert_eq!(nb_features, 2);
    }
    {
        let (labels, nb_features) = label(&data.mapv(|v| v > 0), &Kernel3d::Full.generate());
        assert_eq!(labels, full_result);
        assert_eq!(nb_features, 1);
    }
    {
        let (labels, nb_features) = label(&data.mapv(|v| v > 0), &odd1_kernel);
        assert_eq!(labels, odd1_result);
        assert_eq!(nb_features, 2);
    }
    {
        let (labels, nb_features) = label(&data.mapv(|v| v > 0), &odd2_kernel);
        assert_eq!(labels, odd2_result);
        assert_eq!(nb_features, 6);
    }
}
