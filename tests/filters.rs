use approx::assert_relative_eq;
use ndarray::{arr1, arr2, s, Array1, Array2, Axis};

use ndarray_ndimage::{
    convolve, convolve1d, correlate, correlate1d, gaussian_filter, maximum_filter,
    maximum_filter1d, median_filter, minimum_filter, minimum_filter1d, prewitt, sobel,
    uniform_filter, BorderMode, Mask,
};

#[test] // Results verified with SciPy. (v1.9.0)
fn test_convolve1d() {
    let arr = arr1(&[2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]);
    let arr_odd = arr1(&[2.0, 8.0, 0.0, 4.0, 9.0, 9.0, 0.0]);
    let matrix = arr2(&[
        [1.5, 2.3, 0.7, 1.1, 6.0, 1.7],
        [0.5, 1.3, 0.0, 0.1, 1.2, 0.7],
        [0.4, 1.3, 2.7, 0.1, 0.8, 0.1],
        [2.1, 0.1, 0.7, 0.1, 1.0, 2.8],
        [5.7, 4.0, 1.8, 9.1, 4.8, 2.7],
    ]);

    assert_eq!(
        convolve1d(&arr.mapv(|v| v as u32), &arr1(&[1, 3]), Axis(0), BorderMode::Reflect, -1),
        arr1(&[8, 14, 24, 4, 13, 12, 36, 27])
    );
    assert_eq!(
        convolve1d(&arr.mapv(|v| v as i32), &arr1(&[1, 3]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[14, 24, 4, 13, 12, 36, 27, 0])
    );
    assert_eq!(
        convolve1d(&arr, &arr1(&[1.0, 3.0, 2.0]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[18.0, 28.0, 20.0, 13.0, 20.0, 38.0, 45.0, 18.0])
    );
    assert_eq!(
        convolve1d(&arr, &arr1(&[1.0, 2.0, 0.5, 2.0]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[21.0, 12.0, 25.0, 13.0, 35.5, 24.5, 22.5, 27.0])
    );

    // Symmetric
    assert_eq!(
        convolve1d(&arr, &arr1(&[0.5, 1.5, 0.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[8.0, 13.0, 6.0, 6.5, 8.0, 18.5, 18.0, 4.5])
    );

    // Anti-symmetric
    assert_eq!(
        convolve1d(&arr, &arr1(&[0.5, 1.5, 1.0, -1.5, -0.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[7.0, 6.0, -6.5, 6.0, 13.0, 19.0, -5.0, -13.5])
    );

    // Other modes and dimensions
    assert_relative_eq!(
        convolve1d(&arr_odd, &arr1(&[1.0, 2.0, 0.5, 2.0]), Axis(0), BorderMode::Constant(0.5), 0),
        arr1(&[18.0, 12.0, 33.0, 29.0, 30.5, 23.0, 19.5]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        convolve1d(&arr_odd, &arr1(&[1.0, 3.0, 2.0]), Axis(0), BorderMode::Nearest, 0),
        arr1(&[18.0, 28.0, 20.0, 21.0, 44.0, 45.0, 18.0]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        convolve1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Mirror, 0),
        arr2(&[
            [6.8, 7.6, 3.2, 9.3, 19.7, 11.1],
            [2.8, 3.9, 0.1, 1.5, 4.3, 3.3],
            [2.5, 6.6, 8.2, 1.1, 2.5, 1.1],
            [6.4, 1., 2.2, 1.3, 5.8, 9.4],
            [21.1, 13.8, 14.5, 32.1, 17.1, 12.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        convolve1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Reflect, 0),
        arr2(&[
            [6.8, 7.6, 3.2, 9.3, 19.7, 6.8],
            [2.8, 3.9, 0.1, 1.5, 4.3, 2.8],
            [2.5, 6.6, 8.2, 1.1, 2.5, 0.4],
            [6.4, 1.0, 2.2, 1.3, 5.8, 11.2],
            [21.1, 13.8, 14.5, 32.1, 17.1, 10.8]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        convolve1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Wrap, 0),
        arr2(&[
            [6.8, 7.6, 3.2, 9.3, 19.7, 6.6],
            [2.8, 3.9, 0.1, 1.5, 4.3, 2.6],
            [2.5, 6.6, 8.2, 1.1, 2.5, 0.7],
            [6.4, 1.0, 2.2, 1.3, 5.8, 10.5],
            [21.1, 13.8, 14.5, 32.1, 17.1, 13.8]
        ]),
        epsilon = 1e-7,
    );

    // Do not test "origin != 0", it's only useful to test in correlate
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_correlate1d() {
    let arr = arr1(&[2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]);
    let arr_odd = arr1(&[2.0, 8.0, 0.0, 4.0, 9.0, 9.0, 0.0]);
    let matrix = arr2(&[
        [1.5, 2.3, 0.7, 1.1, 6.0, 1.7],
        [0.5, 1.3, 0.0, 0.1, 1.2, 0.7],
        [0.4, 1.3, 2.7, 0.1, 0.8, 0.1],
        [2.1, 0.1, 0.7, 0.1, 1.0, 2.8],
        [5.7, 4.0, 1.8, 9.1, 4.8, 2.7],
    ]);

    // Non-Symmetric
    assert_eq!(
        correlate1d(&arr.mapv(|v| v as u32), &arr1(&[1, 3]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 26, 8, 12, 7, 28, 36, 9])
    );
    assert_eq!(
        correlate1d(&arr.mapv(|v| v as i32), &arr1(&[1, 3, 2]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[24, 26, 16, 14, 25, 46, 36, 9])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0, 2.0, 1.0]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[26.0, 24.0, 30.0, 17.0, 23.0, 34.0, 46.0, 36.0])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.5, 3.0, 1.5, 0.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[25.0, 21.0, 29.0, 18.5, 18.0, 27.0, 42.0, 40.5])
    );

    // Symmetric
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0, 1.0]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[16.0, 26.0, 12.0, 13.0, 16.0, 37.0, 36.0, 9.0])
    );
    assert_eq!(
        correlate1d(&arr_odd, &arr1(&[1.5, 3.0, 1.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[21.0, 27.0, 18.0, 25.5, 46.5, 40.5, 13.5])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.5, 2.0, 0.5, 2.0, 1.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[33.0, 17.0, 28.5, 29.5, 40.0, 30.5, 24.0, 45.0])
    );

    // Anti-symmetric
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0, -1.0]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[0.0, 26.0, 4.0, 11.0, -2.0, 19.0, 36.0, 9.0])
    );
    assert_eq!(
        correlate1d(&arr_odd, &arr1(&[1.5, 3.0, -1.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[-3.0, 27.0, 6.0, -1.5, 19.5, 40.5, 13.5])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.5, 2.0, 0.5, -2.0, -1.5]), Axis(0), BorderMode::Reflect, 0),
        arr1(&[1.0, 5.0, 9.5, -1.5, -23.0, -5.5, 24.0, 18.0])
    );

    // Other modes and dimensions
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(0), BorderMode::Constant(0.5), 0),
        arr2(&[
            [5.0, 7.4, 2.6, 3.8, 18.5, 5.6],
            [3.0, 6.2, 0.7, 1.4, 9.6, 3.8],
            [1.7, 5.2, 8.1, 0.4, 3.6, 1.0],
            [6.7, 1.6, 4.8, 0.4, 3.8, 8.5],
            [19.2, 12.1, 6.1, 27.4, 15.4, 10.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Constant(0.5), 0),
        arr2(&[
            [5.0, 8.4, 4.4, 4.0, 19.1, 11.1],
            [2.0, 4.4, 1.3, 0.3, 3.7, 3.3],
            [1.7, 4.3, 9.4, 3.0, 2.5, 1.1],
            [6.8, 2.4, 2.2, 1.0, 3.1, 9.4],
            [17.6, 17.7, 9.4, 29.1, 23.5, 12.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(0), BorderMode::Nearest, 0),
        arr2(&[
            [6.0, 9.2, 2.8, 4.4, 24., 6.8],
            [3.0, 6.2, 0.7, 1.4, 9.6, 3.8],
            [1.7, 5.2, 8.1, 0.4, 3.6, 1.0],
            [6.7, 1.6, 4.8, 0.4, 3.8, 8.5],
            [19.2, 12.1, 6.1, 27.4, 15.4, 10.9],
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Nearest, 0),
        arr2(&[
            [6.0, 8.4, 4.4, 4.0, 19.1, 11.1],
            [2.0, 4.4, 1.3, 0.3, 3.7, 3.3],
            [1.6, 4.3, 9.4, 3.0, 2.5, 1.1],
            [8.4, 2.4, 2.2, 1.0, 3.1, 9.4],
            [22.8, 17.7, 9.4, 29.1, 23.5, 12.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(0), BorderMode::Mirror, 0),
        arr2(&[
            [5.0, 8.2, 2.1, 3.4, 19.2, 5.8],
            [3.0, 6.2, 0.7, 1.4, 9.6, 3.8],
            [1.7, 5.2, 8.1, 0.4, 3.6, 1.0],
            [6.7, 1.6, 4.8, 0.4, 3.8, 8.5],
            [19.2, 12.1, 6.1, 27.4, 15.4, 10.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Mirror, 0),
        arr2(&[
            [6.8, 8.4, 4.4, 4., 19.1, 11.1],
            [2.8, 4.4, 1.3, 0.3, 3.7, 3.3],
            [2.5, 4.3, 9.4, 3.0, 2.5, 1.1],
            [6.4, 2.4, 2.2, 1.0, 3.1, 9.4],
            [21.1, 17.7, 9.4, 29.1, 23.5, 12.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Reflect, 0),
        arr2(&[
            [6.0, 8.4, 4.4, 4.0, 19.1, 11.1],
            [2.0, 4.4, 1.3, 0.3, 3.7, 3.3],
            [1.6, 4.3, 9.4, 3.0, 2.5, 1.1],
            [8.4, 2.4, 2.2, 1.0, 3.1, 9.4],
            [22.8, 17.7, 9.4, 29.1, 23.5, 12.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(0), BorderMode::Wrap, 0),
        arr2(&[
            [10.2, 10.9, 3.9, 12.4, 22.8, 7.8],
            [3.0, 6.2, 0.7, 1.4, 9.6, 3.8],
            [1.7, 5.2, 8.1, 0.4, 3.6, 1.0],
            [6.7, 1.6, 4.8, 0.4, 3.8, 8.5],
            [19.2, 12.1, 6.1, 27.4, 15.4, 10.9]
        ]),
        epsilon = 1e-7,
    );
    assert_relative_eq!(
        correlate1d(&matrix, &arr1(&[1.0, 3.0]), Axis(1), BorderMode::Wrap, 0),
        arr2(&[
            [6.2, 8.4, 4.4, 4., 19.1, 11.1],
            [2.2, 4.4, 1.3, 0.3, 3.7, 3.3],
            [1.3, 4.3, 9.4, 3.0, 2.5, 1.1],
            [9.1, 2.4, 2.2, 1.0, 3.1, 9.4],
            [19.8, 17.7, 9.4, 29.1, 23.5, 12.9]
        ]),
        epsilon = 1e-7,
    );

    // origin != 0
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0]), Axis(0), BorderMode::Reflect, -1),
        arr1(&[26.0, 8.0, 12.0, 7.0, 28.0, 36.0, 9.0, 0.0])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0, 2.0]), Axis(0), BorderMode::Reflect, -1),
        arr1(&[26.0, 16.0, 14.0, 25.0, 46.0, 36.0, 9.0, 18.0])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 3.0, 2.0]), Axis(0), BorderMode::Reflect, 1),
        arr1(&[18.0, 24.0, 26.0, 16.0, 14.0, 25.0, 46.0, 36.0])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 0.5, 1.0, 1.5]), Axis(0), BorderMode::Reflect, -2),
        arr1(&[12.0, 13.5, 16.5, 27.0, 14.5, 13.5, 22.5, 22.5])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 0.5, 1.0, 1.5]), Axis(0), BorderMode::Reflect, -1),
        arr1(&[11.0, 12.0, 13.5, 16.5, 27.0, 14.5, 13.5, 22.5])
    );
    assert_eq!(
        correlate1d(&arr, &arr1(&[1.0, 0.5, 1.0, 1.5]), Axis(0), BorderMode::Reflect, 1),
        arr1(&[9.0, 23.0, 11.0, 12.0, 13.5, 16.5, 27.0, 14.5])
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_convolve() {
    let a: Array1<usize> = (0..25).collect();
    let a = a.into_shape_with_order((5, 5)).unwrap();

    let weight = arr2(&[[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    assert_eq!(
        convolve(&a, &weight, BorderMode::Mirror, 1),
        arr2(&[
            [18, 21, 24, 25, 24],
            [33, 36, 39, 40, 39],
            [48, 51, 54, 55, 54],
            [53, 56, 59, 60, 59],
            [48, 51, 54, 55, 54],
        ])
    );
    assert_eq!(
        convolve(&a, &weight, BorderMode::Reflect, 0),
        arr2(&[
            [6, 8, 11, 14, 16],
            [16, 18, 21, 24, 26],
            [31, 33, 36, 39, 41],
            [46, 48, 51, 54, 56],
            [56, 58, 61, 64, 66],
        ])
    );
    assert_eq!(
        convolve(&a, &weight, BorderMode::Wrap, -1),
        arr2(&[
            [42, 40, 38, 41, 44],
            [32, 30, 28, 31, 34],
            [22, 20, 18, 21, 24],
            [37, 35, 33, 36, 39],
            [52, 50, 48, 51, 54],
        ])
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_correlate() {
    let a: Array1<usize> = (0..25).collect();
    let a = a.into_shape_with_order((5, 5)).unwrap();

    let weight = arr2(&[[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    assert_eq!(
        correlate(&a, &weight, BorderMode::Constant(2), 0),
        arr2(&[
            [8, 10, 12, 14, 8],
            [18, 18, 21, 24, 14],
            [28, 33, 36, 39, 24],
            [38, 48, 51, 54, 34],
            [24, 38, 40, 42, 44]
        ])
    );
    assert_eq!(
        correlate(&a, &weight, BorderMode::Nearest, 0),
        arr2(&[
            [6, 8, 11, 14, 16],
            [16, 18, 21, 24, 26],
            [31, 33, 36, 39, 41],
            [46, 48, 51, 54, 56],
            [56, 58, 61, 64, 66]
        ])
    );
    assert_eq!(
        correlate(&a, &weight, BorderMode::Mirror, -1),
        arr2(&[
            [18, 21, 24, 25, 24],
            [33, 36, 39, 40, 39],
            [48, 51, 54, 55, 54],
            [53, 56, 59, 60, 59],
            [48, 51, 54, 55, 54]
        ])
    );
    assert_eq!(
        correlate(&a, &weight, BorderMode::Reflect, 0),
        arr2(&[
            [6, 8, 11, 14, 16],
            [16, 18, 21, 24, 26],
            [31, 33, 36, 39, 41],
            [46, 48, 51, 54, 56],
            [56, 58, 61, 64, 66]
        ])
    );
    assert_eq!(
        correlate(&a, &weight, BorderMode::Wrap, 1),
        arr2(&[
            [42, 40, 38, 41, 44],
            [32, 30, 28, 31, 34],
            [22, 20, 18, 21, 24],
            [37, 35, 33, 36, 39],
            [52, 50, 48, 51, 54]
        ])
    );

    let weight = arr2(&[[0.0, 0.1, 0.0], [0.1, 0.9, 0.1], [0.0, 0.1, 0.0]]);
    assert_relative_eq!(
        correlate(&a.mapv(|v| v as f32), &weight, BorderMode::Reflect, 0),
        arr2(&[
            [0.6, 1.8, 3.1, 4.4, 5.6],
            [6.6, 7.8, 9.1, 10.4, 11.6],
            [13.1, 14.3, 15.6, 16.9, 18.1],
            [19.6, 20.8, 22.1, 23.4, 24.6],
            [25.6, 26.8, 28.1, 29.4, 30.6]
        ]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_median_filter() {
    let mut gt = Mask::from_elem((3, 3, 3), false);
    let mut mask = gt.clone();
    mask[(0, 0, 0)] = true;
    assert_eq!(median_filter(&mask), gt);
    mask[(1, 0, 0)] = true;
    assert_eq!(median_filter(&mask), gt);
    mask[(0, 1, 0)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(0, 0, 0)] = true;
    mask[(0, 0, 1)] = true;
    assert_eq!(median_filter(&mask), gt);

    mask[(1, 1, 0)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(1, 0, 0)] = true;
    gt[(0, 1, 0)] = true;
    gt[(0, 0, 1)] = true;
    mask[(1, 0, 1)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(2, 0, 0)] = true;
    mask[(1, 1, 1)] = true;
    assert_eq!(median_filter(&mask.view()), gt);
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_minmax_filter() {
    // Even tests
    let a = arr1(&[2, 8, 0, 4, 1, 9, 9, 0]);
    assert_eq!(
        minimum_filter1d(&a, 2, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 2, 0, 0, 1, 1, 9, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 3, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 0, 0, 0, 1, 1, 0, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 4, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 0, 0, 0, 0, 1, 0, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 6, Axis(0), BorderMode::Reflect, 0),
        arr1(&[0, 0, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
        maximum_filter1d(&a, 2, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 8, 8, 4, 4, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 3, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 4, 9, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 4, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 8, 9, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 6, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 9, 9, 9, 9, 9])
    );

    // Odd tests
    let a = arr1(&[2, 8, 0, 4, 1, -1, 9, 9, 0]);
    assert_eq!(
        minimum_filter1d(&a, 2, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 2, 0, 0, 1, -1, -1, 9, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 3, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 0, 0, 0, -1, -1, -1, 0, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 4, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 0, 0, 0, -1, -1, -1, -1, 0])
    );
    assert_eq!(
        minimum_filter1d(&a, 6, Axis(0), BorderMode::Reflect, 0),
        arr1(&[0, 0, 0, -1, -1, -1, -1, -1, -1])
    );
    assert_eq!(
        maximum_filter1d(&a, 2, Axis(0), BorderMode::Reflect, 0),
        arr1(&[2, 8, 8, 4, 4, 1, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 3, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 4, 4, 9, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 4, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 8, 4, 9, 9, 9, 9])
    );
    assert_eq!(
        maximum_filter1d(&a, 6, Axis(0), BorderMode::Reflect, 0),
        arr1(&[8, 8, 8, 8, 9, 9, 9, 9, 9])
    );

    let matrix = arr2(&[
        [1.5, 2.3, 0.7, 1.1, 6.0, 1.7],
        [0.5, 1.3, 0.0, 0.1, 1.2, 0.7],
        [0.4, 1.3, 2.7, 0.1, 0.8, 0.1],
        [2.1, 0.1, 0.7, 0.1, 1.0, 2.8],
        [5.7, 4.0, 1.8, 9.1, 4.8, 2.7],
    ]);
    assert_relative_eq!(
        minimum_filter(&matrix, 2, BorderMode::Reflect, 0),
        arr2(&[
            [1.5, 1.5, 0.7, 0.7, 1.1, 1.7],
            [0.5, 0.5, 0.0, 0.0, 0.1, 0.7],
            [0.4, 0.4, 0.0, 0.0, 0.1, 0.1],
            [0.4, 0.1, 0.1, 0.1, 0.1, 0.1],
            [2.1, 0.1, 0.1, 0.1, 0.1, 1.0]
        ])
    );
    assert_relative_eq!(
        minimum_filter(&matrix, 3, BorderMode::Reflect, 0),
        arr2(&[
            [0.5, 0.0, 0.0, 0.0, 0.1, 0.7],
            [0.4, 0.0, 0.0, 0.0, 0.1, 0.1],
            [0.1, 0.0, 0.0, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
        ])
    );
    assert_relative_eq!(
        minimum_filter(&matrix, 6, BorderMode::Reflect, 0),
        Array2::zeros(matrix.dim())
    );
    assert_relative_eq!(
        maximum_filter(&matrix, 2, BorderMode::Reflect, 0),
        arr2(&[
            [1.5, 2.3, 2.3, 1.1, 6.0, 6.0],
            [1.5, 2.3, 2.3, 1.1, 6.0, 6.0],
            [0.5, 1.3, 2.7, 2.7, 1.2, 1.2],
            [2.1, 2.1, 2.7, 2.7, 1.0, 2.8],
            [5.7, 5.7, 4.0, 9.1, 9.1, 4.8]
        ])
    );
    assert_relative_eq!(
        maximum_filter(&matrix, 3, BorderMode::Reflect, 0),
        arr2(&[
            [2.3, 2.3, 2.3, 6.0, 6.0, 6.0],
            [2.3, 2.7, 2.7, 6.0, 6.0, 6.0],
            [2.1, 2.7, 2.7, 2.7, 2.8, 2.8],
            [5.7, 5.7, 9.1, 9.1, 9.1, 4.8],
            [5.7, 5.7, 9.1, 9.1, 9.1, 4.8]
        ])
    );
    assert_relative_eq!(
        maximum_filter(&matrix, 6, BorderMode::Reflect, 0),
        arr2(&[
            [2.7, 2.7, 6.0, 6.0, 6.0, 6.0],
            [2.7, 2.7, 6.0, 6.0, 6.0, 6.0],
            [5.7, 9.1, 9.1, 9.1, 9.1, 9.1],
            [5.7, 9.1, 9.1, 9.1, 9.1, 9.1],
            [5.7, 9.1, 9.1, 9.1, 9.1, 9.1]
        ])
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_gaussian_filter_1d() {
    let mut a: Array1<f32> = (0..7).map(|v| v as f32).collect();
    assert_relative_eq!(
        gaussian_filter(&a, 1.0, 0, BorderMode::Reflect, 4),
        arr1(&[0.42704096, 1.0679559, 2.0048335, 3.0, 3.9951665, 4.932044, 5.572959]),
        epsilon = 1e-5
    );
    a[0] = 0.7;
    assert_relative_eq!(
        gaussian_filter(&a.view(), 2.0, 0, BorderMode::Reflect, 3),
        arr1(&[1.4193099, 1.737984, 2.3200142, 3.0642939, 3.8351974, 4.4778357, 4.845365]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_gaussian_filter_2d() {
    let a: Array1<f32> = (0..70).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((5, 7)).unwrap();
    a[(0, 0)] = 17.0;
    assert_relative_eq!(
        gaussian_filter(&a, 1.0, 0, BorderMode::Reflect, 4),
        arr2(&[
            [13.815777, 11.339161, 10.62479, 12.028319, 13.970364, 15.842661, 17.12449],
            [19.028267, 18.574514, 19.253122, 20.97248, 22.940516, 24.813597, 26.095427],
            [29.490631, 30.42986, 32.06769, 34.004536, 35.990467, 37.864086, 39.14592],
            [41.95432, 43.209373, 45.064693, 47.050846, 49.040836, 50.914577, 52.196407],
            [50.876965, 52.158012, 54.031227, 56.02144, 58.01176, 59.885513, 61.167343],
        ]),
        epsilon = 1e-4
    );
    let a: Array1<f32> = (0..84).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((6, 7)).unwrap();
    a[(0, 0)] = 8.5;
    assert_relative_eq!(
        gaussian_filter(&a, 1.0, 0, BorderMode::Reflect, 2),
        arr2(&[
            [10.078889, 9.458512, 10.006921, 11.707343, 13.707343, 15.598366, 16.892008],
            [17.220367, 17.630152, 18.90118, 20.76284, 22.76284, 24.653864, 25.947506],
            [29.114912, 30.247316, 32.025234, 34.000000, 36.000000, 37.89102, 39.184666],
            [42.815334, 44.10898, 46.000000, 48.000000, 50.000000, 51.89102, 53.184666],
            [56.052494, 57.346134, 59.23716, 61.23716, 63.23716, 65.12818, 66.42182],
            [65.107994, 66.401634, 68.292656, 70.292656, 72.292656, 74.18368, 75.47732],
        ]),
        epsilon = 1e-4
    );

    let a: Array1<f32> = (0..112).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((8, 7)).unwrap();
    a[(0, 0)] = 18.2;
    assert_relative_eq!(
        gaussian_filter(&a, 1.5, 0, BorderMode::Reflect, 3),
        arr2(&[
            [16.712738, 16.30507, 16.362633, 17.34964, 18.918924, 20.453388, 21.402458],
            [22.053278, 22.092232, 22.654442, 23.931578, 25.60057, 27.156698, 28.1087],
            [31.7295, 32.2731, 33.405533, 35.01049, 36.79215, 38.372753, 39.328068],
            [44.08236, 44.91609, 46.376343, 48.169773, 50.0162, 51.61088, 52.5681],
            [57.50711, 58.440548, 60.013466, 61.87167, 63.740356, 65.339874, 66.297745],
            [70.68089, 71.636, 73.2334, 75.10567, 76.979195, 78.579765, 79.53778],
            [81.8913, 82.849335, 84.45004, 86.32423, 88.1984, 89.79911, 90.75715],
            [88.59754, 89.55557, 91.15629, 93.030464, 94.90464, 96.505356, 97.46339],
        ]),
        epsilon = 1e-4
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_gaussian_filter_3d() {
    let a: Array1<f32> = (0..720).map(|v| v as f32 / 50.0).collect();
    let mut a = a.into_shape_with_order((10, 9, 8)).unwrap();
    a[(0, 0, 0)] = 0.2;
    a[(3, 3, 3)] = 1.0;

    let g = gaussian_filter(&a, 1.8, 0, BorderMode::Reflect, 4);
    assert_relative_eq!(
        g.slice(s![0, .., ..]),
        arr2(&[
            [1.647472, 1.651181, 1.659609, 1.673325, 1.691082, 1.709747, 1.725337, 1.734229],
            [1.708805, 1.712651, 1.721257, 1.735376, 1.754014, 1.773838, 1.790377, 1.799745],
            [1.818189, 1.822212, 1.831044, 1.845692, 1.865495, 1.886855, 1.904654, 1.914653],
            [1.95729, 1.961716, 1.971077, 1.986287, 2.006792, 2.028921, 2.04732, 2.057615],
            [2.110379, 2.115686, 2.126213, 2.142124, 2.16256, 2.184116, 2.201956, 2.211958],
            [2.265391, 2.271859, 2.283923, 2.300605, 2.320466, 2.340645, 2.357214, 2.366559],
            [2.409767, 2.417196, 2.430533, 2.447822, 2.467118, 2.486012, 2.501415, 2.51016],
            [2.525863, 2.53382, 2.54786, 2.565478, 2.584446, 2.602611, 2.617354, 2.625759],
            [2.591995, 2.600145, 2.614439, 2.632176, 2.651024, 2.668921, 2.683421, 2.691702],
        ]),
        epsilon = 1e-4
    );
    assert_relative_eq!(
        g.slice(s![9, .., ..]),
        arr2(&[
            [11.68823, 11.69645, 11.71083, 11.72861, 11.74741, 11.76522, 11.77964, 11.78788],
            [11.75407, 11.76228, 11.77665, 11.79442, 11.81323, 11.83105, 11.84548, 11.85373],
            [11.86941, 11.8776, 11.89196, 11.90972, 11.92854, 11.94638, 11.96082, 11.96907],
            [12.01257, 12.02076, 12.03511, 12.05287, 12.07169, 12.08953, 12.10399, 12.11224],
            [12.16671, 12.17491, 12.18926, 12.20702, 12.22584, 12.24368, 12.25812, 12.26638],
            [12.32086, 12.32907, 12.34344, 12.36121, 12.38002, 12.39784, 12.41227, 12.42051],
            [12.46405, 12.47227, 12.48665, 12.50443, 12.52324, 12.54104, 12.55545, 12.56369],
            [12.57941, 12.58764, 12.60204, 12.61982, 12.63862, 12.65641, 12.67081, 12.67905],
            [12.64527, 12.6535, 12.6679, 12.68568, 12.70448, 12.72227, 12.73667, 12.7449],
        ]),
        epsilon = 1e-4
    );
}

#[should_panic]
#[test] // Results verified with SciPy. (v1.9.0)
fn test_gaussian_filter_panic() {
    let a: Array1<f32> = (0..7).map(|v| v as f32).collect();

    let _ = gaussian_filter(&a, 2.0, 0, BorderMode::Reflect, 4);
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_1d() {
    let a: Array1<f32> = (0..7).map(|v| v as f32).collect();
    assert_relative_eq!(
        uniform_filter(&a, 1, BorderMode::Reflect),
        arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        epsilon = 1e-7
    );
    assert_relative_eq!(
        uniform_filter(&a.view(), 2, BorderMode::Reflect),
        arr1(&[0., 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        epsilon = 1e-7
    );
    assert_relative_eq!(
        uniform_filter(&a.view(), 3, BorderMode::Reflect),
        arr1(&[0.33333333, 1., 2., 3., 4., 5., 5.66666667]),
        epsilon = 1e-7
    );
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_2d() {
    let a: Array1<f32> = (0..70).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((5, 7)).unwrap();
    a[(0, 0)] = 17.0;
    assert_relative_eq!(
        uniform_filter(&a, 4, BorderMode::Reflect),
        arr2(&[
            [12.25, 12.75, 12.125, 12., 14., 16., 17.5],
            [15.75, 16.25, 15.625, 15.5, 17.5, 19.5, 21.],
            [24.125, 24.625, 25.0625, 26., 28., 30., 31.5],
            [36., 36.5, 38., 40., 42., 44., 45.5],
            [46.5, 47., 48.5, 50.5, 52.5, 54.5, 56.]
        ]),
        epsilon = 1e-5
    );
    let a: Array1<f32> = (0..84).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((6, 7)).unwrap();
    a[(0, 0)] = 8.5;
    assert_relative_eq!(
        uniform_filter(&a, 5, BorderMode::Reflect),
        arr2(&[
            [14.16, 14.96, 15.88, 17.2, 19.2, 20.8, 21.6],
            [19.76, 20.56, 21.48, 22.8, 24.8, 26.4, 27.2],
            [30.28, 31.08, 32.34, 34., 36., 37.6, 38.4],
            [43.6, 44.4, 46., 48., 50., 51.6, 52.4],
            [54.8, 55.6, 57.2, 59.2, 61.2, 62.8, 63.6],
            [60.4, 61.2, 62.8, 64.8, 66.8, 68.4, 69.2]
        ]),
        epsilon = 1e-5
    );

    let a: Array1<f32> = (0..112).step_by(2).map(|v| v as f32).collect();
    let mut a = a.into_shape_with_order((8, 7)).unwrap();
    a[(0, 0)] = 18.2;
    assert_relative_eq!(
        uniform_filter(&a, 3, BorderMode::Reflect),
        arr2(&[
            [13.42222222, 10.71111111, 8.66666667, 10.66666667, 12.66666667, 14.66666667, 16.],
            [18.71111111, 18.02222222, 18., 20., 22., 24., 25.33333333],
            [28.66666667, 30., 32., 34., 36., 38., 39.33333333],
            [42.66666667, 44., 46., 48., 50., 52., 53.33333333],
            [56.66666667, 58., 60., 62., 64., 66., 67.33333333],
            [70.66666667, 72., 74., 76., 78., 80., 81.33333333],
            [84.66666667, 86., 88., 90., 92., 94., 95.33333333],
            [94., 95.33333333, 97.33333333, 99.33333333, 101.33333333, 103.33333333, 104.66666667]
        ]),
        epsilon = 1e-4
    );
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_3d() {
    let a: Array1<f32> = (0..720).map(|v| v as f32 / 50.0).collect();
    let mut a = a.into_shape_with_order((10, 9, 8)).unwrap();
    a[(0, 0, 0)] = 0.2;
    a[(3, 3, 3)] = 1.0;

    let g = uniform_filter(&a, 6, BorderMode::Reflect);
    assert_relative_eq!(
        g.slice(s![0, .., ..]),
        arr2(&[
            [1.62740741, 1.63074074, 1.64074074, 1.6537037, 1.67, 1.69, 1.70666667, 1.71666667],
            [
                1.65407407, 1.65740741, 1.66740741, 1.68037037, 1.69666667, 1.71666667, 1.73333333,
                1.74333333
            ],
            [
                1.73407407, 1.73740741, 1.74740741, 1.76037037, 1.77666667, 1.79666667, 1.81333333,
                1.82333333
            ],
            [1.8637037, 1.86703704, 1.87703704, 1.89185185, 1.91, 1.93, 1.94666667, 1.95666667],
            [2.02, 2.02333333, 2.03333333, 2.05, 2.07, 2.09, 2.10666667, 2.11666667],
            [2.18, 2.18333333, 2.19333333, 2.21, 2.23, 2.25, 2.26666667, 2.27666667],
            [2.34, 2.34333333, 2.35333333, 2.37, 2.39, 2.41, 2.42666667, 2.43666667],
            [2.47333333, 2.47666667, 2.48666667, 2.50333333, 2.52333333, 2.54333333, 2.56, 2.57],
            [2.55333333, 2.55666667, 2.56666667, 2.58333333, 2.60333333, 2.62333333, 2.64, 2.65]
        ]),
        epsilon = 1e-4
    );
    assert_relative_eq!(
        g.slice(s![9, .., ..]),
        arr2(&[
            [11.46, 11.46333333, 11.47333333, 11.49, 11.51, 11.53, 11.54666667, 11.55666667],
            [
                11.48666667,
                11.49,
                11.5,
                11.51666667,
                11.53666667,
                11.55666667,
                11.57333333,
                11.58333333
            ],
            [
                11.56666667,
                11.57,
                11.58,
                11.59666667,
                11.61666667,
                11.63666667,
                11.65333333,
                11.66333333
            ],
            [11.7, 11.70333333, 11.71333333, 11.73, 11.75, 11.77, 11.78666667, 11.79666667],
            [11.86, 11.86333333, 11.87333333, 11.89, 11.91, 11.93, 11.94666667, 11.95666667],
            [12.02, 12.02333333, 12.03333333, 12.05, 12.07, 12.09, 12.10666667, 12.11666667],
            [12.18, 12.18333333, 12.19333333, 12.21, 12.23, 12.25, 12.26666667, 12.27666667],
            [
                12.31333333,
                12.31666667,
                12.32666667,
                12.34333333,
                12.36333333,
                12.38333333,
                12.4,
                12.41
            ],
            [
                12.39333333,
                12.39666667,
                12.40666667,
                12.42333333,
                12.44333333,
                12.46333333,
                12.48,
                12.49
            ]
        ]),
        epsilon = 1e-4
    );
}

#[should_panic]
#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_panic() {
    let a: Array1<f32> = (0..7).map(|v| v as f32).collect();

    let _ = uniform_filter(&a, 0, BorderMode::Reflect);
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_1d_ints() {
    let a: Array1<i32> = (0..7).collect();
    assert_eq!(uniform_filter(&a, 1, BorderMode::Reflect), arr1(&[0, 1, 2, 3, 4, 5, 6]));
    assert_eq!(uniform_filter(&a.view(), 2, BorderMode::Reflect), arr1(&[0, 0, 1, 2, 3, 4, 5]));
    assert_eq!(uniform_filter(&a.view(), 3, BorderMode::Reflect), arr1(&[0, 1, 2, 3, 4, 5, 5]));
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_2d_ints() {
    let a: Array1<i32> = (0..70).step_by(2).collect();
    let mut a = a.into_shape_with_order((5, 7)).unwrap();
    a[(0, 0)] = 17;
    assert_eq!(
        uniform_filter(&a, 4, BorderMode::Reflect),
        arr2(&[
            [12, 12, 12, 12, 14, 16, 17],
            [15, 16, 15, 15, 17, 19, 20],
            [24, 24, 25, 26, 28, 30, 31],
            [36, 36, 38, 40, 42, 44, 45],
            [46, 46, 48, 50, 52, 54, 55]
        ])
    );

    let a: Array1<i32> = (0..84).step_by(2).collect();
    let mut a = a.into_shape_with_order((6, 7)).unwrap();
    a[(0, 0)] = 8;
    assert_eq!(
        uniform_filter(&a, 5, BorderMode::Reflect),
        arr2(&[
            [13, 14, 15, 17, 19, 20, 21],
            [19, 20, 20, 22, 24, 25, 26],
            [30, 30, 32, 34, 36, 37, 38],
            [43, 44, 46, 48, 50, 51, 52],
            [54, 55, 57, 59, 61, 62, 63],
            [59, 60, 62, 64, 66, 67, 68]
        ])
    );

    let a: Array1<i32> = (0..112).step_by(2).collect();
    let mut a = a.into_shape_with_order((8, 7)).unwrap();
    a[(0, 0)] = 18;
    assert_eq!(
        uniform_filter(&a, 3, BorderMode::Reflect),
        arr2(&[
            [12, 10, 8, 10, 12, 14, 15],
            [18, 18, 18, 20, 22, 24, 25],
            [28, 30, 32, 34, 36, 38, 39],
            [42, 44, 46, 48, 50, 52, 53],
            [56, 58, 60, 62, 64, 66, 67],
            [70, 72, 74, 76, 78, 80, 81],
            [84, 86, 88, 90, 92, 94, 95],
            [93, 95, 97, 99, 101, 103, 104]
        ])
    );
}

#[test] // Results verified with SciPy. (v1.9.1)
fn test_uniform_filter_3d_ints() {
    let a: Array1<i32> = (0..720).map(|v| v / 5).collect();
    let mut a = a.into_shape_with_order((10, 9, 8)).unwrap();
    a[(0, 0, 0)] = 2;
    a[(3, 3, 3)] = 10;

    let g = uniform_filter(&a, 6, BorderMode::Reflect);
    assert_eq!(
        g.slice(s![0, .., ..]),
        arr2(&[
            [15, 15, 15, 15, 15, 15, 15, 16],
            [15, 15, 15, 15, 15, 16, 16, 16],
            [16, 16, 16, 16, 16, 16, 16, 17],
            [17, 17, 17, 17, 17, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 20],
            [20, 20, 20, 20, 21, 21, 21, 21],
            [22, 22, 22, 22, 22, 22, 23, 23],
            [23, 23, 23, 23, 24, 24, 24, 24],
            [24, 24, 24, 24, 24, 25, 25, 25]
        ]),
    );
    assert_eq!(
        g.slice(s![3, .., ..]),
        arr2(&[
            [37, 37, 37, 37, 37, 37, 37, 37],
            [37, 36, 37, 37, 37, 37, 37, 38],
            [38, 37, 37, 37, 38, 38, 38, 38],
            [39, 38, 39, 39, 39, 39, 39, 40],
            [40, 40, 40, 40, 41, 41, 41, 41],
            [42, 41, 42, 42, 42, 42, 42, 43],
            [44, 43, 43, 44, 44, 44, 44, 44],
            [45, 45, 45, 45, 45, 45, 46, 46],
            [46, 46, 46, 46, 46, 46, 46, 47]
        ]),
    );
    assert_eq!(
        g.slice(s![9, .., ..]),
        arr2(&[
            [113, 113, 113, 113, 113, 114, 114, 114],
            [114, 114, 114, 114, 114, 114, 114, 114],
            [114, 114, 114, 114, 114, 115, 115, 115],
            [116, 116, 116, 116, 116, 116, 116, 116],
            [117, 117, 117, 117, 117, 118, 118, 118],
            [119, 119, 119, 119, 119, 119, 119, 120],
            [120, 120, 120, 120, 121, 121, 121, 121],
            [122, 122, 122, 122, 122, 122, 122, 123],
            [123, 123, 123, 123, 123, 123, 123, 123]
        ]),
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_prewitt() {
    let a = arr1(&[2.0, 8.1, 0.5, 4.0, 1.1, 9.0, 9.0, 0.8]);
    assert_relative_eq!(
        prewitt(&a, Axis(0), BorderMode::Reflect),
        arr1(&[6.1, -1.5, -4.1, 0.6, 5.0, 7.9, -8.2, -8.2]),
        epsilon = 1e-5
    );

    let matrix = arr2(&[
        [1.5, 2.3, 0.7, 1.1, 6.0, 1.7],
        [0.5, 1.3, 0.0, 0.1, 1.2, 0.7],
        [0.4, 1.3, 2.7, 0.1, 0.8, 0.1],
        [2.1, 0.1, 0.7, 0.1, 1.0, 2.8],
        [5.7, 4.0, 1.8, 9.1, 4.8, 2.7],
    ]);
    assert_relative_eq!(
        prewitt(&matrix, Axis(0), BorderMode::Reflect),
        arr2(&[
            [-3.0, -2.7, -2.7, -6.5, -6.8, -6.8],
            [-3.2, -0.1, 0.0, -4.2, -7.8, -8.4],
            [2.0, 1.1, -0.5, 0.5, 1.9, 4.0],
            [13.3, 7.1, 10.8, 12.1, 15.6, 9.2],
            [11.1, 8.6, 14.0, 13.9, 12.7, 3.6]
        ]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        prewitt(&matrix, Axis(1), BorderMode::Reflect),
        arr2(&[
            [2.4, -2.1, -3.6, 11.8, 1.8, -9.1],
            [2.5, 1.0, -3.6, 4.6, 1.2, -5.5],
            [-0.3, 0.4, -2.4, -0.4, 3.3, 0.6],
            [-2.8, -3.0, 3.9, 1.4, -3.7, -1.0],
            [-5.4, -9.2, 10.2, 6.3, -10.1, -2.4]
        ]),
        epsilon = 1e-5
    );

    let a: Array1<f32> = (0..720).map(|v| v as f32 / 50.0).collect();
    let mut a = a.into_shape_with_order((10, 9, 8)).unwrap();
    a[(0, 0, 0)] = 0.2;
    a[(3, 3, 3)] = 1.0;

    let ans = prewitt(&a, Axis(0), BorderMode::Reflect);
    assert_relative_eq!(
        ans.slice(s![0, .., ..]),
        arr2(&[
            [12.16, 12.56, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.56, 12.76, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96],
            [12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96, 12.96]
        ]),
        epsilon = 1e-5
    );

    let ans = prewitt(&a, Axis(1), BorderMode::Reflect);
    assert_relative_eq!(
        ans.slice(s![.., 0, ..]),
        arr2(&[
            [0.64, 1.04, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.04, 1.24, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
            [1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44]
        ]),
        epsilon = 1e-5
    );

    let ans = prewitt(&a, Axis(2), BorderMode::Reflect);
    assert_relative_eq!(
        ans.slice(s![.., .., 0]),
        arr2(&[
            [-0.62, -0.22, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [-0.22, -0.02, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
        ]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with SciPy. (v1.9.0)
fn test_sobel() {
    let a = arr1(&[2.0, 8.1, 0.5, 4.0, 1.1, 9.0, 9.0, 0.8]);
    assert_relative_eq!(
        sobel(&a, Axis(0), BorderMode::Reflect),
        arr1(&[6.1, -1.5, -4.1, 0.6, 5.0, 7.9, -8.2, -8.2]),
        epsilon = 1e-5
    );

    let matrix = arr2(&[
        [1.5, 2.3, 0.7, 1.1, 6.0, 1.7],
        [0.5, 1.3, 0.0, 0.1, 1.2, 0.7],
        [0.4, 1.3, 2.7, 0.1, 0.8, 0.1],
        [2.1, 0.1, 0.7, 0.1, 1.0, 2.8],
        [5.7, 4.0, 1.8, 9.1, 4.8, 2.7],
    ]);
    assert_relative_eq!(
        sobel(&matrix, Axis(0), BorderMode::Reflect),
        arr2(&[
            [-4.0, -3.7, -3.4, -7.5, -11.6, -7.8],
            [-4.3, -1.1, 2.0, -5.2, -13.0, -10.0],
            [3.6, -0.1, 0.2, 0.5, 1.7, 6.1],
            [18.6, 9.8, 9.9, 21.1, 19.6, 11.8],
            [14.7, 12.5, 15.1, 22.9, 16.5, 3.5]
        ]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        sobel(&matrix, Axis(1), BorderMode::Reflect),
        arr2(&[
            [3.2, -2.9, -4.8, 17.1, 2.4, -13.4],
            [3.3, 0.5, -4.8, 5.8, 1.8, -6.0],
            [0.6, 2.7, -3.6, -2.3, 3.3, -0.1],
            [-4.8, -4.4, 3.9, 1.7, -1., 0.8],
            [-7.1, -13.1, 15.3, 9.3, -16.5, -4.5]
        ]),
        epsilon = 1e-5
    );
}
