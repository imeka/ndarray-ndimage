use approx::assert_relative_eq;
use ndarray::{arr1, arr2, arr3, Array, Array1, Axis};

use ndarray_ndimage::{spline_filter, spline_filter1d, zoom};

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter_same() {
    for &order in &[2, 3, 4, 5] {
        let data = arr1(&[0.5]);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = arr1(&[0.5, 0.5]);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = arr1(&[0.5, 0.5, 0.5]);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);

        let data = Array::from_elem((3, 3), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = Array::from_elem((3, 4), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = Array::from_elem((4, 3), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);

        let data = Array::from_elem((3, 3, 3), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = Array::from_elem((3, 4, 3), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = Array::from_elem((3, 4, 5), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
        let data = Array::from_elem((5, 4, 3), 0.5);
        assert_relative_eq!(spline_filter(&data, order), data, epsilon = 1e-5);
    }
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter_1d() {
    assert_relative_eq!(
        spline_filter(&arr1(&[0.1, 0.5, 0.5]), 3),
        arr1(&[-0.2, 0.7, 0.4]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter(&arr1(&[0.5, 0.1, 0.5]), 3),
        arr1(&[0.9, -0.3, 0.9]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter(&arr1(&[0.5, 0.5, 0.1]), 3),
        arr1(&[0.4, 0.7, -0.2]),
        epsilon = 1e-5
    );

    let data = arr1(&[0.3, 0.2, 0.5, -0.1, 0.4]);
    assert_relative_eq!(
        spline_filter(&data.view(), 3),
        arr1(&[0.47321429, -0.04642857, 0.9125, -0.60357143, 0.90178571]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter_2d() {
    assert_relative_eq!(
        spline_filter(&arr2(&[[0.1, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), 3),
        arr2(&[[-0.725, 0.85, 0.325], [0.85, 0.4, 0.55], [0.325, 0.55, 0.475]]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter(&arr2(&[[0.3, 0.5, 0.1], [0.6, 0.2, 0.4], [0.3, 0.3, 0.1]]), 3),
        arr2(&[[-0.8, 1.6, -1.1], [1.75, -0.8, 1.45], [-0.5, 1.0, -0.8]]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter(
            &arr2(&[[0.3, 0.5, 0.1], [0.6, 0.2, 0.4], [0.3, 0.3, 0.1], [-0.5, 0.4, 1.1]]),
            3
        ),
        arr2(&[
            [-0.66666667, 1.55333333, -1.14666667],
            [1.48333333, -0.70666667, 1.54333333],
            [0.43333333, 0.67333333, -1.12666667],
            [-1.71666667, 0.41333333, 2.66333333]
        ]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter_3d() {
    let data = (0..27).collect::<Array1<_>>().into_shape((3, 3, 3)).unwrap().mapv(f64::from);

    // Order 2
    assert_relative_eq!(
        spline_filter(&data, 2),
        arr3(&[
            [
                [-4.33333333, -3.0, -1.66666667],
                [-0.33333333, 1.0, 2.33333333],
                [3.66666667, 5.0, 6.33333333]
            ],
            [
                [7.66666667, 9.0, 10.33333333],
                [11.66666667, 13.0, 14.33333333],
                [15.66666667, 17.0, 18.33333333]
            ],
            [
                [19.66666667, 21.0, 22.33333333],
                [23.66666667, 25.0, 26.33333333],
                [27.66666667, 29.0, 30.33333333]
            ]
        ]),
        epsilon = 1e-5
    );

    // Order 3
    assert_relative_eq!(
        spline_filter(&data, 3),
        arr3(&[
            [[-6.5, -5.0, -3.5], [-2.0, -0.5, 1.0], [2.5, 4.0, 5.5]],
            [[7.0, 8.5, 10.0], [11.5, 13.0, 14.5], [16.0, 17.5, 19.0]],
            [[20.5, 22.0, 23.5], [25.0, 26.5, 28.0], [29.5, 31.0, 32.5]]
        ]),
        epsilon = 1e-5
    );

    // Order 4
    assert_relative_eq!(
        spline_filter(&data, 4),
        arr3(&[
            [
                [-8.89473684, -7.21052632, -5.52631579],
                [-3.84210526, -2.15789474, -0.47368421],
                [1.21052632, 2.89473684, 4.57894737]
            ],
            [
                [6.26315789, 7.94736842, 9.63157895],
                [11.31578947, 13.0, 14.68421053],
                [16.36842105, 18.05263158, 19.73684211]
            ],
            [
                [21.42105263, 23.10526316, 24.78947368],
                [26.47368421, 28.15789474, 29.84210526],
                [31.52631579, 33.21052632, 34.89473684]
            ]
        ]),
        epsilon = 1e-5
    );

    // Order 5
    assert_relative_eq!(
        spline_filter(&data, 5),
        arr3(&[
            [[-11.375, -9.5, -7.625], [-5.75, -3.875, -2.0], [-0.125, 1.75, 3.625]],
            [[5.5, 7.375, 9.25], [11.125, 13.0, 14.875], [16.75, 18.625, 20.5]],
            [[22.375, 24.25, 26.125], [28.0, 29.875, 31.75], [33.625, 35.5, 37.375]]
        ]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter1d() {
    let data = arr2(&[[0.5, 0.4], [0.3, 0.4]]);
    assert_relative_eq!(
        spline_filter1d(&data.view(), 2, Axis(0)),
        arr2(&[[0.6, 0.4], [0.2, 0.4]]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter1d(&data, 2, Axis(1)),
        arr2(&[[0.55, 0.35], [0.25, 0.45]]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter1d(&data, 3, Axis(0)),
        arr2(&[[0.7, 0.4], [0.1, 0.4]]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter1d(&data, 3, Axis(1)),
        arr2(&[[0.6, 0.3], [0.2, 0.5]]),
        epsilon = 1e-5
    );

    let data = (0..27).collect::<Array1<_>>().into_shape((3, 3, 3)).unwrap().mapv(f64::from);
    assert_relative_eq!(
        spline_filter1d(&data, 3, Axis(0)),
        arr3(&[
            [[-4.5, -3.5, -2.5], [-1.5, -0.5, 0.5], [1.5, 2.5, 3.5]],
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            [[22.5, 23.5, 24.5], [25.5, 26.5, 27.5], [28.5, 29.5, 30.5]]
        ]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter1d(&data, 3, Axis(1)),
        arr3(&[
            [[-1.5, -0.5, 0.5], [3.0, 4.0, 5.0], [7.5, 8.5, 9.5]],
            [[7.5, 8.5, 9.5], [12.0, 13.0, 14.0], [16.5, 17.5, 18.5]],
            [[16.5, 17.5, 18.5], [21.0, 22.0, 23.0], [25.5, 26.5, 27.5]]
        ]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        spline_filter1d(&data, 3, Axis(2)),
        arr3(&[
            [[-0.5, 1.0, 2.5], [2.5, 4.0, 5.5], [5.5, 7.0, 8.5]],
            [[8.5, 10.0, 11.5], [11.5, 13.0, 14.5], [14.5, 16.0, 17.5]],
            [[17.5, 19.0, 20.5], [20.5, 22.0, 23.5], [23.5, 25.0, 26.5]]
        ]),
        epsilon = 1e-5
    );
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.11.1)
fn test_zoom() {
    let data = (0..27).collect::<Array1<_>>().into_shape((3, 3, 3)).unwrap().mapv(f64::from);
    assert_relative_eq!(
        zoom(&data, [1.5, 1.5, 1.5], true),
        arr3(&[
            [
                [0.0, 0.51851852, 1.48148148, 2.0],
                [1.55555556, 2.07407407, 3.03703704, 3.55555556],
                [4.44444444, 4.96296296, 5.92592593, 6.44444444],
                [6.0, 6.51851852, 7.48148148, 8.0]
            ],
            [
                [4.66666667, 5.18518519, 6.14814815, 6.66666667],
                [6.22222222, 6.74074074, 7.7037037, 8.22222222],
                [9.11111111, 9.62962963, 10.59259259, 11.11111111],
                [10.66666667, 11.18518519, 12.14814815, 12.66666667]
            ],
            [
                [13.33333333, 13.85185185, 14.81481481, 15.33333333],
                [14.88888889, 15.40740741, 16.37037037, 16.88888889],
                [17.77777778, 18.2962963, 19.25925926, 19.77777778],
                [19.33333333, 19.85185185, 20.81481481, 21.33333333]
            ],
            [
                [18.0, 18.51851852, 19.48148148, 20.0],
                [19.55555556, 20.07407407, 21.03703704, 21.55555556],
                [22.44444444, 22.96296296, 23.92592593, 24.44444444],
                [24.0, 24.51851852, 25.48148148, 26.0]
            ]
        ]),
        epsilon = 1e-5
    );

    assert_relative_eq!(
        zoom(&data, [0.75, 0.75, 2.0], true),
        arr3(&[
            [[0.0, 0.208, 0.704, 1.296, 1.792, 2.0], [6.0, 6.208, 6.704, 7.296, 7.792, 8.0]],
            [
                [18.0, 18.208, 18.704, 19.296, 19.792, 20.0],
                [24.0, 24.208, 24.704, 25.296, 25.792, 26.0]
            ]
        ]),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        zoom(&data, [0.5, 0.65, 1.75], false),
        arr3(&[
            [
                [4.33333333, 4.54166667, 5.0, 5.45833333, 5.66666667],
                [8.33333333, 8.54166667, 9.0, 9.45833333, 9.66666667]
            ],
            [
                [16.33333333, 16.54166667, 17.0, 17.45833333, 17.66666667],
                [20.33333333, 20.54166667, 21.0, 21.45833333, 21.66666667]
            ]
        ]),
        epsilon = 1e-5
    );
}
