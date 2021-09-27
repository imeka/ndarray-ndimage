use approx::assert_relative_eq;
use ndarray::{arr3, Array1, Axis};

use ndarray_image::{spline_filter, spline_filter1d};

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter() {
    let data = (0..27).collect::<Array1<_>>().into_shape((3, 3, 3)).unwrap().mapv(f64::from);

    // Order 2
    assert_relative_eq!(
        spline_filter(data.view(), 2),
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
        epsilon = 1e-7
    );

    // Order 3
    assert_relative_eq!(
        spline_filter(data.view(), 3),
        arr3(&[
            [[-6.5, -5.0, -3.5], [-2.0, -0.5, 1.0], [2.5, 4.0, 5.5]],
            [[7.0, 8.5, 10.0], [11.5, 13.0, 14.5], [16.0, 17.5, 19.0]],
            [[20.5, 22.0, 23.5], [25.0, 26.5, 28.0], [29.5, 31.0, 32.5]]
        ]),
        epsilon = 1e-7
    );

    // Order 4
    assert_relative_eq!(
        spline_filter(data.view(), 4),
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
        epsilon = 1e-7
    );

    // Order 5
    assert_relative_eq!(
        spline_filter(data.view(), 5),
        arr3(&[
            [[-11.375, -9.5, -7.625], [-5.75, -3.875, -2.0], [-0.125, 1.75, 3.625]],
            [[5.5, 7.375, 9.25], [11.125, 13.0, 14.875], [16.75, 18.625, 20.5]],
            [[22.375, 24.25, 26.125], [28.0, 29.875, 31.75], [33.625, 35.5, 37.375]]
        ]),
        epsilon = 1e-7
    );
}

#[test] // Results verified with the `spline_filter` function from SciPy. (v1.7.0)
fn test_spline_filter1d() {
    let data = (0..27).collect::<Array1<_>>().into_shape((3, 3, 3)).unwrap().mapv(f64::from);

    let new_data = spline_filter1d(data.view(), 3, Axis(0));
    assert_relative_eq!(
        new_data,
        arr3(&[
            [[-4.5, -3.5, -2.5], [-1.5, -0.5, 0.5], [1.5, 2.5, 3.5]],
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            [[22.5, 23.5, 24.5], [25.5, 26.5, 27.5], [28.5, 29.5, 30.5]]
        ]),
        epsilon = 1e-10
    );

    let new_data = spline_filter1d(data.view(), 3, Axis(1));
    assert_relative_eq!(
        new_data,
        arr3(&[
            [[-1.5, -0.5, 0.5], [3.0, 4.0, 5.0], [7.5, 8.5, 9.5]],
            [[7.5, 8.5, 9.5], [12.0, 13.0, 14.0], [16.5, 17.5, 18.5]],
            [[16.5, 17.5, 18.5], [21.0, 22.0, 23.0], [25.5, 26.5, 27.5]]
        ]),
        epsilon = 1e-10
    );

    let new_data = spline_filter1d(data.view(), 3, Axis(2));
    assert_relative_eq!(
        new_data,
        arr3(&[
            [[-0.5, 1.0, 2.5], [2.5, 4.0, 5.5], [5.5, 7.0, 8.5]],
            [[8.5, 10.0, 11.5], [11.5, 13.0, 14.5], [14.5, 16.0, 17.5]],
            [[17.5, 19.0, 20.5], [20.5, 22.0, 23.5], [23.5, 25.0, 26.5]]
        ]),
        epsilon = 1e-10
    );
}