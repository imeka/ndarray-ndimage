use ndarray::{s, Array, Array1, Array2, ArrayBase, Axis, Data, Dimension, Zip};
use num_traits::{Float, FromPrimitive};

use crate::{array_like, BorderMode};

use super::{con_corr::inner_correlate1d, symmetry::SymmetryStateCheck};

/// Gaussian filter for n-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `order` - The order of the filter along all axes. An order of 0 corresponds to a convolution
///   with a Gaussian kernel. A positive order corresponds to a convolution with that derivative of
///   a Gaussian.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
/// * `truncate` - Truncate the filter at this many standard deviations.
///
/// **Panics** if one of the axis' lengths is lower than `truncate * sigma + 0.5`.
pub fn gaussian_filter<S, A, D>(
    data: &ArrayBase<S, D>,
    sigma: A,
    order: usize,
    mode: BorderMode<A>,
    truncate: usize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive + 'static,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    // We need 2 buffers because
    // * We're reading neignbors so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = array_like(&data, data.dim(), A::zero());

    let weights = weights(sigma, order, truncate);
    for d in 0..data.ndim() {
        inner_correlate1d(&data, &weights, Axis(d), mode, 0, &mut output);
        if d != data.ndim() - 1 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}

/// Gaussian filter for 1-dimensional arrays.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `axis` - The axis of input along which to calculate.
/// * `order` - The order of the filter along all axes. An order of 0 corresponds to a convolution
///   with a Gaussian kernel. A positive order corresponds to a convolution with that derivative of
///   a Gaussian.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
/// * `truncate` - Truncate the filter at this many standard deviations.
///
/// **Panics** if the axis length is lower than `truncate * sigma + 0.5`.
pub fn gaussian_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    sigma: A,
    axis: Axis,
    order: usize,
    mode: BorderMode<A>,
    truncate: usize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive + 'static,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let weights = weights(sigma, order, truncate);
    let mut output = array_like(&data, data.dim(), A::zero());
    inner_correlate1d(data, &weights, axis, mode, 0, &mut output);
    output
}

/// Computes a 1-D Gaussian convolution kernel.
fn weights<A>(sigma: A, order: usize, truncate: usize) -> Vec<A>
where
    A: Float + FromPrimitive + 'static,
{
    // Make the radius of the filter equal to truncate standard deviations
    let radius = (A::from(truncate).unwrap() * sigma + A::from(0.5).unwrap()).to_isize().unwrap();

    let sigma2 = sigma.powi(2);
    let phi_x = {
        let m05 = A::from(-0.5).unwrap();
        let mut phi_x: Vec<_> =
            (-radius..=radius).map(|x| (m05 / sigma2 * A::from(x.pow(2)).unwrap()).exp()).collect();
        let sum = phi_x.iter().fold(A::zero(), |acc, &v| acc + v);
        phi_x.iter_mut().for_each(|v| *v = *v / sum);
        phi_x
    };

    if order == 0 {
        phi_x
    } else {
        let mut q = Array1::zeros(order + 1);
        q[0] = A::one();

        let q_d = {
            let mut q_d = Array2::<A>::zeros((order + 1, order + 1));
            for (e, i) in q_d.slice_mut(s![..order, 1..]).diag_mut().iter_mut().zip(1..) {
                *e = A::from(i).unwrap();
            }

            q_d.slice_mut(s![1.., ..order]).diag_mut().fill(-sigma2.recip());
            q_d
        };

        for _ in 0..order {
            q = q_d.dot(&q);
        }

        (-radius..=radius)
            .zip(phi_x.into_iter())
            .map(|(x, phi_x)| {
                Zip::indexed(&q)
                    .fold(A::zero(), |acc, i, &q| acc + q * A::from(x.pow(i as u32)).unwrap())
                    * phi_x
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weights() {
        assert_relative_eq!(
            weights(1.0, 0, 3).as_slice(),
            &[0.00443304, 0.05400558, 0.24203622, 0.39905027, 0.24203622, 0.05400558, 0.00443304][..],
            epsilon = 1e-7
        );
        assert_relative_eq!(
            weights(1.0, 0, 4).as_slice(),
            &[
                0.00013383, 0.00443186, 0.05399112, 0.24197144, 0.39894346, 0.24197144, 0.05399112,
                0.00443186, 0.00013383,
            ][..],
            epsilon = 1e-7
        );

        // Different orders
        assert_relative_eq!(
            weights(1.0, 1, 3).as_slice(),
            &[0.01329914, 0.10801116, 0.24203622, 0.0, -0.24203622, -0.10801116, -0.01329914][..],
            epsilon = 1e-7
        );
        assert_relative_eq!(
            weights(1.0, 1, 4).as_slice(),
            &[
                0.00053532,
                0.01329558,
                0.10798225,
                0.24197144,
                0.0,
                -0.24197144,
                -0.10798225,
                -0.01329558,
                -0.00053532,
            ][..],
            epsilon = 1e-7
        );
        assert_relative_eq!(
            weights(1.0, 2, 3).as_slice(),
            &[0.03546438, 0.16201674, 0.0, -0.39905027, 0.0, 0.16201674, 0.03546438][..],
            epsilon = 1e-7
        );
        assert_relative_eq!(
            weights(0.75, 3, 3).as_slice(),
            &[0.39498175, -0.84499983, 0.0, 0.84499983, -0.39498175][..],
            epsilon = 1e-7
        );
    }
}
