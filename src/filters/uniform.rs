use ndarray::{Array, ArrayBase, Axis, Data, Dimension};
use num_traits::{Float, FromPrimitive};

use crate::{array_like, BorderMode};

use super::{con_corr::inner_correlate1d, symmetry::SymmetryStateCheck};

/// Uniform filter for n-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode.
///
/// * `data` - The input N-D data.
/// * `size` - The len
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
///
/// **Panics** if `size` is zero, or one of the axis' lengths is lower than `size`.
pub fn uniform_filter<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    mode: BorderMode<A>,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive + 'static,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let weights = weights(size);
    let half = weights.len() / 2;

    // We need 2 buffers because
    // * We're reading neighbours so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = array_like(&data, data.dim(), A::zero());

    for d in 0..data.ndim() {
        // TODO This can be made to work if the padding modes (`reflect`, `symmetric`, `wrap`) are
        // more robust. One just needs to reflect the input data several times if the `weights`
        // length is greater than the input array. It works in SciPy because they are looping on a
        // size variable instead of running the algo only once like we do.
        let n = data.len_of(Axis(d));
        if half > n {
            panic!("Data size is too small for the inputs (sigma and truncate)");
        }

        inner_correlate1d(&data, &weights, Axis(d), mode, 0, &mut output);
        if d != data.ndim() - 1 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}

/// Uniform filter for 1-dimensional arrays.
///
/// * `data` - The input N-D data.
/// * `size` - Length of the uniform filter.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
///
/// **Panics** if `size` is zero, or the axis length is lower than `size`.
pub fn uniform_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive + 'static,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let weights = weights(size);
    let mut output = array_like(&data, data.dim(), A::zero());
    inner_correlate1d(data, &weights, axis, mode, 0, &mut output);
    output
}

fn weights<A>(size: usize) -> Vec<A>
where
    A: Float + FromPrimitive + 'static,
{
    if size == 0 {
        panic!("Size is zero");
    }
    [A::one() / A::from_usize(size).unwrap()].repeat(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weights() {
        assert_relative_eq!(weights(5).as_slice(), &[0.2, 0.2, 0.2, 0.2, 0.2][..], epsilon = 1e-7);
        assert_relative_eq!(weights(1).as_slice(), &[1.0][..], epsilon = 1e-7);
    }

    #[should_panic]
    #[test]
    fn test_weights_zero() {
        weights::<f64>(0);
    }
}
