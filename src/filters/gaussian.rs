use ndarray::{Array, ArrayBase, Axis, Data, Dimension};
use num_traits::{Float, FromPrimitive};

use crate::array_like;

/// Gaussian filter for n-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `truncate` - Truncate the filter at this many standard deviations.
///
/// **Panics** if one of the axis' lengths is lower than `truncate * sigma + 0.5`.
pub fn gaussian_filter<S, A, D>(data: &ArrayBase<S, D>, sigma: A, truncate: usize) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive + 'static,
    D: Dimension,
{
    // We need 2 buffers because
    // * We're reading neignbors so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = array_like(&data, data.dim(), A::zero());

    let weights = weights(sigma, truncate);
    for d in 0..data.ndim() {
        _gaussian_filter1d(&data, &weights, Axis(d), &mut output);
        if d != data.ndim() - 1 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}

/// Gaussian filter for 1-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `truncate` - Truncate the filter at this many standard deviations.
/// * `axis` - The axis of input along which to calculate.
///
/// **Panics** if the axis length is lower than `truncate * sigma + 0.5`.
pub fn gaussian_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    sigma: A,
    truncate: usize,
    axis: Axis,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + FromPrimitive,
    D: Dimension,
{
    let weights = weights(sigma, truncate);
    let mut output = array_like(&data, data.dim(), A::zero());
    _gaussian_filter1d(data, &weights, axis, &mut output);
    output
}

fn _gaussian_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &[A],
    axis: Axis,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Float + FromPrimitive,
    D: Dimension,
{
    let half = weights.len() / 2;
    let middle_weight = weights[half];

    // TODO This can be made to work if the buffer code (see below) is more robust. It works in
    // SciPy. One just needs to reflect the input data several times. However, this buffer
    // exists only to handle the missing edges, so I really wonder if we could avoid it with
    // some clever indexing. Might be super complex though.
    let n = data.len_of(axis);
    if half > n {
        panic!("Data size is too small for the inputs (sigma and truncate)");
    }

    let mut buffer = vec![A::zero(); n + 2 * half];
    let input_it = data.lanes(axis).into_iter();
    let output_it = output.lanes_mut(axis).into_iter();
    for (input, mut o) in input_it.zip(output_it) {
        // TODO Remove this unsafe! This is easy to remove but I wasn't able to remove it and stay
        // fast. For more information, please read the thread at
        // https://users.rust-lang.org/t/scipy-gaussian-filter-port/62661
        unsafe {
            // Prepare the 'reflect' buffer
            let mut pos_b = 0;
            let mut pos_i = half - 1;
            for _ in 0..half {
                *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                pos_b += 1;
                pos_i = pos_i.wrapping_sub(1);
            }
            let mut pos_i = 0;
            for _ in 0..n {
                *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                pos_b += 1;
                pos_i += 1;
            }
            pos_i = n - 1;
            for _ in 0..half {
                *buffer.get_unchecked_mut(pos_b) = *input.uget(pos_i);
                pos_b += 1;
                pos_i = pos_i.wrapping_sub(1);
            }

            // Convolve the input data with the weights array.
            for idx in 0..n {
                let s = half + idx;
                let mut pos_l = s - 1;
                let mut pos_r = s + 1;

                let mut sum = *buffer.get_unchecked(s) * middle_weight;
                for &w in &weights[half + 1..] {
                    sum = sum + (*buffer.get_unchecked(pos_l) + *buffer.get_unchecked(pos_r)) * w;
                    pos_l = pos_l.wrapping_sub(1);
                    pos_r += 1;
                }
                *o.uget_mut(idx) = sum;
            }
        }
    }
}

/// Computes a 1-D Gaussian convolution kernel.
fn weights<A>(sigma: A, truncate: usize) -> Vec<A>
where
    A: Float + FromPrimitive,
{
    // Make the radius of the filter equal to truncate standard deviations
    let radius = (A::from(truncate).unwrap() * sigma + A::from(0.5).unwrap()).to_isize().unwrap();

    let sigma2 = sigma.powi(2);
    let m05 = A::from(-0.5).unwrap();
    let mut phi_x: Vec<_> =
        (-radius..=radius).map(|x| (m05 / sigma2 * A::from(x.pow(2)).unwrap()).exp()).collect();
    let sum = phi_x.iter().fold(A::zero(), |acc, &v| acc + v);
    phi_x.iter_mut().for_each(|v| *v = *v / sum);
    phi_x
}
