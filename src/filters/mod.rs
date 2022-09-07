use ndarray::{s, Array, ArrayBase, Axis, Data, Dimension, Ix3, Zip};
use num_traits::{Float, ToPrimitive};

use crate::{array_like, dim_minus, Mask, PadMode};

pub mod con_corr;
pub mod min_max;

// TODO We might want to offer all NumPy mode (use PadMode instead)
/// Method that will be used to determines how the input array is extended beyond its boundaries.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BorderMode<T> {
    /// The input is extended by filling all values beyond the edge with the same constant value,
    ///
    /// `[1, 2, 3] -> [T, T, 1, 2, 3, T, T]`
    Constant(T),

    /// The input is extended by replicating the last pixel.
    ///
    /// `[1, 2, 3] -> [1, 1, 1, 2, 3, 3, 3]`
    Nearest,

    /// The input is extended by reflecting about the center of the last pixel.
    ///
    /// `[1, 2, 3] -> [3, 2, 1, 2, 3, 2, 1]`
    Mirror,

    /// The input is extended by reflecting about the edge of the last pixel.
    ///
    /// `[1, 2, 3] -> [2, 1, 1, 2, 3, 3, 2]`
    Reflect,

    /// The input is extended by wrapping around to the opposite edge.
    ///
    /// `[1, 2, 3] -> [2, 3, 1, 2, 3, 1, 2]`
    Wrap,
}

impl<T: Copy> BorderMode<T> {
    fn to_pad_mode(&self) -> PadMode<T> {
        match *self {
            BorderMode::Constant(t) => PadMode::Constant(t),
            BorderMode::Nearest => PadMode::Edge,
            BorderMode::Mirror => PadMode::Reflect,
            BorderMode::Reflect => PadMode::Symmetric,
            BorderMode::Wrap => PadMode::Wrap,
        }
    }
}

fn origin_check(len: usize, origin: isize, left: usize, right: usize) -> [usize; 2] {
    let len = len as isize;
    assert!(
        origin >= -len / 2 && origin <= (len - 1) / 2,
        "origin must satisfy: -(len(weights) / 2) <= origin <= (len(weights) - 1) / 2"
    );
    [(left as isize + origin) as usize, (right as isize - origin) as usize]
}

/// Binary median filter.
///
/// A 3x3 structuring element (`Kernel3d::Full`) is used except on the borders, where a smaller
/// structuring element is used.
pub fn median_filter<S>(mask: &ArrayBase<S, Ix3>) -> Mask
where
    S: Data<Elem = bool>,
{
    let range = |i, max| {
        if i == 0 {
            0..2
        } else if i == max {
            max - 1..max + 1
        } else {
            i - 1..i + 2
        }
    };

    let (width, height, depth) = dim_minus(mask, 1);
    let ranges_x: Vec<_> = (0..=width).map(|x| range(x, width)).collect();
    let ranges_y: Vec<_> = (0..=height).map(|y| range(y, height)).collect();
    let ranges_z: Vec<_> = (0..=depth).map(|z| range(z, depth)).collect();

    // `from_shape_fn` is strangely much slower here
    let mut new_mask = array_like(mask, mask.dim(), false);
    Zip::indexed(&mut new_mask).for_each(|idx, new_mask| {
        let r_x = &ranges_x[idx.0];
        let r_y = &ranges_y[idx.1];
        let r_z = &ranges_z[idx.2];

        // For binary images, the median filter can be replaced with a simple majority vote
        let nb_required = ((r_x.len() * r_y.len() * r_z.len()) as u8 - 1) / 2;
        *new_mask = mask
            .slice(s![r_x.clone(), r_y.clone(), r_z.clone()])
            .iter()
            .fold(0, |acc, &m| acc + m as u8)
            > nb_required;
    });
    new_mask
}

/// Gaussian filter for n-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode and 0 order.
///
/// * `data` - The input N-D data.
/// * `sigma` - Standard deviation for Gaussian kernel.
/// * `truncate` - Truncate the filter at this many standard deviations.
///
/// **Panics** if one of the axis' lengths is lower than `truncate * sigma + 0.5`.
pub fn gaussian_filter<S, A, D>(data: &ArrayBase<S, D>, sigma: A, truncate: A) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + ToPrimitive,
    D: Dimension,
{
    // We need 2 buffers because
    // * We're reading neignbors so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = data.to_owned();

    let weights = weights(sigma, truncate);
    for d in 0..data.ndim() {
        _gaussian_filter1d(&data, &weights, Axis(d), &mut output);
        // TODO No need to do this at last iter
        data.assign(&output);
    }
    output
}

/// Gaussian filter for 1-dimensional arrays.
///
/// Currently hardcoded with the `PadMode::Reflect` padding mode and 0 order.
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
    truncate: A,
    axis: Axis,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + ToPrimitive,
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
    A: Float + ToPrimitive,
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
fn weights<A>(sigma: A, truncate: A) -> Vec<A>
where
    A: Float,
{
    // Make the radius of the filter equal to truncate standard deviations
    let radius = (truncate * sigma + A::from(0.5).unwrap()).to_isize().unwrap();

    let sigma2 = sigma.powi(2);
    let mut phi_x: Vec<_> = (-radius..=radius)
        .map(|x| (A::from(-0.5).unwrap() / sigma2 * A::from(x.pow(2)).unwrap()).exp())
        .collect();
    let sum = phi_x.iter().fold(A::zero(), |acc, &v| acc + v);
    phi_x.iter_mut().for_each(|v| *v = *v / sum);
    phi_x
}
