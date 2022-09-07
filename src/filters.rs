use std::{collections::VecDeque, fmt::Display};

use ndarray::{
    s, Array, Array1, ArrayBase, Axis, Data, Dimension, Ix1, Ix3, ScalarOperand, ShapeBuilder, Zip,
};
use num_traits::{Float, FromPrimitive, Num, ToPrimitive};

use crate::{array_like, dim_minus, pad, pad_to, Mask, PadMode};

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

/// Calculate a 1-D convolution along the given axis.
///
/// The lines of the array along the given axis are convolved with the given weights.
///
/// * `data` - The input N-D data.
/// * `weights` - 1-D sequence of numbers.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///    centers the filter over the pixel, with positive values shifting the filter to the left, and
///    negative ones to the right.
pub fn convolve1d<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &ArrayBase<S, Ix1>,
    axis: Axis,
    mode: BorderMode<A>,
    mut origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    // TODO Should be Num, not Float
    A: Float + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    if weights.is_empty() {
        panic!("No filter weights given");
    }
    if weights.len() == 1 {
        return data.to_owned() * weights[0];
    }

    let weights = Zip::from(weights.slice(s![..; -1])).map_collect(|&w| w);

    origin = -origin;
    if weights.len() % 2 == 0 {
        origin -= 1;
    }

    _correlate1d(data, weights.as_slice().unwrap(), axis, mode, origin)
}

/// Calculate a 1-D correlation along the given axis.
///
/// The lines of the array along the given axis are correlated with the given weights.
///
/// * `data` - The input N-D data.
/// * `weights` - 1-D sequence of numbers.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///    centers the filter over the pixel, with positive values shifting the filter to the left, and
///    negative ones to the right.
pub fn correlate1d<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &ArrayBase<S, Ix1>,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    // TODO Should be Num, not Float
    A: Float + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    if weights.is_empty() {
        panic!("No filter weights given");
    }
    if weights.len() == 1 {
        return data.to_owned() * weights[0];
    }

    match weights.as_slice_memory_order() {
        Some(s) => _correlate1d(data, s, axis, mode, origin),
        None => {
            let weights = weights.to_owned();
            _correlate1d(data, weights.as_slice_memory_order().unwrap(), axis, mode, origin)
        }
    }
}

fn _correlate1d<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &[A],
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    // TODO Should be Num, not Float
    A: Float + FromPrimitive,
    D: Dimension,
{
    let symmetry_state = symmetry_state(weights);
    let size1 = weights.len() / 2;
    let size2 = weights.len() - size1 - 1;
    let size_2 = 2 * size1;

    let mode = mode.to_pad_mode();
    let n = data.len_of(axis);
    let pad = vec![origin_check(weights.len(), origin, size1, size2)];
    let mut buffer = Array1::from_elem(n + pad[0][0] + pad[0][1], mode.init());

    let mut output = data.to_owned();
    Zip::from(data.lanes(axis)).and(output.lanes_mut(axis)).for_each(|input, o| {
        pad_to(&input, &pad, mode, &mut buffer);
        let buffer = buffer.as_slice_memory_order().unwrap();

        match symmetry_state {
            SymmetryState::NonSymmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    // An unsafe here actually help
                    // acc + unsafe { *buffer.get_unchecked(i) } * w
                    *o = weights.iter().zip(i..).fold(A::zero(), |acc, (&w, i)| acc + buffer[i] * w)
                });
            }
            SymmetryState::Symmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    let middle = buffer[size1 + i] * weights[size1];
                    let mut left = i;
                    let mut right = i + size_2;
                    *o = weights[..size1].iter().fold(middle, |acc, &w| {
                        let ans = acc + (buffer[left] + buffer[right]) * w;
                        left += 1;
                        right -= 1;
                        ans
                    })
                });
            }
            SymmetryState::AntiSymmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    let middle = buffer[size1 + i] * weights[size1];
                    let mut left = i;
                    let mut right = i + size_2;
                    *o = weights[..size1].iter().fold(middle, |acc, &w| {
                        let ans = acc + (buffer[left] - buffer[right]) * w;
                        left += 1;
                        right -= 1;
                        ans
                    })
                });
            }
        }
    });

    output
}

#[derive(PartialEq)]
enum SymmetryState {
    NonSymmetric,
    Symmetric,
    AntiSymmetric,
}

fn symmetry_state<A>(arr: &[A]) -> SymmetryState
where
    A: Float,
{
    // Test for symmetry or anti-symmetry
    let mut state = SymmetryState::NonSymmetric;
    let filter_size = arr.len();
    let size1 = filter_size / 2;
    if filter_size & 1 > 0 {
        state = SymmetryState::Symmetric;
        for ii in 1..=size1 {
            if (arr[ii + size1] - arr[size1 - ii]).abs() > A::epsilon() {
                state = SymmetryState::NonSymmetric;
                break;
            }
        }
        if state == SymmetryState::NonSymmetric {
            state = SymmetryState::AntiSymmetric;
            for ii in 1..=size1 {
                if (arr[ii + size1] + arr[size1 - ii]).abs() > A::epsilon() {
                    state = SymmetryState::NonSymmetric;
                    break;
                }
            }
        }
    }
    state
}

/// Multidimensional convolution.
///
/// The array is convolved with the given kernel.
///
/// * `data` - The input N-D data.
/// * `weights` - Array of weights, same number of dimensions as `data`.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///    centers the filter over the pixel, with positive values shifting the filter to the left, and
///    negative ones to the right.
pub fn convolve<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &ArrayBase<S, D>,
    mode: BorderMode<A>,
    mut origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    if weights.is_empty() {
        panic!("No filter weights given");
    }

    let rev_weights;
    let s = match weights.as_slice() {
        Some(s) => s,
        None => {
            rev_weights = weights.to_owned();
            rev_weights.as_slice().unwrap()
        }
    };
    let rev_weights: Array1<_> = s.iter().rev().cloned().collect();
    let rev_weights = rev_weights.into_shape(weights.dim()).unwrap();

    origin = -origin;
    if weights.len() % 2 == 0 {
        origin -= 1;
    }
    _correlate(data, rev_weights, mode, origin)
}

/// Multidimensional correlation.
///
/// The array is correlated with the given kernel.
///
/// * `data` - The input N-D data.
/// * `weights` - Array of weights, same number of dimensions as `data`.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`BorderMode`](crate::BorderMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///    centers the filter over the pixel, with positive values shifting the filter to the left, and
///    negative ones to the right.
pub fn correlate<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: &ArrayBase<S, D>,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    // TODO Any way to not allocate weights for nothing?
    _correlate(data, weights.to_owned(), mode, origin)
}

fn _correlate<S, A, D>(
    data: &ArrayBase<S, D>,
    weights: Array<A, D>,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    let n = weights.shape()[0] / 2;
    let padded = pad(data, &[origin_check(weights.shape()[0], origin, n, n)], mode.to_pad_mode());
    let strides = padded.strides();
    let starting_idx_at = |idx: <D as Dimension>::Pattern| {
        let idx = idx.into_shape().raw_dim().clone();
        (0..data.ndim()).fold(0, |offset, d| offset + idx[d] * strides[d] as usize)
    };
    let padded = padded.as_slice_memory_order().unwrap();

    // Find the offsets for all non-zero values of the kernel
    let offsets: Vec<_> = weights
        .indexed_iter()
        .filter_map(|(idx, &k)| (k != A::zero()).then(|| (k, starting_idx_at(idx))))
        .collect();
    // Because we're working with a non-padded and a padded image, the offsets are not enough; we
    // must adjust them with a starting index. Otherwise, only the first row is right.
    Array::from_shape_fn(data.dim(), |idx| {
        let start = starting_idx_at(idx);
        offsets.iter().fold(A::zero(), |acc, &(k, offset)| acc + k * padded[start + offset])
    })
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

/// Calculate a 1-D maximum filter along the given axis.
///
/// The lines of the array along the given axis are filtered with a maximum filter of given size.
///
/// * `data` - The input N-D data.
/// * `size` - Length along which to calculate 1D maximum.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///   centers the filter over the pixel, with positive values shifting the filter to the left, and
///   negative ones to the right.
pub fn maximum_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive + Display,
    D: Dimension,
{
    let lower = |a, b| a <= b;
    let higher = |a, b| a >= b;
    min_or_max_filter(data, size, axis, mode, origin, higher, lower)
}

/// Calculate a 1-D minimum filter along the given axis.
///
/// The lines of the array along the given axis are filtered with a minimum filter of given size.
///
/// * `data` - The input N-D data.
/// * `size` - Length along which to calculate 1D minimum.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///   centers the filter over the pixel, with positive values shifting the filter to the left, and
///   negative ones to the right.
pub fn minimum_filter1d<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive + Display,
    D: Dimension,
{
    let lower = |a, b| a <= b;
    let higher = |a, b| a >= b;
    min_or_max_filter(data, size, axis, mode, origin, lower, higher)
}

fn min_or_max_filter<S, A, D, F1, F2>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
    f1: F1,
    f2: F2,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive + Display,
    D: Dimension,
    F1: Fn(A, A) -> bool,
    F2: Fn(A, A) -> bool,
{
    if size == 0 {
        panic!("Incorrect filter size (0)");
    }
    if size == 1 {
        return data.to_owned();
    }

    let size1 = size / 2;
    let size2 = size - size1 - 1;
    let mode = mode.to_pad_mode();
    let n = data.len_of(axis);
    let pad = vec![origin_check(size, origin, size1, size2)];
    let mut buffer = Array1::from_elem(n + pad[0][0] + pad[0][1], mode.init());

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Pair<A> {
        val: A,
        death: usize,
    }
    let mut ring = VecDeque::<Pair<A>>::with_capacity(size);

    let mut output = data.to_owned();
    Zip::from(data.lanes(axis)).and(output.lanes_mut(axis)).for_each(|input, mut o| {
        pad_to(&input, &pad, mode, &mut buffer);
        let buffer = buffer.as_slice_memory_order().unwrap();

        let mut o_idx = 0;
        ring.push_back(Pair { val: buffer[0], death: size });
        for (&v, i) in buffer[1..].iter().zip(1..) {
            if ring[0].death == i {
                ring.pop_front().unwrap();
            }

            if f1(v, ring[0].val) {
                ring[0] = Pair { val: v, death: size + i };
                while ring.len() > 1 {
                    ring.pop_back().unwrap();
                }
            } else {
                while f2(ring.back().unwrap().val, v) {
                    ring.pop_back().unwrap();
                }
                ring.push_back(Pair { val: v, death: size + i });
            }
            if i >= size - 1 {
                o[o_idx] = ring[0].val;
                o_idx += 1;
            }
        }
        ring.pop_back();
    });
    output
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
