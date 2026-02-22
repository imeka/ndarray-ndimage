use ndarray::{
    s, Array, Array1, ArrayRef, ArrayRef1, Axis, Dimension, ScalarOperand, ShapeArg, Zip
};
use num_traits::{FromPrimitive, Num, Signed};

use super::{
    origin_check,
    symmetry::{symmetry_state, SymmetryState, SymmetryStateCheck},
};
use crate::{array_like, pad, pad_to, BorderMode};

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
pub fn convolve1d<A, D>(
    data: &ArrayRef<A, D>,
    weights: &ArrayRef1<A>,
    axis: Axis,
    mode: BorderMode<A>,
    mut origin: isize,
) -> Array<A, D>
where
    A: Copy + Num + ScalarOperand + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
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

    let mut output = data.to_owned();
    inner_correlate1d(data, weights.as_slice().unwrap(), axis, mode, origin, &mut output);
    output
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
pub fn correlate1d<A, D>(
    data: &ArrayRef<A, D>,
    weights: &ArrayRef1<A>,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    A: Copy + Num + FromPrimitive + ScalarOperand + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    if weights.is_empty() {
        panic!("No filter weights given");
    }
    if weights.len() == 1 {
        return data.to_owned() * weights[0];
    }

    let mut output = data.to_owned();
    match weights.as_slice_memory_order() {
        Some(s) => inner_correlate1d(data, s, axis, mode, origin, &mut output),
        None => {
            let weights = weights.to_owned();
            let weights = weights.as_slice_memory_order().unwrap();
            inner_correlate1d(data, weights, axis, mode, origin, &mut output)
        }
    };
    output
}

pub(crate) fn inner_correlate1d<A, D>(
    data: &ArrayRef<A, D>,
    weights: &[A],
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
    output: &mut Array<A, D>,
) where
    A: Copy + Num + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
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

    Zip::from(data.lanes(axis)).and(output.lanes_mut(axis)).for_each(|input, o| {
        pad_to(&input, &pad, mode, &mut buffer);
        let buffer = buffer.as_slice_memory_order().unwrap();

        match symmetry_state {
            SymmetryState::NonSymmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    *o = weights
                        .iter()
                        .zip(&buffer[i..])
                        .fold(A::zero(), |acc, (&w, &b)| acc + b * w)
                });
            }
            SymmetryState::Symmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    let middle = buffer[size1 + i] * weights[size1];
                    let mut left = i;
                    let mut right = i + size_2;
                    *o = weights[..size1].iter().fold(middle, |acc, &w| {
                        // let ans = acc + (buffer[left] + buffer[right]) * w;
                        let ans = unsafe {
                            (*buffer.get_unchecked(left) + *buffer.get_unchecked(right)) * w
                        };
                        left += 1;
                        right -= 1;
                        acc + ans
                    })
                });
            }
            SymmetryState::AntiSymmetric => {
                Zip::indexed(o).for_each(|i, o| {
                    let middle = buffer[size1 + i] * weights[size1];
                    let mut left = i;
                    let mut right = i + size_2;
                    *o = weights[..size1].iter().fold(middle, |acc, &w| {
                        // let ans = acc + (buffer[left] - buffer[right]) * w;
                        let ans = unsafe {
                            (*buffer.get_unchecked(left) - *buffer.get_unchecked(right)) * w
                        };
                        left += 1;
                        right -= 1;
                        acc + ans
                    })
                });
            }
        }
    });
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
pub fn convolve<A, D>(
    data: &ArrayRef<A, D>,
    weights: &ArrayRef<A, D>,
    mode: BorderMode<A>,
    mut origin: isize,
) -> Array<A, D>
where
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
    let rev_weights = rev_weights.into_shape_with_order(weights.dim()).unwrap();

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
pub fn correlate<A, D>(
    data: &ArrayRef<A, D>,
    weights: &ArrayRef<A, D>,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    // TODO Any way to not allocate weights for nothing?
    _correlate(data, weights.to_owned(), mode, origin)
}

fn _correlate<A, D>(
    data: &ArrayRef<A, D>,
    weights: Array<A, D>,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    let n = weights.shape()[0] / 2;
    let padded = pad(data, &[origin_check(weights.shape()[0], origin, n, n)], mode.to_pad_mode());
    let strides = padded.strides();
    let starting_idx_at = |idx: <D as Dimension>::Pattern| {
        let (idx, _) = idx.into_shape_and_order();
        let idx = idx.clone();
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

/// Calculate a Prewitt filter.
///
/// * `data` - The input N-D data.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
pub fn prewitt<A, D>(data: &ArrayRef<A, D>, axis: Axis, mode: BorderMode<A>) -> Array<A, D>
where
    A: Copy + Signed + ScalarOperand + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let second_weights = [A::one(); 3];
    inner_prewitt_sobel(data, axis, mode, &second_weights)
}

/// Calculate a Prewitt filter.
///
/// * `data` - The input N-D data.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
pub fn sobel<A, D>(data: &ArrayRef<A, D>, axis: Axis, mode: BorderMode<A>) -> Array<A, D>
where
    A: Copy + Signed + ScalarOperand + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let second_weights = [A::one(), A::from_u8(2).unwrap(), A::one()];
    inner_prewitt_sobel(data, axis, mode, &second_weights)
}

fn inner_prewitt_sobel<A, D>(
    data: &ArrayRef<A, D>,
    axis: Axis,
    mode: BorderMode<A>,
    second_weights: &[A],
) -> Array<A, D>
where
    A: Copy + Signed + ScalarOperand + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    let weights = [-A::one(), A::zero(), A::one()];
    let mut output = array_like(&data, data.dim(), A::zero());
    inner_correlate1d(&data.view(), &weights, axis, mode, 0, &mut output);
    if data.ndim() == 1 {
        return output;
    }

    let indices: Vec<_> = (0..data.ndim()).filter(|&d| d != axis.index()).collect();
    let mut data = output.clone();
    for (i, d) in indices.into_iter().enumerate() {
        let axis = Axis(d);
        inner_correlate1d(&data, second_weights, axis, mode, 0, &mut output);
        if i != data.ndim() - 2 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}
