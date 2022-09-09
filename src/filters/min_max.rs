use std::collections::VecDeque;

use ndarray::{Array, Array1, ArrayBase, Axis, Data, Dimension, ScalarOperand, Zip};
use num_traits::{FromPrimitive, Num};

use crate::{array_like, filters::origin_check, pad_to, BorderMode};

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
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    let mut output = data.to_owned();
    maximum_filter1d_to(data, size, axis, mode, origin, &mut output);
    output
}

/// Calculate a multidimensional maximum filter.
///
/// * `data` - The input N-D data.
/// * `size` - Length along which to calculate 1D maximum.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///   centers the filter over the pixel, with positive values shifting the filter to the left, and
///   negative ones to the right.
pub fn maximum_filter<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    // We need 2 buffers because
    // * We're reading neignbors so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = array_like(&data, data.dim(), A::zero());

    for d in 0..data.ndim() {
        maximum_filter1d_to(&data, size, Axis(d), mode, origin, &mut output);
        if d < data.ndim() - 1 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}

/// Calculate a 1-D maximum filter along the given axis.
///
/// See `maximum_filter1d`.
pub fn maximum_filter1d_to<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    let lower = |a, b| a <= b;
    let higher = |a, b| a >= b;
    min_or_max_filter(data, size, axis, mode, origin, higher, lower, output);
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
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    let mut output = data.to_owned();
    minimum_filter1d_to(data, size, axis, mode, origin, &mut output);
    output
}

/// Calculate a multidimensional minimum filter.
///
/// * `data` - The input N-D data.
/// * `size` - Length along which to calculate 1D minimum.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
/// * `origin` - Controls the placement of the filter on the input array’s pixels. A value of 0
///   centers the filter over the pixel, with positive values shifting the filter to the left, and
///   negative ones to the right.
pub fn minimum_filter<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    mode: BorderMode<A>,
    origin: isize,
) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    // We need 2 buffers because
    // * We're reading neignbors so we can't read and write on the same location.
    // * The process is applied for each axis on the result of the previous process.
    // * It's uglier (using &mut) but much faster than allocating for each axis.
    let mut data = data.to_owned();
    let mut output = array_like(&data, data.dim(), A::zero());

    for d in 0..data.ndim() {
        minimum_filter1d_to(&data, size, Axis(d), mode, origin, &mut output);
        if d < data.ndim() - 1 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}

/// Calculate a 1-D minimum filter along the given axis.
///
/// See `minimum_filter1d`.
pub fn minimum_filter1d_to<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    let lower = |a, b| a <= b;
    let higher = |a, b| a >= b;
    min_or_max_filter(data, size, axis, mode, origin, lower, higher, output);
}

/// MINLIST algorithm from Richard Harter
fn min_or_max_filter<S, A, D, F1, F2>(
    data: &ArrayBase<S, D>,
    filter_size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    origin: isize,
    f1: F1,
    f2: F2,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + Num + PartialOrd + ScalarOperand + FromPrimitive,
    D: Dimension,
    F1: Fn(A, A) -> bool,
    F2: Fn(A, A) -> bool,
{
    if filter_size == 0 {
        panic!("Incorrect filter size (0)");
    }
    if filter_size == 1 {
        output.assign(data);
        return;
    }

    let size1 = filter_size / 2;
    let size2 = filter_size - size1 - 1;
    let mode = mode.to_pad_mode();
    let n = data.len_of(axis);
    let pad = vec![origin_check(filter_size, origin, size1, size2)];
    let mut buffer = Array1::from_elem(n + pad[0][0] + pad[0][1], mode.init());

    #[derive(Copy, Clone, PartialEq)]
    struct Pair<A> {
        val: A,
        death: usize,
    }
    let mut ring = VecDeque::<Pair<A>>::with_capacity(filter_size);

    // The original algorihtm has been modfied to fit the `VecDeque` which makes `minpair` and
    // `last` useless. Moreover, we need to clear the `ring` at the end because there's always
    // at least one element left. There can be more with greater `filter_size`.
    Zip::from(data.lanes(axis)).and(output.lanes_mut(axis)).for_each(|input, mut o| {
        pad_to(&input, &pad, mode, &mut buffer);
        let buffer = buffer.as_slice_memory_order().unwrap();

        let mut o_idx = 0;
        ring.push_back(Pair { val: buffer[0], death: filter_size });
        for (&v, i) in buffer[1..].iter().zip(1..) {
            if ring[0].death == i {
                ring.pop_front().unwrap();
            }

            if f1(v, ring[0].val) {
                ring[0] = Pair { val: v, death: filter_size + i };
                while ring.len() > 1 {
                    ring.pop_back().unwrap();
                }
            } else {
                while f2(ring.back().unwrap().val, v) {
                    ring.pop_back().unwrap();
                }
                ring.push_back(Pair { val: v, death: filter_size + i });
            }
            if i >= filter_size - 1 {
                o[o_idx] = ring[0].val;
                o_idx += 1;
            }
        }
        ring.clear();
    });
}
