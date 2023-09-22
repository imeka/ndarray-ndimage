use ndarray::{s, Array, Array1, ArrayBase, Axis, Data, Dimension, Zip};
use num_traits::{FromPrimitive, Num};

use crate::{array_like, pad_to, BorderMode};

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
    A: Copy + Num + FromPrimitive + PartialOrd + 'static,
    D: Dimension,
{
    let half = size / 2;

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

        inner_uniform1d(&data, size, Axis(d), mode, &mut output);
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
    A: Copy + Num + FromPrimitive + PartialOrd + 'static,
    D: Dimension,
{
    let mut output = array_like(&data, data.dim(), A::zero());
    inner_uniform1d(data, size, axis, mode, &mut output);
    output
}

pub(crate) fn inner_uniform1d<S, A, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: BorderMode<A>,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd,
    D: Dimension,
{
    let size1 = size / 2;
    let size2 = size - size1 - 1;
    let size_as_a = A::from_usize(size).unwrap();

    let mode = mode.to_pad_mode();
    let n = data.len_of(axis);
    let pad = vec![[size1, size2]];
    let mut buffer = Array1::from_elem(n + size - 1, mode.init());

    Zip::from(data.lanes(axis)).and(output.lanes_mut(axis)).for_each(|input, o| {
        pad_to(&input, &pad, mode, &mut buffer);
        let mut accumulator = buffer.slice(s![..size - 1]).sum();

        // Optimise the filter by keeping a running total, to which add the newest item entering the
        // window, and then subtract the element which has fallen out of the window.
        Zip::from(o).and(buffer.slice(s![size - 1..])).and(buffer.slice(s![..n])).for_each(
            |o, &leading_edge, &trailing_edge| {
                accumulator = accumulator + leading_edge;
                *o = accumulator / size_as_a;
                accumulator = accumulator - trailing_edge;
            },
        );
    });
}
