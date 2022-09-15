//! This modules defines some image padding methods for 3D images.

use std::borrow::Cow;

use ndarray::{
    s, Array, Array1, ArrayBase, ArrayView1, Axis, AxisDescription, Data, Dimension, Slice, Zip,
};
use ndarray_stats::QuantileExt;
use num_traits::{FromPrimitive, Num, Zero};

use crate::array_like;

#[derive(Copy, Clone, Debug, PartialEq)]
/// Method that will be used to select the padded values.
pub enum PadMode<T> {
    /// Pads with a constant value.
    ///
    /// `[1, 2, 3] -> [T, T, 1, 2, 3, T, T]`
    Constant(T),

    /// Pads with the edge values of array.
    ///
    /// `[1, 2, 3] -> [1, 1, 1, 2, 3, 3, 3]`
    Edge,

    /// Pads with the maximum value of all or part of the vector along each axis.
    ///
    /// `[1, 2, 3] -> [3, 3, 1, 2, 3, 3, 3]`
    Maximum,

    /// Pads with the mean value of all or part of the vector along each axis.
    ///
    /// `[1, 2, 3] -> [2, 2, 1, 2, 3, 2, 2]`
    Mean,

    /// Pads with the median value of all or part of the vector along each axis.
    ///
    /// `[1, 2, 3] -> [2, 2, 1, 2, 3, 2, 2]`
    Median,

    /// Pads with the minimum value of all or part of the vector along each axis.
    ///
    /// `[1, 2, 3] -> [1, 1, 1, 2, 3, 1, 1]`
    Minimum,

    /// Pads with the reflection of the vector mirrored on the first and last values of the vector
    /// along each axis.
    ///
    /// `[1, 2, 3] -> [3, 2, 1, 2, 3, 2, 1]`
    Reflect,

    /// Pads with the reflection of the vector mirrored along the edge of the array.
    ///
    /// `[1, 2, 3] -> [2, 1, 1, 2, 3, 3, 2]`
    Symmetric,

    /// Pads with the wrap of the vector along the axis. The first values are used to pad the end
    /// and the end values are used to pad the beginning.
    ///
    /// `[1, 2, 3] -> [2, 3, 1, 2, 3, 1, 2]`
    Wrap,
}

impl<T: PartialEq> PadMode<T> {
    pub(crate) fn init(&self) -> T
    where
        T: Copy + Zero,
    {
        match *self {
            PadMode::Constant(init) => init,
            _ => T::zero(),
        }
    }

    fn action(&self) -> PadAction {
        match *self {
            PadMode::Constant(_) => PadAction::StopAfterCopy,
            PadMode::Maximum | PadMode::Mean | PadMode::Median | PadMode::Minimum => {
                PadAction::ByLane
            }
            PadMode::Reflect | PadMode::Symmetric | PadMode::Wrap => PadAction::ByIndices,
            PadMode::Edge => PadAction::BySides,
        }
    }

    fn dynamic_value(&self, lane: ArrayView1<T>, buffer: &mut Array1<T>) -> T
    where
        T: Clone + Copy + FromPrimitive + Num + PartialOrd,
    {
        match *self {
            PadMode::Minimum => *lane.min().expect("Can't find min because of NaN values"),
            PadMode::Mean => lane.mean().expect("Can't find mean because of NaN values"),
            PadMode::Median => {
                buffer.assign(&lane);
                buffer.as_slice_mut().unwrap().sort_unstable_by(|a, b| {
                    a.partial_cmp(b).expect("Can't find median because of NaN values")
                });
                let n = buffer.len();
                let h = (n - 1) / 2;
                if n & 1 > 0 {
                    buffer[h]
                } else {
                    (buffer[h] + buffer[h + 1]) / T::from_u32(2).unwrap()
                }
            }
            PadMode::Maximum => *lane.max().expect("Can't find max because of NaN values"),
            _ => panic!("Only Minimum, Median and Maximum have a dynamic value"),
        }
    }

    fn needs_buffer(&self) -> bool {
        *self == PadMode::Median
    }

    fn indices(&self, size: usize, pad_left: usize, pad_right: usize) -> (Vec<usize>, Vec<usize>) {
        match *self {
            PadMode::Reflect => (
                (1..=pad_left).rev().map(|i| i + pad_left).collect(),
                (size - pad_right - 1..size - 1).rev().map(|i| i + pad_left).collect(),
            ),
            PadMode::Symmetric => (
                (0..pad_left).rev().map(|i| i + pad_left).collect(),
                (size - pad_right..size).rev().map(|i| i + pad_left).collect(),
            ),
            PadMode::Wrap => (
                (size - pad_left..size).map(|i| i + pad_left).collect(),
                (0..pad_right).map(|i| i + pad_left).collect(),
            ),
            _ => panic!("Only Reflect, Symmetric and Wrap have indices"),
        }
    }
}

#[derive(PartialEq)]
enum PadAction {
    StopAfterCopy,
    ByLane,
    ByIndices,
    BySides,
}

/// Pad an image.
///
/// * `data` - A N-D array of the data to pad.
/// * `pad` - Number of values padded to the edges of each axis.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`PadMode`](crate::PadMode) enum for more information.
pub fn pad<S, A, D>(data: &ArrayBase<S, D>, pad: &[[usize; 2]], mode: PadMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + FromPrimitive + Num + PartialOrd,
    D: Dimension,
{
    let pad = read_pad(data.ndim(), pad);
    let mut new_dim = data.raw_dim();
    for (ax, (&ax_len, pad)) in data.shape().iter().zip(pad.iter()).enumerate() {
        new_dim[ax] = ax_len + pad[0] + pad[1];
    }

    let mut padded = array_like(&data, new_dim, mode.init());
    pad_to(data, &pad, mode, &mut padded);
    padded
}

/// Pad an image.
///
/// Write the result in the already_allocated array `output`.
///
/// * `data` - A N-D array of the data to pad.
/// * `pad` - Number of values padded to the edges of each axis.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`PadMode`](crate::PadMode) enum for more information.
/// * `output` - An already allocated N-D array used to write the results.
pub fn pad_to<S, A, D>(
    data: &ArrayBase<S, D>,
    pad: &[[usize; 2]],
    mode: PadMode<A>,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + FromPrimitive + Num + PartialOrd,
    D: Dimension,
{
    let pad = read_pad(data.ndim(), pad);

    // Select portion of padded array that needs to be copied from the original array.
    output
        .view_mut()
        .slice_each_axis_mut(|ad| {
            let AxisDescription { axis, len, .. } = ad;
            let d = axis.index();
            Slice::from(pad[d][0]..len - pad[d][1])
        })
        .assign(data);

    match mode.action() {
        PadAction::StopAfterCopy => { /* Nothing */ }
        PadAction::ByIndices => {
            for d in 0..data.ndim() {
                let pad = pad[d];
                let (left_indices, right_indices) = mode.indices(data.shape()[d], pad[0], pad[1]);
                Zip::from(output.lanes_mut(Axis(d))).for_each(|mut lane| {
                    for (i, &ii) in left_indices.iter().enumerate() {
                        lane[i] = lane[ii];
                    }
                    for (i, &ii) in right_indices.iter().enumerate() {
                        lane[i] = lane[ii];
                    }
                });
            }
        }
        PadAction::ByLane => {
            for d in 0..data.ndim() {
                let start = pad[d][0];
                let end = start + data.shape()[d];
                let data_zone = s![start..end];
                let real_end = output.shape()[d];
                let mut buffer =
                    if mode.needs_buffer() { Array1::zeros(end - start) } else { Array1::zeros(0) };
                Zip::from(output.lanes_mut(Axis(d))).for_each(|mut lane| {
                    let v = mode.dynamic_value(lane.slice(data_zone), &mut buffer);
                    for i in 0..start {
                        lane[i] = v;
                    }
                    for i in end..real_end {
                        lane[i] = v;
                    }
                });
            }
        }
        PadAction::BySides => {
            for d in 0..data.ndim() {
                let start = pad[d][0];
                let end = start + data.shape()[d];
                Zip::from(output.lanes_mut(Axis(d))).for_each(|mut lane| {
                    let left = lane[start];
                    let right = lane[end - 1];
                    lane.slice_mut(s![..start]).fill(left);
                    lane.slice_mut(s![end..]).fill(right);
                });
            }
        }
    }
}

fn read_pad(nb_dim: usize, pad: &[[usize; 2]]) -> Cow<[[usize; 2]]> {
    if pad.len() == 1 && pad.len() < nb_dim {
        // The user provided a single padding for all dimensions
        Cow::from(vec![pad[0]; nb_dim])
    } else if pad.len() == nb_dim {
        Cow::from(pad)
    } else {
        panic!("Inconsistant number of dimensions and pad arrays");
    }
}
