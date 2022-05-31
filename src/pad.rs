//! This modules defines some image padding methods for 3D images.

use ndarray::{
    s, Array, Array1, ArrayBase, ArrayView1, Axis, Data, Dimension, NdIndex, ShapeBuilder, Slice,
    Zip,
};
use ndarray_stats::QuantileExt;
use num_traits::{FromPrimitive, Num, Zero};

use crate::array_like;

#[derive(Copy, Clone, PartialEq)]
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
    fn init(&self) -> T
    where
        T: Copy + Zero,
    {
        match self {
            PadMode::Constant(init) => *init,
            _ => T::zero(),
        }
    }

    fn action(&self) -> PadAction {
        match self {
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
        match self {
            PadMode::Minimum => *lane.min().unwrap(),
            PadMode::Mean => lane.mean().unwrap(),
            PadMode::Median => {
                buffer.assign(&lane);
                buffer.as_slice_mut().unwrap().sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = buffer.len();
                let h = (n - 1) / 2;
                let dadc = if n & 1 > 0 {
                    buffer[h]
                } else {
                    (buffer[h] + buffer[h + 1]) / T::from_u32(2).unwrap()
                };
                dadc
            }
            PadMode::Maximum => *lane.max().unwrap(),
            _ => panic!("Only Minimum, Median and Maximum have a dynamic value"),
        }
    }

    fn needs_buffer(&self) -> bool {
        *self == PadMode::Median
    }

    fn indices(&self, size: usize, padding: usize) -> Vec<u16> {
        match self {
            PadMode::Reflect => {
                let mut v = Vec::with_capacity(2 * padding + size);
                let size = size as u16;
                let padding = padding as u16;
                v.extend((1..=padding).rev());
                v.extend(0..size);
                v.extend((size - padding - 1..size - 1).rev());
                v
            }
            PadMode::Symmetric => {
                let mut v = Vec::with_capacity(2 * padding + size);
                let size = size as u16;
                let padding = padding as u16;
                v.extend((0..padding).rev());
                v.extend(0..size);
                v.extend((size - padding..size).rev());
                v
            }
            PadMode::Wrap => {
                let mut v = Vec::with_capacity(2 * padding + size);
                let size = size as u16;
                let padding = padding as u16;
                v.extend(size - padding..size);
                v.extend(0..size);
                v.extend(0..padding);
                v
            }
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
/// * `data` - A 3D view to the data to pad.
/// * `pad_width` - Number of values padded to the edges of each axis.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`PadMode`](crate::PadMode) enum for more information.
pub fn pad<S, A, D, Sh>(data: &ArrayBase<S, D>, pad: Sh, mode: PadMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Clone + Copy + FromPrimitive + Num + PartialOrd,
    D: Dimension + Copy,
    Sh: ShapeBuilder<Dim = D>,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    let pad = pad.into_shape().raw_dim().clone();
    let new_dim = data.raw_dim() + pad.clone() + pad.clone();
    let mut padded = array_like(&data, new_dim, mode.init());
    let padded_dim = padded.raw_dim();

    let action = mode.action();
    if action != PadAction::ByIndices {
        // Select portion of padded array that needs to be copied from the original array.
        let mut orig_portion = padded.view_mut();
        for d in 0..data.ndim() {
            orig_portion.slice_axis_inplace(Axis(d), Slice::from(pad[d]..padded_dim[d] - pad[d]));
        }
        orig_portion.assign(data);
    }

    match action {
        PadAction::StopAfterCopy => { /* Nothing */ }
        PadAction::ByIndices => {
            let indices: Vec<_> =
                (0..data.ndim()).map(|d| mode.indices(data.len_of(Axis(d)), pad[d])).collect();
            Zip::indexed(&mut padded).for_each(|idx, v| {
                let mut idx = idx.into_shape().raw_dim().clone();
                for d in 0..data.ndim() {
                    idx[d] = indices[d][idx[d]] as usize;
                }
                *v = data[idx.into_pattern()];
            });
        }
        PadAction::ByLane => {
            for d in 0..data.ndim() {
                let start = pad[d];
                let end = start + data.shape()[d];
                let mut buffer =
                    if mode.needs_buffer() { Array1::zeros(end - start) } else { Array1::zeros(0) };
                Zip::from(padded.lanes_mut(Axis(d))).for_each(|mut lane| {
                    let v = mode.dynamic_value(lane.slice(s![start..end]), &mut buffer);
                    lane.slice_mut(s![..start]).fill(v);
                    lane.slice_mut(s![end..]).fill(v);
                });
            }
        }
        PadAction::BySides => {
            for d in 0..data.ndim() {
                let start = pad[d];
                let end = start + data.shape()[d];
                Zip::from(padded.lanes_mut(Axis(d))).for_each(|mut lane| {
                    let left = lane[start];
                    let right = lane[end - 1];
                    lane.slice_mut(s![..start]).fill(left);
                    lane.slice_mut(s![end..]).fill(right);
                });
            }
        }
    }
    padded
}
