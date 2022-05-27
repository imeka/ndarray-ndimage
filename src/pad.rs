//! This modules defines some image padding methods for 3D images.

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, NdIndex, ShapeBuilder, Slice, Zip};
use num_traits::Zero;

use crate::array_like;

#[derive(Copy, Clone, PartialEq)]
/// Method that will be used to select the padded values.
pub enum PadMode<T> {
    /// Pads with a constant value.
    ///
    /// `[1, 2, 3] -> [T, T, 1, 2, 3, T, T]`
    Constant(T),

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

impl<T: Copy + Zero> PadMode<T> {
    fn init(&self) -> T {
        match self {
            PadMode::Constant(init) => *init,
            _ => T::zero(),
        }
    }

    fn action(&self) -> PadAction {
        match self {
            PadMode::Constant(_) => PadAction::StopAfterCopy,
            PadMode::Reflect | PadMode::Symmetric | PadMode::Wrap => PadAction::ByIndices,
        }
    }

    fn indices(&self, size: usize, padding: usize) -> Vec<u16> {
        match self {
            PadMode::Constant(_) => panic!(),
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
        }
    }
}

#[derive(PartialEq)]
enum PadAction {
    StopAfterCopy,
    ByIndices,
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
    A: Zero + Clone + Copy,
    D: Dimension + Copy,
    Sh: ShapeBuilder<Dim = D>,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    let pad = pad.into_shape().raw_dim().clone();
    let new_dim = data.raw_dim() + pad.clone() + pad.clone();
    let mut padded = array_like(&data, new_dim, mode.init());
    let padded_dim = padded.raw_dim();

    if mode.action() == PadAction::StopAfterCopy {
        // Select portion of padded array that needs to be copied from the original array.
        let mut orig_portion = padded.view_mut();
        for d in 0..data.ndim() {
            orig_portion.slice_axis_inplace(Axis(d), Slice::from(pad[d]..padded_dim[d] - pad[d]));
        }
        orig_portion.assign(data);
        return padded;
    }

    let indices: Vec<_> =
        (0..data.ndim()).map(|d| mode.indices(data.len_of(Axis(d)), pad[d])).collect();
    Zip::indexed(&mut padded).for_each(|idx, v| {
        let mut idx = idx.into_shape().raw_dim().clone();
        for d in 0..data.ndim() {
            idx[d] = indices[d][idx[d]] as usize;
        }
        *v = data[idx.into_pattern()];
    });

    padded
}
