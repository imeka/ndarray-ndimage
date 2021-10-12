//! This modules defines some image padding methods for 3D images.

use ndarray::{Array3, ArrayView3, ShapeBuilder};
use num_traits::Zero;

type PadSize = (usize, usize, usize);

/// Method that will be used to select the padded values.
pub enum PadMode {
    /// Pads with the reflection of the vector mirrored on the first and last values of the vector
    /// along each axis.
    Reflect,

    /// Pads with the reflection of the vector mirrored along the edge of the array.
    Symmetric,

    /// Pads with the wrap of the vector along the axis. The first values are used to pad the end
    /// and the end values are used to pad the beginning.
    Wrap,
}

/// Pad an image.
///
/// * `data` - A 3D view to the data to pad.
/// * `pad_width` - Number of values padded to the edges of each axis.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`PadMode`](crate::PadMode) enum for more information.
pub fn pad<T>(data: ArrayView3<T>, pad_width: PadSize, mode: PadMode) -> Array3<T>
where
    T: Zero + Clone + Copy,
{
    let indices = |size: usize, padding: usize| match mode {
        PadMode::Reflect => reflect_indices(size, padding),
        PadMode::Symmetric => symmetric_indices(size, padding),
        PadMode::Wrap => wrap_indices(size, padding),
    };

    let (width, height, depth) = data.dim();
    pad_by_indices(
        data,
        pad_width,
        indices(width, pad_width.0),
        indices(height, pad_width.1),
        indices(depth, pad_width.2),
    )
}

fn reflect_indices(size: usize, padding: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(2 * padding + size);
    let size = size as u16;
    let padding = padding as u16;
    v.extend((1..=padding).rev());
    v.extend(0..size);
    v.extend((size - padding - 1..size - 1).rev());
    v
}

fn symmetric_indices(size: usize, padding: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(2 * padding + size);
    let size = size as u16;
    let padding = padding as u16;
    v.extend((0..padding).rev());
    v.extend(0..size);
    v.extend((size - padding..size).rev());
    v
}

fn wrap_indices(size: usize, padding: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(2 * padding + size);
    let size = size as u16;
    let padding = padding as u16;
    v.extend(size - padding..size);
    v.extend(0..size);
    v.extend(0..padding);
    v
}

fn pad_by_indices<T>(
    data: ArrayView3<T>,
    pad: PadSize,
    indices_i: Vec<u16>,
    indices_j: Vec<u16>,
    indices_k: Vec<u16>,
) -> Array3<T>
where
    T: Zero + Clone + Copy,
{
    let width = data.dim().0 + 2 * pad.0;
    let height = data.dim().1 + 2 * pad.1;
    let depth = data.dim().2 + 2 * pad.2;
    let mut out = Array3::zeros((width, height, depth).f());

    // Using zip or Array3::from_shape_fn doesn't make this faster. Directly copying `data` into
    // the center of `out` and only working on the borders *may* be faster, but
    // * this solution is by far simpler and cleaner than any other
    // * it takes around 8ms for a 100x100x100 image
    // so lets keep it this way.

    for i in 0..width {
        let idx_i = indices_i[i] as usize;
        for j in 0..height {
            let idx_j = indices_j[j] as usize;
            for k in 0..depth {
                let idx_k = indices_k[k] as usize;
                out[(i, j, k)] = data[(idx_i, idx_j, idx_k)];
            }
        }
    }
    out
}
