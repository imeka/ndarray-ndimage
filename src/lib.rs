#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::{Array3, ArrayBase, DataOwned, Dimension, ShapeBuilder};

mod filters;
mod interpolation;
mod measurements;
mod morphology;
mod pad;

pub use filters::{gaussian_filter, gaussian_filter1d, median_filter};
pub use interpolation::{spline_filter, spline_filter1d};
pub use measurements::{label, label_histogram, most_frequent_label};
pub use morphology::{binary_dilation, binary_erosion};
pub use pad::{pad, PadMode};

/// 3D mask
pub type Mask = Array3<bool>;

/// 3D common kernels. Also called Structuring Element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel3d {
    /// Diamond/star kernel (center and sides).
    Star,
    /// 3x3 cube.
    Full,
}

/// Utilitary function that returns a new *n*-dimensional of dimension `shape` with the same
/// datatype and memory order as the input `arr`.
pub fn array_like<S, A, D, Sh>(arr: &ArrayBase<S, D>, shape: Sh, elem: A) -> ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    A: Clone,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    // TODO `is_standard_layout` only works on owned arrays. Change it if using `ArrayBase`.
    if arr.is_standard_layout() {
        ArrayBase::from_elem(shape, elem)
    } else {
        ArrayBase::from_elem(shape.f(), elem)
    }
}

/// Utilitary function that returns the mask dimension minus 1 on all dimensions.
pub fn dim_minus_1(mask: &Mask) -> (usize, usize, usize) {
    let (width, height, depth) = mask.dim();
    (width - 1, height - 1, depth - 1)
}
