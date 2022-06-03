#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::{Array, Array3, ArrayBase, Data, Dimension, Ix3, ShapeBuilder};

mod filters;
mod interpolation;
mod measurements;
mod morphology;
mod pad;

pub use filters::{gaussian_filter, gaussian_filter1d, median_filter};
pub use interpolation::{spline_filter, spline_filter1d};
pub use measurements::{label, label_histogram, largest_connected_components, most_frequent_label};
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

/// Utilitary function that returns a new *n*-dimensional array of dimension `shape` with the same
/// datatype and memory order as the input `arr`.
pub fn array_like<S, A, D, Sh>(arr: &ArrayBase<S, D>, shape: Sh, elem: A) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Clone,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    // TODO `is_standard_layout` only works on owned arrays. Change it if using `ArrayBase`.
    if arr.is_standard_layout() {
        Array::from_elem(shape, elem)
    } else {
        Array::from_elem(shape.f(), elem)
    }
}

/// Utilitary function that returns the mask dimension minus 1 on all dimensions.
pub fn dim_minus_1<S, A>(mask: &ArrayBase<S, Ix3>) -> (usize, usize, usize)
where
    S: Data<Elem = A>,
    A: Clone,
{
    let (width, height, depth) = mask.dim();
    (width - 1, height - 1, depth - 1)
}
