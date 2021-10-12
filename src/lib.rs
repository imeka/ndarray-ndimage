#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::{Array3, ShapeBuilder};

mod filters;
mod interpolation;
mod measurements;
mod morphology;
mod pad;

pub use filters::median_filter;
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

/// Utilitary function that returns a new mask of dimension `dim` with the same memory order as
/// the input `mask`.
pub fn mask_like(mask: &Mask, dim: (usize, usize, usize), init: bool) -> Mask {
    // TODO `is_standard_layout` only works on owned arrays. Change it if using `ArrayBase`.
    if mask.is_standard_layout() {
        Mask::from_elem(dim, init)
    } else {
        Mask::from_elem(dim.f(), init)
    }
}

/// Utilitary function that returns the mask dimension minus 1 on all dimensions.
pub fn dim_minus_1(mask: &Mask) -> (usize, usize, usize) {
    let (width, height, depth) = mask.dim();
    (width - 1, height - 1, depth - 1)
}
