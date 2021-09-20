#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::Array3;

mod morphology;

pub use morphology::{binary_dilation, binary_erosion};

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

/// Utilitary function that returns the mask dimension minus 1 on all dimensions.
pub fn dim_minus_1(mask: &Mask) -> (usize, usize, usize) {
    let (width, height, depth) = mask.dim();
    (width - 1, height - 1, depth - 1)
}
