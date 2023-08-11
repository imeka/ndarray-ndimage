#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::{arr3, Array, Array3, ArrayBase, Data, Dimension, Ix3, ShapeBuilder};

mod filters;
mod interpolation;
mod measurements;
mod morphology;
mod pad;

pub use filters::{
    con_corr::{convolve, convolve1d, correlate, correlate1d, prewitt, sobel},
    gaussian::{gaussian_filter, gaussian_filter1d},
    median::median_filter,
    min_max::{
        maximum_filter, maximum_filter1d, maximum_filter1d_to, minimum_filter, minimum_filter1d,
        minimum_filter1d_to,
    },
    BorderMode,
};
pub use interpolation::{spline_filter, spline_filter1d};
pub use measurements::{label, label_histogram, largest_connected_components, most_frequent_label};
pub use morphology::{binary_closing, binary_dilation, binary_erosion, binary_opening};
pub use pad::{pad, pad_to, PadMode};

/// 3D mask
pub type Mask = Array3<bool>;

/// 3D common kernels. Also called Structuring Element.
#[derive(Clone, Debug, PartialEq)]
pub enum Kernel3d {
    /// Diamond/star kernel (center and sides).
    ///
    /// Equivalent to SciPy `generate_binary_structure(3, 1)`.
    Star,
    /// Ball kernel (center and sides).
    ///
    /// Equivalent to SciPy `generate_binary_structure(3, 2)`.
    Ball,
    /// 3x3x3 cube.
    ///
    /// Equivalent to SciPy `generate_binary_structure(3, 3)`.
    Full,
}

impl Kernel3d {
    /// Generate a binary 3D kernel.
    pub fn generate(&self) -> Array3<bool> {
        match self {
            Kernel3d::Star => arr3(&[
                [[false, false, false], [false, true, false], [false, false, false]],
                [[false, true, false], [true, true, true], [false, true, false]],
                [[false, false, false], [false, true, false], [false, false, false]],
            ]),
            Kernel3d::Ball => arr3(&[
                [[false, true, false], [true, true, true], [false, true, false]],
                [[true, true, true], [true, true, true], [true, true, true]],
                [[false, true, false], [true, true, true], [false, true, false]],
            ]),
            Kernel3d::Full => Array3::from_elem((3, 3, 3), true),
        }
    }
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
pub fn dim_minus<S, A>(mask: &ArrayBase<S, Ix3>, n: usize) -> (usize, usize, usize)
where
    S: Data<Elem = A>,
    A: Clone,
{
    let (width, height, depth) = mask.dim();
    (width - n, height - n, depth - n)
}

// TODO Use x.round_ties_even() when available on stable
// https://github.com/rust-lang/rust/issues/96710
fn round_ties_even(x: f64) -> f64 {
    let i = x as i32;
    let f = (x - i as f64).abs();
    if f == 0.5 {
        if i & 1 == 1 {
            // -1.5, 1.5, 3.5, ...
            (x.abs() + 0.5).copysign(x)
        } else {
            (x.abs() - 0.5).copysign(x)
        }
    } else {
        x.round()
    }
}

#[cfg(test)]
mod tests {
    use crate::round_ties_even;

    #[test]
    fn test_round_ties_even() {
        assert_eq!(round_ties_even(-2.51), -3.0);
        assert_eq!(round_ties_even(-2.5), -2.0);
        assert_eq!(round_ties_even(-1.5), -2.0);
        assert_eq!(round_ties_even(-0.5), -0.0);
        assert_eq!(round_ties_even(-0.1), 0.0);
        assert_eq!(round_ties_even(-0.0), 0.0);
        assert_eq!(round_ties_even(0.0), 0.0);
        assert_eq!(round_ties_even(0.1), 0.0);
        assert_eq!(round_ties_even(0.5), 0.0);
        assert_eq!(round_ties_even(1.5), 2.0);
        assert_eq!(round_ties_even(2.5), 2.0);
        assert_eq!(round_ties_even(2.51), 3.0);
    }
}
