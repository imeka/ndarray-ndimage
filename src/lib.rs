#![warn(missing_docs, trivial_casts, trivial_numeric_casts, unused_qualifications)]

//! The `ndarray-image` crate provides multidimensional image processing for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].

use ndarray::{arr3, Array, Array3, ArrayBase, ArrayView3, Data, Dimension, Ix3, ShapeBuilder};

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
#[derive(Clone, PartialEq)]
pub enum Kernel3d<'a> {
    /// Diamond/star kernel (center and sides).
    ///
    /// Equivalent to `generate_binary_structure(3, 1)`.
    Star,
    /// Ball kernel (center and sides).
    ///
    /// Equivalent to `generate_binary_structure(3, 2)`.
    Ball,
    /// 3x3x3 cube.
    ///
    /// Equivalent to `generate_binary_structure(3, 3)`.
    Full,
    /// Generic kernel (owned data) of any 3D size.
    ///
    /// The generic kernels are incredibly slower on all morphological operations.
    GenericOwned(Array3<bool>),
    /// Generic kernel (shared data) of any 3D size.
    ///
    /// The generic kernels are incredibly slower on all morphological operations.
    GenericView(ArrayView3<'a, bool>),
}

impl<'a> std::fmt::Debug for Kernel3d<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernel3d::Star => write!(f, "Star {:?}", self.dim()),
            Kernel3d::Ball => write!(f, "Ball {:?}", self.dim()),
            Kernel3d::Full => write!(f, "Full {:?}", self.dim()),
            Kernel3d::GenericOwned(k) => write!(f, "Generic (owned) {:?}", k.dim()),
            Kernel3d::GenericView(k) => write!(f, "Generic (view) {:?}", k.dim()),
        }
    }
}

impl<'a> Kernel3d<'a> {
    /// Return the kernel dimension.
    pub fn dim(&self) -> (usize, usize, usize) {
        match self {
            Kernel3d::Star | Kernel3d::Ball | Kernel3d::Full => (3, 3, 3),
            Kernel3d::GenericOwned(k) => k.dim(),
            Kernel3d::GenericView(k) => k.dim(),
        }
    }

    /// Return the actual 3D array.
    pub fn array(&self) -> Array3<bool> {
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
            Kernel3d::GenericOwned(k) => k.clone(),
            Kernel3d::GenericView(k) => k.to_owned(),
        }
    }

    /// Return the 3-tuple radius of the kernel.
    pub fn radii(&self) -> (usize, usize, usize) {
        let dim = self.dim();
        ((dim.0 - 1) / 2, (dim.1 - 1) / 2, (dim.2 - 1) / 2)
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
