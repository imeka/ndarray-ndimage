use ndarray::{s, ArrayBase, ArrayView3, ArrayViewMut3, Data, Ix3, Zip};

use crate::{array_like, Kernel3d, Mask};

impl<'a> Kernel3d<'a> {
    /// Erode the mask when at least one of the values doesn't respect the kernel.
    ///
    /// An erosion is defined either as `all(!(!w & k))` or `!any(!w & k)`. Thus, an empty kernel
    /// will always produce a full mask.
    #[rustfmt::skip]
    fn erode(&self, from: ArrayView3<bool>, into: ArrayViewMut3<bool>) {
        let windows = from.windows(self.dim());
        match self {
            Kernel3d::Full => Zip::from(windows).map_assign_into(into, |w| {
                // This is incredibly ugly but this is much faster (3x) than the Zip
                // |w| Zip::from(w).all(|&m| m)
                   w[(0, 0, 0)] && w[(0, 0, 1)] && w[(0, 0, 2)]
                && w[(0, 1, 0)] && w[(0, 1, 1)] && w[(0, 1, 2)]
                && w[(0, 2, 0)] && w[(0, 2, 1)] && w[(0, 2, 2)]
                && w[(1, 0, 0)] && w[(1, 0, 1)] && w[(1, 0, 2)]
                && w[(1, 1, 0)] && w[(1, 1, 1)] && w[(1, 1, 2)]
                && w[(1, 2, 0)] && w[(1, 2, 1)] && w[(1, 2, 2)]
                && w[(2, 0, 0)] && w[(2, 0, 1)] && w[(2, 0, 2)]
                && w[(2, 1, 0)] && w[(2, 1, 1)] && w[(2, 1, 2)]
                && w[(2, 2, 0)] && w[(2, 2, 1)] && w[(2, 2, 2)]
            }),
            Kernel3d::Ball => Zip::from(windows).map_assign_into(into, |w| {
                                   w[(0, 0, 1)]
                && w[(0, 1, 0)] && w[(0, 1, 1)] && w[(0, 1, 2)]
                                && w[(0, 2, 1)]
                && w[(1, 0, 0)] && w[(1, 0, 1)] && w[(1, 0, 2)]
                && w[(1, 1, 0)] && w[(1, 1, 1)] && w[(1, 1, 2)]
                && w[(1, 2, 0)] && w[(1, 2, 1)] && w[(1, 2, 2)]
                                && w[(2, 0, 1)]
                && w[(2, 1, 0)] && w[(2, 1, 1)] && w[(2, 1, 2)]
                                && w[(2, 2, 1)]
            }),
            Kernel3d::Star => Zip::from(windows).map_assign_into(into, |w| {
                // This ugly condition is equivalent to
                // *mask = !w.iter().zip(&kernel).any(|(w, k)| !w & k)
                // but it's extremely faster because there's no branch misprediction
                                   w[(0, 1, 1)]
                                && w[(1, 0, 1)]
                && w[(1, 1, 0)] && w[(1, 1, 1)] && w[(1, 1, 2)]
                                && w[(1, 2, 1)]
                                && w[(2, 1, 1)]
            }),
            Kernel3d::GenericOwned(kernel) => Zip::from(windows).map_assign_into(into, |w| {
                // TODO Maybe use Zip::any when available
                // !Zip::from(w).and(kernel).any(|(&w, &k)| !w & k)
                Zip::from(w).and(kernel).all(|&w, &k| !(!w & k))
            }),
            Kernel3d::GenericView(kernel) => Zip::from(windows).map_assign_into(into, |w| {
                // TODO Maybe use Zip::any when available
                // !Zip::from(w).and(kernel).any(|(&w, &k)| !w & k)
                Zip::from(w).and(kernel).all(|&w, &k| !(!w & k))
            })
        }
    }

    /// Dilate the mask when at least one of the values respect the kernel:
    ///
    /// A dilation is defined as `any(w & k)`. Thus, an empty kernel will always produce an empty
    /// mask.
    #[rustfmt::skip]
    fn dilate(&self, from: ArrayView3<bool>, into: ArrayViewMut3<bool>) {
        let windows = from.windows(self.dim());
        match self {
            Kernel3d::Full => Zip::from(windows).map_assign_into(into, |w| {
                // This is incredibly ugly but this is much faster (6x) than the Zip
                // |w| w.iter().any(|&w| w))
                   w[(0, 0, 0)] || w[(0, 0, 1)] || w[(0, 0, 2)]
                || w[(0, 1, 0)] || w[(0, 1, 1)] || w[(0, 1, 2)]
                || w[(0, 2, 0)] || w[(0, 2, 1)] || w[(0, 2, 2)]
                || w[(1, 0, 0)] || w[(1, 0, 1)] || w[(1, 0, 2)]
                || w[(1, 1, 0)] || w[(1, 1, 1)] || w[(1, 1, 2)]
                || w[(1, 2, 0)] || w[(1, 2, 1)] || w[(1, 2, 2)]
                || w[(2, 0, 0)] || w[(2, 0, 1)] || w[(2, 0, 2)]
                || w[(2, 1, 0)] || w[(2, 1, 1)] || w[(2, 1, 2)]
                || w[(2, 2, 0)] || w[(2, 2, 1)] || w[(2, 2, 2)]
            }),
            Kernel3d::Ball => Zip::from(windows).map_assign_into(into, |w| {
                                   w[(0, 0, 1)]
                || w[(0, 1, 0)] || w[(0, 1, 1)] || w[(0, 1, 2)]
                                || w[(0, 2, 1)]
                || w[(1, 0, 0)] || w[(1, 0, 1)] || w[(1, 0, 2)]
                || w[(1, 1, 0)] || w[(1, 1, 1)] || w[(1, 1, 2)]
                || w[(1, 2, 0)] || w[(1, 2, 1)] || w[(1, 2, 2)]
                                || w[(2, 0, 1)]
                || w[(2, 1, 0)] || w[(2, 1, 1)] || w[(2, 1, 2)]
                                || w[(2, 2, 1)]
            }),
            Kernel3d::Star => Zip::from(windows).map_assign_into(into, |w| {
                // This ugly condition is equivalent to
                // |(w, k)| w & k
                // but it's around 5x faster because there's no branch misprediction
                                   w[(0, 1, 1)]
                                || w[(1, 0, 1)]
                || w[(1, 1, 0)] || w[(1, 1, 1)] || w[(1, 1, 2)]
                                || w[(1, 2, 1)]
                                || w[(2, 1, 1)]
            }),
            Kernel3d::GenericOwned(kernel) => Zip::from(windows).map_assign_into(into, |w| {
                // TODO Use Zip::any when available
                // Zip::from(w).and(kernel).any(|idx, &w, &k| w & k)
                w.iter().zip(kernel).any(|(w, k)| w & k)
            }),
            Kernel3d::GenericView(kernel) => Zip::from(windows).map_assign_into(into, |w| {
                // TODO Use Zip::any when available
                // Zip::from(w).and(kernel).any(|idx, &w, &k| w & k)
                w.iter().zip(kernel).any(|(w, k)| w & k)
            })
        }
    }
}

/// Binary erosion of a 3D binary image.
///
/// * `mask` - Binary image to be eroded.
/// * `kernel` - Structuring element used for the erosion.
/// * `iterations` - The erosion is repeated iterations times.
pub fn binary_erosion<S>(mask: &ArrayBase<S, Ix3>, kernel: &Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    let (w, h, d) = mask.dim();
    let (r_x, r_y, r_z) = kernel.radius();

    // By definition, all borders are set to 0
    let mut eroded_mask = Mask::from_elem(mask.dim(), false);
    let zone = s![r_x..w - r_x, r_y..h - r_y, r_z..d - r_z];
    eroded_mask.slice_mut(zone).assign(&mask.slice(zone));

    kernel.erode(mask.view(), eroded_mask.slice_mut(zone));
    if iterations == 1 {
        return eroded_mask;
    }

    let iterations = iterations - 1;
    let mut previous = eroded_mask.clone();
    for it in 0..iterations {
        let from = previous.slice(s![it..w - it, it..h - it, it..d - it]);
        let zone = s![it + r_x..w - it - r_x, it + r_y..h - it - r_y, it + r_z..d - it - r_z];
        kernel.erode(from, eroded_mask.slice_mut(zone));

        if it != iterations {
            previous = eroded_mask.clone();
        }
    }

    eroded_mask
}

/// Binary dilation of a 3D binary image.
///
/// * `mask` - Binary image to be dilated.
/// * `kernel` - Structuring element used for the dilation.
/// * `iterations` - The dilation is repeated iterations times.
pub fn binary_dilation<S>(mask: &ArrayBase<S, Ix3>, kernel: &Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    let (w, h, d) = mask.dim();
    let (r_x, r_y, r_z) = kernel.radius();
    let crop = s![r_x..w + r_x, r_y..h + r_y, r_z..d + r_z];
    let mut new_mask = array_like(mask, (w + 2 * r_x, h + 2 * r_y, d + 2 * r_z), false);
    new_mask.slice_mut(crop).assign(mask);

    let mut previous = new_mask.clone();
    kernel.dilate(previous.view(), new_mask.slice_mut(crop));

    for _ in 1..iterations {
        previous = new_mask.clone();
        kernel.dilate(previous.view(), new_mask.slice_mut(crop));
    }

    new_mask.slice(crop).to_owned()
}

/// Binary opening of a 3D binary image.
///
/// The opening of an input image by a structuring element is the dilation of the erosion of the
/// image by the structuring element.
///
/// * `mask` - Binary image to be opened.
/// * `kernel` - Structuring element used for the opening.
/// * `iterations` - The erosion step of the opening, then the dilation step are each repeated
///   iterations times.
pub fn binary_opening<S>(mask: &ArrayBase<S, Ix3>, kernel: &Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    let eroded = binary_erosion(mask, kernel, iterations);
    binary_dilation(&eroded, kernel, iterations)
}

/// Binary closing of a 3D binary image.
///
/// The closing of an input image by a structuring element is the erosion of the dilation of the
/// image by the structuring element.
///
/// * `mask` - Binary image to be closed.
/// * `kernel` - Structuring element used for the closing.
/// * `iterations` - The dilation step of the closing, then the erosion step are each repeated
///   iterations times.
pub fn binary_closing<S>(mask: &ArrayBase<S, Ix3>, kernel: &Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    let dilated = binary_dilation(mask, kernel, iterations);
    binary_erosion(&dilated, kernel, iterations)
}
