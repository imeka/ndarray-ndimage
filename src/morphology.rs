use ndarray::{s, Array3, ArrayBase, Data, Ix3, Zip};

use crate::{array_like, dim_minus_1, Kernel3d, Mask};

/// Binary erosion of a 3D binary image.
///
/// * `mask` - Binary image to be eroded.
/// * `kernel` - Structuring element used for the erosion.
pub fn binary_erosion<S>(mask: &ArrayBase<S, Ix3>, kernel: Kernel3d) -> Mask
where
    S: Data<Elem = bool>,
{
    // By definition, all borders are set to 0
    let (width, height, depth) = dim_minus_1(mask);
    let mut eroded_mask = Array3::from_elem(mask.dim(), false);
    let zone = s![1..width, 1..height, 1..depth];
    eroded_mask.slice_mut(zone).assign(&mask.slice(zone));

    // Erode the mask when at least one of the values doesn't respect the kernel.
    // An erosion is defined either as `all(!(!w & k))` or `!any(!w & k)`.
    // Note that an empty kernel will always produce a full mask.
    let zone = eroded_mask.slice_mut(s![1..width, 1..height, 1..depth]);
    if kernel == Kernel3d::Full {
        Zip::from(mask.windows((3, 3, 3))).map_assign_into(zone, |w| !w.iter().any(|w| !w));
    } else {
        Zip::from(mask.windows((3, 3, 3))).map_assign_into(zone, |w| {
            // This ugly condition is equivalent to
            // *mask = !w.iter().zip(&kernel).any(|(w, k)| !w & k)
            // but it's around 5x faster because there's no branch misprediction
            !(!w[(0, 1, 1)]
                || !w[(1, 0, 1)]
                || !w[(1, 1, 0)]
                || !w[(1, 1, 1)]
                || !w[(1, 1, 2)]
                || !w[(1, 2, 1)]
                || !w[(2, 1, 1)])
        });
    }
    eroded_mask
}

/// Binary dilation of a 3D binary image.
///
/// * `mask` - Binary image to be dilated.
/// * `kernel` - Structuring element used for the dilation.
pub fn binary_dilation<S>(mask: &ArrayBase<S, Ix3>, kernel: Kernel3d) -> Mask
where
    S: Data<Elem = bool>,
{
    let (width, height, depth) = mask.dim();
    let crop = s![1..=width, 1..=height, 1..=depth];
    let mut new_mask = array_like(mask, (width + 2, height + 2, depth + 2), false);
    new_mask.slice_mut(crop).assign(mask);
    let mask = new_mask.clone();

    // Dilate the mask when at least one of the values respect the kernel: `any(w & k)`.
    // Note that an empty kernel will always produce an empty mask.
    let zone = new_mask.slice_mut(crop);
    if kernel == Kernel3d::Full {
        Zip::from(mask.windows((3, 3, 3))).map_assign_into(zone, |w| w.iter().any(|&w| w));
    } else {
        Zip::from(mask.windows((3, 3, 3))).map_assign_into(zone, |w| {
            // This ugly condition is equivalent to
            // *mask = w.iter().zip(&kernel).any(|(w, k)| w & k)
            // but it's around 5x faster because there's no branch misprediction
            w[(0, 1, 1)]
                || w[(1, 0, 1)]
                || w[(1, 1, 0)]
                || w[(1, 1, 1)]
                || w[(1, 1, 2)]
                || w[(1, 2, 1)]
                || w[(2, 1, 1)]
        });
    }
    new_mask.slice(crop).to_owned()
}
