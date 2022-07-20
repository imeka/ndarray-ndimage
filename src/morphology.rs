use ndarray::{s, ArrayBase, ArrayView3, ArrayViewMut3, Data, Ix3, Zip};

use crate::{array_like, dim_minus, Kernel3d, Mask};

/// Binary erosion of a 3D binary image.
///
/// * `mask` - Binary image to be eroded.
/// * `kernel` - Structuring element used for the erosion.
/// * `iterations` - The erosion is repeated iterations times.
pub fn binary_erosion<S>(mask: &ArrayBase<S, Ix3>, kernel: Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    // Erode the mask when at least one of the values doesn't respect the kernel.
    // An erosion is defined either as `all(!(!w & k))` or `!any(!w & k)`.
    // An empty kernel will always produce a full mask.
    let erode = |from: ArrayView3<bool>, into: ArrayViewMut3<bool>| {
        if kernel == Kernel3d::Full {
            Zip::from(from.windows((3, 3, 3))).map_assign_into(into, |w| !w.iter().any(|w| !w));
        } else {
            Zip::from(from.windows((3, 3, 3))).map_assign_into(into, |w| {
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
    };

    // By definition, all borders are set to 0
    let (width, height, depth) = dim_minus(mask, 1);
    let mut eroded_mask = Mask::from_elem(mask.dim(), false);
    let zone = s![1..width, 1..height, 1..depth];
    eroded_mask.slice_mut(zone).assign(&mask.slice(zone));

    erode(mask.view(), eroded_mask.slice_mut(zone));

    if iterations > 1 {
        let mut previous = eroded_mask.clone();
        for it in 1..iterations {
            let (width, height, depth) = dim_minus(mask, it - 1);
            let from = previous.slice(s![it - 1..width, it - 1..height, it - 1..depth]);
            let (width, height, depth) = dim_minus(mask, it);
            let zone = s![it..width, it..height, it..depth];

            erode(from, eroded_mask.slice_mut(zone));

            if it != iterations {
                previous = eroded_mask.clone();
            }
        }
    }

    eroded_mask
}

/// Binary dilation of a 3D binary image.
///
/// * `mask` - Binary image to be dilated.
/// * `kernel` - Structuring element used for the dilation.
/// * `iterations` - The erosion is repeated iterations times.
pub fn binary_dilation<S>(mask: &ArrayBase<S, Ix3>, kernel: Kernel3d, iterations: usize) -> Mask
where
    S: Data<Elem = bool>,
{
    // Dilate the mask when at least one of the values respect the kernel: `any(w & k)`.
    // Note that an empty kernel will always produce an empty mask.
    let dilate = |from: ArrayView3<bool>, into: ArrayViewMut3<bool>| {
        if kernel == Kernel3d::Full {
            Zip::from(from.windows((3, 3, 3))).map_assign_into(into, |w| w.iter().any(|&w| w));
        } else {
            Zip::from(from.windows((3, 3, 3))).map_assign_into(into, |w| {
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
    };
    let (width, height, depth) = mask.dim();
    let crop = s![1..=width, 1..=height, 1..=depth];
    let mut new_mask = array_like(mask, (width + 2, height + 2, depth + 2), false);
    new_mask.slice_mut(crop).assign(mask);

    let mut previous = new_mask.clone();
    dilate(previous.view(), new_mask.slice_mut(crop));

    for _ in 1..iterations {
        previous = new_mask.clone();
        dilate(previous.view(), new_mask.slice_mut(crop));
    }

    new_mask.slice(crop).to_owned()
}
