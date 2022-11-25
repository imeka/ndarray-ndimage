use ndarray::{s, ArrayBase, ArrayView3, ArrayViewMut3, Data, DataMut, Ix3, Zip};

use crate::{array_like, Kernel3d, Mask};

impl<'a> Kernel3d<'a> {
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

    /// Builds the kernel indices and offsets.
    ///
    /// For example, on a 4x5x6 image with the `Kernel3d::Star`
    /// - `((0, 0, -1), -1)`
    /// - `((0, -1, 0), -6)`
    /// - `((-1, 0, 0), -30)`
    /// - `((1, 0, 0), 30)`
    /// - `((0, 1, 0), 6)`
    /// - `((0, 0, 1), 1)`
    ///
    /// The center is ignored for optimization reasons.
    fn indices_offsets(&self, strides: &[isize]) -> Vec<((isize, isize, isize), isize)> {
        let rad = self.radii();
        let rad = (rad.0 as isize, rad.1 as isize, rad.2 as isize);

        let mut i_o = vec![];
        Zip::indexed(&self.array()).for_each(|idx, &k| {
            if k {
                let idx = (idx.0 as isize - rad.0, idx.1 as isize - rad.1, idx.2 as isize - rad.2);
                let offset = idx.0 * strides[0] + idx.1 * strides[1] + idx.2 * strides[2];
                if offset != 0 {
                    i_o.push((idx, offset));
                }
            }
        });
        i_o
    }

    fn center_is_true(&self) -> bool {
        let dim = self.dim();
        let idx_center = (dim.0 / 2, dim.1 / 2, dim.2 / 2);
        match self {
            Kernel3d::Star | Kernel3d::Ball | Kernel3d::Full => true,
            Kernel3d::GenericOwned(kernel) => kernel[idx_center],
            Kernel3d::GenericView(kernel) => kernel[idx_center],
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
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    let strides = mask.strides();
    let i_o = kernel.indices_offsets(strides);
    let center_is_true = kernel.center_is_true();
    let is_fortran = strides[0] < strides[2];

    let mut eroded = mask.to_owned();
    erode(mask, &mut eroded, &i_o, center_is_true, is_fortran);
    if iterations > 1 {
        let mut buffer = eroded.clone();
        for it in 1..iterations {
            erode(&buffer, &mut eroded, &i_o, center_is_true, is_fortran);
            if it != iterations - 1 {
                buffer = eroded.clone();
            }
        }
    }
    eroded
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
    let (r_x, r_y, r_z) = kernel.radii();
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
/// Unlike other libraries, the **border values** of the:
/// - dilation is always `false`, to avoid dilating the borders
/// - erosion is always `true`, to avoid *border effects*
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
/// Unlike other libraries, the **border values** of the:
/// - dilation is always `false`, to avoid dilating the borders
/// - erosion is always `true`, to avoid *border effects*
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

fn erode<S, SMut>(
    mask: &ArrayBase<S, Ix3>,
    out: &mut ArrayBase<SMut, Ix3>,
    i_o: &[((isize, isize, isize), isize)],
    center_is_true: bool,
    is_fortran: bool,
) where
    S: Data<Elem = bool>,
    SMut: DataMut<Elem = bool>,
{
    let dim = (mask.dim().0 as isize, mask.dim().1 as isize, mask.dim().2 as isize);
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();

    let mut x = 0;
    let mut y = 0;
    let mut z = 0;
    for (i, (&m, o)) in mask.iter().zip(out).enumerate() {
        if center_is_true && !m {
            *o = false;
        } else {
            *o = true;
            let ii = i as isize;
            for &(idx, offset) in i_o {
                // TODO Those `contains` make us slower than ScilPy. Their algo builds a list of
                // offsets (much longer than our `i_o`) and they are able to always have the right
                // offsets (see `NI_FILTER_NEXT2`). They have a single condition here instead of 3*2
                // conditions. The complexity is moved after this loop, which is perfect for bigger
                // kernels.
                if (0..dim.2).contains(&(x + idx.2))
                    && (0..dim.1).contains(&(y + idx.1))
                    && (0..dim.0).contains(&(z + idx.0))
                {
                    if !mask[(ii + offset) as usize] {
                        *o = false;
                        break;
                    }
                }
            }
        }

        // Calculate the next index in `mask`
        if is_fortran {
            z = z + 1;
            if z == dim.0 {
                z = 0;
                y = y + 1;
                if y == dim.1 {
                    y = 0;
                    x = x + 1;
                }
            }
        } else {
            x = x + 1;
            if x == dim.2 {
                x = 0;
                y = y + 1;
                if y == dim.1 {
                    y = 0;
                    z = z + 1;
                }
            }
        }
    }
}
