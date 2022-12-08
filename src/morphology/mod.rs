mod offsets;

use ndarray::{Array3, ArrayBase, ArrayView3, ArrayViewMut3, Data, Ix3};

use crate::Mask;
use offsets::Offsets;

/// Binary erosion of a 3D binary image.
///
/// * `mask` - Binary image to be eroded.
/// * `kernel` - Structuring element used for the erosion.
/// * `iterations` - The erosion is repeated iterations times.
pub fn binary_erosion<SM, SK>(
    mask: &ArrayBase<SM, Ix3>,
    kernel: &ArrayBase<SK, Ix3>,
    iterations: usize,
) -> Mask
where
    SM: Data<Elem = bool>,
    SK: Data<Elem = bool>,
{
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    let mut last_indices = (iterations > 1).then_some(vec![]);
    let mut eroded = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view(), false);
    erode(mask.view(), &mut eroded.view_mut(), &mut offsets, &mut last_indices);

    if let Some(mut last_indices) = last_indices {
        for _ in 1..iterations {
            if last_indices.is_empty() {
                return eroded;
            }
            erode_from_indices(&mut eroded, &mut offsets, &mut last_indices);
        }
    }
    eroded
}

/// Binary dilation of a 3D binary image.
///
/// * `mask` - Binary image to be dilated.
/// * `kernel` - Structuring element used for the dilation.
/// * `iterations` - The dilation is repeated iterations times.
pub fn binary_dilation<SM, SK>(
    mask: &ArrayBase<SM, Ix3>,
    kernel: &ArrayBase<SK, Ix3>,
    iterations: usize,
) -> Mask
where
    SM: Data<Elem = bool>,
    SK: Data<Elem = bool>,
{
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    let mut last_indices = (iterations > 1).then_some(vec![]);
    let mut dilated = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view(), true);
    dilate(mask.view(), &mut dilated, &mut offsets, &mut last_indices);

    if let Some(mut last_indices) = last_indices {
        for _ in 1..iterations {
            if last_indices.is_empty() {
                return dilated;
            }
            dilate_from_indices(&mut dilated, &mut offsets, &mut last_indices);
        }
    }
    dilated
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
pub fn binary_opening<SM, SK>(
    mask: &ArrayBase<SM, Ix3>,
    kernel: &ArrayBase<SK, Ix3>,
    iterations: usize,
) -> Mask
where
    SM: Data<Elem = bool>,
    SK: Data<Elem = bool>,
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
pub fn binary_closing<SM, SK>(
    mask: &ArrayBase<SM, Ix3>,
    kernel: &ArrayBase<SK, Ix3>,
    iterations: usize,
) -> Mask
where
    SM: Data<Elem = bool>,
    SK: Data<Elem = bool>,
{
    let dilated = binary_dilation(mask, kernel, iterations);
    binary_erosion(&dilated, kernel, iterations)
}

fn erode(
    mask: ArrayView3<bool>,
    out: &mut ArrayViewMut3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Option<Vec<isize>>,
) {
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();
    let center_is_true = offsets.center_is_true();
    let ooi_offset = mask.len() as isize;

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if center_is_true && !m {
            *o = false;
        } else {
            *o = true;
            for &offset in offsets.range() {
                // Is offset the special value "Out Of Image"?
                if offset == ooi_offset {
                    // The offsets are sorted so we can quit as soon as we see the `ooi_offset`
                    break;
                } else {
                    if !mask[(i + offset) as usize] {
                        *o = false;
                        break;
                    }
                }
            }

            // If we have more than one iteration, note all modfied indices
            if let Some(last_indices) = last_indices {
                if *o != m {
                    // Remember that `i` IS the neighbor, not the "center"
                    last_indices.push(i);
                }
            }
        }
        offsets.next();
        i += 1;
    }
}

fn erode_from_indices(
    out: &mut Array3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Vec<isize>,
) {
    let out = out.as_slice_memory_order_mut().unwrap();
    let ooi_offset = out.len() as isize;

    let mut new_indices = vec![];
    for &i in &*last_indices {
        offsets.move_to(i);
        for &offset in offsets.range() {
            if offset == ooi_offset {
                break;
            } else {
                let out = &mut out[(i + offset) as usize];
                if *out {
                    new_indices.push(i + offset);
                }
                *out = false;
            }
        }
    }
    *last_indices = new_indices;
}

// Even if `erode` and `dilate` could share the same code (as SciPy does), it produces much slower
// code in practice. See previous function for some documentation.
fn dilate(
    mask: ArrayView3<bool>,
    out: &mut Array3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Option<Vec<isize>>,
) {
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();
    let center_is_true = offsets.center_is_true();
    let ooi_offset = mask.len() as isize;

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if center_is_true && m {
            *o = true;
        } else {
            *o = false;
            for &offset in offsets.range() {
                if offset == ooi_offset {
                    break;
                } else {
                    if mask[(i + offset) as usize] {
                        *o = true;
                        break;
                    }
                }
            }

            if let Some(last_indices) = last_indices {
                if *o != m {
                    last_indices.push(i);
                }
            }
        }
        offsets.next();
        i += 1;
    }
}

fn dilate_from_indices(
    out: &mut Array3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Vec<isize>,
) {
    let out = out.as_slice_memory_order_mut().unwrap();
    let ooi_offset = out.len() as isize;

    let mut new_indices = vec![];
    for &i in &*last_indices {
        offsets.move_to(i);
        for &offset in offsets.range() {
            if offset == ooi_offset {
                break;
            } else {
                let out = &mut out[(i + offset) as usize];
                if !*out {
                    new_indices.push(i + offset);
                }
                *out = true;
            }
        }
    }
    *last_indices = new_indices;
}
