mod offsets;

use ndarray::{Array3, ArrayRef3, ArrayView3, ArrayViewMut3};

use crate::Mask;
use offsets::Offsets;

/// Binary erosion of a 3D binary image.
///
/// * `mask` - Binary image to be eroded.
/// * `kernel` - Structuring element used for the erosion. Must be of odd length. The center must
///   be `true`.
/// * `iterations` - The erosion is repeated iterations times.
pub fn binary_erosion(
    mask: &ArrayRef3<bool>,
    kernel: &ArrayRef3<bool>,
    iterations: usize,
) -> Mask
{
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    // We can't really reserve a good number of elements here. It could be anything.
    let mut last_indices = (iterations > 1).then_some(vec![]);
    let mut eroded = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view(), false);
    erode(mask.view(), &mut eroded.view_mut(), &mut offsets, &mut last_indices);

    if let Some(mut last_indices) = last_indices {
        for it in 1..iterations {
            if last_indices.is_empty() {
                break;
            }

            let save_next_indices = it < iterations - 1;
            next_it(&mut eroded, &mut offsets, &mut last_indices, save_next_indices, true, false);
        }
    }
    eroded
}

/// Binary dilation of a 3D binary image.
///
/// * `mask` - Binary image to be dilated.
/// * `kernel` - Structuring element used for the erosion. Must be of odd length. The center must
///   be `true`.
/// * `iterations` - The dilation is repeated iterations times.
pub fn binary_dilation(
    mask: &ArrayRef3<bool>,
    kernel: &ArrayRef3<bool>,
    iterations: usize,
) -> Mask
{
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    // We can't really reserve a good number of elements here. It could be anything.
    let mut last_indices = (iterations > 1).then_some(vec![]);
    let mut dilated = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view(), true);
    dilate(mask.view(), &mut dilated, &mut offsets, &mut last_indices);

    if let Some(mut last_indices) = last_indices {
        for it in 1..iterations {
            if last_indices.is_empty() {
                break;
            }

            let save_next_indices = it < iterations - 1;
            next_it(&mut dilated, &mut offsets, &mut last_indices, save_next_indices, false, true);
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
pub fn binary_opening(
    mask: &ArrayRef3<bool>,
    kernel: &ArrayRef3<bool>,
    iterations: usize,
) -> Mask
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
pub fn binary_closing(
    mask: &ArrayRef3<bool>,
    kernel: &ArrayRef3<bool>,
    iterations: usize,
) -> Mask
{
    let dilated = binary_dilation(mask, kernel, iterations);
    binary_erosion(&dilated, kernel, iterations)
}

/// Actual erosion work.
///
/// `out` MUST be a clone of `mask`, otherwise this won't work. We're not setting any values
/// uselessly, to be as fast as possible.
fn erode(
    mask: ArrayView3<bool>,
    out: &mut ArrayViewMut3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Option<Vec<isize>>,
) {
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();
    let ooi_offset = mask.len() as isize;

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if m {
            for &offset in offsets.range() {
                // Is offset the special value "Out Of Image"?
                if offset == ooi_offset {
                    // The offsets are sorted so we can quit as soon as we see the `ooi_offset`
                    break;
                } else {
                    // unsafe { !*mask.get_unchecked((i + offset) as usize) }
                    if !mask[(i + offset) as usize] {
                        *o = false;
                        // If we have more than one iteration, note all modified indices
                        if let Some(last_indices) = last_indices {
                            // Remember that `i` IS the neighbor, not the "center", so we're adding
                            // `i` here, not `i + offset`.
                            last_indices.push(i);
                        }
                        break;
                    }
                }
            }
        }
        offsets.next();
        i += 1;
    }
}

/// Actual dilation work.
///
/// `out` MUST be a clone of `mask`, otherwise this won't work. We're not setting any values
/// uselessly, to be as fast as possible.
fn dilate(
    mask: ArrayView3<bool>,
    out: &mut Array3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Option<Vec<isize>>,
) {
    // Even if `erode` and `dilate` could share the same code (as SciPy does), it produces much
    // slower code in practice. See previous function for some documentation.
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();
    let ooi_offset = mask.len() as isize;

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if !m {
            for &offset in offsets.range() {
                if offset == ooi_offset {
                    break;
                } else {
                    // unsafe { *mask.get_unchecked((i + offset) as usize) }
                    if mask[(i + offset) as usize] {
                        *o = true;
                        if let Some(last_indices) = last_indices {
                            last_indices.push(i);
                        }
                        break;
                    }
                }
            }
        }
        offsets.next();
        i += 1;
    }
}

/// Common function do compute another iteration of dilate or erode.
///
/// Use only when `dilate` or `erode` have been called and the `last_indices` collected.
///
/// - Use `false` and `true` for dilate.
/// - Use `true` and `false` for erode.
fn next_it(
    out: &mut Array3<bool>,
    offsets: &mut Offsets,
    last_indices: &mut Vec<isize>,
    save_next_indices: bool,
    b1: bool,
    b2: bool,
) {
    let out = out.as_slice_memory_order_mut().unwrap();
    let ooi_offset = out.len() as isize;

    // Again, it's complex to guess the right number of elements. I think the same number as last
    // time + ?% makes sense, but it could also be empty.
    let mut new_indices = vec![];
    for &i in &*last_indices {
        offsets.move_to(i);
        for &offset in offsets.range() {
            if offset == ooi_offset {
                break;
            } else {
                let out = &mut out[(i + offset) as usize];
                if save_next_indices && *out == b1 {
                    // This time, `i` is the center and `i + offset` is the neighbor
                    new_indices.push(i + offset);
                }
                *out = b2;
            }
        }
    }
    *last_indices = new_indices;
}
