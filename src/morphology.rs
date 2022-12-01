use ndarray::{ArrayBase, ArrayView3, ArrayViewMut3, Data, Ix3};

use crate::Mask;

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

    let mut eroded = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view());
    erode(mask.view(), &mut eroded.view_mut(), &mut offsets);
    if iterations > 1 {
        let mut buffer = eroded.clone();
        for it in 1..iterations {
            erode(buffer.view(), &mut eroded.view_mut(), &mut offsets);
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

    let mut dilated = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel.view());
    dilate(mask.view(), &mut dilated.view_mut(), &mut offsets);
    if iterations > 1 {
        let mut buffer = dilated.clone();
        for it in 1..iterations {
            dilate(buffer.view(), &mut dilated.view_mut(), &mut offsets);
            if it != iterations - 1 {
                buffer = dilated.clone();
            }
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

fn erode(mask: ArrayView3<bool>, out: &mut ArrayViewMut3<bool>, offsets: &mut Offsets) {
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if offsets.center_is_true && !m {
            *o = false;
        } else {
            *o = true;
            for &offset in offsets.range() {
                if offset != offsets.ooi {
                    if !mask[(i + offset) as usize] {
                        *o = false;
                        break;
                    }
                }
            }
        }

        // TODO 'f' order could be made cache-friendly (and faster than SciPy)
        // We currently iterate on 'x' (axis 2) even in 'f' order, which makes us jump all around
        // in memory. The goal would be to:
        // - Reorder the offsets list in `Offsets` so that it works with the usual 'c' code
        // - Remove `data_backstrides` in `Offsets::new`
        // - `i += 1` below, remove `inc` in `Offsets::next`
        i += offsets.next();
    }
}

// Even if `erode` and `dilate` could share the same code (as SciPy does), it produces much slower
// code in practice.
fn dilate(mask: ArrayView3<bool>, out: &mut ArrayViewMut3<bool>, offsets: &mut Offsets) {
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        if offsets.center_is_true && m {
            *o = true;
        } else {
            *o = false;
            for &offset in offsets.range() {
                if offset != offsets.ooi {
                    if mask[(i + offset) as usize] {
                        *o = true;
                        break;
                    }
                }
            }
        }
        i += offsets.next();
    }
}

struct Offsets {
    dim_m1: Vec<usize>,
    mask_strides: Vec<isize>,
    mask_backstrides: Vec<isize>,
    offsets: Vec<isize>,
    center_is_true: bool,
    ooi: isize, // Out Of Image offset value

    strides: Vec<usize>,
    backstrides: Vec<usize>,
    bounds: Vec<std::ops::Range<usize>>,
    n: usize,

    coordinates: Vec<usize>,
    at: usize,
}

impl Offsets {
    fn new<S>(mask: &ArrayBase<S, Ix3>, kernel: ArrayView3<bool>) -> Offsets
    where
        S: Data<Elem = bool>,
    {
        let mask_shape = mask.shape();
        let mask_strides = mask.strides().to_vec();
        let mask_backstrides = if mask_strides[0] > mask_strides[1] {
            // 'c' order
            vec![0, mask_strides[0] - mask_strides[1], mask_strides[1] - mask_strides[2]]
        } else {
            // 'f' order
            vec![
                0,
                mask_strides[1] * (mask_shape[1] - 1) as isize,
                mask_strides[2] * (mask_shape[2] - 1) as isize,
            ]
        };

        let (offsets, n, ooi) = build_offsets(mask_shape, &mask_strides, kernel.view());
        let dim_m1: Vec<_> = mask_shape.iter().map(|&len| len - 1).collect();

        let kernel_shape = kernel.shape();
        let mut strides = vec![0; mask.ndim()];
        strides[mask.ndim() - 1] = n;
        for d in (0..mask.ndim() - 1).rev() {
            strides[d] = strides[d + 1] * kernel_shape[d];
        }
        let backstrides = strides.iter().zip(kernel_shape).map(|(&s, &l)| (l - 1) * s).collect();
        let bounds = (0..mask.ndim())
            .map(|d| {
                let radius = (kernel_shape[d] - 1) / 2;
                radius..dim_m1[d] - radius
            })
            .collect();

        let center_is_true = kernel.as_slice_memory_order().unwrap()[kernel.len() / 2];

        //println!("Strides: {:?}", strides);
        //println!("Backstrides: {:?}", backstrides);
        //println!("Bounds: {:?}", bounds);

        Offsets {
            dim_m1,
            mask_strides,
            mask_backstrides,
            offsets,
            center_is_true,
            ooi,
            strides,
            backstrides,
            bounds,
            n,
            coordinates: vec![0; mask.ndim()],
            at: 0,
        }
    }

    /// Return all offsets defined at the current positon
    fn range(&self) -> &[isize] {
        &self.offsets[self.at..self.at + self.n]
    }

    fn next(&mut self) -> isize {
        let mut inc = 0;
        for d in (0..3).rev() {
            if self.coordinates[d] < self.dim_m1[d] {
                if !self.bounds[d].contains(&self.coordinates[d]) {
                    self.at += self.strides[d];
                }
                self.coordinates[d] += 1;
                inc += self.mask_strides[d];
                break;
            } else {
                self.coordinates[d] = 0;
                self.at -= self.backstrides[d];
                inc -= self.mask_backstrides[d];
            }
        }
        inc
    }
}

/// Builds the kernel offsets.
fn build_offsets(
    shape: &[usize],
    strides: &[isize],
    kernel: ArrayView3<bool>,
) -> (Vec<isize>, usize, isize) {
    let kernel_shape = kernel.shape();
    let radii: Vec<_> = kernel_shape.iter().map(|&len| (len - 1) / 2).collect();
    let mut indices = vec![];
    for x in 0..kernel_shape[2] {
        for y in 0..kernel_shape[1] {
            for z in 0..kernel_shape[0] {
                if kernel[(z, y, x)] {
                    indices.push((
                        z as isize - radii[0] as isize,
                        y as isize - radii[1] as isize,
                        x as isize - radii[2] as isize,
                    ))
                }
            }
        }
    }
    //for idx in &indices {
    //    println!("{:?}", idx);
    //}

    /*let indices: Vec<_> = self
    .array()
    .indexed_iter()
    .filter_map(|(idx, &b)| {
        b.then_some((
            idx.0 as isize - radii[0] as isize,
            idx.1 as isize - radii[1] as isize,
            idx.2 as isize - radii[2] as isize,
        ))
    })
    .collect();*/

    let shape = [shape[0] as isize, shape[1] as isize, shape[2] as isize];
    let ooi_offset = shape.iter().fold(1, |acc, &s| acc * s);
    let build_pos = |d: usize| {
        let mut pos = Vec::with_capacity(kernel.shape()[d]);
        let radius = radii[d] as isize;
        pos.extend(0..radius);
        pos.push(shape[d] / 2);
        pos.extend(shape[d] - radius..shape[d]);
        pos
    };
    let z_pos = build_pos(2);
    let y_pos = build_pos(1);
    let x_pos = build_pos(0);

    let mut offsets = vec![];
    for &z in &z_pos {
        for &y in &y_pos {
            for &x in &x_pos {
                for &idx2 in &indices {
                    let idx = (x + idx2.0, y + idx2.1, z + idx2.2);
                    let offset = if !(0..shape[0]).contains(&idx.0)
                        || !(0..shape[1]).contains(&idx.1)
                        || !(0..shape[2]).contains(&idx.2)
                    {
                        // This voxel in the current kernel is out of image
                        ooi_offset
                    } else {
                        // TODO Should we remove the center?
                        idx2.0 * strides[2] + idx2.1 * strides[1] + idx2.2 * strides[0]
                    };
                    offsets.push(offset)
                }
            }
        }
    }
    //for chunk in offsets.chunks(indices.len()) {
    //    println!("{:?}", chunk);
    //}
    (offsets, indices.len(), ooi_offset)
}
