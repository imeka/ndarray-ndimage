use ndarray::{ArrayBase, Data, DataMut, Ix3};

use crate::{Kernel3d, Mask};

impl<'a> Kernel3d<'a> {
    /// Builds the kernel offsets.
    fn offsets(&self, shape: &[usize], strides: &[isize]) -> (Vec<isize>, usize, isize) {
        let dim = self.dim();
        let array = self.array();
        let radii = self.radii();
        let mut indices = vec![];
        for x in 0..dim.2 {
            for y in 0..dim.1 {
                for z in 0..dim.0 {
                    if array[(z, y, x)] {
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
            let mut pos = Vec::with_capacity(array.shape()[d]);
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

    let mut eroded = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel);
    dilate_or_erode(mask, &mut eroded, &mut offsets, true, true, false);
    if iterations > 1 {
        let mut buffer = eroded.clone();
        for it in 1..iterations {
            dilate_or_erode(&buffer, &mut eroded, &mut offsets, true, true, false);
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
    mask.as_slice_memory_order()
        .expect("Morphological operations can only be called on arrays with contiguous memory.");

    let mut dilated = mask.to_owned();
    let mut offsets = Offsets::new(mask, kernel);
    dilate_or_erode(mask, &mut dilated, &mut offsets, true, false, true);
    if iterations > 1 {
        let mut buffer = dilated.clone();
        for it in 1..iterations {
            dilate_or_erode(&buffer, &mut dilated, &mut offsets, true, false, true);
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

fn dilate_or_erode<S, SMut>(
    mask: &ArrayBase<S, Ix3>,
    out: &mut ArrayBase<SMut, Ix3>,
    offsets: &mut Offsets,
    border_value: bool,
    true_: bool,
    false_: bool,
) where
    S: Data<Elem = bool>,
    SMut: DataMut<Elem = bool>,
{
    let mask = mask.as_slice_memory_order().unwrap();
    let out = out.as_slice_memory_order_mut().unwrap();

    let mut i = 0;
    for (&m, o) in mask.iter().zip(out) {
        //println!("{:?}  {:?}  {}", offsets.coordinates, offsets.range(), i);
        if offsets.center_is_true && m == false_ {
            *o = false_;
        } else {
            *o = true_;
            for &offset in offsets.range() {
                if offset == offsets.ooi {
                    if !border_value {
                        *o = false_;
                        break;
                    }
                } else {
                    let t = if mask[(i + offset) as usize] { true_ } else { false_ };
                    if !t {
                        *o = false_;
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

struct Offsets {
    dim_m1: Vec<usize>,
    data_strides: Vec<isize>,
    data_backstrides: Vec<isize>,
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
    fn new<S>(mask: &ArrayBase<S, Ix3>, kernel: &Kernel3d) -> Offsets
    where
        S: Data<Elem = bool>,
    {
        let shape = mask.shape();
        let data_strides = mask.strides().to_vec();
        let data_backstrides = if data_strides[0] > data_strides[1] {
            // 'c' order
            vec![0, data_strides[0] - data_strides[1], data_strides[1] - data_strides[2]]
        } else {
            // 'f' order
            vec![
                0,
                data_strides[1] * (shape[1] - 1) as isize,
                data_strides[2] * (shape[2] - 1) as isize,
            ]
        };

        let (offsets, n, ooi) = kernel.offsets(shape, &data_strides);
        let radii = kernel.radii();
        let dim_m1: Vec<_> = shape.iter().map(|&len| len - 1).collect();

        let array = kernel.array();
        let mut strides = vec![0; mask.ndim()];
        strides[mask.ndim() - 1] = n;
        for d in (0..mask.ndim() - 1).rev() {
            strides[d] = strides[d + 1] * array.shape()[d];
        }
        let backstrides = strides.iter().zip(array.shape()).map(|(&s, &l)| (l - 1) * s).collect();
        let bounds = (0..mask.ndim()).map(|d| radii[d]..dim_m1[d] - radii[d]).collect();

        //println!("Strides: {:?}", strides);
        //println!("Backstrides: {:?}", backstrides);
        //println!("Bounds: {:?}", bounds);

        Offsets {
            dim_m1,
            data_strides,
            data_backstrides,
            offsets,
            center_is_true: kernel.center_is_true(),
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
                inc += self.data_strides[d];
                break;
            } else {
                self.coordinates[d] = 0;
                self.at -= self.backstrides[d];
                inc -= self.data_backstrides[d];
            }
        }
        inc
    }
}
