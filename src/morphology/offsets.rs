use ndarray::{ArrayBase, ArrayView3, Data, Ix3};

pub struct Offsets {
    mask_strides: Vec<isize>,
    dim_m1: Vec<usize>,
    offsets: Vec<isize>,
    center_is_true: bool,
    axes: [usize; 3],

    strides: Vec<usize>,
    backstrides: Vec<usize>,
    bounds: Vec<std::ops::Range<usize>>,
    n: usize,

    pub coordinates: Vec<usize>,
    at: usize,
}

impl Offsets {
    pub fn new<S>(mask: &ArrayBase<S, Ix3>, kernel: ArrayView3<bool>, is_dilate: bool) -> Offsets
    where
        S: Data<Elem = bool>,
    {
        let mask_shape = mask.shape();
        let mask_strides = mask.strides().to_vec();
        let axes = if mask_strides[0] > mask_strides[2] { [2, 1, 0] } else { [0, 1, 2] };
        let (offsets, n) = build_offsets(mask_shape, &mask_strides, kernel.view(), is_dilate);
        let dim_m1: Vec<_> = mask_shape.iter().map(|&len| len - 1).collect();

        let kernel_shape = kernel.shape();
        let center_is_true =
            kernel[(kernel_shape[0] / 2, kernel_shape[1] / 2, kernel_shape[2] / 2)];

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

        Offsets {
            mask_strides,
            dim_m1,
            offsets,
            center_is_true,
            axes,
            strides,
            backstrides,
            bounds,
            n,
            coordinates: vec![0; mask.ndim()],
            at: 0,
        }
    }

    /// Return all offsets defined at the current positon
    pub fn range(&self) -> &[isize] {
        &self.offsets[self.at..self.at + self.n]
    }

    pub fn move_to(&mut self, idx: isize) {
        //print!("{}  ", idx);
        let mut idx = idx as usize;
        for d in [0, 1, 2] {
            let s = self.mask_strides[d] as usize;
            self.coordinates[d] = idx / s;
            idx -= self.coordinates[d] * s;
        }
        //print!("{:?}  ", self.coordinates);
        //if self.coordinates == [5, 5, 6] {
        //    print!("");
        //}

        self.at = 0;
        for &d in &self.axes {
            let (start, end) = (self.bounds[d].start, self.bounds[d].end);
            let c = self.coordinates[d];
            let j = if c < start {
                c
            } else if c > end && end >= start {
                c + start - end
            } else {
                start
            };
            self.at += self.strides[d] * j;
        }
        //println!("{:?}", self.range());
    }

    pub fn next(&mut self) {
        for &d in &self.axes {
            if self.coordinates[d] < self.dim_m1[d] {
                if !self.bounds[d].contains(&self.coordinates[d]) {
                    self.at += self.strides[d];
                }
                self.coordinates[d] += 1;
                break;
            } else {
                self.coordinates[d] = 0;
                self.at -= self.backstrides[d];
            }
        }
    }

    pub fn center_is_true(&self) -> bool {
        self.center_is_true
    }
}

/// Builds the kernel offsets.
fn build_offsets(
    shape: &[usize],
    strides: &[isize],
    kernel: ArrayView3<bool>,
    is_dilate: bool,
) -> (Vec<isize>, usize) {
    let radii: Vec<_> = kernel.shape().iter().map(|&len| (len - 1) / 2).collect();
    let indices = build_indices(kernel, &radii, is_dilate);

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
    let z_pos = build_pos(0);
    let y_pos = build_pos(1);
    let x_pos = build_pos(2);

    let mut offsets = vec![];
    for &z in &z_pos {
        for &y in &y_pos {
            for &x in &x_pos {
                for idx2 in &indices {
                    let idx = [z + idx2[0], y + idx2[1], x + idx2[2]];
                    let offset = if idx.iter().zip(shape).any(|(i, s)| !(0..s).contains(i)) {
                        // This voxel in the current kernel is out of image
                        ooi_offset
                    } else {
                        idx2.iter().zip(strides).fold(0, |acc, (i, s)| acc + i * s)
                    };
                    offsets.push(offset)
                }
            }
        }
    }

    // Sort all chunks:
    // - This will enhance cache locality
    // - The `ooi_offset` are all glued at the end, so we can `break` when we see one
    for chunk in offsets.chunks_mut(indices.len()) {
        chunk.sort();
    }

    (offsets, indices.len())
}

fn build_indices(kernel: ArrayView3<bool>, radii: &[usize], is_dilate: bool) -> Vec<[isize; 3]> {
    let radii = [radii[0] as isize, radii[1] as isize, radii[2] as isize];
    kernel
        .indexed_iter()
        .filter_map(|(idx, &b)| {
            if !b {
                return None;
            }

            // Do not add index (0, 0, 0) because it represents offset 0 which it's useless for
            // both `dilate` and `erode`, thanks to the `center_is_true` condition.
            let centered =
                [idx.0 as isize - radii[0], idx.1 as isize - radii[1], idx.2 as isize - radii[2]];
            (centered != [0, 0, 0]).then_some(if is_dilate {
                // dilate works by applying offsets on all voxels (checking the state of the
                // neighbors), not by applying the kernel on all voxels. This frame of reference
                // switch implies that we must reverse the indices.
                [-1 * centered[0], -1 * centered[1], -1 * centered[2]]
            } else {
                // erosion doesn work "normally" so we don't need to reverse anything
                centered
            })
        })
        .collect()
}
