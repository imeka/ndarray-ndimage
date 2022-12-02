use ndarray::{ArrayBase, ArrayView3, Data, Ix3};

pub struct Offsets {
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
    pub fn new<S>(mask: &ArrayBase<S, Ix3>, kernel: ArrayView3<bool>) -> Offsets
    where
        S: Data<Elem = bool>,
    {
        let mask_shape = mask.shape();
        let mask_strides = mask.strides().to_vec();
        let (offsets, n) = build_offsets(mask_shape, &mask_strides, kernel.view());
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

        Offsets {
            dim_m1,
            offsets,
            center_is_true: kernel.as_slice_memory_order().unwrap()[kernel.len() / 2],
            axes: if mask_strides[0] > mask_strides[2] { [2, 1, 0] } else { [0, 1, 2] },
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
) -> (Vec<isize>, usize) {
    let radii: Vec<_> = kernel.shape().iter().map(|&len| (len - 1) / 2).collect();
    let indices = build_indices(kernel, &radii);

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

    // Sort all chunks so that
    // - Enhance cache locality
    // - The `ooi_offset` are all glued at the end, so we can `break` when we see one
    for chunk in offsets.chunks_mut(indices.len()) {
        chunk.sort();
        //println!("{:?}", chunk);
    }

    (offsets, indices.len())
}

fn build_indices(kernel: ArrayView3<bool>, radii: &[usize]) -> Vec<(isize, isize, isize)> {
    let indices_: Vec<_> = kernel
        .indexed_iter()
        .filter_map(|(idx, &b)| {
            b.then_some((
                idx.0 as isize - radii[0] as isize,
                idx.1 as isize - radii[1] as isize,
                idx.2 as isize - radii[2] as isize,
            ))
        })
        .collect();
    let mut indices = Vec::with_capacity(indices_.len());
    indices.extend(indices_[..indices_.len() / 2].iter().rev());
    indices.push(indices_[indices_.len() / 2]);
    indices.extend(indices_[indices_.len() / 2 + 1..].iter().rev());
    indices
}
