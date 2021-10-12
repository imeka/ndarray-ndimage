use ndarray::{s, Array3, Axis, Zip};

use crate::Mask;

const BACKGROUND: u16 = 0;
const FOREGROUND: u16 = 1;

/// Calculates the histogram of a label image
pub fn label_histogram(labels: &Array3<u16>, nb_features: usize) -> Vec<usize> {
    let mut count = vec![0; nb_features + 1];
    Zip::from(labels).for_each(|&l| {
        count[l as usize] += 1;
    });
    count
}

/// Returns the most frequent label and its index.
///
/// Ignores the background label. A blank label image will return None.
pub fn most_frequent_label(labels: &Array3<u16>, nb_features: usize) -> Option<(u16, usize)> {
    let hist = label_histogram(labels, nb_features);
    let (max, max_index) =
        hist[1..].iter().enumerate().fold((0, 0), |acc, (i, &nb)| acc.max((nb, i)));
    (max > 0).then(|| ((max_index + 1) as u16, max))
}

/// Returns the bigest zone of `mask` in a new mask.
pub fn largest_connected_components(mask: &Mask) -> Option<Mask> {
    let (labels, nb_features) = label(mask);
    let (right_label, _) = most_frequent_label(&labels, nb_features)?;
    Some(labels.mapv(|l| l == right_label))
}

/// Labels features of 3D binary images.
///
/// Currently support only the Star kernel (`Kernel3d::Star`).
///
/// Returns the labels and the number of features.
pub fn label(data: &Mask) -> (Array3<u16>, usize) {
    let len = data.dim().2;
    let mut line_buffer = vec![BACKGROUND; len + 2];
    let mut neighbors = vec![BACKGROUND; len + 2];

    let mut next_region = FOREGROUND + 1;
    let mut equivalences: Vec<_> = (0..next_region).collect();

    // We only handle 3D data with a 6-connectivity kernel for now, but this algo can handle
    // N-dimensional data and any symmetric kernels.
    // https://github.com/scipy/scipy/blob/v0.16.1/scipy/ndimage/src/_ni_label.pyx
    // N-D: Use a loop in `is_valid` and change the `labels` indexing (might be hard in Rust)
    // Kernels: Add new `kernel_data` and choose the right one
    let kernel_data = [
        //((false, false, false), [0, 0]),
        ([false, true, false], [0, 1]),
        //((false, false, false), [0, 0]),
        ([false, true, false], [1, 0]),
    ];
    let nb_neighbors = kernel_data.len();

    let use_previous = true;
    let width = data.dim().0 as isize;
    let height = data.dim().1 as isize;
    let mut labels = Array3::from_elem(data.dim(), BACKGROUND);
    Zip::indexed(data.lanes(Axis(2))).for_each(|idx, data| {
        for (&v, b) in data.iter().zip(&mut line_buffer[1..]) {
            *b = if !v { BACKGROUND } else { FOREGROUND }
        }

        let mut needs_self_labeling = true;
        for (i, (kernel, coordinates)) in kernel_data.iter().enumerate() {
            // Check that the neighbor line is in bounds
            if let Some((x, y)) = is_valid(&[idx.0, idx.1], coordinates, &[width, height]) {
                // Copy the interesting neighbor labels to `neighbors`
                for (&v, b) in labels.slice(s![x, y, ..]).iter().zip(&mut neighbors[1..]) {
                    *b = v;
                }

                let label_unlabeled = i == nb_neighbors - 1;
                next_region = label_line_with_neighbor(
                    &mut line_buffer,
                    &neighbors,
                    &mut equivalences,
                    *kernel,
                    use_previous,
                    label_unlabeled,
                    next_region,
                );
                if label_unlabeled {
                    needs_self_labeling = false;
                }
            }
        }

        if needs_self_labeling {
            // We didn't call label_line_with_neighbor above with label_unlabeled=True, so call it
            // now in such a way as to cause unlabeled regions to get a label.
            next_region = label_line_with_neighbor(
                &mut line_buffer,
                &neighbors,
                &mut equivalences,
                [false, false, false],
                use_previous,
                true,
                next_region,
            );
        }

        // Copy the results (`line_buffer`) to the output labels image
        Zip::from(&line_buffer[1..=len])
            .map_assign_into(labels.slice_mut(s![idx.0, idx.1, ..]), |&b| b);
    });

    // Compact and apply the equivalences
    let nb_features = compact_equivalences(&mut equivalences, next_region);
    labels.mapv_inplace(|l| equivalences[l as usize]);

    (labels, nb_features)
}

fn is_valid(idx: &[usize; 2], coords: &[isize; 2], dims: &[isize; 2]) -> Option<(usize, usize)> {
    let valid = |i, c, d| -> Option<usize> {
        let a = i as isize + (c - 1);
        if a >= 0 && a < d {
            if c == 0 {
                return Some(i - 1);
            }
        } else {
            return None;
        }
        Some(i)
    };
    // Returns `Some((x, y))` only if both calls succeded
    valid(idx[0], coords[0], dims[0])
        .and_then(|x| valid(idx[1], coords[1], dims[1]).and_then(|y| Some((x, y))))
}

fn label_line_with_neighbor(
    line: &mut [u16],
    neighbors: &[u16],
    equivalences: &mut Vec<u16>,
    kernel: [bool; 3],
    use_previous: bool,
    label_unlabeled: bool,
    mut next_region: u16,
) -> u16 {
    let mut previous = line[0];
    for (n, l) in neighbors.windows(3).zip(&mut line[1..]) {
        if *l != BACKGROUND {
            for (&n, &k) in n.iter().zip(&kernel) {
                if k {
                    *l = take_label_or_merge(*l, n, equivalences);
                }
            }
            if label_unlabeled {
                if use_previous {
                    *l = take_label_or_merge(*l, previous, equivalences);
                }
                // Still needs a label?
                if *l == FOREGROUND {
                    *l = next_region;
                    equivalences.push(next_region);
                    next_region += 1;
                }
            }
        }
        previous = *l;
    }
    next_region
}

/// Take the label of a neighbor, or mark them for merging
fn take_label_or_merge(current: u16, neighbor: u16, equivalences: &mut [u16]) -> u16 {
    if neighbor == BACKGROUND {
        current
    } else if current == FOREGROUND {
        neighbor // neighbor is not BACKGROUND
    } else if current != neighbor {
        mark_for_merge(neighbor, current, equivalences)
    } else {
        current
    }
}

/// Mark two labels to be merged
fn mark_for_merge(mut a: u16, mut b: u16, equivalences: &mut [u16]) -> u16 {
    // Find smallest root for each of a and b
    let original_a = a;
    while a != equivalences[a as usize] {
        a = equivalences[a as usize];
    }
    let original_b = b;
    while b != equivalences[b as usize] {
        b = equivalences[b as usize];
    }
    let lowest_label = a.min(b);

    // Merge roots
    equivalences[a as usize] = lowest_label;
    equivalences[b as usize] = lowest_label;

    // Merge every step to minlabel
    a = original_a;
    while a != lowest_label {
        let a_copy = a;
        a = equivalences[a as usize];
        equivalences[a_copy as usize] = lowest_label;
    }
    b = original_b;
    while b != lowest_label {
        let b_copy = b;
        b = equivalences[b as usize];
        equivalences[b_copy as usize] = lowest_label;
    }

    lowest_label
}

/// Compact the equivalences vector
fn compact_equivalences(equivalences: &mut [u16], next_region: u16) -> usize {
    let no_labelling = next_region == 2;
    let mut dest_label = if no_labelling { 0 } else { 1 };
    for i in 2..next_region as usize {
        if equivalences[i] == i as u16 {
            equivalences[i] = dest_label;
            dest_label += 1;
        } else {
            // We've compacted every label below this, and equivalences has an invariant that it
            // always points downward. Therefore, we can fetch the final label by two steps of
            // indirection.
            equivalences[i] = equivalences[equivalences[i] as usize];
        }
    }
    if no_labelling {
        0
    } else {
        *equivalences.iter().max().unwrap() as usize
    }
}
