use ndarray::{s, Array3, ArrayBase, Axis, Data, Ix3, Zip};
use num_traits::{Bounded, FromPrimitive, NumAssignOps, ToPrimitive, Unsigned};

use crate::Mask;

pub trait LabelType:
    Copy + FromPrimitive + ToPrimitive + Ord + Unsigned + NumAssignOps + Bounded
{
    fn background() -> Self;
    fn foreground() -> Self;
}

impl<T> LabelType for T
where
    T: Copy + FromPrimitive + ToPrimitive + Ord + Unsigned + NumAssignOps + Bounded,
{
    fn background() -> Self {
        T::zero()
    }
    fn foreground() -> Self {
        T::one()
    }
}

/// Calculates the histogram of a label image.
///
/// * `labels` - 3D labels image, returned by the `label` function.
/// * `nb_features` - Number of unique labels, returned by the `label` function.
pub fn label_histogram<S>(labels: &ArrayBase<S, Ix3>, nb_features: usize) -> Vec<usize>
where
    S: Data,
    S::Elem: LabelType,
{
    let mut count = vec![0; nb_features + 1];
    Zip::from(labels).for_each(|&l| {
        count[l.to_usize().unwrap()] += 1;
    });
    count
}

/// Returns the most frequent label and its index.
///
/// Ignores the background label. A blank label image will return None.
///
/// * `labels` - 3D labels image, returned by the `label` function.
/// * `nb_features` - Number of unique labels, returned by the `label` function.
pub fn most_frequent_label<S>(
    labels: &ArrayBase<S, Ix3>,
    nb_features: usize,
) -> Option<(S::Elem, usize)>
where
    S: Data,
    S::Elem: LabelType,
{
    let hist = label_histogram(labels, nb_features);
    let (max, max_index) =
        hist[1..].iter().enumerate().fold((0, 0), |acc, (i, &nb)| acc.max((nb, i)));
    (max > 0).then(|| (S::Elem::from_usize(max_index + 1).unwrap(), max))
}

/// Returns a new mask, containing the biggest zone of `mask`.
///
/// * `mask` - Binary image to be labeled and studied.
/// * `structure` - Structuring element used for the labeling. Must be 3x3x3 (e.g. the result
///   of [`Kernel3d::generate`](crate::Kernel3d::generate)) and centrosymmetric. The center must be `true`.
///
/// The labeling is done using `u16`, this may be too small when `mask` has more than [`u16::MAX`] elements.
pub fn largest_connected_components<S>(
    mask: &ArrayBase<S, Ix3>,
    structure: &ArrayBase<S, Ix3>,
) -> Option<Mask>
where
    S: Data<Elem = bool>,
{
    let (labels, nb_features) = label::<_, u16>(mask, structure);
    let (right_label, _) = most_frequent_label(&labels, nb_features)?;
    Some(labels.mapv(|l| l == right_label))
}

/// Labels features of 3D binary images.
///
/// Returns the labels and the number of features.
///
/// * `mask` - Binary image to be labeled. `false` values are considered the background.
/// * `structure` - Structuring element used for the labeling. Must be 3x3x3 (e.g. the result
///   of [`Kernel3d::generate`](crate::Kernel3d::generate)) and centrosymmetric. The center must be `true`.
///
/// The return type of `label` can be specified using turbofish syntax:
///
/// ```
/// // Will use `u16` as the label type
/// ndarray_ndimage::label::<_, u16>(
///     &ndarray::Array3::from_elem((100, 100, 100), true),
///     &ndarray_ndimage::Kernel3d::Star.generate()
/// );
/// ```
///
/// As a rough rule of thumb, the maximum value of the label type should be larger than `data.len()`.
/// This is the worst case, the exact bound will depend on the kernel used. If the label type overflows
/// while assigning labels, a panic will occur.
pub fn label<S, O>(data: &ArrayBase<S, Ix3>, structure: &ArrayBase<S, Ix3>) -> (Array3<O>, usize)
where
    S: Data<Elem = bool>,
    O: LabelType,
{
    assert!(structure.shape() == &[3, 3, 3], "`structure` must be size 3 in all dimensions");
    assert!(structure == structure.slice(s![..;-1, ..;-1, ..;-1]), "`structure is not symmetric");

    let len = data.dim().2;
    let mut line_buffer = vec![O::background(); len + 2];
    let mut neighbors = vec![O::background(); len + 2];

    let mut next_region = O::foreground() + O::one();
    let mut equivalences: Vec<_> =
        (0..next_region.to_usize().unwrap()).map(|x| O::from_usize(x).unwrap()).collect();

    // We only handle 3D data for now, but this algo can handle N-dimensional data.
    // https://github.com/scipy/scipy/blob/v0.16.1/scipy/ndimage/src/_ni_label.pyx
    // N-D: Use a loop in `is_valid` and change the `labels` indexing (might be hard in Rust)

    let nb_neighbors = structure.len() / (3 * 2);
    let kernel_data: Vec<([bool; 3], [isize; 2])> = structure
        .lanes(Axis(2))
        .into_iter()
        .zip(0isize..)
        // Only consider lanes before the center
        .take(nb_neighbors)
        // Filter out kernel lanes with no `true` elements (since that are no-ops)
        .filter(|(lane, _)| lane.iter().any(|x| *x))
        .map(|(lane, i)| {
            let kernel = [lane[0], lane[1], lane[2]];
            // Convert i into coordinates
            let y = i / 3;
            let x = i - y * 3;
            (kernel, [y, x])
        })
        .collect();

    let use_previous = structure[(1, 1, 0)];
    let width = data.dim().0 as isize;
    let height = data.dim().1 as isize;
    let mut labels = Array3::from_elem(data.dim(), O::background());
    Zip::indexed(data.lanes(Axis(2))).for_each(|idx, data| {
        for (&v, b) in data.iter().zip(&mut line_buffer[1..]) {
            *b = if !v { O::background() } else { O::foreground() }
        }

        let mut needs_self_labeling = true;
        for (i, (kernel, coordinates)) in kernel_data.iter().enumerate() {
            // Check that the neighbor line is in bounds
            if let Some((x, y)) = is_valid(&[idx.0, idx.1], coordinates, &[width, height]) {
                // Copy the interesting neighbor labels to `neighbors`
                for (&v, b) in labels.slice(s![x, y, ..]).iter().zip(&mut neighbors[1..]) {
                    *b = v;
                }

                let label_unlabeled = i == kernel_data.len() - 1;
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
    labels.mapv_inplace(|l| equivalences[l.to_usize().unwrap()]);

    (labels, nb_features)
}

fn is_valid(idx: &[usize; 2], coords: &[isize; 2], dims: &[isize; 2]) -> Option<(usize, usize)> {
    let valid = |i, c, d| -> Option<usize> {
        let a = i as isize + (c - 1);
        if a >= 0 && a < d {
            Some(a as usize)
        } else {
            None
        }
    };
    // Returns `Some((x, y))` only if both calls succeeded
    valid(idx[0], coords[0], dims[0])
        .and_then(|x| valid(idx[1], coords[1], dims[1]).and_then(|y| Some((x, y))))
}

fn label_line_with_neighbor<O>(
    line: &mut [O],
    neighbors: &[O],
    equivalences: &mut Vec<O>,
    kernel: [bool; 3],
    use_previous: bool,
    label_unlabeled: bool,
    mut next_region: O,
) -> O
where
    O: LabelType,
{
    let mut previous = line[0];
    for (n, l) in neighbors.windows(3).zip(&mut line[1..]) {
        if *l != O::background() {
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
                if *l == O::foreground() {
                    *l = next_region;
                    equivalences.push(next_region);
                    assert!(next_region < O::max_value(), "Overflow when assigning label");
                    next_region += O::one();
                }
            }
        }
        previous = *l;
    }
    next_region
}

/// Take the label of a neighbor, or mark them for merging
fn take_label_or_merge<O>(current: O, neighbor: O, equivalences: &mut [O]) -> O
where
    O: LabelType,
{
    if neighbor == O::background() {
        current
    } else if current == O::foreground() {
        neighbor // neighbor is not background
    } else if current != neighbor {
        mark_for_merge(neighbor, current, equivalences)
    } else {
        current
    }
}

/// Mark two labels to be merged
fn mark_for_merge<O>(mut a: O, mut b: O, equivalences: &mut [O]) -> O
where
    O: LabelType,
{
    // Find smallest root for each of a and b
    let original_a = a;
    while a != equivalences[a.to_usize().unwrap()] {
        a = equivalences[a.to_usize().unwrap()];
    }
    let original_b = b;
    while b != equivalences[b.to_usize().unwrap()] {
        b = equivalences[b.to_usize().unwrap()];
    }
    let lowest_label = a.min(b);

    // Merge roots
    equivalences[a.to_usize().unwrap()] = lowest_label;
    equivalences[b.to_usize().unwrap()] = lowest_label;

    // Merge every step to minlabel
    a = original_a;
    while a != lowest_label {
        let a_copy = a;
        a = equivalences[a.to_usize().unwrap()];
        equivalences[a_copy.to_usize().unwrap()] = lowest_label;
    }
    b = original_b;
    while b != lowest_label {
        let b_copy = b;
        b = equivalences[b.to_usize().unwrap()];
        equivalences[b_copy.to_usize().unwrap()] = lowest_label;
    }

    lowest_label
}

/// Compact the equivalences vector
fn compact_equivalences<O>(equivalences: &mut [O], next_region: O) -> usize
where
    O: LabelType,
{
    let no_labelling = next_region == O::from_usize(2).unwrap();
    let mut dest_label = if no_labelling { 0 } else { 1 };
    for i in 2..next_region.to_usize().unwrap() {
        if equivalences[i] == O::from_usize(i).unwrap() {
            equivalences[i] = O::from_usize(dest_label).unwrap();
            dest_label = dest_label + 1;
        } else {
            // We've compacted every label below this, and equivalences has an invariant that it
            // always points downward. Therefore, we can fetch the final label by two steps of
            // indirection.
            equivalences[i] = equivalences[equivalences[i].to_usize().unwrap()];
        }
    }
    if no_labelling {
        0
    } else {
        equivalences.iter().max().unwrap().to_usize().unwrap()
    }
}
