use ndarray::{s, ArrayBase, Data, Ix3, Zip};

use crate::{array_like, dim_minus, Mask};

/// Binary median filter.
///
/// A 3x3 structuring element (`Kernel3d::Full`) is used except on the borders, where a smaller
/// structuring element is used.
pub fn median_filter<S>(mask: &ArrayBase<S, Ix3>) -> Mask
where
    S: Data<Elem = bool>,
{
    let range = |i, max| {
        if i == 0 {
            0..2
        } else if i == max {
            max - 1..max + 1
        } else {
            i - 1..i + 2
        }
    };

    let (width, height, depth) = dim_minus(mask, 1);
    let ranges_x: Vec<_> = (0..=width).map(|x| range(x, width)).collect();
    let ranges_y: Vec<_> = (0..=height).map(|y| range(y, height)).collect();
    let ranges_z: Vec<_> = (0..=depth).map(|z| range(z, depth)).collect();

    // `from_shape_fn` is strangely much slower here
    let mut new_mask = array_like(mask, mask.dim(), false);
    Zip::indexed(&mut new_mask).for_each(|idx, new_mask| {
        let r_x = &ranges_x[idx.0];
        let r_y = &ranges_y[idx.1];
        let r_z = &ranges_z[idx.2];

        // For binary images, the median filter can be replaced with a simple majority vote
        let nb_required = ((r_x.len() * r_y.len() * r_z.len()) as u8 - 1) / 2;
        *new_mask = mask
            .slice(s![r_x.clone(), r_y.clone(), r_z.clone()])
            .iter()
            .fold(0, |acc, &m| acc + m as u8)
            > nb_required;
    });
    new_mask
}
