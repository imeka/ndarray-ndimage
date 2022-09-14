use ndarray::{Array, ArrayBase, Axis, Data, Dimension, ScalarOperand};
use num_traits::{FromPrimitive, Signed};

use super::{con_corr::inner_correlate1d, symmetry::SymmetryStateCheck};
use crate::{array_like, BorderMode};

/// Calculate a Prewitt filter.
///
/// * `data` - The input N-D data.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
pub fn prewitt<S, A, D>(data: &ArrayBase<S, D>, axis: Axis, mode: BorderMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + Signed + ScalarOperand + FromPrimitive + PartialOrd,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    // TODO Warn the user to NOT call this function with unsigned data
    let mut weights = [-A::one(), A::zero(), A::one()];
    let mut output = array_like(&data, data.dim(), A::zero());
    inner_correlate1d(&data.view(), &weights, axis, mode, 0, &mut output);
    if data.ndim() == 1 {
        return output;
    }

    weights = [A::one(); 3];
    let indices: Vec<_> = (0..data.ndim()).filter(|&d| d != axis.index()).collect();
    let mut data = output.clone();
    for (i, d) in indices.into_iter().enumerate() {
        inner_correlate1d(&data, &weights, Axis(d), mode, 0, &mut output);
        if i != data.ndim() - 2 {
            std::mem::swap(&mut output, &mut data);
        }
    }
    output
}
