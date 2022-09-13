use ndarray::{Array, ArrayBase, Axis, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use super::{con_corr::inner_correlate1d, symmetry::SymmetryStateCheck};
use crate::BorderMode;

/// Calculate a Prewitt filter.
///
/// * `data` - The input N-D data.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
pub fn sobel<S, A, D>(data: &ArrayBase<S, D>, axis: Axis, mode: BorderMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + ScalarOperand + FromPrimitive,
    for<'a> &'a [A]: SymmetryStateCheck,
    D: Dimension,
{
    // TODO Warn the user to NOT call this function with unsigned data
    let mut weights = [-A::one(), A::zero(), A::one()];
    let mut output = inner_correlate1d(&data.view(), &weights, axis, mode, 0);
    if data.ndim() == 1 {
        return output;
    }

    weights = [A::one(), A::from(2).unwrap(), A::one()];
    for d in 0..data.ndim() {
        if d != axis.index() {
            let axis = Axis(d);
            output = inner_correlate1d(&output.view(), &weights, axis, mode, 0);
        }
    }
    output
}
