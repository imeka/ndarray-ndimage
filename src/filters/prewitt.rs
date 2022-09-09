use ndarray::{arr1, Array, Array1, ArrayBase, Axis, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::{correlate1d, BorderMode};

/// Calculate a Prewitt filter.
///
/// * `data` - The input N-D data.
/// * `axis` - The axis of input along which to calculate.
/// * `mode` - Method that will be used to select the padded values. See the
///   [`CorrelateMode`](crate::CorrelateMode) enum for more information.
pub fn prewitt<S, A, D>(data: &ArrayBase<S, D>, axis: Axis, mode: BorderMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Float + ScalarOperand + FromPrimitive,
    D: Dimension,
{
    // TODO Warn the user to NOT call this function with unsigned data
    let weights = arr1(&[-A::one(), A::zero(), A::one()]);
    let mut output = correlate1d(&data.view(), &weights.view(), axis, mode, 0);
    if data.ndim() == 1 {
        return output;
    }

    let weights = Array1::ones(3);
    for d in 0..data.ndim() {
        if d != axis.index() {
            let axis = Axis(d);
            output = correlate1d(&output.view(), &weights.view(), axis, mode, 0);
        }
    }
    output
}
