use ndarray::{arr1, s, Array, Array1, ArrayBase, ArrayViewMut1, Axis, Data, Dimension};
use num_traits::ToPrimitive;

/// Multidimensional spline filter.
///
/// The multidimensional filter is implemented as a sequence of one-dimensional spline filters. The
/// input `data` will be processed in `f64` and returned as such.
///
/// * `data` - The input N-D data.
/// * `order` - The order of the spline.
///
/// **Panics** if `order` isn't in the range \[2, 5\].
pub fn spline_filter<S, A, D>(data: &ArrayBase<S, D>, order: usize) -> Array<f64, D>
where
    S: Data<Elem = A>,
    A: ToPrimitive,
    D: Dimension,
{
    let mut data = data.map(|v| v.to_f64().unwrap());
    if data.len() == 1 {
        return data;
    }

    let poles = get_filter_poles(order);
    let gain = filter_gain(&poles);
    for axis in 0..data.ndim() {
        _spline_filter1d(&mut data, Axis(axis), &poles, gain);
    }
    data
}

/// Calculate a 1-D spline filter along the given axis.
///
/// The lines of the array along the given axis are filtered by a spline filter. The input `data`
/// will be processed in `f64` and returned as such.
///
/// * `data` - The input N-D data.
/// * `order` - The order of the spline.
/// * `axis` - The axis along which the spline filter is applied.
///
/// **Panics** if `order` isn't in the range \[0, 5\].
pub fn spline_filter1d<S, A, D>(data: &ArrayBase<S, D>, order: usize, axis: Axis) -> Array<f64, D>
where
    S: Data<Elem = A>,
    A: ToPrimitive,
    D: Dimension,
{
    let mut data = data.map(|v| v.to_f64().unwrap());
    if order == 0 || order == 1 || data.len() == 1 {
        return data;
    }

    let poles = get_filter_poles(order);
    let gain = filter_gain(&poles);

    _spline_filter1d(&mut data, axis, &poles, gain);
    data
}

fn _spline_filter1d<D>(data: &mut Array<f64, D>, axis: Axis, poles: &Array1<f64>, gain: f64)
where
    D: Dimension,
{
    for mut line in data.lanes_mut(axis) {
        for val in line.iter_mut() {
            *val *= gain;
        }
        for &pole in poles {
            set_initial_causal_coefficient(&mut line, pole);
            for i in 1..line.len() {
                line[i] += pole * line[i - 1];
            }

            set_initial_anticausal_coefficient(&mut line, pole);
            for i in (0..line.len() - 1).rev() {
                line[i] = pole * (line[i + 1] - line[i]);
            }
        }
    }
}

fn get_filter_poles(order: usize) -> Array1<f64> {
    match order {
        1 => panic!("Can't use 'spline_filter' with order 1"),
        2 => arr1(&[8.0f64.sqrt() - 3.0]),
        3 => arr1(&[3.0f64.sqrt() - 2.0]),
        4 => arr1(&[
            (664.0 - 438976.0f64.sqrt()).sqrt() + 304.0f64.sqrt() - 19.0,
            (664.0 + 438976.0f64.sqrt()).sqrt() - 304.0f64.sqrt() - 19.0,
        ]),
        5 => arr1(&[
            (67.5 - 4436.25f64.sqrt()).sqrt() + 26.25f64.sqrt() - 6.5,
            (67.5 + 4436.25f64.sqrt()).sqrt() - 26.25f64.sqrt() - 6.5,
        ]),
        _ => panic!("Order must be between 2 and 5"),
    }
}

fn filter_gain(poles: &Array1<f64>) -> f64 {
    let mut gain = 1.0;
    for pole in poles {
        gain *= (1.0 - pole) * (1.0 - 1.0 / pole);
    }
    gain
}

fn set_initial_causal_coefficient(line: &mut ArrayViewMut1<f64>, pole: f64) {
    let mut z_i = pole;
    let tolerance: f64 = 1e-15;
    let last_coefficient = (tolerance.ln().ceil() / pole.abs().ln()) as usize;
    if last_coefficient < line.len() {
        let mut sum = line[0];
        // All values from line[1..last_coefficient]
        for val in line.iter().take(last_coefficient).skip(1) {
            sum += z_i * val;
            z_i *= pole;
        }
        line[0] = sum;
    } else {
        let inv_z = 1.0 / pole;
        let z_n_1 = pole.powi(line.len() as i32 - 1);
        let mut z_2n_2_i = z_n_1 * z_n_1 * inv_z;

        let mut sum = line[0] + (line[line.len() - 1] * z_n_1);
        for v in line.slice(s![1..line.len() - 1]) {
            sum += (z_i + z_2n_2_i) * v;
            z_i *= pole;
            z_2n_2_i *= inv_z;
        }
        line[0] = sum / (1.0 - z_n_1 * z_n_1);
    }
}

fn set_initial_anticausal_coefficient(line: &mut ArrayViewMut1<f64>, pole: f64) {
    let lm1 = line.len() - 1;
    line[lm1] = pole / (pole * pole - 1.0) * (pole * line[line.len() - 2] + line[lm1]);
}
