use std::ops::{Add, Sub};

use ndarray::{Array, Array2, ArrayBase, Data, Ix3, Zip};
use num_traits::{FromPrimitive, Num, ToPrimitive};

use crate::{array_like, pad, round_ties_even, spline_filter, BorderMode, PadMode};

/// Shift an array.
///
/// The array is shifted using spline interpolation of the requested order. Points outside the
/// boundaries of the input are filled according to the given mode.
///
/// * `data` - A 3D array of the data to shift.
/// * `shift` - The shift along the axes.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `prefilter` - Determines if the input array is prefiltered with spline_filter before
///   interpolation. The default is `true`, which will create a temporary `f64` array of filtered
///   values if `order > 1`. If setting this to `false`, the output will be slightly blurred if
///   `order > 1`, unless the input is prefiltered.
pub fn shift<S, A>(
    data: &ArrayBase<S, Ix3>,
    shift: [f64; 3],
    mode: BorderMode<A>,
    prefilter: bool,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let dim = [data.dim().0, data.dim().1, data.dim().2];
    let shift = shift.map(|s| -s);
    let reslicer = ZoomShiftReslicer::new(dim, dim, [1.0, 1.0, 1.0], shift, mode);

    let out = array_like(&data, data.raw_dim(), A::zero());
    run_zoom_shift(data, mode, prefilter, &reslicer, out)
}

/// Zoom an array.
///
/// The array is zoomed using spline interpolation of the requested order.
///
/// * `data` - A 3D array of the data to zoom
/// * `zoom` - The zoom factor along the axes.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `prefilter` - Determines if the input array is prefiltered with spline_filter before
///   interpolation. The default is `true`, which will create a temporary `f64` array of filtered
///   values if `order > 1`. If setting this to `false`, the output will be slightly blurred if
///   `order > 1`, unless the input is prefiltered.
pub fn zoom<S, A>(
    data: &ArrayBase<S, Ix3>,
    zoom: [f64; 3],
    mode: BorderMode<A>,
    prefilter: bool,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let i_dim = [data.dim().0, data.dim().1, data.dim().2];
    let mut o_dim = data.raw_dim();
    for (ax, (&ax_len, zoom)) in data.shape().iter().zip(zoom.iter()).enumerate() {
        o_dim[ax] = round_ties_even(ax_len as f64 * zoom) as usize;
    }
    let o_dim = [o_dim[0], o_dim[1], o_dim[2]];

    let mut nom = data.raw_dim();
    let mut div = o_dim.clone();
    for ax in 0..data.ndim() {
        nom[ax] -= 1;
        div[ax] -= 1;
    }
    let zoom = [
        nom[0] as f64 / div[0] as f64,
        nom[1] as f64 / div[1] as f64,
        nom[2] as f64 / div[2] as f64,
    ];
    let reslicer = ZoomShiftReslicer::new(i_dim, o_dim, zoom, [0.0, 0.0, 0.0], mode);

    let out = array_like(&data, o_dim, A::zero());
    run_zoom_shift(data, mode, prefilter, &reslicer, out)
}

fn run_zoom_shift<S, A>(
    data: &ArrayBase<S, Ix3>,
    mode: BorderMode<A>,
    prefilter: bool,
    reslicer: &ZoomShiftReslicer,
    mut out: Array<A, Ix3>,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let order = 3;
    if prefilter && order > 1 {
        let data = match mode {
            BorderMode::Nearest => {
                let padded = pad(data, &[[12, 12]], PadMode::Edge);
                spline_filter(&padded, order, mode)
            }
            _ => spline_filter(data, order, mode),
        };
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(&data, idx)).unwrap();
        });
    } else {
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(&data, idx)).unwrap();
        });
    }
    out
}

/// Zoom shift transformation (only scaling and translation).
///
/// Hardcoded to use spline interpolation of order 3 and mirror mode.
struct ZoomShiftReslicer {
    offsets: [Vec<isize>; 3],
    edge_offsets: [Array2<isize>; 3],
    is_edge_case: [Vec<bool>; 3],
    splvals: [Array2<f64>; 3],
    zeros: [Vec<bool>; 3],
}

impl ZoomShiftReslicer {
    /// Build all necessary data to call `interpolate`.
    pub fn new<A>(
        idim: [usize; 3],
        odim: [usize; 3],
        zooms: [f64; 3],
        shifts: [f64; 3],
        mode: BorderMode<A>,
    ) -> ZoomShiftReslicer
    where
        A: Copy,
    {
        let order = 3;

        let offsets = [vec![0; odim[0]], vec![0; odim[1]], vec![0; odim[2]]];
        let edge_offsets = [
            Array2::zeros((odim[0], order + 1)),
            Array2::zeros((odim[1], order + 1)),
            Array2::zeros((odim[2], order + 1)),
        ];
        let is_edge_case = [vec![false; odim[0]], vec![false; odim[1]], vec![false; odim[2]]];
        let splvals = [
            Array2::zeros((odim[0], order + 1)),
            Array2::zeros((odim[1], order + 1)),
            Array2::zeros((odim[2], order + 1)),
        ];
        let zeros = [vec![false; odim[0]], vec![false; odim[1]], vec![false; odim[2]]];

        let mut reslicer =
            ZoomShiftReslicer { offsets, edge_offsets, is_edge_case, splvals, zeros };
        reslicer.build_arrays(idim, odim, zooms, shifts, order, mode);
        reslicer
    }

    fn build_arrays<A>(
        &mut self,
        idim: [usize; 3],
        odim: [usize; 3],
        zooms: [f64; 3],
        shifts: [f64; 3],
        order: usize,
        mode: BorderMode<A>,
    ) where
        A: Copy,
    {
        // Modes without an anlaytic prefilter or explicit prepadding use mirror extension
        let spline_mode = match mode {
            BorderMode::Constant(_) | BorderMode::Wrap => BorderMode::Mirror,
            _ => mode,
        };
        let nb_prepad = match mode {
            BorderMode::Nearest => 12,
            _ => 0,
        };
        let iorder = order as isize;
        let idim = [
            idim[0] as isize + 2 * nb_prepad,
            idim[1] as isize + 2 * nb_prepad,
            idim[2] as isize + 2 * nb_prepad,
        ];
        let nb_prepad = nb_prepad as f64;

        for axis in 0..3 {
            let splvals = &mut self.splvals[axis];
            let offsets = &mut self.offsets[axis];
            let edge_offsets = &mut self.edge_offsets[axis];
            let is_edge_case = &mut self.is_edge_case[axis];
            let zeros = &mut self.zeros[axis];
            let len = idim[axis] as f64;
            for from in 0..odim[axis] {
                let mut to = (from as f64 + shifts[axis]) * zooms[axis] + nb_prepad;
                match mode {
                    BorderMode::Nearest => {}
                    _ => to = map_coordinates(to, idim[axis] as f64, mode),
                };
                if to > -1.0 {
                    if order & 1 == 0 {
                        to += 0.5;
                    }

                    let x = to - to.floor();
                    let y = x;
                    let z = 1.0 - x;
                    splvals[(from, 0)] = z * z * z / 6.0;
                    splvals[(from, 1)] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
                    splvals[(from, 2)] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
                    splvals[(from, order)] =
                        1.0 - (splvals[(from, 0)] + splvals[(from, 1)] + splvals[(from, 2)]);

                    let start = to.floor() as isize - iorder / 2;
                    offsets[from] = start;
                    if start < 0 || start + iorder >= idim[axis] {
                        is_edge_case[from] = true;
                        for o in 0..=order {
                            let x = (start + o as isize) as f64;
                            let idx = map_coordinates(x, len, spline_mode) as isize;
                            edge_offsets[(from, o)] = idx - start;
                        }
                    }
                } else {
                    zeros[from] = true;
                }
            }
        }
    }

    /// Spline interpolation with up-to 8 neighbors of a point.
    pub fn interpolate<A, S>(&self, data: &ArrayBase<S, Ix3>, start: (usize, usize, usize)) -> f64
    where
        S: Data<Elem = A>,
        A: ToPrimitive + Add<Output = A> + Sub<Output = A> + Copy,
    {
        if self.zeros[0][start.0] || self.zeros[1][start.1] || self.zeros[2][start.2] {
            return 0.0;
        }

        // Linear interpolation use a 4x4x4 block. This is simple enough, but we must adjust this
        // block when the `start` is near the edges.
        let valid_index = |original_offset, is_edge, start, d: usize, v| {
            (original_offset + if is_edge { self.edge_offsets[d][(start, v)] } else { v as isize })
                as usize
        };

        let original_offset = self.offsets[0][start.0];
        let is_edge = self.is_edge_case[0][start.0];
        let xs = [
            valid_index(original_offset, is_edge, start.0, 0, 0),
            valid_index(original_offset, is_edge, start.0, 0, 1),
            valid_index(original_offset, is_edge, start.0, 0, 2),
            valid_index(original_offset, is_edge, start.0, 0, 3),
        ];

        let original_offset = self.offsets[1][start.1];
        let is_edge = self.is_edge_case[1][start.1];
        let ys = [
            valid_index(original_offset, is_edge, start.1, 1, 0),
            valid_index(original_offset, is_edge, start.1, 1, 1),
            valid_index(original_offset, is_edge, start.1, 1, 2),
            valid_index(original_offset, is_edge, start.1, 1, 3),
        ];

        let is_edge = self.is_edge_case[2][start.2];
        let original_offset = self.offsets[2][start.2];
        let zs = [
            valid_index(original_offset, is_edge, start.2, 2, 0),
            valid_index(original_offset, is_edge, start.2, 2, 1),
            valid_index(original_offset, is_edge, start.2, 2, 2),
            valid_index(original_offset, is_edge, start.2, 2, 3),
        ];

        let mut t = 0.0;
        for (z, &idx_z) in zs.iter().enumerate() {
            let spline_z = self.splvals[2][(start.2, z)];
            for (y, &idx_y) in ys.iter().enumerate() {
                let spline_yz = self.splvals[1][(start.1, y)] * spline_z;
                for (x, &idx_x) in xs.iter().enumerate() {
                    let spline_xyz = self.splvals[0][(start.0, x)] * spline_yz;
                    t += data[(idx_x, idx_y, idx_z)].to_f64().unwrap() * spline_xyz;
                }
            }
        }
        t
    }
}

fn map_coordinates<A>(mut idx: f64, len: f64, mode: BorderMode<A>) -> f64 {
    match mode {
        BorderMode::Constant(_) => {
            if idx < 0.0 || idx >= len {
                idx = -1.0;
            }
        }
        BorderMode::Nearest => {
            if idx < 0.0 {
                idx = 0.0;
            } else {
                idx = len - 1.0;
            }
        }
        BorderMode::Mirror => {
            let s2 = 2.0 * len - 2.0;
            if idx < 0.0 {
                idx = s2 * (-idx / s2).floor() + idx;
                idx = if idx <= 1.0 - len { idx + s2 } else { -idx };
            } else {
                idx -= s2 * (idx / s2).floor();
                if idx >= len {
                    idx = s2 - idx;
                }
            }
        }
        BorderMode::Reflect => {
            let s2 = 2.0 * len;
            if idx < 0.0 {
                if idx < -s2 {
                    idx = s2 * (-idx / s2).floor() + idx;
                }
                idx = if idx < -len { idx + s2 } else { -idx - 1.0 };
            } else {
                idx -= s2 * (idx / s2).floor();
                if idx >= len {
                    idx = s2 - idx - 1.0;
                }
            }
        }
        BorderMode::Wrap => {
            let s = len - 1.0;
            if idx < 0.0 {
                idx += s * ((-idx / s).floor() + 1.0);
            } else {
                idx -= s * (idx / s).floor();
            }
        }
    };
    idx
}
