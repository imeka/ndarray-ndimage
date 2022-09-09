use crate::PadMode;

pub mod con_corr;
pub mod gaussian;
pub mod median;
pub mod min_max;

// TODO We might want to offer all NumPy mode (use PadMode instead)
/// Method that will be used to determines how the input array is extended beyond its boundaries.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BorderMode<T> {
    /// The input is extended by filling all values beyond the edge with the same constant value,
    ///
    /// `[1, 2, 3] -> [T, T, 1, 2, 3, T, T]`
    Constant(T),

    /// The input is extended by replicating the last pixel.
    ///
    /// `[1, 2, 3] -> [1, 1, 1, 2, 3, 3, 3]`
    Nearest,

    /// The input is extended by reflecting about the center of the last pixel.
    ///
    /// `[1, 2, 3] -> [3, 2, 1, 2, 3, 2, 1]`
    Mirror,

    /// The input is extended by reflecting about the edge of the last pixel.
    ///
    /// `[1, 2, 3] -> [2, 1, 1, 2, 3, 3, 2]`
    Reflect,

    /// The input is extended by wrapping around to the opposite edge.
    ///
    /// `[1, 2, 3] -> [2, 3, 1, 2, 3, 1, 2]`
    Wrap,
}

impl<T: Copy> BorderMode<T> {
    fn to_pad_mode(&self) -> PadMode<T> {
        match *self {
            BorderMode::Constant(t) => PadMode::Constant(t),
            BorderMode::Nearest => PadMode::Edge,
            BorderMode::Mirror => PadMode::Reflect,
            BorderMode::Reflect => PadMode::Symmetric,
            BorderMode::Wrap => PadMode::Wrap,
        }
    }
}

fn origin_check(len: usize, origin: isize, left: usize, right: usize) -> [usize; 2] {
    let len = len as isize;
    assert!(
        origin >= -len / 2 && origin <= (len - 1) / 2,
        "origin must satisfy: -(len(weights) / 2) <= origin <= (len(weights) - 1) / 2"
    );
    [(left as isize + origin) as usize, (right as isize - origin) as usize]
}
