#[derive(Debug, PartialEq)]
pub enum SymmetryState {
    NonSymmetric,
    Symmetric,
    AntiSymmetric,
}

#[inline(always)]
pub fn symmetry_state<A: SymmetryStateCheck>(a: A) -> SymmetryState {
    a.symmetry_state()
}

pub trait SymmetryStateCheck {
    fn symmetry_state(self) -> SymmetryState;
}

macro_rules! impl_symmetry_state_for_unsigned {
    ( $( $self:ty ),* ) => {
        $(
            impl<'a> SymmetryStateCheck for &'a [$self] {
                fn symmetry_state(self) -> SymmetryState {
                    // Test for symmetry
                    let mut state = SymmetryState::NonSymmetric;
                    let filter_size = self.len();
                    let half = filter_size / 2;
                    if filter_size & 1 > 0 {
                        state = SymmetryState::Symmetric;
                        for ii in 1..=half {
                            if self[ii + half] != self[half - ii] {
                                state = SymmetryState::NonSymmetric;
                                break;
                            }
                        }
                    }
                    state
                }
            }
        )*
    }
}

macro_rules! impl_symmetry_state_for_signed {
    ( $( $self:ty ),* ) => {
        $(
            impl<'a> SymmetryStateCheck for &'a [$self] {
                fn symmetry_state(self) -> SymmetryState {
                    // Test for symmetry or anti-symmetry
                    let mut state = SymmetryState::NonSymmetric;
                    let filter_size = self.len();
                    let half = filter_size / 2;
                    if filter_size & 1 > 0 {
                        state = SymmetryState::Symmetric;
                        for ii in 1..=half {
                            if self[ii + half] != self[half - ii] {
                                state = SymmetryState::NonSymmetric;
                                break;
                            }
                        }
                        if state == SymmetryState::NonSymmetric {
                            state = SymmetryState::AntiSymmetric;
                            for ii in 1..=half {
                                if self[ii + half] != -self[half - ii] {
                                    state = SymmetryState::NonSymmetric;
                                    break;
                                }
                            }
                        }
                    }
                    state
                }
            }
        )*
    }
}

macro_rules! impl_symmetry_state_for_fp {
    ( $( $self:ty ),* ) => {
        $(
            impl<'a> SymmetryStateCheck for &'a [$self] {
                fn symmetry_state(self) -> SymmetryState {
                    // Test for symmetry or anti-symmetry
                    let mut state = SymmetryState::NonSymmetric;
                    let filter_size = self.len();
                    let half = filter_size / 2;
                    if filter_size & 1 > 0 {
                        state = SymmetryState::Symmetric;
                        for ii in 1..=half {
                            if (self[ii + half] - self[half - ii]).abs() > <$self>::EPSILON {
                                state = SymmetryState::NonSymmetric;
                                break;
                            }
                        }
                        if state == SymmetryState::NonSymmetric {
                            state = SymmetryState::AntiSymmetric;
                            for ii in 1..=half {
                                if (self[ii + half] + self[half - ii]).abs() > <$self>::EPSILON {
                                    state = SymmetryState::NonSymmetric;
                                    break;
                                }
                            }
                        }
                    }
                    state
                }
            }
        )*
    }
}

impl_symmetry_state_for_unsigned!(u8, u16, u32, u64);
impl_symmetry_state_for_signed!(i8, i16, i32, i64);
impl_symmetry_state_for_fp!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // Results verified with SciPy. (v1.9.0)
    fn test_symmetry_state() {
        assert_eq!(symmetry_state(&[1.0, 1.0][..]), SymmetryState::NonSymmetric);
        assert_eq!(symmetry_state(&[1.0, 1.0, 2.0][..]), SymmetryState::NonSymmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, 0.0][..]), SymmetryState::NonSymmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, 1.000000001][..]), SymmetryState::NonSymmetric);

        assert_eq!(symmetry_state(&[-1.0, 2.0, -1.0][..]), SymmetryState::Symmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, 1.0][..]), SymmetryState::Symmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, 1.0, 2.0, 1.0][..]), SymmetryState::Symmetric);

        assert_eq!(symmetry_state(&[-1.0, 2.0, 1.0][..]), SymmetryState::AntiSymmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, -1.0][..]), SymmetryState::AntiSymmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, 1.0, -2.0, -1.0][..]), SymmetryState::AntiSymmetric);
        assert_eq!(symmetry_state(&[1.0, 2.0, -1.0, -2.0, -1.0][..]), SymmetryState::AntiSymmetric);
    }
}
