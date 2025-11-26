use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    UnsortedKeys,
    EmptyInput,
    InvalidEpsilon,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsortedKeys => write!(f, "keys must be sorted and non-decreasing"),
            Error::EmptyInput => write!(f, "input data cannot be empty"),
            Error::InvalidEpsilon => write!(f, "epsilon must be greater than 0"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
