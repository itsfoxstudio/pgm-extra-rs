//! Support for bytes and string keys via prefix extraction.
//!
//! PGM-Index internally uses numeric keys for its linear models.
//! For bytes/strings, we extract an 8-byte prefix as a u64 for the model,
//! while keeping the full key for exact comparisons.

use core::cmp::Ordering;

/// A fixed-size prefix extracted from bytes/strings for use as a PGM key.
///
/// This extracts the first 8 bytes (big-endian) to create a u64,
/// that preserves lexicographic ordering for the prefix.
/// The full original data should be kept separately for exact lookups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct Prefix(pub u64);

impl Prefix {
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        let len = bytes.len().min(8);
        buf[..len].copy_from_slice(&bytes[..len]);
        Self(u64::from_be_bytes(buf))
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[cfg(feature = "std")]
impl std::str::FromStr for Prefix {
    type Err = core::convert::Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_bytes(s.as_bytes()))
    }
}

impl From<&[u8]> for Prefix {
    #[inline]
    fn from(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl PartialOrd for Prefix {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Prefix {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl crate::index::Key for Prefix {
    type Unsigned = u64;

    #[inline]
    fn to_unsigned(self) -> Self::Unsigned {
        self.0
    }

    #[inline]
    fn to_f64_fast(self) -> f64 {
        self.0 as f64
    }

    #[inline]
    fn to_i64_fast(self) -> i64 {
        self.0 as i64
    }
}

impl num_traits::Zero for Prefix {
    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl num_traits::Bounded for Prefix {
    #[inline]
    fn min_value() -> Self {
        Self(u64::MIN)
    }

    #[inline]
    fn max_value() -> Self {
        Self(u64::MAX)
    }
}

impl num_traits::ToPrimitive for Prefix {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Some(self.0)
    }
}

impl core::ops::Add for Prefix {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl core::ops::Sub for Prefix {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Static;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_bytes_prefix_ordering() {
        let a = Prefix::from_bytes(b"apple");
        let b = Prefix::from_bytes(b"banana");
        let c = Prefix::from_bytes(b"cherry");

        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_bytes_prefix_from_bytes() {
        let bytes = b"hello world";
        let prefix = Prefix::from_bytes(bytes);

        let expected = u64::from_be_bytes(*b"hello wo");
        assert_eq!(prefix.as_u64(), expected);
    }

    #[test]
    fn test_bytes_prefix_short() {
        let short = Prefix::from_bytes(b"hi");
        let mut expected = [0u8; 8];
        expected[0] = b'h';
        expected[1] = b'i';
        assert_eq!(short.as_u64(), u64::from_be_bytes(expected));
    }

    #[test]
    fn test_pgm_with_string_prefixes() {
        let strings: Vec<&str> = vec![
            "aardvark",
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "honeydew",
            "jackfruit",
        ];

        let prefixes: Vec<Prefix> = strings
            .iter()
            .map(|s| Prefix::from_bytes(s.as_bytes()))
            .collect();
        let index = Static::new(&prefixes, 2, 2).unwrap();

        let query = Prefix::from_bytes(b"cherry");
        let approx = index.search(&query);

        let found_idx = prefixes[approx.lo..approx.hi]
            .binary_search(&query)
            .map(|i| approx.lo + i);

        assert_eq!(found_idx, Ok(3));
    }

    #[test]
    fn test_bytes_prefix_equality() {
        let a = Prefix::from_bytes(b"test");
        let b = Prefix::from_bytes(b"test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_long_strings_same_prefix() {
        let a = Prefix::from_bytes(b"abcdefgh_suffix1");
        let b = Prefix::from_bytes(b"abcdefgh_suffix2");

        assert_eq!(a, b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_str() {
        let prefix: Prefix = "hello".parse().unwrap();
        assert_eq!(prefix, Prefix::from_bytes(b"hello"));
    }
}
