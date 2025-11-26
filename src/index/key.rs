use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{Add, Sub};
use num_traits::{Bounded, NumCast, ToPrimitive, Zero};

use crate::util::bytes::Prefix;

pub trait Key:
    Copy
    + Clone
    + Ord
    + Default
    + Send
    + Sync
    + ToPrimitive
    + Bounded
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
{
    type Unsigned: Copy + Ord + ToPrimitive + NumCast;

    fn to_unsigned(self) -> Self::Unsigned;

    fn to_f64_fast(self) -> f64;

    fn to_i64_fast(self) -> i64;
}

macro_rules! impl_key_unsigned {
    ($($t:ty),*) => {
        $(
            impl Key for $t {
                type Unsigned = $t;

                #[inline]
                fn to_unsigned(self) -> Self::Unsigned {
                    self
                }

                #[inline]
                fn to_f64_fast(self) -> f64 {
                    self as f64
                }

                #[inline]
                fn to_i64_fast(self) -> i64 {
                    self as i64
                }
            }
        )*
    };
}

macro_rules! impl_key_signed {
    ($(($signed:ty, $unsigned:ty)),*) => {
        $(
            impl Key for $signed {
                type Unsigned = $unsigned;

                #[inline]
                fn to_unsigned(self) -> Self::Unsigned {
                    const OFFSET: $unsigned = <$signed>::MIN as $unsigned;
                    (self as $unsigned).wrapping_sub(OFFSET)
                }

                #[inline]
                fn to_f64_fast(self) -> f64 {
                    self as f64
                }

                #[inline]
                fn to_i64_fast(self) -> i64 {
                    self as i64
                }
            }
        )*
    };
}

impl_key_unsigned!(u8, u16, u32, u64, u128, usize);
impl_key_signed!(
    (i8, u8),
    (i16, u16),
    (i32, u32),
    (i64, u64),
    (i128, u128),
    (isize, usize)
);

/// Trait for types that can be indexed by a PGM-Index.
///
/// The PGM-Index requires numeric keys for its piecewise linear models.
/// This trait defines how a value is converted to such a key.
///
/// - For numeric types (u64, i32, etc.), the value IS the key.
/// - For string/bytes types, an 8-byte prefix is extracted as the key.
///
/// # Implementing for custom types
///
/// ```
/// use pgm_extra::index::key::Indexable;
///
/// struct UserId(u64);
///
/// impl Indexable for UserId {
///     type Key = u64;
///     fn index_key(&self) -> u64 { self.0 }
/// }
/// ```
pub trait Indexable {
    /// The numeric key type used for indexing.
    type Key: Key;

    /// Extract the index key from this value.
    fn index_key(&self) -> Self::Key;
}

macro_rules! impl_indexable_numeric {
    ($($t:ty),*) => {
        $(
            impl Indexable for $t {
                type Key = $t;

                #[inline]
                fn index_key(&self) -> Self::Key {
                    *self
                }
            }
        )*
    };
}

impl_indexable_numeric!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl Indexable for String {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        Prefix::from_bytes(self.as_bytes())
    }
}

impl Indexable for str {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        Prefix::from_bytes(self.as_bytes())
    }
}

impl Indexable for [u8] {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        Prefix::from_bytes(self)
    }
}

impl Indexable for Vec<u8> {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        Prefix::from_bytes(self)
    }
}

impl<const N: usize> Indexable for [u8; N] {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        Prefix::from_bytes(self)
    }
}

// References delegate to the underlying type
impl<T: Indexable + ?Sized> Indexable for &T {
    type Key = T::Key;

    #[inline]
    fn index_key(&self) -> Self::Key {
        (*self).index_key()
    }
}

impl<T: Indexable + ?Sized> Indexable for &mut T {
    type Key = T::Key;

    #[inline]
    fn index_key(&self) -> Self::Key {
        (**self).index_key()
    }
}

impl<T: Indexable + ?Sized> Indexable for alloc::boxed::Box<T> {
    type Key = T::Key;

    #[inline]
    fn index_key(&self) -> Self::Key {
        (**self).index_key()
    }
}

impl<T: Indexable + ?Sized> Indexable for alloc::rc::Rc<T> {
    type Key = T::Key;

    #[inline]
    fn index_key(&self) -> Self::Key {
        (**self).index_key()
    }
}

impl<T: Indexable + ?Sized> Indexable for alloc::sync::Arc<T> {
    type Key = T::Key;

    #[inline]
    fn index_key(&self) -> Self::Key {
        (**self).index_key()
    }
}

// BytesPrefix indexes itself
impl Indexable for Prefix {
    type Key = Prefix;

    #[inline]
    fn index_key(&self) -> Prefix {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_unsigned_monotonic() {
        let vals: Vec<u64> = vec![0, 1, 100, 1000, u64::MAX];
        for w in vals.windows(2) {
            assert!(w[0].to_unsigned() < w[1].to_unsigned());
        }
    }

    #[test]
    fn test_signed_monotonic() {
        let vals: Vec<i64> = vec![i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX];
        for w in vals.windows(2) {
            assert!(
                w[0].to_unsigned() < w[1].to_unsigned(),
                "{} -> {} should be < {} -> {}",
                w[0],
                w[0].to_unsigned(),
                w[1],
                w[1].to_unsigned()
            );
        }
    }

    #[test]
    fn test_numeric_indexable() {
        assert_eq!(42u64.index_key(), 42u64);
        assert_eq!((-10i32).index_key(), -10i32);
    }

    #[test]
    fn test_string_indexable() {
        let s = String::from("hello");
        let key = s.index_key();
        assert_eq!(key, Prefix::from_bytes(b"hello"));
    }

    #[test]
    fn test_str_indexable() {
        let s = "world";
        let key = s.index_key();
        assert_eq!(key, Prefix::from_bytes(b"world"));
    }

    #[test]
    fn test_bytes_indexable() {
        let bytes: Vec<u8> = vec![1, 2, 3, 4, 5];
        let key = bytes.index_key();
        assert_eq!(key, Prefix::from_bytes(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_reference_indexable() {
        let val = 100u64;
        let ref_val = &val;
        assert_eq!(ref_val.index_key(), 100u64);

        let s = "test";
        assert_eq!(s.index_key(), Prefix::from_bytes(b"test"));
    }
}