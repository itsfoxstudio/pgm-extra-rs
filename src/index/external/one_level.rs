//! Single-level PGM-Index.
//!
//! A simpler variant with only one level of linear models.
//! Good for smaller datasets or when minimal memory is preferred.

use alloc::vec::Vec;
use core::ops::RangeBounds;

use crate::error::Error;

use crate::index::key::Indexable;
use crate::index::model::build_segments;
use crate::index::segment::Segment;

use crate::util::ApproxPos;
use crate::util::range::range_to_indices;
use crate::util::search::{pgm_add_eps, pgm_sub_eps};

/// A single-level PGM-Index.
///
/// This is a simpler variant of the multi-level index that uses only
/// one level of segments. It's suitable for smaller datasets or when
/// you want to minimize memory usage at the cost of slightly longer
/// segment search time.
///
/// # Example
///
/// ```
/// use pgm_extra::index::external::OneLevel;
///
/// let keys: Vec<u64> = (0..1000).collect();
/// let index = OneLevel::new(&keys, 8).unwrap();
///
/// assert!(index.contains(&keys, &500));
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T::Key: serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct OneLevel<T: Indexable> {
    epsilon: usize,
    len: usize,
    segments: Vec<Segment<T::Key>>,
}

impl<T: Indexable> OneLevel<T>
where
    T::Key: Ord,
{
    /// Build a new single-level PGM-Index from sorted data.
    pub fn new(data: &[T], epsilon: usize) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }
        if epsilon == 0 {
            return Err(Error::InvalidEpsilon);
        }

        debug_assert!(
            data.windows(2)
                .all(|w| w[0].index_key() <= w[1].index_key()),
            "data must be sorted by index key"
        );

        let keys: Vec<T::Key> = data.iter().map(|v| v.index_key()).collect();
        let segments = build_segments(&keys, epsilon);

        Ok(Self {
            segments,
            epsilon,
            len: keys.len(),
        })
    }

    #[inline]
    fn find_segment(&self, key: &T::Key) -> usize {
        let pos = self
            .segments
            .partition_point(|s| s.key <= *key)
            .saturating_sub(1);
        pos.min(self.segments.len().saturating_sub(1))
    }

    /// Get an approximate position for the given value.
    #[inline]
    pub fn search(&self, value: &T) -> ApproxPos {
        let key = value.index_key();
        self.search_by_key(&key)
    }

    /// Get an approximate position for the given key.
    #[inline]
    pub fn search_by_key(&self, key: &T::Key) -> ApproxPos {
        let seg_idx = self.find_segment(key);
        let segment = &self.segments[seg_idx];

        let pos = segment.predict(*key).min(self.len.saturating_sub(1));
        let lo = pgm_sub_eps(pos, self.epsilon);
        let hi = pgm_add_eps(pos, self.epsilon, self.len);

        ApproxPos::new(pos, lo, hi)
    }

    /// Find the first position where `data[pos] >= value`.
    #[inline]
    pub fn lower_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        let key = value.index_key();
        let approx = self.search_by_key(&key);
        let len = data.len();

        if len == 0 {
            return 0;
        }

        let pos = approx.pos.min(len - 1);
        let k = &data[pos];

        if *k == *value {
            let mut i = pos;
            while i > 0 && data[i - 1] == *value {
                i -= 1;
            }
            return i;
        }

        if *k < *value {
            if pos + 1 < len && data[pos + 1] >= *value {
                return pos + 1;
            }
        } else if pos > 0 && data[pos - 1] < *value {
            return pos;
        }

        // For values not found directly, fall back to full binary search in
        // the predicted range. Extend bounds slightly to handle edge cases.
        let lo = approx.lo.saturating_sub(self.epsilon);
        let hi = (approx.hi + self.epsilon).min(len);
        let slice = &data[lo..hi];
        lo + slice.partition_point(|x| x < value)
    }

    /// Find the first position where `data[pos] > value`.
    #[inline]
    pub fn upper_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        let idx = self.lower_bound(data, value);
        let mut i = idx;
        while i < data.len() && data[i] == *value {
            i += 1;
        }
        i
    }

    /// Check if the value exists in the data.
    #[inline]
    pub fn contains(&self, data: &[T], value: &T) -> bool
    where
        T: Ord,
    {
        let idx = self.lower_bound(data, value);
        idx < data.len() && data[idx] == *value
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    #[inline]
    pub fn epsilon(&self) -> usize {
        self.epsilon
    }

    pub fn size_in_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.segments.capacity() * core::mem::size_of::<Segment<T::Key>>()
    }

    /// Returns the (start, end) indices for iterating over data in the given range.
    #[inline]
    pub fn range_indices<R>(&self, data: &[T], range: R) -> (usize, usize)
    where
        T: Ord,
        R: RangeBounds<T>,
    {
        range_to_indices(
            range,
            data.len(),
            |v| self.lower_bound(data, v),
            |v| self.upper_bound(data, v),
        )
    }

    /// Returns an iterator over data in the given range.
    #[inline]
    pub fn range<'a, R>(&self, data: &'a [T], range: R) -> impl DoubleEndedIterator<Item = &'a T>
    where
        T: Ord,
        R: RangeBounds<T>,
    {
        let (start, end) = self.range_indices(data, range);
        data[start..end].iter()
    }
}

impl<T: Indexable> crate::index::External<T> for OneLevel<T>
where
    T::Key: Ord,
{
    #[inline]
    fn search(&self, value: &T) -> ApproxPos {
        self.search(value)
    }

    #[inline]
    fn lower_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        self.lower_bound(data, value)
    }

    #[inline]
    fn upper_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        self.upper_bound(data, value)
    }

    #[inline]
    fn contains(&self, data: &[T], value: &T) -> bool
    where
        T: Ord,
    {
        self.contains(data, value)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn segments_count(&self) -> usize {
        self.segments_count()
    }

    #[inline]
    fn epsilon(&self) -> usize {
        self.epsilon()
    }

    #[inline]
    fn size_in_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_one_level_basic() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = OneLevel::new(&keys, 8).unwrap();

        assert_eq!(index.len(), 1000);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_one_level_search() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = OneLevel::new(&keys, 8).unwrap();

        for &key in &[0u64, 100, 500, 999] {
            let idx = index.lower_bound(&keys, &key);
            assert_eq!(idx, key as usize);
        }
    }

    #[test]
    fn test_one_level_contains() {
        let keys: Vec<u64> = (0..100).map(|i| i * 2).collect();
        let index = OneLevel::new(&keys, 8).unwrap();

        assert!(index.contains(&keys, &0));
        assert!(index.contains(&keys, &50));
        assert!(!index.contains(&keys, &1));
        assert!(!index.contains(&keys, &51));
    }

    #[test]
    fn test_one_level_empty_error() {
        let keys: Vec<u64> = vec![];
        let result = OneLevel::new(&keys, 8);
        assert!(matches!(result, Err(Error::EmptyInput)));
    }
}
