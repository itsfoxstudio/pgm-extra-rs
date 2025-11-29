//! Multi-level recursive PGM-Index.
//!
//! This is the primary external-keys index with recursive segment levels
//! for optimal query performance on large datasets.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::RangeBounds;

use crate::error::Error;

use crate::index::Segment;
use crate::index::key::Indexable;
use crate::index::model::build_segments;

use crate::util::ApproxPos;
use crate::util::range::range_to_indices;
use crate::util::search::{pgm_add_eps, pgm_sub_eps};

const LINEAR_SEARCH_THRESHOLD_SEGMENTS: usize = 32;

/// A multi-level recursive PGM-Index.
///
/// This index builds multiple levels of linear models for efficient lookups.
/// It does not own the data; the keys must be stored separately and passed
/// to query methods.
///
/// # Type Parameters
///
/// - `T`: The value type that implements [`Indexable`]. The index internally
///   stores segments of `T::Key` for the linear models.
///
/// # Example
///
/// ```
/// use pgm_extra::index::external::Static;
///
/// let keys: Vec<u64> = (0..10000).collect();
/// let index = Static::new(&keys, 64, 4).unwrap();
///
/// assert!(index.contains(&keys, &5000));
/// assert_eq!(index.lower_bound(&keys, &5000), 5000);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T::Key: serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct Static<T: Indexable> {
    epsilon: usize,
    epsilon_recursive: usize,
    len: usize,
    levels_offsets: Vec<usize>,
    segments: Vec<Segment<T::Key>>,
}

impl<T: Indexable> Static<T>
where
    T::Key: Ord,
{
    /// Build a new multi-level PGM-Index from sorted data.
    ///
    /// # Parameters
    ///
    /// - `data`: A sorted slice of values
    /// - `epsilon`: Error bound for the bottom level (larger = fewer segments)
    /// - `epsilon_recursive`: Error bound for upper levels
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty or `epsilon` is 0.
    pub fn new(data: &[T], epsilon: usize, epsilon_recursive: usize) -> Result<Self, Error> {
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
        Self::build_from_keys(&keys, epsilon, epsilon_recursive)
    }

    /// Build from pre-extracted keys (internal use).
    pub(crate) fn build_from_keys(
        keys: &[T::Key],
        epsilon: usize,
        epsilon_recursive: usize,
    ) -> Result<Self, Error> {
        let bottom_segments = build_segments(keys, epsilon);

        if bottom_segments.is_empty() {
            return Err(Error::EmptyInput);
        }

        let mut levels: Vec<Vec<Segment<T::Key>>> = vec![bottom_segments];

        while epsilon_recursive > 0 && levels.last().unwrap().len() > 1 {
            let prev_level = levels.last().unwrap();
            let super_keys: Vec<T::Key> = prev_level.iter().map(|s| s.key).collect();
            let upper_segments = build_segments(&super_keys, epsilon_recursive);

            if upper_segments.len() >= prev_level.len() {
                break;
            }

            levels.push(upper_segments);
        }

        let total_segments: usize = levels.iter().map(|l| l.len()).sum();
        let mut segments = Vec::with_capacity(total_segments);
        let mut levels_offsets = Vec::with_capacity(levels.len() + 1);

        levels_offsets.push(0);
        for level in levels.iter().rev() {
            segments.extend_from_slice(level);
            levels_offsets.push(segments.len());
        }

        Ok(Self {
            segments,
            levels_offsets,
            epsilon,
            epsilon_recursive,
            len: keys.len(),
        })
    }

    #[cfg(feature = "parallel")]
    pub fn new_parallel(
        data: &[T],
        epsilon: usize,
        epsilon_recursive: usize,
    ) -> Result<Self, Error> {
        use crate::index::model::build_segments_parallel;

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

        let bottom_segments = build_segments_parallel(&keys, epsilon);

        if bottom_segments.is_empty() {
            return Err(Error::EmptyInput);
        }

        let mut levels: Vec<Vec<Segment<T::Key>>> = vec![bottom_segments];

        while epsilon_recursive > 0 && levels.last().unwrap().len() > 1 {
            let prev_level = levels.last().unwrap();
            let super_keys: Vec<T::Key> = prev_level.iter().map(|s| s.key).collect();
            let upper_segments = build_segments(&super_keys, epsilon_recursive);

            if upper_segments.len() >= prev_level.len() {
                break;
            }

            levels.push(upper_segments);
        }

        let total_segments: usize = levels.iter().map(|l| l.len()).sum();
        let mut segments = Vec::with_capacity(total_segments);
        let mut levels_offsets = Vec::with_capacity(levels.len() + 1);

        levels_offsets.push(0);
        for level in levels.iter().rev() {
            segments.extend_from_slice(level);
            levels_offsets.push(segments.len());
        }

        Ok(Self {
            segments,
            levels_offsets,
            epsilon,
            epsilon_recursive,
            len: keys.len(),
        })
    }

    #[inline]
    fn search_segment(&self, level: usize, key: &T::Key, lo: usize, hi: usize) -> usize {
        let level_start = self.levels_offsets[level];
        let level_end = self.levels_offsets[level + 1];
        let level_size = level_end - level_start;

        let lo = lo.min(level_size);
        let hi = hi.min(level_size);

        if hi <= lo {
            return lo;
        }

        let abs_lo = level_start + lo;
        let abs_hi = level_start + hi;

        if abs_hi - abs_lo <= LINEAR_SEARCH_THRESHOLD_SEGMENTS {
            let mut idx = abs_lo;
            while idx + 1 < abs_hi && self.segments[idx + 1].key <= *key {
                idx += 1;
            }
            idx - level_start
        } else {
            let slice = &self.segments[abs_lo..abs_hi];
            let pos = slice.partition_point(|s| s.key <= *key);
            let pos = pos.saturating_sub(1);
            lo + pos
        }
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
        let num_levels = self.levels_offsets.len() - 1;

        if num_levels == 0 {
            return ApproxPos::new(0, 0, self.len);
        }

        let mut seg_lo = 0usize;
        let mut seg_hi = self.levels_offsets[1];

        for level in 0..num_levels - 1 {
            let level_start = self.levels_offsets[level];
            let level_size = self.levels_offsets[level + 1] - level_start;

            let local_idx = self.search_segment(level, key, seg_lo, seg_hi.min(level_size));
            let segment = &self.segments[level_start + local_idx];

            let next_level_start = self.levels_offsets[level + 1];
            let next_level_size = self.levels_offsets[level + 2] - next_level_start;
            let predicted = segment.predict(*key).min(next_level_size.saturating_sub(1));

            seg_lo = pgm_sub_eps(predicted, self.epsilon_recursive + 1);
            seg_hi = pgm_add_eps(predicted, self.epsilon_recursive, next_level_size);
        }

        let bottom_level = num_levels - 1;
        let bottom_start = self.levels_offsets[bottom_level];
        let bottom_size = self.levels_offsets[bottom_level + 1] - bottom_start;

        let local_idx = self.search_segment(bottom_level, key, seg_lo, seg_hi.min(bottom_size));
        let segment = &self.segments[bottom_start + local_idx];

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
        if data[pos] == *value {
            let mut i = pos;
            while i > 0 && data[i - 1] == *value {
                i -= 1;
            }
            return i;
        }

        if data[pos] < *value {
            if pos + 1 < len && data[pos + 1] >= *value {
                return pos + 1;
            }
        } else if pos > 0 && data[pos - 1] < *value {
            return pos;
        }

        let lo = approx.lo;
        let hi = approx.hi.min(len);
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
        let key = value.index_key();
        let approx = self.search_by_key(&key);
        let len = data.len();

        if len == 0 {
            return false;
        }

        let pos = approx.pos.min(len - 1);
        if data[pos] == *value {
            return true;
        }

        let lo = approx.lo;
        let hi = approx.hi.min(len);
        data[lo..hi].binary_search(value).is_ok()
    }

    /// Number of elements the index was built for.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of segments across all levels.
    #[inline]
    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    /// Number of levels in the index.
    #[inline]
    pub fn height(&self) -> usize {
        self.levels_offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn epsilon(&self) -> usize {
        self.epsilon
    }

    #[inline]
    pub fn epsilon_recursive(&self) -> usize {
        self.epsilon_recursive
    }

    /// Approximate memory usage in bytes.
    pub fn size_in_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.segments.capacity() * core::mem::size_of::<Segment<T::Key>>()
            + self.levels_offsets.capacity() * core::mem::size_of::<usize>()
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

impl<T: Indexable> crate::index::External<T> for Static<T>
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
    use alloc::vec::Vec;

    #[test]
    fn test_pgm_index_basic() {
        let keys: Vec<u64> = (0..10000).collect();
        let index = Static::new(&keys, 64, 4).unwrap();

        assert_eq!(index.len(), 10000);
        assert!(!index.is_empty());
        assert!(index.height() >= 1);
    }

    #[test]
    fn test_pgm_index_search() {
        let keys: Vec<u64> = (0..10000).collect();
        let index = Static::new(&keys, 64, 4).unwrap();

        for &key in &[0u64, 100, 5000, 9999] {
            let idx = index.lower_bound(&keys, &key);
            assert_eq!(idx, key as usize, "Failed for key {}", key);
        }
    }

    #[test]
    fn test_pgm_index_sparse() {
        let keys: Vec<u64> = (0..1000).map(|i| i * 1000).collect();
        let index = Static::new(&keys, 16, 4).unwrap();

        for (i, &key) in keys.iter().enumerate() {
            let idx = index.lower_bound(&keys, &key);
            assert_eq!(idx, i, "Failed for key {} at index {}", key, i);
        }
    }

    #[test]
    fn test_pgm_index_contains() {
        let keys: Vec<u64> = (0..100).map(|i| i * 2).collect();
        let index = Static::new(&keys, 8, 4).unwrap();

        assert!(index.contains(&keys, &0));
        assert!(index.contains(&keys, &100));
        assert!(!index.contains(&keys, &1));
        assert!(!index.contains(&keys, &99));
    }

    #[test]
    fn test_pgm_index_signed() {
        let keys: Vec<i64> = (-500..500).collect();
        let index = Static::new(&keys, 16, 4).unwrap();

        for &key in &[-500i64, -100, 0, 100, 499] {
            let expected = (key + 500) as usize;
            let idx = index.lower_bound(&keys, &key);
            assert_eq!(idx, expected, "Failed for key {}", key);
        }
    }

    #[test]
    fn test_pgm_index_duplicates() {
        let keys: Vec<u64> = vec![1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5];
        let index = Static::new(&keys, 4, 2).unwrap();

        assert_eq!(index.lower_bound(&keys, &1), 0);
        assert_eq!(index.lower_bound(&keys, &2), 2);
        assert_eq!(index.lower_bound(&keys, &5), 8);
    }

    #[test]
    fn test_empty_input_error() {
        let keys: Vec<u64> = vec![];
        let result = Static::new(&keys, 64, 4);
        assert_eq!(result.unwrap_err(), Error::EmptyInput);
    }

    #[test]
    fn test_invalid_epsilon_error() {
        let keys: Vec<u64> = vec![1, 2, 3];
        let result = Static::new(&keys, 0, 4);
        assert_eq!(result.unwrap_err(), Error::InvalidEpsilon);
    }

    #[test]
    fn test_single_element() {
        let keys: Vec<u64> = vec![42];
        let index = Static::new(&keys, 64, 4).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.height(), 1);
        assert!(index.contains(&keys, &42));
        assert!(!index.contains(&keys, &0));
        assert!(!index.contains(&keys, &100));
        assert_eq!(index.lower_bound(&keys, &42), 0);
        assert_eq!(index.lower_bound(&keys, &0), 0);
        assert_eq!(index.lower_bound(&keys, &100), 1);
    }

    #[test]
    fn test_epsilon_recursive_zero() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = Static::new(&keys, 64, 0).unwrap();

        assert_eq!(index.height(), 1);
        assert!(index.contains(&keys, &500));
        assert_eq!(index.lower_bound(&keys, &500), 500);
    }

    #[test]
    fn test_very_small_epsilon() {
        let keys: Vec<u64> = (0..100).collect();
        let index = Static::new(&keys, 1, 1).unwrap();

        for &key in &[0u64, 50, 99] {
            assert!(index.contains(&keys, &key));
            assert_eq!(index.lower_bound(&keys, &key), key as usize);
        }
    }

    #[test]
    fn test_very_large_epsilon() {
        let keys: Vec<u64> = (0..100).collect();
        let index = Static::new(&keys, 1000, 1000).unwrap();

        assert_eq!(index.segments_count(), 1);
        for &key in &[0u64, 50, 99] {
            assert!(index.contains(&keys, &key));
        }
    }

    #[test]
    fn test_upper_bound() {
        let keys: Vec<u64> = vec![1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5];
        let index = Static::new(&keys, 4, 2).unwrap();

        assert_eq!(index.upper_bound(&keys, &1), 2);
        assert_eq!(index.upper_bound(&keys, &2), 5);
        assert_eq!(index.upper_bound(&keys, &5), 12);
        assert_eq!(index.upper_bound(&keys, &0), 0);
        assert_eq!(index.upper_bound(&keys, &6), 12);
    }

    #[test]
    fn test_range_all_variants() {
        let keys: Vec<u64> = (0..100).collect();
        let index = Static::new(&keys, 16, 4).unwrap();

        let range_full: Vec<_> = index.range(&keys, ..).copied().collect();
        assert_eq!(range_full.len(), 100);

        let range_from: Vec<_> = index.range(&keys, 90..).copied().collect();
        assert_eq!(range_from, (90..100).collect::<Vec<_>>());

        let range_to: Vec<_> = index.range(&keys, ..10).copied().collect();
        assert_eq!(range_to, (0..10).collect::<Vec<_>>());

        let range_to_inclusive: Vec<_> = index.range(&keys, ..=10).copied().collect();
        assert_eq!(range_to_inclusive, (0..=10).collect::<Vec<_>>());

        let range_bounded: Vec<_> = index.range(&keys, 10..20).copied().collect();
        assert_eq!(range_bounded, (10..20).collect::<Vec<_>>());

        let range_bounded_inclusive: Vec<_> = index.range(&keys, 10..=20).copied().collect();
        assert_eq!(range_bounded_inclusive, (10..=20).collect::<Vec<_>>());
    }

    #[test]
    fn test_range_empty() {
        let keys: Vec<u64> = (0..100).collect();
        let index = Static::new(&keys, 16, 4).unwrap();

        let empty: Vec<_> = index.range(&keys, 200..300).copied().collect();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_size_in_bytes() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = Static::new(&keys, 64, 4).unwrap();

        assert!(index.size_in_bytes() > 0);
    }
}
