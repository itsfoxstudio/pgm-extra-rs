//! Cached PGM-Index with hot-key lookup optimization.
//!
//! Wraps a multi-level index with a small cache for frequently accessed keys.

use core::ops::RangeBounds;

use crate::error::Error;
use crate::index::Static;
use crate::index::key::Indexable;
use crate::util::ApproxPos;
use crate::util::cache::{FastHash, HotCache};
use crate::util::range::range_to_indices;

/// A PGM-Index wrapper with a small hot-key cache.
///
/// This struct uses interior mutability (`Cell`) to update the cache on read operations.
/// Therefore, it is `!Sync` and cannot be shared across threads without synchronization
/// (e.g., `Mutex<Cached>`).
///
/// Note: When serialized with serde, only the inner index is saved. The cache is
/// recreated empty on deserialization.
///
/// # Example
///
/// ```
/// use pgm_extra::index::external::Cached;
///
/// let keys: Vec<u64> = (0..10000).collect();
/// let index = Cached::new(&keys, 64, 4).unwrap();
///
/// // First lookup - cache miss, populates cache
/// assert!(index.contains(&keys, &5000));
///
/// // Second lookup - cache hit
/// assert!(index.contains(&keys, &5000));
/// ```
#[derive(Debug)]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T::Key: serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct Cached<T: Indexable>
where
    T::Key: FastHash + core::default::Default,
{
    inner: Static<T>,
    #[cfg_attr(feature = "rkyv", rkyv(with = rkyv::with::Skip))]
    #[cfg_attr(feature = "serde", serde(skip, default))]
    cache: HotCache<T::Key>,
}

impl<T: Indexable> Cached<T>
where
    T::Key: Ord + FastHash + core::default::Default,
{
    /// Build a new cached PGM-Index from sorted data.
    pub fn new(data: &[T], epsilon: usize, epsilon_recursive: usize) -> Result<Self, Error> {
        let inner = Static::new(data, epsilon, epsilon_recursive)?;
        Ok(Self {
            inner,
            cache: HotCache::new(),
        })
    }

    /// Wrap an existing index with a cache.
    pub fn from_index(index: Static<T>) -> Self {
        Self {
            inner: index,
            cache: HotCache::new(),
        }
    }

    /// Get an approximate position for the given value.
    #[inline]
    pub fn search(&self, value: &T) -> ApproxPos {
        self.inner.search(value)
    }

    /// Find the first position where `data[pos] >= value`.
    #[inline]
    pub fn lower_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        let key = value.index_key();

        if let Some(pos) = self.cache.lookup(&key)
            && pos < data.len()
            && data[pos] == *value
        {
            return pos;
        }

        let result = self.inner.lower_bound(data, value);

        if result < data.len() && data[result] == *value {
            self.cache.insert(key, result);
        }

        result
    }

    /// Find the first position where `data[pos] > value`.
    #[inline]
    pub fn upper_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord,
    {
        self.inner.upper_bound(data, value)
    }

    /// Check if the value exists in the data.
    #[inline]
    pub fn contains(&self, data: &[T], value: &T) -> bool
    where
        T: Ord,
    {
        let key = value.index_key();

        if let Some(pos) = self.cache.lookup(&key)
            && pos < data.len()
            && data[pos] == *value
        {
            return true;
        }

        let result = self.inner.contains(data, value);

        if result {
            let pos = self.inner.lower_bound(data, value);
            self.cache.insert(key, pos);
        }

        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn segments_count(&self) -> usize {
        self.inner.segments_count()
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    #[inline]
    pub fn epsilon(&self) -> usize {
        self.inner.epsilon()
    }

    #[inline]
    pub fn epsilon_recursive(&self) -> usize {
        self.inner.epsilon_recursive()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes() + core::mem::size_of::<HotCache<T::Key>>()
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get a reference to the inner index.
    pub fn inner(&self) -> &Static<T> {
        &self.inner
    }

    /// Consume and return the inner index.
    pub fn into_inner(self) -> Static<T> {
        self.inner
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

impl<T: Indexable> From<Static<T>> for Cached<T>
where
    T::Key: Ord + FastHash + core::default::Default,
{
    fn from(index: Static<T>) -> Self {
        Self::from_index(index)
    }
}

impl<T: Indexable> From<Cached<T>> for Static<T>
where
    T::Key: Ord + FastHash + core::default::Default,
{
    fn from(cached: Cached<T>) -> Self {
        cached.into_inner()
    }
}

impl<T: Indexable> crate::index::External<T> for Cached<T>
where
    T::Key: Ord + crate::util::cache::FastHash + core::default::Default,
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
    fn test_cached_index_basic() {
        let keys: Vec<u64> = (0..10000).collect();
        let index = Cached::new(&keys, 64, 4).unwrap();

        assert_eq!(index.len(), 10000);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_cached_index_hit() {
        let keys: Vec<u64> = (0..1000).collect();
        let index = Cached::new(&keys, 64, 4).unwrap();

        let key = 500u64;
        let pos1 = index.lower_bound(&keys, &key);
        assert_eq!(pos1, 500);

        let pos2 = index.lower_bound(&keys, &key);
        assert_eq!(pos2, 500);
    }

    #[test]
    fn test_cached_contains() {
        let keys: Vec<u64> = (0..100).map(|i| i * 2).collect();
        let index = Cached::new(&keys, 8, 4).unwrap();

        assert!(index.contains(&keys, &0));
        assert!(index.contains(&keys, &100));

        assert!(index.contains(&keys, &0));

        assert!(!index.contains(&keys, &1));
        assert!(!index.contains(&keys, &99));
    }

    #[test]
    fn test_cached_clear() {
        let keys: Vec<u64> = (0..100).collect();
        let index = Cached::new(&keys, 16, 4).unwrap();

        let _ = index.lower_bound(&keys, &50);
        index.clear_cache();
    }
}
