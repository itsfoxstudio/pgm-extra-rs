//! Dynamic PGM-Index supporting insertions and deletions.
//!
//! This index owns its data and supports modifications with automatic rebuilding.

use alloc::vec::Vec;
use core::ops::RangeBounds;
use std::collections::BTreeSet;

use crate::error::Error;
use crate::index::external::Static;
use crate::index::key::Indexable;

/// A dynamic PGM-Index that supports insertions and deletions.
///
/// This index owns its data and maintains insert/delete buffers that are
/// periodically merged into the main index. When the buffers exceed a
/// threshold, the index is automatically rebuilt.
///
/// # Example
///
/// ```
/// use pgm_extra::index::owned::Dynamic;
///
/// let mut index: Dynamic<u64> = Dynamic::new(16, 4);
///
/// index.insert(5);
/// index.insert(3);
/// index.insert(7);
///
/// assert!(index.contains(&3));
/// assert!(index.contains(&5));
/// assert!(!index.contains(&4));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound = "T: serde::Serialize + serde::de::DeserializeOwned, T::Key: serde::Serialize + serde::de::DeserializeOwned"
    )
)]
pub struct Dynamic<T: Indexable + Ord>
where
    T::Key: Ord,
{
    base_index: Option<Static<T>>,
    base_data: Vec<T>,
    delete_buffer: BTreeSet<T>,
    epsilon: usize,
    epsilon_recursive: usize,
    insert_buffer: BTreeSet<T>,
    rebuild_threshold: usize,
}

impl<T: Indexable + Ord + std::fmt::Debug> std::fmt::Debug for Dynamic<T>
where
    T::Key: Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dynamic")
            .field("base_data_len", &self.base_data.len())
            .field("insert_buffer_len", &self.insert_buffer.len())
            .field("delete_buffer_len", &self.delete_buffer.len())
            .field("epsilon", &self.epsilon)
            .field("epsilon_recursive", &self.epsilon_recursive)
            .finish()
    }
}

impl<T: Indexable + Ord + Copy> Dynamic<T>
where
    T::Key: Ord,
{
    /// Create a new empty dynamic index.
    pub fn new(epsilon: usize, epsilon_recursive: usize) -> Self {
        Self {
            base_index: None,
            base_data: Vec::new(),
            delete_buffer: BTreeSet::new(),
            epsilon: epsilon.max(1),
            epsilon_recursive: epsilon_recursive.max(1),
            insert_buffer: BTreeSet::new(),
            rebuild_threshold: 1024,
        }
    }

    /// Create a dynamic index from pre-sorted data.
    pub fn from_sorted(
        data: Vec<T>,
        epsilon: usize,
        epsilon_recursive: usize,
    ) -> Result<Self, Error> {
        let epsilon = epsilon.max(1);
        let epsilon_recursive = epsilon_recursive.max(1);

        if data.is_empty() {
            return Ok(Self::new(epsilon, epsilon_recursive));
        }

        let base_index = Static::new(&data, epsilon, epsilon_recursive)?;
        let rebuild_threshold = (data.len() / 10).max(1024);

        Ok(Self {
            base_index: Some(base_index),
            base_data: data,
            delete_buffer: BTreeSet::new(),
            epsilon,
            epsilon_recursive,
            insert_buffer: BTreeSet::new(),
            rebuild_threshold,
        })
    }

    /// Set the threshold for automatic rebuilding.
    pub fn with_rebuild_threshold(mut self, threshold: usize) -> Self {
        self.rebuild_threshold = threshold.max(1);
        self
    }

    /// Insert a value into the index.
    pub fn insert(&mut self, value: T) {
        self.insert_buffer.insert(value);
        self.maybe_rebuild();
    }

    /// Remove a value from the index.
    pub fn remove(&mut self, value: &T) -> bool {
        if self.insert_buffer.remove(value) {
            return true;
        }

        if self
            .base_index
            .as_ref()
            .is_some_and(|idx| idx.contains(&self.base_data, value))
        {
            self.delete_buffer.insert(*value);
            self.maybe_rebuild();
            return true;
        }

        false
    }

    /// Check if a value exists in the index.
    pub fn contains(&self, value: &T) -> bool {
        if self.insert_buffer.contains(value) {
            return true;
        }

        if self.delete_buffer.contains(value) {
            return false;
        }

        (self.base_index.as_ref()).is_some_and(|idx| idx.contains(&self.base_data, value))
    }

    /// Find the smallest value >= the given value.
    pub fn lower_bound(&self, value: &T) -> Option<T> {
        let base_result = self.base_index.as_ref().and_then(|idx| {
            let pos = idx.lower_bound(&self.base_data, value);
            if pos < self.base_data.len() {
                let v = self.base_data[pos];
                if !self.delete_buffer.contains(&v) {
                    return Some(v);
                }
                for i in (pos + 1)..self.base_data.len() {
                    let v = self.base_data[i];
                    if !self.delete_buffer.contains(&v) {
                        return Some(v);
                    }
                }
            }
            None
        });

        let buffer_result = self.insert_buffer.range(value..).next().copied();

        match (base_result, buffer_result) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Get the number of elements in the index.
    pub fn len(&self) -> usize {
        let base_len = self.base_data.len();
        let inserts = self.insert_buffer.len();
        let deletes = self.delete_buffer.len();
        base_len + inserts - deletes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of pending operations in buffers.
    pub fn pending_operations(&self) -> usize {
        self.insert_buffer.len() + self.delete_buffer.len()
    }

    fn maybe_rebuild(&mut self) {
        if self.pending_operations() < self.rebuild_threshold {
            return;
        }

        self.force_rebuild();
    }

    /// Force an immediate rebuild of the index.
    pub fn force_rebuild(&mut self) {
        let mut all_data: Vec<T> = self
            .base_data
            .iter()
            .copied()
            .filter(|v| !self.delete_buffer.contains(v))
            .collect();

        all_data.extend(self.insert_buffer.iter().copied());
        all_data.sort();

        if all_data.is_empty() {
            self.base_data = Vec::new();
            self.base_index = None;
        } else {
            match Static::new(&all_data, self.epsilon, self.epsilon_recursive) {
                Ok(new_index) => {
                    self.base_data = all_data;
                    self.base_index = Some(new_index);
                }
                Err(_) => {
                    self.base_data = all_data;
                    self.base_index = None;
                }
            }
        }

        self.insert_buffer.clear();
        self.delete_buffer.clear();
    }

    /// Iterate over all values in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        let base_iter = self
            .base_data
            .iter()
            .copied()
            .filter(|v| !self.delete_buffer.contains(v));

        let buffer_iter = self.insert_buffer.iter().copied();

        MergedIterator::new(base_iter, buffer_iter)
    }

    /// Iterate over values in the given range.
    pub fn range<'a, R>(&'a self, range: R) -> impl Iterator<Item = T> + 'a
    where
        R: RangeBounds<T> + Clone + 'a,
    {
        let range_clone = range.clone();
        let base_iter = self
            .base_data
            .iter()
            .copied()
            .filter(|v| !self.delete_buffer.contains(v))
            .filter(move |v| range_clone.contains(v));

        let buffer_iter = self
            .insert_buffer
            .range((range.start_bound().cloned(), range.end_bound().cloned()))
            .copied();

        MergedIterator::new(base_iter, buffer_iter)
    }

    /// Approximate memory usage in bytes.
    pub fn size_in_bytes(&self) -> usize {
        let base_size =
            core::mem::size_of::<Self>() + self.base_data.capacity() * core::mem::size_of::<T>();

        let index_size = self
            .base_index
            .as_ref()
            .map_or(0, |idx| idx.size_in_bytes());

        let buffer_size =
            (self.insert_buffer.len() + self.delete_buffer.len()) * core::mem::size_of::<T>() * 3;

        base_size + index_size + buffer_size
    }
}

struct MergedIterator<I1, I2, T>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
    T: Ord,
{
    iter1: core::iter::Peekable<I1>,
    iter2: core::iter::Peekable<I2>,
}

impl<I1, I2, T> MergedIterator<I1, I2, T>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
    T: Ord,
{
    fn new(iter1: I1, iter2: I2) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
        }
    }
}

impl<I1, I2, T> Iterator for MergedIterator<I1, I2, T>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
    T: Ord + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.iter1.peek(), self.iter2.peek()) {
            (Some(&a), Some(&b)) => {
                if a < b {
                    self.iter1.next()
                } else if a > b {
                    self.iter2.next()
                } else {
                    self.iter2.next();
                    self.iter1.next()
                }
            }
            (Some(_), None) => self.iter1.next(),
            (None, Some(_)) => self.iter2.next(),
            (None, None) => None,
        }
    }
}

impl<T: Indexable + Ord + Copy> core::iter::Extend<T> for Dynamic<T>
where
    T::Key: Ord,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.insert_buffer.extend(iter);
        self.maybe_rebuild();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_empty() {
        let index: Dynamic<u64> = Dynamic::new(16, 4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_dynamic_insert() {
        let mut index: Dynamic<u64> = Dynamic::new(16, 4);

        index.insert(5);
        index.insert(3);
        index.insert(7);

        assert!(index.contains(&3));
        assert!(index.contains(&5));
        assert!(index.contains(&7));
        assert!(!index.contains(&4));
    }

    #[test]
    fn test_dynamic_extend() {
        let mut index: Dynamic<u64> = Dynamic::new(16, 4);

        index.extend(5..7);

        assert!(index.contains(&5));
        assert!(index.contains(&6));
        assert!(!index.contains(&7));
    }

    #[test]
    fn test_dynamic_remove() {
        let data: Vec<u64> = (0..100).collect();
        let mut index = Dynamic::from_sorted(data, 16, 4).unwrap();

        assert!(index.contains(&50));
        assert!(index.remove(&50));
        assert!(!index.contains(&50));
    }

    #[test]
    fn test_dynamic_rebuild() {
        let mut index: Dynamic<u64> = Dynamic::new(16, 4).with_rebuild_threshold(10);

        for i in 0..20 {
            index.insert(i);
        }

        assert_eq!(index.pending_operations(), 0);

        for i in 0..20 {
            assert!(index.contains(&i), "Missing value {}", i);
        }
    }

    #[test]
    fn test_dynamic_iter() {
        let mut index: Dynamic<u64> = Dynamic::new(16, 4);

        index.insert(3);
        index.insert(1);
        index.insert(2);

        let collected: Vec<u64> = index.iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }
}