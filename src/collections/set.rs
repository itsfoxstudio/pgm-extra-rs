//! An owned set optimized for read-heavy workloads backed by PGM-Index.
//!
//! `Set` is a drop-in replacement for `BTreeSet` in read-heavy workloads
//! where you build the set once and perform many lookups.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::ops::RangeBounds;

use crate::error::Error;
use crate::index::external;
use crate::index::key::Indexable;
use crate::util::range::range_to_indices;

/// An owned set optimized for read-heavy workloads, backed by a PGM-index.
///
/// This is a BTreeSet-like container that owns its data and provides
/// efficient lookups using a learned index. Mutations are supported but
/// trigger O(n) index rebuilds; for frequent updates use [`crate::Dynamic`].
///
/// Works with any type that implements [`Indexable`]:
/// - Numeric types (u64, i32, etc.) are indexed directly
/// - String/bytes types use prefix extraction
///
/// # Example
///
/// ```
/// use pgm_extra::Set;
///
/// // Numeric set
/// let nums: Vec<u64> = (0..10000).collect();
/// let set = Set::from_sorted_unique(nums, 64, 4).unwrap();
/// assert!(set.contains(&5000));
///
/// // String set
/// let words = vec!["apple", "banana", "cherry"];
/// let set = Set::from_sorted_unique(words, 64, 4).unwrap();
/// assert!(set.contains(&"banana"));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound = "T: serde::Serialize + serde::de::DeserializeOwned, T::Key: serde::Serialize + serde::de::DeserializeOwned"
    )
)]
pub struct Set<T: Indexable> {
    data: Vec<T>,
    index: Option<external::Static<T>>,
    epsilon: usize,
    epsilon_recursive: usize,
}

impl<T: Indexable + Ord> Set<T>
where
    T::Key: Ord,
{
    /// Create a set from pre-sorted, unique values.
    ///
    /// # Panics
    ///
    /// Debug builds will panic if values are not sorted or contain duplicates.
    pub fn from_sorted_unique(
        data: Vec<T>,
        epsilon: usize,
        epsilon_recursive: usize,
    ) -> Result<Self, Error> {
        debug_assert!(
            data.windows(2).all(|w| w[0] < w[1]),
            "data must be sorted and unique"
        );

        let index = if data.is_empty() {
            None
        } else {
            Some(external::Static::new(&data, epsilon, epsilon_recursive)?)
        };
        Ok(Self {
            data,
            index,
            epsilon,
            epsilon_recursive,
        })
    }

    /// Create a set from an unsorted iterator.
    ///
    /// Values are sorted and deduplicated (like `BTreeSet::from_iter`).
    pub fn build<I>(iter: I, epsilon: usize, epsilon_recursive: usize) -> Result<Self, Error>
    where
        I: IntoIterator<Item = T>,
    {
        let mut data: Vec<T> = iter.into_iter().collect();
        data.sort();
        data.dedup();

        Self::from_sorted_unique(data, epsilon, epsilon_recursive)
    }

    /// Create an empty set with the given epsilon values.
    pub fn empty(epsilon: usize, epsilon_recursive: usize) -> Self {
        Self {
            data: Vec::new(),
            index: None,
            epsilon,
            epsilon_recursive,
        }
    }

    /// Create a set with default epsilon values (64, 4).
    pub fn new(data: Vec<T>) -> Result<Self, Error> {
        Self::build(data, 64, 4)
    }

    /// Check if the set contains the value.
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.get(value).is_some()
    }

    /// Get a reference to the value if it exists.
    #[inline]
    pub fn get(&self, value: &T) -> Option<&T> {
        let index = self.index.as_ref()?;
        let approx = index.search(value);

        let lo = approx.lo;
        let hi = approx.hi.min(self.data.len());

        for i in lo..hi {
            match self.data[i].cmp(value) {
                Ordering::Equal => return Some(&self.data[i]),
                Ordering::Greater => return None,
                Ordering::Less => continue,
            }
        }
        None
    }

    /// Find the index of the first value >= the given value.
    #[inline]
    pub fn lower_bound(&self, value: &T) -> usize {
        match &self.index {
            Some(index) => index.lower_bound(&self.data, value),
            None => 0,
        }
    }

    /// Find the index of the first value > the given value.
    #[inline]
    pub fn upper_bound(&self, value: &T) -> usize {
        match &self.index {
            Some(index) => index.upper_bound(&self.data, value),
            None => 0,
        }
    }

    /// Returns an iterator over values in the given range.
    #[inline]
    pub fn range<R>(&self, range: R) -> impl DoubleEndedIterator<Item = &T>
    where
        R: RangeBounds<T>,
    {
        let (start, end) = range_to_indices(
            range,
            self.data.len(),
            |v| self.lower_bound(v),
            |v| self.upper_bound(v),
        );
        self.data[start..end].iter()
    }

    /// Get the first (smallest) value.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }

    /// Get the last (largest) value.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        self.data.last()
    }

    /// Iterate over all values in sorted order.
    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        self.data.iter()
    }

    /// Get the number of values.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of segments in the underlying index.
    #[inline]
    pub fn segments_count(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.segments_count())
    }

    /// Get the height of the underlying index.
    #[inline]
    pub fn height(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.height())
    }

    /// Get the epsilon value.
    #[inline]
    pub fn epsilon(&self) -> usize {
        self.epsilon
    }

    /// Get the epsilon_recursive value.
    #[inline]
    pub fn epsilon_recursive(&self) -> usize {
        self.epsilon_recursive
    }

    /// Approximate memory usage in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.size_in_bytes())
            + self.data.capacity() * core::mem::size_of::<T>()
    }

    /// Get a reference to the underlying data slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consume the set and return the underlying data.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Get a reference to the underlying index.
    #[inline]
    pub fn index(&self) -> Option<&external::Static<T>> {
        self.index.as_ref()
    }

    /// Insert a value into the set.
    ///
    /// Returns `true` if the value was newly inserted, `false` if it already existed.
    ///
    /// **Note**: This rebuilds the entire index, making it O(n) per insertion.
    /// For bulk insertions, prefer collecting into a new set or using `extend`.
    /// For frequent mutations, consider using `index::owned::Dynamic` directly.
    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }

        let mut data = core::mem::take(&mut self.data);
        data.push(value);
        data.sort();

        if let Ok(new_set) = Self::from_sorted_unique(data, self.epsilon, self.epsilon_recursive) {
            *self = new_set;
        }
        true
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint(&self, other: &Set<T>) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }

        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };

        for value in smaller.iter() {
            if larger.contains(value) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if `self` is a subset of `other`.
    pub fn is_subset(&self, other: &Set<T>) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if `self` is a superset of `other`.
    pub fn is_superset(&self, other: &Set<T>) -> bool {
        other.is_subset(self)
    }

    /// Returns an iterator over values in `self` but not in `other`.
    pub fn difference<'a>(&'a self, other: &'a Set<T>) -> impl Iterator<Item = &'a T> {
        self.iter().filter(move |v| !other.contains(v))
    }

    /// Returns an iterator over values in `self` or `other` but not both.
    pub fn symmetric_difference<'a>(&'a self, other: &'a Set<T>) -> impl Iterator<Item = &'a T> {
        self.difference(other).chain(other.difference(self))
    }

    /// Returns an iterator over values in both `self` and `other`.
    pub fn intersection<'a>(&'a self, other: &'a Set<T>) -> impl Iterator<Item = &'a T> {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        smaller.iter().filter(move |v| larger.contains(v))
    }

    /// Returns an iterator over values in either `self` or `other`.
    pub fn union<'a>(&'a self, other: &'a Set<T>) -> impl Iterator<Item = &'a T> {
        MergeIter::new(self.data.iter(), other.data.iter())
    }
}

/// Iterator that merges two sorted iterators, yielding unique elements.
pub struct MergeIter<'a, T> {
    a: core::slice::Iter<'a, T>,
    b: core::slice::Iter<'a, T>,
    peeked_a: Option<&'a T>,
    peeked_b: Option<&'a T>,
}

impl<'a, T: Ord> MergeIter<'a, T> {
    fn new(mut a: core::slice::Iter<'a, T>, mut b: core::slice::Iter<'a, T>) -> Self {
        let peeked_a = a.next();
        let peeked_b = b.next();
        Self {
            a,
            b,
            peeked_a,
            peeked_b,
        }
    }
}

impl<'a, T: Ord> Iterator for MergeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.peeked_a, self.peeked_b) {
            (Some(a), Some(b)) => match a.cmp(b) {
                Ordering::Less => {
                    self.peeked_a = self.a.next();
                    Some(a)
                }
                Ordering::Greater => {
                    self.peeked_b = self.b.next();
                    Some(b)
                }
                Ordering::Equal => {
                    self.peeked_a = self.a.next();
                    self.peeked_b = self.b.next();
                    Some(a)
                }
            },
            (Some(a), None) => {
                self.peeked_a = self.a.next();
                Some(a)
            }
            (None, Some(b)) => {
                self.peeked_b = self.b.next();
                Some(b)
            }
            (None, None) => None,
        }
    }
}

impl<T: Ord> FusedIterator for MergeIter<'_, T> {}

// Standard trait implementations

impl<T: Indexable + Clone> Clone for Set<T>
where
    T::Key: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            index: self.index.clone(),
            epsilon: self.epsilon,
            epsilon_recursive: self.epsilon_recursive,
        }
    }
}

impl<T: Indexable + fmt::Debug> fmt::Debug for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.data.iter()).finish()
    }
}

impl<T: Indexable + Ord + PartialEq> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Indexable + Ord + Eq> Eq for Set<T> {}

impl<T: Indexable + Ord + PartialOrd> PartialOrd for Set<T>
where
    T::Key: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Indexable + Ord> Ord for Set<T>
where
    T::Key: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

impl<T: Indexable + Hash> Hash for Set<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T: Indexable + Ord> IntoIterator for Set<T>
where
    T::Key: Ord,
{
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: Indexable> IntoIterator for &'a Set<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T: Indexable + Ord> FromIterator<T> for Set<T>
where
    T::Key: Ord,
{
    /// Creates a Set from an iterator.
    ///
    /// Returns an empty set if the iterator is empty.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::build(iter, 64, 4).unwrap_or_else(|_| Self::empty(64, 4))
    }
}

impl<T: Indexable + Ord> core::iter::Extend<T> for Set<T>
where
    T::Key: Ord,
{
    /// Extends the set with elements from an iterator.
    ///
    /// **Note**: This rebuilds the entire index, making it O(n) per call.
    /// For bulk insertions, prefer collecting into a new set.
    /// For frequent mutations, consider using `index::owned::Dynamic` directly.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut data = core::mem::take(&mut self.data);
        data.extend(iter);
        data.sort();
        data.dedup();

        match Self::from_sorted_unique(data, self.epsilon, self.epsilon_recursive) {
            Ok(new_set) => *self = new_set,
            Err(_) => {
                *self = Self::empty(self.epsilon, self.epsilon_recursive);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    #[test]
    fn test_set_numeric() {
        let data: Vec<u64> = (0..1000).collect();
        let set = Set::from_sorted_unique(data, 64, 4).unwrap();

        assert_eq!(set.len(), 1000);
        assert!(set.contains(&500));
        assert!(!set.contains(&1001));
    }

    #[test]
    fn test_set_strings() {
        let data = vec!["apple", "banana", "cherry", "date"];
        let set = Set::from_sorted_unique(data, 64, 4).unwrap();

        assert!(set.contains(&"banana"));
        assert!(set.contains(&"cherry"));
        assert!(!set.contains(&"elderberry"));
    }

    #[test]
    fn test_set_owned_strings() {
        let data: Vec<String> = vec!["alpha", "beta", "gamma"]
            .into_iter()
            .map(String::from)
            .collect();
        let set = Set::from_sorted_unique(data, 64, 4).unwrap();

        assert!(set.contains(&String::from("beta")));
        assert!(!set.contains(&String::from("delta")));
    }

    #[test]
    fn test_set_build() {
        let data = vec![5u64, 3, 1, 4, 1, 5, 9, 2, 6];
        let set = Set::build(data, 4, 2).unwrap();

        assert_eq!(set.len(), 7);
        assert!(set.contains(&1));
        assert!(set.contains(&9));

        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_set_first_last() {
        let data: Vec<u64> = vec![10, 20, 30, 40, 50];
        let set = Set::from_sorted_unique(data, 4, 2).unwrap();

        assert_eq!(set.first(), Some(&10));
        assert_eq!(set.last(), Some(&50));
    }

    #[test]
    fn test_set_range() {
        let data: Vec<u64> = (0..100).collect();
        let set = Set::from_sorted_unique(data, 16, 4).unwrap();

        let range: Vec<_> = set.range(10..20).copied().collect();
        assert_eq!(range, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_set_iter() {
        let data: Vec<u64> = (0..10).collect();
        let set = Set::from_sorted_unique(data, 4, 2).unwrap();

        let forward: Vec<_> = set.iter().copied().collect();
        let backward: Vec<_> = set.iter().rev().copied().collect();

        assert_eq!(forward, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(backward, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_set_operations() {
        let set1 = Set::build(vec![1u64, 2, 3, 4, 5], 4, 2).unwrap();
        let set2 = Set::build(vec![4u64, 5, 6, 7, 8], 4, 2).unwrap();

        let intersection: Vec<_> = set1.intersection(&set2).copied().collect();
        assert_eq!(intersection, vec![4, 5]);

        let difference: Vec<_> = set1.difference(&set2).copied().collect();
        assert_eq!(difference, vec![1, 2, 3]);

        assert!(!set1.is_disjoint(&set2));

        let set3 = Set::build(vec![10u64, 11], 4, 2).unwrap();
        assert!(set1.is_disjoint(&set3));
    }

    #[test]
    fn test_set_collect() {
        let set: Set<u64> = (0..100).collect();
        assert_eq!(set.len(), 100);
        assert!(set.contains(&50));
    }

    #[test]
    fn test_set_empty() {
        let set: Set<u64> = Set::empty(64, 4);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(&0));
        assert_eq!(set.first(), None);
        assert_eq!(set.last(), None);
    }

    #[test]
    fn test_set_collect_empty() {
        let set: Set<u64> = core::iter::empty().collect();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_set_insert() {
        let mut set = Set::build(vec![1u64, 3, 5], 4, 2).unwrap();
        assert_eq!(set.len(), 3);

        assert!(set.insert(2));
        assert_eq!(set.len(), 4);
        assert!(set.contains(&2));

        assert!(!set.insert(2));
        assert_eq!(set.len(), 4);

        assert!(set.insert(4));
        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_set_insert_into_empty() {
        let mut set: Set<u64> = Set::empty(64, 4);
        assert!(set.insert(42));
        assert_eq!(set.len(), 1);
        assert!(set.contains(&42));
    }

    #[test]
    fn test_set_extend_empty() {
        let mut set: Set<u64> = Set::empty(64, 4);
        set.extend(vec![3, 1, 2]);
        assert_eq!(set.len(), 3);
        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }
}