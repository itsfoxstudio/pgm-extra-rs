//! An owned map optimized for read-heavy workloads backed by PGM-Index.
//!
//! `Map` is a drop-in replacement for `BTreeMap` in read-heavy workloads
//! where you build the map once and perform many lookups.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::RangeBounds;

use crate::error::Error;
use crate::index::external;
use crate::index::key::Indexable;
use crate::util::range::range_to_indices;

/// An owned map optimized for read-heavy workloads, backed by a PGM-index.
///
/// This is a BTreeMap-like container that owns its data and provides
/// efficient lookups using a learned index. Mutations are supported but
/// trigger O(n) index rebuilds; for frequent updates use [`crate::Dynamic`].
///
/// Works with any key type that implements [`Indexable`]:
/// - Numeric types (u64, i32, etc.) are indexed directly
/// - String/bytes types use prefix extraction
///
/// # Example
///
/// ```
/// use pgm_extra::Map;
///
/// // Numeric keys
/// let entries: Vec<(u64, &str)> = vec![(1, "one"), (2, "two"), (3, "three")];
/// let map = Map::from_sorted_unique(entries, 64, 4).unwrap();
/// assert_eq!(map.get(&2), Some(&"two"));
///
/// // String keys
/// let entries = vec![("apple", 1), ("banana", 2), ("cherry", 3)];
/// let map = Map::from_sorted_unique(entries, 64, 4).unwrap();
/// assert_eq!(map.get(&"banana"), Some(&2));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound = "K: serde::Serialize + serde::de::DeserializeOwned, V: serde::Serialize + serde::de::DeserializeOwned, K::Key: serde::Serialize + serde::de::DeserializeOwned"
    )
)]
pub struct Map<K: Indexable, V> {
    keys: Vec<K>,
    values: Vec<V>,
    index: Option<external::Static<K>>,
    epsilon: usize,
    epsilon_recursive: usize,
}

impl<K: Indexable + Ord, V> Map<K, V>
where
    K::Key: Ord,
{
    /// Create a map from pre-sorted, unique key-value pairs.
    ///
    /// # Panics
    ///
    /// Debug builds will panic if keys are not sorted or contain duplicates.
    pub fn from_sorted_unique(
        entries: Vec<(K, V)>,
        epsilon: usize,
        epsilon_recursive: usize,
    ) -> Result<Self, Error> {
        let (keys, values): (Vec<K>, Vec<V>) = entries.into_iter().unzip();

        debug_assert!(
            keys.windows(2).all(|w| w[0] < w[1]),
            "keys must be sorted and unique"
        );

        let index = if keys.is_empty() {
            None
        } else {
            Some(external::Static::new(&keys, epsilon, epsilon_recursive)?)
        };
        Ok(Self {
            keys,
            values,
            index,
            epsilon,
            epsilon_recursive,
        })
    }

    /// Create a map from an unsorted iterator of key-value pairs.
    ///
    /// If duplicate keys exist, the last value wins (like `BTreeMap::from_iter`).
    pub fn build<I>(iter: I, epsilon: usize, epsilon_recursive: usize) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut entries: Vec<(K, V)> = iter.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries.dedup_by(|a, b| a.0 == b.0);

        Self::from_sorted_unique(entries, epsilon, epsilon_recursive)
    }

    pub fn empty(epsilon: usize, epsilon_recursive: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            index: None,
            epsilon,
            epsilon_recursive,
        }
    }

    pub fn new(entries: Vec<(K, V)>) -> Result<Self, Error> {
        Self::build(entries, 64, 4)
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.index.as_ref()?;
        let approx = index.search(key);

        let lo = approx.lo;
        let hi = approx.hi.min(self.keys.len());

        for i in lo..hi {
            match self.keys[i].cmp(key) {
                Ordering::Equal => return Some(&self.values[i]),
                Ordering::Greater => return None,
                Ordering::Less => continue,
            }
        }
        None
    }

    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        let index = self.index.as_ref()?;
        let approx = index.search(key);

        let lo = approx.lo;
        let hi = approx.hi.min(self.keys.len());

        for i in lo..hi {
            match self.keys[i].cmp(key) {
                Ordering::Equal => return Some((&self.keys[i], &self.values[i])),
                Ordering::Greater => return None,
                Ordering::Less => continue,
            }
        }
        None
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Find the index of the first key >= the given key.
    #[inline]
    pub fn lower_bound(&self, key: &K) -> usize {
        match &self.index {
            Some(index) => index.lower_bound(&self.keys, key),
            None => 0,
        }
    }

    /// Find the index of the first key > the given key.
    #[inline]
    pub fn upper_bound(&self, key: &K) -> usize {
        match &self.index {
            Some(index) => index.upper_bound(&self.keys, key),
            None => 0,
        }
    }

    #[inline]
    pub fn range<R>(&self, range: R) -> impl DoubleEndedIterator<Item = (&K, &V)>
    where
        R: RangeBounds<K>,
    {
        let (start, end) = range_to_indices(
            range,
            self.keys.len(),
            |k| self.lower_bound(k),
            |k| self.upper_bound(k),
        );
        self.keys[start..end]
            .iter()
            .zip(self.values[start..end].iter())
    }

    #[inline]
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        if self.keys.is_empty() {
            None
        } else {
            Some((&self.keys[0], &self.values[0]))
        }
    }

    #[inline]
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        if self.keys.is_empty() {
            None
        } else {
            let last = self.keys.len() - 1;
            Some((&self.keys[last], &self.values[last]))
        }
    }

    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + DoubleEndedIterator {
        self.keys.iter().zip(self.values.iter())
    }

    #[inline]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + DoubleEndedIterator {
        self.keys.iter()
    }

    #[inline]
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator {
        self.values.iter()
    }

    #[inline]
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> + DoubleEndedIterator {
        self.values.iter_mut()
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let index = self.index.as_ref()?;
        let approx = index.search(key);

        let lo = approx.lo;
        let hi = approx.hi.min(self.keys.len());

        for i in lo..hi {
            match self.keys[i].cmp(key) {
                Ordering::Equal => return Some(&mut self.values[i]),
                Ordering::Greater => return None,
                Ordering::Less => continue,
            }
        }
        None
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
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
            + self.keys.capacity() * core::mem::size_of::<K>()
            + self.values.capacity() * core::mem::size_of::<V>()
    }

    /// Get a reference to the underlying keys slice.
    #[inline]
    pub fn keys_slice(&self) -> &[K] {
        &self.keys
    }

    /// Get a reference to the underlying values slice.
    #[inline]
    pub fn values_slice(&self) -> &[V] {
        &self.values
    }

    /// Consume the map and return the underlying key-value pairs.
    #[inline]
    pub fn into_vec(self) -> Vec<(K, V)> {
        self.keys.into_iter().zip(self.values).collect()
    }

    /// Get a reference to the underlying index.
    #[inline]
    pub fn index(&self) -> Option<&external::Static<K>> {
        self.index.as_ref()
    }

    /// Insert a key-value pair into the map.
    ///
    /// Returns the old value if the key already existed, or `None` if it was newly inserted.
    ///
    /// **Note**: This rebuilds the entire index, making it O(n) per insertion.
    /// For bulk insertions, prefer collecting into a new map.
    /// For frequent mutations, consider using `index::owned::Dynamic` directly.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(existing) = self.get_mut(&key) {
            return Some(core::mem::replace(existing, value));
        }

        let mut entries: Vec<(K, V)> = core::mem::take(&mut self.keys)
            .into_iter()
            .zip(core::mem::take(&mut self.values))
            .collect();
        entries.push((key, value));
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        if let Ok(new_map) = Self::from_sorted_unique(entries, self.epsilon, self.epsilon_recursive)
        {
            *self = new_map;
        }
        None
    }
}

impl<K: Indexable + Ord, V> core::ops::Index<&K> for Map<K, V>
where
    K::Key: Ord,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the map.
    #[inline]
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("key not found")
    }
}

// Standard trait implementations

impl<K: Indexable + Clone, V: Clone> Clone for Map<K, V>
where
    K::Key: Clone,
{
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
            index: self.index.clone(),
            epsilon: self.epsilon,
            epsilon_recursive: self.epsilon_recursive,
        }
    }
}

impl<K: Indexable + fmt::Debug, V: fmt::Debug> fmt::Debug for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.keys.iter().zip(self.values.iter()))
            .finish()
    }
}

impl<K: Indexable + Ord + PartialEq, V: PartialEq> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys && self.values == other.values
    }
}

impl<K: Indexable + Ord + Eq, V: Eq> Eq for Map<K, V> {}

impl<K: Indexable + Hash, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in self.keys.iter().zip(self.values.iter()) {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: Indexable + Ord, V> IntoIterator for Map<K, V>
where
    K::Key: Ord,
{
    type Item = (K, V);
    type IntoIter = alloc::vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.keys
            .into_iter()
            .zip(self.values)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, K: Indexable, V> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = core::iter::Zip<core::slice::Iter<'a, K>, core::slice::Iter<'a, V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.keys.iter().zip(self.values.iter())
    }
}

impl<K: Indexable + Ord, V> FromIterator<(K, V)> for Map<K, V>
where
    K::Key: Ord,
{
    /// Creates a Map from an iterator.
    ///
    /// Returns an empty map if the iterator is empty.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self::build(iter, 64, 4).unwrap_or_else(|_| Self::empty(64, 4))
    }
}

impl<K: Indexable + Ord, V> core::iter::Extend<(K, V)> for Map<K, V>
where
    K::Key: Ord,
{
    /// Extends the map with key-value pairs from an iterator.
    ///
    /// If duplicate keys exist, the last value wins.
    ///
    /// **Note**: This rebuilds the entire index, making it O(n) per call.
    /// For bulk insertions, prefer collecting into a new map.
    /// For frequent mutations, consider using `index::owned::Dynamic` directly.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let mut entries: Vec<(K, V)> = core::mem::take(&mut self.keys)
            .into_iter()
            .zip(core::mem::take(&mut self.values))
            .collect();
        entries.extend(iter);
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries.dedup_by(|a, b| a.0 == b.0);

        match Self::from_sorted_unique(entries, self.epsilon, self.epsilon_recursive) {
            Ok(new_map) => *self = new_map,
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
    fn test_map_numeric() {
        let entries: Vec<(u64, i32)> = (0..1000).map(|i| (i, i as i32 * 2)).collect();
        let map = Map::from_sorted_unique(entries, 64, 4).unwrap();

        assert_eq!(map.len(), 1000);
        assert_eq!(map.get(&500), Some(&1000));
        assert_eq!(map.get(&1001), None);
    }

    #[test]
    fn test_map_string_keys() {
        let entries = vec![("apple", 1), ("banana", 2), ("cherry", 3), ("date", 4)];
        let map = Map::from_sorted_unique(entries, 64, 4).unwrap();

        assert_eq!(map.get(&"banana"), Some(&2));
        assert_eq!(map.get(&"cherry"), Some(&3));
        assert_eq!(map.get(&"elderberry"), None);
    }

    #[test]
    fn test_map_owned_string_keys() {
        let entries: Vec<(String, i32)> =
            vec![("alpha".into(), 1), ("beta".into(), 2), ("gamma".into(), 3)];
        let map = Map::from_sorted_unique(entries, 64, 4).unwrap();

        assert_eq!(map.get(&String::from("beta")), Some(&2));
        assert!(map.contains_key(&String::from("alpha")));
    }

    #[test]
    fn test_map_get_key_value() {
        let entries: Vec<(u64, &str)> = vec![(1, "a"), (2, "b")];
        let map = Map::from_sorted_unique(entries, 4, 2).unwrap();

        assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
        assert_eq!(map.get_key_value(&3), None);
    }

    #[test]
    fn test_map_build() {
        let entries = vec![(5u64, "e"), (3, "c"), (1, "a"), (4, "d"), (2, "b")];
        let map = Map::build(entries, 4, 2).unwrap();

        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&1), Some(&"a"));
        assert_eq!(map.get(&5), Some(&"e"));

        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_map_first_last() {
        let entries: Vec<(u64, &str)> = vec![(10, "ten"), (20, "twenty"), (30, "thirty")];
        let map = Map::from_sorted_unique(entries, 4, 2).unwrap();

        assert_eq!(map.first_key_value(), Some((&10, &"ten")));
        assert_eq!(map.last_key_value(), Some((&30, &"thirty")));
    }

    #[test]
    fn test_map_iter() {
        let entries: Vec<(u64, i32)> = vec![(1, 10), (2, 20), (3, 30)];
        let map = Map::from_sorted_unique(entries, 4, 2).unwrap();

        let collected: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(collected, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn test_map_index_operator() {
        let entries: Vec<(u64, &str)> = vec![(1, "one"), (2, "two")];
        let map = Map::from_sorted_unique(entries, 4, 2).unwrap();

        assert_eq!(map[&1], "one");
        assert_eq!(map[&2], "two");
    }

    #[test]
    fn test_map_collect() {
        let map: Map<u64, i32> = vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&20));
    }

    #[test]
    fn test_map_get_mut() {
        let mut map: Map<u64, i32> = vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();

        if let Some(v) = map.get_mut(&2) {
            *v = 200;
        }
        assert_eq!(map.get(&2), Some(&200));

        assert!(map.get_mut(&999).is_none());
    }

    #[test]
    fn test_map_values_mut() {
        let mut map: Map<u64, i32> = vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();

        for v in map.values_mut() {
            *v *= 10;
        }

        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&200));
        assert_eq!(map.get(&3), Some(&300));
    }

    #[test]
    fn test_map_empty() {
        let map: Map<u64, i32> = Map::empty(64, 4);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&0), None);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn test_map_collect_empty() {
        let map: Map<u64, i32> = core::iter::empty().collect();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_map_insert() {
        let mut map = Map::build(vec![(1u64, "a"), (3, "c"), (5, "e")], 4, 2).unwrap();
        assert_eq!(map.len(), 3);

        assert_eq!(map.insert(2, "b"), None);
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&2), Some(&"b"));

        assert_eq!(map.insert(2, "B"), Some("b"));
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&2), Some(&"B"));

        assert_eq!(map.insert(4, "d"), None);
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_map_insert_into_empty() {
        let mut map: Map<u64, &str> = Map::empty(64, 4);
        assert_eq!(map.insert(42, "forty-two"), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&42), Some(&"forty-two"));
    }

    #[test]
    fn test_map_extend() {
        let mut map: Map<u64, i32> = vec![(1, 10), (2, 20)].into_iter().collect();
        map.extend(vec![(3, 30), (4, 40)]);
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&3), Some(&30));
        assert_eq!(map.get(&4), Some(&40));
    }

    #[test]
    fn test_map_extend_empty() {
        let mut map: Map<u64, i32> = Map::empty(64, 4);
        map.extend(vec![(3, 30), (1, 10), (2, 20)]);
        assert_eq!(map.len(), 3);
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }
}
