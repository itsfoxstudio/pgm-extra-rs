use alloc::vec::Vec;
use core::cell::Cell;

const CACHE_SIZE: usize = 64;

#[derive(Clone, Copy)]
struct CacheEntry<K: Copy> {
    key: K,
    pos: usize,
    valid: bool,
}

impl<K: Copy + Default> Default for CacheEntry<K> {
    fn default() -> Self {
        Self {
            key: K::default(),
            pos: 0,
            valid: false,
        }
    }
}

pub trait FastHash: Copy + Eq {
    fn fast_hash(&self) -> usize;
}

macro_rules! impl_fast_hash_int {
    ($($t:ty),*) => {
        $(
            impl FastHash for $t {
                #[inline(always)]
                fn fast_hash(&self) -> usize {
                    let x = *self as u64;
                    let x = x.wrapping_mul(0x9E3779B97F4A7C15);
                    (x >> 58) as usize
                }
            }
        )*
    };
}

impl_fast_hash_int!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl FastHash for u128 {
    #[inline(always)]
    fn fast_hash(&self) -> usize {
        let x = (*self as u64).wrapping_mul(0x9E3779B97F4A7C15);
        (x >> 58) as usize
    }
}

impl FastHash for i128 {
    #[inline(always)]
    fn fast_hash(&self) -> usize {
        let x = (*self as u64).wrapping_mul(0x9E3779B97F4A7C15);
        (x >> 58) as usize
    }
}

pub struct HotCache<K: Copy + Default + FastHash> {
    entries: Vec<Cell<CacheEntry<K>>>,
}

impl<K: Copy + Default + FastHash> core::fmt::Debug for HotCache<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HotCache")
            .field("size", &self.entries.len())
            .finish()
    }
}

impl<K: Copy + Default + FastHash> HotCache<K> {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(CACHE_SIZE);
        for _ in 0..CACHE_SIZE {
            entries.push(Cell::new(CacheEntry::default()));
        }
        Self { entries }
    }

    #[inline(always)]
    fn hash_key(key: &K) -> usize {
        key.fast_hash()
    }

    #[inline]
    pub fn lookup(&self, key: &K) -> Option<usize> {
        let idx = Self::hash_key(key);
        if idx >= self.entries.len() {
            return None;
        }

        let entry = self.entries[idx].get();

        if entry.valid && entry.key == *key {
            Some(entry.pos)
        } else {
            None
        }
    }

    #[inline]
    pub fn insert(&self, key: K, pos: usize) {
        let idx = Self::hash_key(&key);
        if idx >= self.entries.len() {
            return;
        }

        self.entries[idx].set(CacheEntry {
            key,
            pos,
            valid: true,
        });
    }

    pub fn clear(&self) {
        for cell in &self.entries {
            let mut entry = cell.get();
            entry.valid = false;
            cell.set(entry);
        }
    }
}

impl<K: Copy + Default + FastHash> Default for HotCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache: HotCache<u64> = HotCache::new();

        assert_eq!(cache.lookup(&42), None);

        cache.insert(42, 100);
        assert_eq!(cache.lookup(&42), Some(100));

        cache.insert(42, 200);
        assert_eq!(cache.lookup(&42), Some(200));
    }

    #[test]
    fn test_cache_miss() {
        let cache: HotCache<u64> = HotCache::new();
        cache.insert(42, 100);

        assert_eq!(cache.lookup(&43), None);
    }

    #[test]
    fn test_cache_collision() {
        let cache: HotCache<u64> = HotCache::new();

        cache.insert(0, 100);
        cache.insert(64, 200);

        let hit0 = cache.lookup(&0);
        let hit64 = cache.lookup(&64);

        assert!(hit0.is_some() && hit64.is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache: HotCache<u64> = HotCache::new();
        cache.insert(42, 100);
        cache.clear();
        assert_eq!(cache.lookup(&42), None);
    }
}
