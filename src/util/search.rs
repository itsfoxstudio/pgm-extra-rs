#[inline]
pub fn pgm_sub_eps(pos: usize, eps: usize) -> usize {
    pos.saturating_sub(eps)
}

#[inline]
pub fn pgm_add_eps(pos: usize, eps: usize, size: usize) -> usize {
    let upper = pos.saturating_add(eps).saturating_add(2);
    if upper >= size { size } else { upper }
}

#[inline]
pub fn binary_search_branchless<K: Ord>(keys: &[K], key: &K, lo: usize, hi: usize) -> usize {
    if lo >= hi || lo >= keys.len() {
        return lo;
    }

    let hi = hi.min(keys.len());
    let slice = &keys[lo..hi];

    let mut size = slice.len();
    if size == 0 {
        return lo;
    }

    let mut base = 0usize;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        base = if slice[mid] < *key { mid } else { base };
        size -= half;
    }

    let result = base + (slice[base] < *key) as usize;
    lo + result
}

#[inline]
pub fn linear_search<K: Ord>(keys: &[K], key: &K, lo: usize, hi: usize) -> usize {
    let hi = hi.min(keys.len());
    if hi <= lo {
        return lo;
    }

    let slice = &keys[lo..hi];
    let len = slice.len();
    let mut i = 0;

    while i + 4 <= len {
        if slice[i] >= *key {
            return lo + i;
        }
        if slice[i + 1] >= *key {
            return lo + i + 1;
        }
        if slice[i + 2] >= *key {
            return lo + i + 2;
        }
        if slice[i + 3] >= *key {
            return lo + i + 3;
        }
        i += 4;
    }

    while i < len {
        if slice[i] >= *key {
            return lo + i;
        }
        i += 1;
    }

    hi
}

const LINEAR_SEARCH_THRESHOLD: usize = 64;

#[inline]
pub fn linear_search_simd<K: Ord + Copy>(keys: &[K], key: &K, lo: usize, hi: usize) -> usize {
    let hi = hi.min(keys.len());
    if hi <= lo {
        return lo;
    }

    if let Some(pos) = &keys[lo..hi].iter().position(|k| k >= key) {
        return lo + pos;
    }

    hi
}

#[inline]
pub fn adaptive_search<K: Ord>(keys: &[K], key: &K, lo: usize, hi: usize) -> usize {
    let hi = hi.min(keys.len());
    if hi <= lo {
        return lo;
    }

    if (hi - lo) <= LINEAR_SEARCH_THRESHOLD {
        return linear_search(keys, key, lo, hi);
    }

    binary_search_branchless(keys, key, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_pgm_sub_eps() {
        assert_eq!(pgm_sub_eps(10, 3), 7);
        assert_eq!(pgm_sub_eps(3, 10), 0);
        assert_eq!(pgm_sub_eps(0, 5), 0);
    }

    #[test]
    fn test_pgm_add_eps() {
        assert_eq!(pgm_add_eps(10, 3, 100), 15);
        assert_eq!(pgm_add_eps(95, 10, 100), 100);
        assert_eq!(pgm_add_eps(0, 5, 10), 7);
    }

    #[test]
    fn test_binary_search_branchless() {
        let keys = vec![1, 3, 5, 7, 9, 11, 13, 15];
        assert_eq!(binary_search_branchless(&keys, &5, 0, 8), 2);
        assert_eq!(binary_search_branchless(&keys, &6, 0, 8), 3);
        assert_eq!(binary_search_branchless(&keys, &1, 0, 8), 0);
        assert_eq!(binary_search_branchless(&keys, &0, 0, 8), 0);
    }

    #[test]
    fn test_linear_search() {
        let keys = vec![1, 3, 5, 7, 9];
        assert_eq!(linear_search(&keys, &5, 0, 5), 2);
        assert_eq!(linear_search(&keys, &6, 0, 5), 3);
        assert_eq!(linear_search(&keys, &0, 0, 5), 0);
        assert_eq!(linear_search(&keys, &100, 0, 5), 5);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_linear_search_u64() {
        let keys: Vec<u64> = (0..100).collect();

        for &key in &[0u64, 50, 99, 100] {
            let expected = linear_search(&keys, &key, 0, 100);
            let simd_result = linear_search_simd(&keys, &key, 0, 100);
            assert_eq!(simd_result, expected, "Mismatch for key {}", key);
        }

        assert_eq!(linear_search_simd(&keys, &25, 10, 50), 25);
        assert_eq!(linear_search_simd(&keys, &5, 10, 50), 10);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_linear_search_u32() {
        let keys: Vec<u32> = (0..100).collect();

        for &key in &[0u32, 50, 99, 100] {
            let expected = linear_search(&keys, &key, 0, 100);
            let simd_result = linear_search_simd(&keys, &key, 0, 100);
            assert_eq!(simd_result, expected, "Mismatch for key {}", key);
        }
    }
}