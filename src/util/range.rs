use core::ops::{Bound, RangeBounds};

/// Convert a start bound to an index using lower_bound/upper_bound.
#[inline]
pub fn start_bound_to_idx<K, F, G>(
    bound: Bound<&K>,
    len: usize,
    lower_bound: F,
    upper_bound: G,
) -> usize
where
    F: Fn(&K) -> usize,
    G: Fn(&K) -> usize,
{
    match bound {
        Bound::Included(key) => lower_bound(key).min(len),
        Bound::Excluded(key) => upper_bound(key).min(len),
        Bound::Unbounded => 0,
    }
}

/// Convert an end bound to an index using lower_bound/upper_bound.
#[inline]
pub fn end_bound_to_idx<K, F, G>(
    bound: Bound<&K>,
    len: usize,
    lower_bound: F,
    upper_bound: G,
) -> usize
where
    F: Fn(&K) -> usize,
    G: Fn(&K) -> usize,
{
    match bound {
        Bound::Included(key) => upper_bound(key).min(len),
        Bound::Excluded(key) => lower_bound(key).min(len),
        Bound::Unbounded => len,
    }
}

/// Compute the (start, end) indices for a range.
#[inline]
pub fn range_to_indices<K, R, F, G>(
    range: R,
    len: usize,
    lower_bound: F,
    upper_bound: G,
) -> (usize, usize)
where
    R: RangeBounds<K>,
    F: Fn(&K) -> usize + Copy,
    G: Fn(&K) -> usize + Copy,
{
    let start = start_bound_to_idx(range.start_bound(), len, lower_bound, upper_bound);
    let end = end_bound_to_idx(range.end_bound(), len, lower_bound, upper_bound);
    (start, end.max(start))
}
