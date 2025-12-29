//! # PGM-Extra
//!
//! A Rust implementation of the PGM-index, a learned index structure that uses
//! piecewise linear models to approximate the cumulative distribution function
//! of sorted data for fast lookups.
//!
//! ## Quick Start
//!
//! For most use cases, use [`Set`] or [`Map`] which own their data:
//!
//! ```rust
//! use pgm_extra::{Set, Map};
//!
//! // Set (like BTreeSet)
//! let set: Set<u64> = (0..10000).collect();
//! assert!(set.contains(&5000));
//!
//! // Map (like BTreeMap)
//! let map: Map<u64, &str> = vec![(1, "one"), (2, "two")].into_iter().collect();
//! assert_eq!(map.get(&1), Some(&"one"));
//!
//! // Works with strings too!
//! let words: Set<&str> = vec!["apple", "banana", "cherry"].into_iter().collect();
//! assert!(words.contains(&"banana"));
//! ```
//!
//! ## Index Types
//!
//! ### Owned collections (recommended for most uses)
//!
//! - [`Set`]: BTreeSet-like set optimized for read-heavy workloads
//! - [`Map`]: BTreeMap-like map optimized for read-heavy workloads
//!
//! ### External-keys indices (for advanced use cases)
//!
//! - [`Static`]: Multi-level recursive index (also available as [`index::external::Static`])
//! - [`OneLevel`]: Single-level index for smaller datasets
//! - [`Cached`]: Multi-level with hot-key cache for repeated lookups
//! - [`Dynamic`]: Mutable index with insert/delete (requires `std` feature)
//!
//! ## Features
//!
//! - `std` (default): Enables `Dynamic` index and other std-only features
//! - `parallel`: Enables parallel index construction
//! - `simd`: Enables SIMD-accelerated search routines
//! - `serde`: Enables serialization/deserialization with serde
//! - `rkyv`: Enables zero-copy serialization/deserialization with rkyv
//!
//! ## Performance
//!
//! The PGM-index provides O(log n / log epsilon) query time with significantly
//! smaller space overhead compared to traditional B-trees. Construction is O(n).

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod collections;
pub mod error;
pub mod index;
pub mod util;

// Primary exports
pub use collections::map::Map;
pub use collections::set::Set;

// Re-export commonly used types
pub use error::Error;

// Re-export index types at crate root for convenience
pub use index::external::Cached as CachedIndex;
pub use index::external::OneLevel as OneLevelIndex;
pub use index::external::Static as StaticIndex;

// Re-export index types at crate root for convenience
pub use index::external::Cached;
pub use index::external::OneLevel;
pub use index::external::Static;
#[cfg(feature = "std")]
pub use index::owned::Dynamic;
#[cfg(feature = "std")]
pub use index::owned::Dynamic as DynamicIndex;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_integration_basic() {
        let data: Vec<u64> = (0..10000).collect();
        let index = index::Builder::new()
            .epsilon(64)
            .epsilon_recursive(4)
            .build(&data)
            .unwrap();

        for i in (0..10000).step_by(100) {
            let pos = index.lower_bound(&data, &i);
            assert_eq!(pos, i as usize);
        }
    }

    #[test]
    fn test_integration_signed() {
        let data: Vec<i64> = (-5000..5000).collect();
        let index = index::external::Static::new(&data, 64, 4).unwrap();

        for i in (-5000i64..5000).step_by(100) {
            let pos = index.lower_bound(&data, &i);
            let expected = (i + 5000) as usize;
            assert_eq!(pos, expected, "Failed for key {}", i);
        }

        assert!(index.contains(&data, &-5000));
        assert!(!index.contains(&data, &5000));
    }

    #[test]
    fn test_integration_sparse() {
        let data: Vec<u64> = (0..1000).map(|i| i * i).collect();
        let index = index::external::Static::new(&data, 32, 4).unwrap();

        for (i, &key) in data.iter().enumerate() {
            let pos = index.lower_bound(&data, &key);
            assert_eq!(pos, i, "Failed for key {} at index {}", key, i);
        }
    }

    #[test]
    fn test_missing_keys() {
        let data: Vec<u64> = (0..100).map(|i| i * 2).collect();
        let index = index::external::Static::new(&data, 8, 4).unwrap();

        let pos = index.lower_bound(&data, &1);
        assert_eq!(pos, 1);

        let pos = index.lower_bound(&data, &199);
        assert_eq!(pos, 100);
    }

    #[test]
    fn test_set_api() {
        let set: Set<u64> = (0..1000).collect();
        assert!(set.contains(&500));
        assert!(!set.contains(&1001));
    }

    #[test]
    fn test_map_api() {
        let map: Map<u64, i32> = (0..100).map(|i| (i, i as i32 * 2)).collect();
        assert_eq!(map.get(&50), Some(&100));
    }
}
