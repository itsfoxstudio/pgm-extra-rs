//! External-keys PGM indices.
//!
//! These indices store only the learned model metadata. The actual data is stored
//! externally and passed to query methods as a slice.
//!
//! Use these when:
//! - You want to share the data between multiple indices
//! - You want fine control over data storage
//! - You need to update the data without rebuilding the index
//!
//! # Index Types
//!
//! - [`Static`]: Multi-level recursive index for best query performance
//! - [`OneLevel`]: Single-level index with lower memory overhead
//! - [`Cached`]: Multi-level index with hot-key cache
//!
//! # Example
//!
//! ```
//! use pgm_extra::index::external::{Static, OneLevel, Cached};
//!
//! let data: Vec<u64> = (0..10000).collect();
//!
//! // Multi-level for large datasets
//! let pgm = Static::new(&data, 64, 4).unwrap();
//! assert!(pgm.contains(&data, &5000));
//!
//! // Single-level for smaller datasets
//! let one = OneLevel::new(&data, 64).unwrap();
//! assert!(one.contains(&data, &5000));
//!
//! // Cached for hot-key workloads
//! let cached = Cached::new(&data, 64, 4).unwrap();
//! assert!(cached.contains(&data, &5000));
//! ```

mod cached;
mod one_level;
mod r#static;

pub use cached::Cached;
pub use one_level::OneLevel;
pub use r#static::Static;

use crate::index::key::Indexable;
use crate::util::ApproxPos;

/// Trait for external-keys PGM indices that operate over a sorted slice.
///
/// This trait provides a common interface for [`crate::index::external::Static`],
/// [`crate::index::external::OneLevel`], and [`crate::index::external::Cached`],
/// allowing generic code to work with any of them.
///
/// # Example
///
/// ```
/// use pgm_extra::index::external;
/// use pgm_extra::index::External;
/// use pgm_extra::index::key::Indexable;
///
/// fn search_any<T, I>(index: &I, data: &[T], value: &T) -> bool
/// where
///     T: Indexable + Ord,
///     T::Key: Ord,
///     I: External<T>,
/// {
///     index.contains(data, value)
/// }
///
/// let data: Vec<u64> = (0..1000).collect();
/// let pgm = external::Static::new(&data, 64, 4).unwrap();
/// let one_level = external::OneLevel::new(&data, 64).unwrap();
///
/// assert!(search_any(&pgm, &data, &500));
/// assert!(search_any(&one_level, &data, &500));
/// ```
pub trait External<T: Indexable> {
    /// Get an approximate position for the value.
    fn search(&self, value: &T) -> ApproxPos;

    /// Find the first position where `data[pos] >= value`.
    fn lower_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord;

    /// Find the first position where `data[pos] > value`.
    fn upper_bound(&self, data: &[T], value: &T) -> usize
    where
        T: Ord;

    /// Check if the value exists in the sorted slice.
    fn contains(&self, data: &[T], value: &T) -> bool
    where
        T: Ord;

    /// Number of elements the index was built for.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of segments in the index.
    fn segments_count(&self) -> usize;

    /// Epsilon value used for the bottom level.
    fn epsilon(&self) -> usize;

    /// Approximate memory usage in bytes.
    fn size_in_bytes(&self) -> usize;
}
