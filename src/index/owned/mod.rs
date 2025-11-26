//! Owned-data PGM indices.
//!
//! These indices own their data, providing a more convenient API similar to
//! `BTreeSet` and `BTreeMap`.
//!
//! # Index Types
//!
//! - [`Dynamic`]: Mutable index supporting insertions and deletions (std-only)
//!
//! For read-only owned sets and maps, see:
//! - [`crate::Set`]: Read-only set, `BTreeSet` replacement
//! - [`crate::Map`]: Read-only map, `BTreeMap` replacement

#[cfg(feature = "std")]
mod dynamic;

#[cfg(feature = "std")]
pub use dynamic::Dynamic;
