//! PGM-Index implementations.
//!
//! This module contains various PGM-Index implementations organized by their
//! data ownership model:
//!
//! ## External-keys indices ([`external`])
//!
//! These indices store only the learned model metadata. Data is stored externally
//! and passed to query methods:
//!
//! - [`external::Static`]: Multi-level recursive index (primary implementation)
//! - [`external::OneLevel`]: Single-level index for smaller datasets
//! - [`external::Cached`]: Multi-level with hot-key cache for repeated lookups
//!
//! ## Owned-data indices ([`owned`])
//!
//! These indices own their data, similar to standard library collections:
//!
//! - [`owned::Dynamic`]: Mutable index with insert/delete (requires `std` feature)
//!
//! For read-only owned collections, see [`crate::Set`] and [`crate::Map`].
//!
//! ## Traits
//!
//! - [`crate::index::External`]: Common interface for external-keys indices
//! - [`crate::index::key::Indexable`]: Trait for types that can be indexed

pub(crate) mod builder;
pub mod key;

pub mod external;
pub mod model;
#[cfg(feature = "std")]
pub mod owned;
pub mod segment;

pub use builder::Builder;
pub use external::External;
pub use key::{Indexable, Key};
pub use segment::Segment;

pub use external::{Cached, OneLevel, Static};

#[cfg(feature = "std")]
pub use owned::Dynamic;