/// Approximate position returned by index search.
///
/// When searching for a key, the index returns an approximate position
/// along with guaranteed bounds where the key must exist (if present).
///
/// The actual key position (if it exists) is guaranteed to be in `[lo, hi)`.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ApproxPos {
    /// The predicted position of the key.
    pub pos: usize,
    /// The lowest index guaranteed to contain the key if it exists.
    pub lo: usize,
    /// One past the highest index guaranteed to contain the key if it exists.
    pub hi: usize,
}

impl ApproxPos {
    #[inline]
    pub fn new(pos: usize, lo: usize, hi: usize) -> Self {
        Self { pos, lo, hi }
    }
}
