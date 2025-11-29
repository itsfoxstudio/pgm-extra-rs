use crate::index::Key;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment<K: Key> {
    pub key: K,
    pub slope: f64,
    pub intercept: f64,
}

impl<K: Key> Segment<K> {
    #[inline]
    pub fn new(key: K, slope: f64, intercept: f64) -> Self {
        Self {
            key,
            slope,
            intercept,
        }
    }

    #[inline(always)]
    pub fn predict(&self, key: K) -> usize {
        let diff = key.to_f64_fast() - self.key.to_f64_fast();
        let pos = self.intercept + self.slope * diff;
        pos.max(0.0) as usize
    }

    #[inline(always)]
    pub fn predict_f64(&self, key: K) -> f64 {
        let diff = key.to_f64_fast() - self.key.to_f64_fast();
        self.intercept + self.slope * diff
    }
}

impl<K: Key> Default for Segment<K> {
    fn default() -> Self {
        Self {
            key: K::default(),
            slope: 0.0,
            intercept: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_predict() {
        let seg = Segment::new(0u64, 1.0, 0.0);
        assert_eq!(seg.predict(0), 0);
        assert_eq!(seg.predict(10), 10);
        assert_eq!(seg.predict(100), 100);
    }

    #[test]
    fn test_segment_with_intercept() {
        let seg = Segment::new(10u64, 0.5, 5.0);
        assert_eq!(seg.predict(10), 5);
        assert_eq!(seg.predict(20), 10);
    }

    #[test]
    fn test_segment_size() {
        assert_eq!(core::mem::size_of::<Segment<u64>>(), 24);
        assert_eq!(core::mem::size_of::<Segment<u32>>(), 24);
    }
}
