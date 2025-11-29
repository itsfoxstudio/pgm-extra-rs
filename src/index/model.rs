use alloc::vec;
use alloc::vec::Vec;

use crate::index::{Key, Segment};

#[inline(always)]
fn key_diff<K: Key>(a: K, b: K) -> f64 {
    a.to_f64_fast() - b.to_f64_fast()
}

struct SegmentBuilder {
    end_idx: usize,
    slope_lo: f64,
    slope_hi: f64,
}

impl SegmentBuilder {
    fn new() -> Self {
        Self {
            end_idx: 0,
            slope_lo: f64::NEG_INFINITY,
            slope_hi: f64::INFINITY,
        }
    }

    fn add_point(&mut self, x_diff: f64, y: f64, epsilon: f64) -> bool {
        if x_diff == 0.0 {
            self.end_idx += 1;
            return true;
        }

        let y_actual = y;
        let slope_min = (y_actual - epsilon) / x_diff;
        let slope_max = (y_actual + epsilon) / x_diff;

        let new_lo = self.slope_lo.max(slope_min);
        let new_hi = self.slope_hi.min(slope_max);

        if new_lo > new_hi {
            return false;
        }

        self.slope_lo = new_lo;
        self.slope_hi = new_hi;
        self.end_idx += 1;
        true
    }

    fn get_slope(&self) -> f64 {
        if self.slope_lo.is_infinite() && self.slope_hi.is_infinite() {
            0.0
        } else if self.slope_lo.is_infinite() {
            self.slope_hi
        } else if self.slope_hi.is_infinite() {
            self.slope_lo
        } else {
            (self.slope_lo + self.slope_hi) / 2.0
        }
    }
}

pub fn build_segments<K: Key>(keys: &[K], epsilon: usize) -> Vec<Segment<K>> {
    if keys.is_empty() {
        return Vec::new();
    }

    if keys.len() == 1 {
        return vec![Segment::new(keys[0], 0.0, 0.0)];
    }

    let eps = epsilon as f64;
    let mut segments = Vec::with_capacity(keys.len() / epsilon.max(1) + 1);

    let mut segment_start_idx = 0usize;
    let mut first_key = keys[0];
    let mut builder = SegmentBuilder::new();

    for (i, &key) in keys.iter().enumerate() {
        let x_diff = key_diff(key, first_key);
        let y = (i - segment_start_idx) as f64;

        if !builder.add_point(x_diff, y, eps) {
            let slope = builder.get_slope();
            let segment = Segment::new(first_key, slope, segment_start_idx as f64);
            segments.push(segment);

            segment_start_idx = i;
            first_key = key;
            builder = SegmentBuilder::new();
            builder.add_point(0.0, 0.0, eps);
        }
    }

    let slope = builder.get_slope();
    let segment = Segment::new(first_key, slope, segment_start_idx as f64);
    segments.push(segment);

    segments
}

#[cfg(feature = "parallel")]
pub fn build_segments_parallel<K: Key>(keys: &[K], epsilon: usize) -> Vec<Segment<K>> {
    use rayon::prelude::*;

    const PARALLEL_THRESHOLD: usize = 100_000;

    if keys.len() < PARALLEL_THRESHOLD {
        return build_segments(keys, epsilon);
    }

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = keys.len().div_ceil(num_threads);

    let chunks: Vec<_> = keys.chunks(chunk_size).collect();
    let offsets: Vec<usize> = chunks
        .iter()
        .scan(0usize, |acc, chunk| {
            let offset = *acc;
            *acc += chunk.len();
            Some(offset)
        })
        .collect();

    let partial_results: Vec<Vec<Segment<K>>> = chunks
        .par_iter()
        .zip(offsets.par_iter())
        .map(|(chunk, &offset)| build_segments_with_offset(chunk, epsilon, offset))
        .collect();

    partial_results.into_iter().flatten().collect()
}

#[cfg(feature = "parallel")]
fn build_segments_with_offset<K: Key>(
    keys: &[K],
    epsilon: usize,
    offset: usize,
) -> Vec<Segment<K>> {
    if keys.is_empty() {
        return Vec::new();
    }

    if keys.len() == 1 {
        return vec![Segment::new(keys[0], 0.0, offset as f64)];
    }

    let eps = epsilon as f64;
    let mut segments = Vec::with_capacity(keys.len() / epsilon.max(1) + 1);

    let mut segment_start_idx = 0usize;
    let mut first_key = keys[0];
    let mut builder = SegmentBuilder::new();

    for (i, &key) in keys.iter().enumerate() {
        let x_diff = key_diff(key, first_key);
        let y = (i - segment_start_idx) as f64;

        if !builder.add_point(x_diff, y, eps) {
            let slope = builder.get_slope();
            let segment = Segment::new(first_key, slope, (offset + segment_start_idx) as f64);
            segments.push(segment);

            segment_start_idx = i;
            first_key = key;
            builder = SegmentBuilder::new();
            builder.add_point(0.0, 0.0, eps);
        }
    }

    let slope = builder.get_slope();
    let segment = Segment::new(first_key, slope, (offset + segment_start_idx) as f64);
    segments.push(segment);

    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_build_segments_uniform() {
        let keys: Vec<u64> = (0..1000).collect();
        let segments = build_segments(&keys, 8);

        assert!(!segments.is_empty());
        for seg in &segments {
            assert!(seg.slope >= 0.0);
        }
    }

    #[test]
    fn test_build_segments_single() {
        let keys: Vec<u64> = vec![42];
        let segments = build_segments(&keys, 8);
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_build_segments_empty() {
        let keys: Vec<u64> = vec![];
        let segments = build_segments(&keys, 8);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_build_segments_quadratic() {
        let keys: Vec<u64> = (0..1000).map(|i| i * i).collect();
        let segments = build_segments(&keys, 32);

        assert!(
            segments.len() > 1,
            "Quadratic data should need multiple segments"
        );
    }

    #[test]
    fn test_epsilon_guarantee() {
        let keys: Vec<u64> = (0..10000).map(|i| i * 7).collect();
        let epsilon = 16;
        let segments = build_segments(&keys, epsilon);

        for (i, &key) in keys.iter().enumerate() {
            let seg_idx = segments.partition_point(|s| s.key <= key).saturating_sub(1);
            let seg = &segments[seg_idx];
            let predicted = seg.predict(key);
            let error = (predicted as i64 - i as i64).unsigned_abs() as usize;
            assert!(
                error <= epsilon + 1,
                "Error {} > epsilon {} for key {} at index {}",
                error,
                epsilon,
                key,
                i
            );
        }
    }
}
