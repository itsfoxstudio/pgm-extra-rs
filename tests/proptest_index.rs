use pgm_extra::index::{Builder, external::OneLevel, external::Static};
#[cfg(feature = "std")]
use pgm_extra::index::owned::Dynamic;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn pgm_lower_bound_matches_binary_search(
        keys in prop::collection::vec(0u64..1_000_000, 1..5000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let index = Static::new(&keys, 64, 4).unwrap();

        for probe in [0u64, 1, 500_000, 999_999, 1_000_001] {
            let got = index.lower_bound(&keys, &probe);
            let expected = keys.partition_point(|x| *x < probe);
            prop_assert_eq!(got, expected, "Failed for probe {}", probe);
        }
    }

    #[test]
    fn pgm_contains_correct(
        keys in prop::collection::vec(0u64..10_000, 1..1000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let index = Static::new(&keys, 32, 4).unwrap();

        for &key in &keys {
            prop_assert!(index.contains(&keys, &key), "Should contain {}", key);
        }

        for probe in [10_001u64, 20_000, 50_000] {
            if !keys.contains(&probe) {
                prop_assert!(!index.contains(&keys, &probe), "Should not contain {}", probe);
            }
        }
    }

    #[test]
    fn one_level_matches_multi_level(
        keys in prop::collection::vec(0u64..100_000, 100..2000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let one_level = OneLevel::new(&keys, 64).unwrap();
        let multi_level = Static::new(&keys, 64, 4).unwrap();

        let min_key = keys.first().copied().unwrap_or(0);
        let max_key = keys.last().copied().unwrap_or(0);
        let probes = vec![
            min_key,
            max_key,
            min_key.saturating_add(1),
            max_key.saturating_sub(1),
            (min_key + max_key) / 2,
        ];

        for probe in probes {
            let one_result = one_level.lower_bound(&keys, &probe);
            let multi_result = multi_level.lower_bound(&keys, &probe);
            prop_assert_eq!(one_result, multi_result, "Mismatch for probe {}", probe);
        }
    }

    #[test]
    fn epsilon_guarantee_holds(
        epsilon in 4usize..128,
        keys in prop::collection::vec(0u64..1_000_000, 100..5000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let index = Static::new(&keys, epsilon, epsilon.min(32)).unwrap();

        for (actual_pos, &key) in keys.iter().enumerate() {
            let approx = index.search(&key);

            let lo_ok = approx.lo <= actual_pos;
            let hi_ok = approx.hi >= actual_pos;

            prop_assert!(
                lo_ok && hi_ok,
                "Epsilon guarantee violated for key {} at pos {}: predicted range [{}, {}]",
                key, actual_pos, approx.lo, approx.hi
            );
        }
    }

    #[test]
    fn signed_integers_work(
        keys in prop::collection::vec(-50_000i64..50_000, 100..2000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let index = Static::new(&keys, 64, 4).unwrap();

        for &key in &keys {
            prop_assert!(index.contains(&keys, &key));
        }

        for (i, &key) in keys.iter().enumerate() {
            let pos = index.lower_bound(&keys, &key);
            prop_assert_eq!(pos, i, "Wrong position for key {}", key);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn dynamic_insert_contains(
        initial in prop::collection::vec(0u64..10_000, 0..500),
        inserts in prop::collection::vec(0u64..10_000, 1..100)
    ) {
        let mut initial = initial;
        initial.sort();
        initial.dedup();

        let mut index = if initial.is_empty() {
            Dynamic::new(32, 4)
        } else {
            Dynamic::from_sorted(initial.clone(), 32, 4).unwrap()
        };

        for &key in &inserts {
            index.insert(key);
        }

        for &key in &initial {
            prop_assert!(index.contains(&key), "Should contain initial key {}", key);
        }

        for &key in &inserts {
            prop_assert!(index.contains(&key), "Should contain inserted key {}", key);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn dynamic_remove_works(
        keys in prop::collection::vec(0u64..1000, 50..200)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.len() < 10 {
            return Ok(());
        }

        let mut index = Dynamic::from_sorted(keys.clone(), 16, 4).unwrap();

        let to_remove: Vec<u64> = keys.iter().step_by(3).copied().collect();

        for &key in &to_remove {
            index.remove(&key);
        }

        for &key in &to_remove {
            prop_assert!(!index.contains(&key), "Should not contain removed key {}", key);
        }

        for &key in &keys {
            if !to_remove.contains(&key) {
                prop_assert!(index.contains(&key), "Should still contain non-removed key {}", key);
            }
        }
    }

    #[test]
    fn builder_produces_valid_index(
        epsilon in 4usize..256,
        epsilon_rec in 2usize..64,
        keys in prop::collection::vec(0u64..100_000, 100..2000)
    ) {
        let mut keys = keys;
        keys.sort();
        keys.dedup();

        if keys.is_empty() {
            return Ok(());
        }

        let builder = Builder::new()
            .epsilon(epsilon)
            .epsilon_recursive(epsilon_rec);

        let index = builder.build(&keys).unwrap();

        prop_assert_eq!(index.epsilon(), epsilon);

        for &key in keys.iter().take(100) {
            prop_assert!(index.contains(&keys, &key));
        }
    }
}

#[test]
fn test_external_trait_generic() {
    use pgm_extra::index::External;

    fn search_any<I: External<u64>>(index: &I, data: &[u64], value: &u64) -> bool {
        index.contains(data, value)
    }

    let data: Vec<u64> = (0..1000).collect();
    let static_idx = Static::new(&data, 64, 4).unwrap();
    let one_level = OneLevel::new(&data, 64).unwrap();

    assert!(search_any(&static_idx, &data, &500));
    assert!(search_any(&one_level, &data, &500));
    assert!(!search_any(&static_idx, &data, &2000));
}

#[cfg(feature = "std")]
#[test]
fn test_dynamic_lower_bound() {
    let mut index: Dynamic<u64> = Dynamic::new(16, 4);
    index.extend(vec![10, 20, 30, 40, 50]);

    assert_eq!(index.lower_bound(&25), Some(30));
    assert_eq!(index.lower_bound(&10), Some(10));
    assert_eq!(index.lower_bound(&100), None);
}

#[cfg(feature = "std")]
#[test]
fn test_dynamic_range() {
    let mut index: Dynamic<u64> = Dynamic::new(16, 4);
    index.extend(0..100);

    let range_items: Vec<u64> = index.range(10..20).collect();
    assert_eq!(range_items, (10..20).collect::<Vec<_>>());
}