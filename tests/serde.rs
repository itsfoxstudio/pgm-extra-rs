#![cfg(feature = "serde")]

use pgm_extra::{Map, Set};
use pgm_extra::index::external::Static;

#[test]
fn serde_roundtrip_static() {
    let data: Vec<u64> = (0..1000).collect();
    let index = Static::new(&data, 64, 4).unwrap();

    let json = serde_json::to_string(&index).unwrap();
    let de: Static<u64> = serde_json::from_str(&json).unwrap();

    assert_eq!(index.len(), de.len());
    assert_eq!(index.height(), de.height());
    assert_eq!(index.segments_count(), de.segments_count());

    for key in [0u64, 10, 500, 999] {
        assert_eq!(
            index.lower_bound(&data, &key),
            de.lower_bound(&data, &key)
        );
    }
}

#[test]
fn serde_roundtrip_set() {
    let set: Set<u64> = (0..100).collect();
    let json = serde_json::to_string(&set).unwrap();
    let de: Set<u64> = serde_json::from_str(&json).unwrap();

    assert_eq!(set.len(), de.len());
    for v in 0..100 {
        assert_eq!(set.contains(&v), de.contains(&v));
    }
}

#[test]
fn serde_roundtrip_map() {
    let map: Map<u64, String> = (0..100).map(|i| (i, format!("v{i}"))).collect();
    let json = serde_json::to_string(&map).unwrap();
    let de: Map<u64, String> = serde_json::from_str(&json).unwrap();

    assert_eq!(map.len(), de.len());
    for k in 0..100 {
        assert_eq!(map.get(&k), de.get(&k));
    }
}

#[test]
fn serde_roundtrip_set_strings() {
    let words: Vec<String> = vec!["apple", "banana", "cherry", "date", "elderberry"]
        .into_iter()
        .map(String::from)
        .collect();
    let set: Set<String> = words.into_iter().collect();
    let json = serde_json::to_string(&set).unwrap();
    let de: Set<String> = serde_json::from_str(&json).unwrap();

    assert_eq!(set.len(), de.len());
    assert!(de.contains(&String::from("banana")));
    assert!(!de.contains(&String::from("fig")));
}

#[test]
fn serde_roundtrip_signed_integers() {
    let data: Vec<i64> = (-500..500).collect();
    let index = Static::new(&data, 64, 4).unwrap();

    let json = serde_json::to_string(&index).unwrap();
    let de: Static<i64> = serde_json::from_str(&json).unwrap();

    for key in [-500i64, -100, 0, 100, 499] {
        assert_eq!(
            index.lower_bound(&data, &key),
            de.lower_bound(&data, &key)
        );
    }
}
