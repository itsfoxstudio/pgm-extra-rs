#![cfg(feature = "rkyv")]

use pgm_extra::index::external::{Cached, Static};
use pgm_extra::{Map, Set};
use rkyv::rancor::Error;
use rkyv::{access, deserialize, to_bytes};

#[test]
fn rkyv_roundtrip_static() {
    let data: Vec<u64> = (0..1000).collect();
    let index = Static::new(&data, 64, 4).expect("failed to create index");

    let bytes = to_bytes::<Error>(&index).expect("failed to serialize");
    let archived =
        access::<rkyv::Archived<Static<u64>>, Error>(&bytes).expect("failed to access archive");
    let de: Static<u64> =
        deserialize::<Static<u64>, Error>(archived).expect("failed to deserialize");

    assert_eq!(index.len(), de.len());
    assert_eq!(index.height(), de.height());
    assert_eq!(index.segments_count(), de.segments_count());

    for key in [0u64, 10, 500, 999] {
        assert_eq!(index.lower_bound(&data, &key), de.lower_bound(&data, &key));
    }
}

#[test]
fn rkyv_roundtrip_cached() {
    let data: Vec<u64> = (0..1000).collect();
    let index = Cached::new(&data, 64, 4).expect("failed to create cached index");

    let bytes = to_bytes::<Error>(&index).expect("failed to serialize");
    let archived =
        access::<rkyv::Archived<Cached<u64>>, Error>(&bytes).expect("failed to access archive");
    let de: Cached<u64> =
        deserialize::<Cached<u64>, Error>(archived).expect("failed to deserialize");

    assert_eq!(index.len(), de.len());

    for key in [0u64, 10, 500, 999] {
        assert_eq!(index.lower_bound(&data, &key), de.lower_bound(&data, &key));
    }
}

#[test]
fn rkyv_roundtrip_set() {
    let set: Set<u64> = (0..100).collect();

    let bytes = to_bytes::<Error>(&set).expect("failed to serialize");
    let archived =
        access::<rkyv::Archived<Set<u64>>, Error>(&bytes).expect("failed to access archive");
    let de: Set<u64> = deserialize::<Set<u64>, Error>(archived).expect("failed to deserialize");

    assert_eq!(set.len(), de.len());
    for v in 0..100 {
        assert_eq!(set.contains(&v), de.contains(&v));
    }
}

#[test]
fn rkyv_roundtrip_map() {
    let map: Map<u64, String> = (0..100).map(|i| (i, format!("v{i}"))).collect();

    let bytes = to_bytes::<Error>(&map).expect("failed to serialize");
    let archived = access::<rkyv::Archived<Map<u64, String>>, Error>(&bytes)
        .expect("failed to access archive");
    let de: Map<u64, String> =
        deserialize::<Map<u64, String>, Error>(archived).expect("failed to deserialize");

    assert_eq!(map.len(), de.len());
    for k in 0..100 {
        assert_eq!(map.get(&k), de.get(&k));
    }
}

#[test]
fn rkyv_roundtrip_signed_integers() {
    let data: Vec<i64> = (-500..500).collect();
    let index = Static::new(&data, 64, 4).expect("failed to create index");

    let bytes = to_bytes::<Error>(&index).expect("failed to serialize");
    let archived =
        access::<rkyv::Archived<Static<i64>>, Error>(&bytes).expect("failed to access archive");
    let de: Static<i64> =
        deserialize::<Static<i64>, Error>(archived).expect("failed to deserialize");

    for key in [-500i64, -100, 0, 100, 499] {
        assert_eq!(index.lower_bound(&data, &key), de.lower_bound(&data, &key));
    }
}

#[test]
fn rkyv_access_without_deserialize_static() {
    let data: Vec<u64> = (0..1000).collect();
    let index = Static::new(&data, 64, 4).expect("failed to create index");

    let bytes = to_bytes::<Error>(&index).expect("failed to serialize");

    let archived =
        access::<rkyv::Archived<Static<u64>>, Error>(&bytes).expect("failed to access archive");
    let archived2 =
        access::<rkyv::Archived<Static<u64>>, Error>(&bytes).expect("failed to access archive");

    assert!(core::ptr::eq(archived, archived2));
}

#[test]
fn rkyv_access_without_deserialize_cached() {
    let data: Vec<u64> = (0..1000).collect();
    let index = Cached::new(&data, 64, 4).expect("failed to create cached index");

    let bytes = to_bytes::<Error>(&index).expect("failed to serialize");

    let archived =
        access::<rkyv::Archived<Cached<u64>>, Error>(&bytes).expect("failed to access archive");
    let archived2 =
        access::<rkyv::Archived<Cached<u64>>, Error>(&bytes).expect("failed to access archive");

    assert!(core::ptr::eq(archived, archived2));
}

#[test]
fn rkyv_access_without_deserialize_set() {
    let set: Set<u64> = (0..100).collect();

    let bytes = to_bytes::<Error>(&set).expect("failed to serialize");

    let archived =
        access::<rkyv::Archived<Set<u64>>, Error>(&bytes).expect("failed to access archive");
    let archived2 =
        access::<rkyv::Archived<Set<u64>>, Error>(&bytes).expect("failed to access archive");

    assert!(core::ptr::eq(archived, archived2));
}

#[test]
fn rkyv_access_without_deserialize_map() {
    let map: Map<u64, String> = (0..100).map(|i| (i, format!("v{i}"))).collect();

    let bytes = to_bytes::<Error>(&map).expect("failed to serialize");

    let archived = access::<rkyv::Archived<Map<u64, String>>, Error>(&bytes)
        .expect("failed to access archive");
    let archived2 = access::<rkyv::Archived<Map<u64, String>>, Error>(&bytes)
        .expect("failed to access archive");

    assert!(core::ptr::eq(archived, archived2));
}
