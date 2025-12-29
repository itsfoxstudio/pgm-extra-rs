# PGM Extra

[![Crates.io](https://img.shields.io/crates/v/pgm-extra.svg)](https://crates.io/crates/pgm-extra)
[![Documentation](https://docs.rs/pgm-extra/badge.svg)](https://docs.rs/pgm-extra)
[![License](https://img.shields.io/crates/l/pgm-extra.svg)](LICENSE)

A high-performance Rust implementation of the PGM-index (Piecewise Geometric Model index).
Includes multiple variants of the index, as well as drop-in replacements for `BTreeSet` and `BTreeMap`.

Based on the paper: [The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds](https://dl.acm.org/doi/10.14778/3389133.3389135) by Paolo Ferragina and Giorgio Vinciguerra.


## Features

- **High-Performance**: Outperforms traditional B-trees on point queries while using less memory
- **Cache Efficient**: Adaptive linear / binary search based on epsilon for optimal cache utilization
- **Multi-Level Indexing**: Recursive index structure with configurable epsilon at each level
- **Generic**: Works with all integer types (signed and unsigned)
- **Dynamic Variant**: Support for insertions and deletions with auto-rebuild
- **Parallel Construction**: Optional parallel index building using Rayon
- **No-std support**: Enables usage on embedded / WASM


## Performance

- **Query time**: O(log n / log ε)
- **Space**: O(n / ε) segments
- **Construction**: O(n) time
- **Expected outcome**: speedup of 2-4x over binary search on large datasets

*ε is the error bound, which controls the trade-off between index size and query performance.*


### Benchmark Results

Benchmarks run on Apple M1 Pro. Dataset: 1M random u64 keys.

#### Query Performance (10K queries, 1M keys)

| Data Structure | Time (µs) | Throughput | vs BTreeSet |
|----------------|-----------|------------|-------------|
| HashMap | 68.2 | 146.6 M/s | 12.5x faster |
| Binary Search | 266.8 | 37.5 M/s | 3.2x faster |
| **PGM (ε=256)** | **489.5** | **20.4 M/s** | **1.7x faster** |
| **PGM (ε=64)** | **520.3** | **19.2 M/s** | **1.6x faster** |
| **PGM (ε=16)** | **656.6** | **15.2 M/s** | **1.3x faster** |
| BTreeSet | 851.6 | 11.7 M/s | baseline |
| BTreeMap | 906.1 | 11.0 M/s | 0.9x |

#### Memory Overhead (index only, 1M keys)

| Data Structure | Memory | B/elem | vs BTreeSet |
|----------------|--------|--------|-------------|
| **PGM (ε=256)** | **7.7 MB** | **8.09** | **3.2x smaller** |
| **PGM (ε=64)** | **8.0 MB** | **8.38** | **3.1x smaller** |
| **PGM (ε=16)** | **9.1 MB** | **9.55** | **2.7x smaller** |
| BTreeSet | 25.0 MB | 26.18 | baseline |
| HashMap | 34.0 MB | 35.65 | 1.4x larger |
| BTreeMap | 40.2 MB | 42.18 | 1.6x larger |

#### Range Queries (1K queries, scan 100 elements)

| Method | Time (µs) | vs BTreeSet |
|--------|-----------|-------------|
| **PGM + scan** | **120.5** | **1.4x faster** |
| Binary Search + scan | 129.9 | 1.3x faster |
| BTreeSet range | 174.3 | baseline |


## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pgm-extra = "0.1"
```

Available features:
- `std`: Enables `Dynamic` index and other std-only features
- `simd`: Enables SIMD for linear search
- `parallel`: Enables parallel index construction
- `serde`: Enables serialization/deserialization with serde
- `rkyv`: Enables zero-copy serialization/deserialization with rkyv


## Quick Start

For most use cases, use `Set` or `Map` which own their data:

```rust
use pgm_extra::{Set, Map};

// Set (replacement for BTreeSet, includes same read & write interface)
//     (Even though it supports write operations, it's recommended to be used as read-only, due to index rebuild on mutation)
let set: Set<u64> = (0..1_000_000).collect();
assert!(set.contains(&500_000));

// Map (replacement for BTreeMap, includes same read & write interface)
//     (Even though it supports write operations, it's recommended to be used as read-only, due to index rebuild on mutation)
let map: Map<u64, &str> = vec![(1, "one"), (2, "two")].into_iter().collect();
assert_eq!(map.get(&1), Some(&"one"));
```


### Using index types directly

For advanced use cases where you manage data separately:

```rust
use pgm_extra::index::{Static, Builder};

let data: Vec<u64> = (0..1_000_000).collect();

// Build index with custom epsilon
let index = Builder::new()
    .epsilon(128)          // Error bound for data level
    .epsilon_recursive(16) // Error bound for recursive levels
    .build(&data)
    .unwrap();

// Point query - find position of key
let key = 500_000u64;
let pos = index.lower_bound(&data, &key);
assert_eq!(data[pos], key);

// Check existence
assert!(index.contains(&data, &key));
assert!(!index.contains(&data, &1_000_001));

// Get approximate position with bounds
let approx = index.search(&key);
println!("Position ~{} in range [{}, {}]", approx.pos, approx.lo, approx.hi);
```


## Index Types


### Static

The default multi-level recursive index.
Best for large datasets where query performance is critical.

```rust
use pgm_extra::index::Static;

let data: Vec<u64> = (0..100_000).collect();
let index = Static::new(&data, 64, 4).unwrap();

println!("Height: {}", index.height());
println!("Segments: {}", index.segments_count());
println!("Size: {} bytes", index.size_in_bytes());
```


### OneLevel

Simple single-level variant.
Lower overhead for smaller datasets.

```rust
use pgm_extra::index::OneLevel;

let data: Vec<u64> = (0..10_000).collect();
let index = OneLevel::new(&data, 64).unwrap();
```


### Dynamic (requires `std` feature)

Dynamic variant where mutations are required.
Supports insertions and deletions with automatic rebuilding.

```rust
use pgm_extra::index::Dynamic;

let mut index = Dynamic::new(64, 4);

// Insert keys (mutation)
index.extend(0..1000);

assert!(index.contains(&500));

index.remove(&500);
assert!(!index.contains(&500));

// Iterate in sorted order
for key in index.iter() {
    println!("{}", key);
}
```


## Epsilon Selection

The `epsilon` parameter controls the trade-off between index size and query performance:


| Epsilon | Segments | Query Speed | Use Case |
|---------|----------|-------------|----------|
| 4-16 | Many | Fastest | Cache-critical workloads |
| 32-64 | Moderate | Fast | Default, good balance |
| 128-256 | Few | Moderate | Memory-constrained |
| 512+ | Minimal | Slower | Extreme memory savings |


## Parallel Construction

Enable the `parallel` feature for multi-threaded index building:

```rust
use pgm_extra::index::Static;

let data: Vec<u64> = (0..10_000_000).collect();
let index = Static::new_parallel(&data, 64, 4).expect("failed to build index");
```


## Benchmarks

Run benchmarks with:

```bash
just bench
```


## Signed Integers

The index correctly handles signed integers by mapping them to unsigned space.
As shown in the example below, there are no collisions on absolute values of the keys.

```rust
use pgm_extra::index::Static;

let data: Vec<i64> = (-5000..5000).collect();
let index = Static::new(&data, 64, 4).unwrap();

assert!(index.contains(&data, &-100));
assert!(index.contains(&data, &100));

assert!(index.contains(&data, &-5000));
assert!(!index.contains(&data, &5000));
```


## FAQ

### Why is SIMD not manually implemented?

Manual SIMD (e.g., using AVX2 or NEON intrinsics) was tested and found to offer **no performance benefit** over the current implementation.
LLVM's ability to **auto-vectorize** the loop makes it possible to write high-performance code without unsafe blocks or architecture-specific code.


## References

- [Original C++ Implementation](https://github.com/gvinciguerra/PGM-index)
- [The PGM-index Paper](https://dl.acm.org/doi/10.14778/3389133.3389135)


## License

MIT License

Copyright (c) 2025 Fox Studio (Oskar Cieslik)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
