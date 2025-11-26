use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pgm_extra::{
    index::Segment, index::external::Cached, index::external::OneLevel, index::external::Static,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn format_bytes(bytes: usize) -> String {
    format!("{:>14.9} MB", (bytes as f64) / (1024.0 * 1024.0))
}

fn generate_uniform_data(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<u64> = (0..n).map(|_| rng.r#gen()).collect();
    data.sort();
    data.dedup();
    data
}

fn generate_dense_data(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn generate_sparse_data(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i * 1000).collect()
}

fn generate_queries(data: &[u64], num_queries: usize, lookup_ratio: f64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_lookups = (num_queries as f64 * lookup_ratio) as usize;

    let mut queries = Vec::with_capacity(num_queries);

    for _ in 0..num_lookups {
        let idx = rng.gen_range(0..data.len());
        queries.push(data[idx]);
    }

    let min_key = data.first().copied().unwrap_or(0);
    let max_key = data.last().copied().unwrap_or(u64::MAX);
    for _ in num_lookups..num_queries {
        queries.push(rng.gen_range(min_key..=max_key));
    }

    queries
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for &n in &[100_000usize, 1_000_000] {
        let data = generate_uniform_data(n, 42);

        group.throughput(Throughput::Elements(n as u64));

        // PGM-Index construction
        for &epsilon in &[16, 64, 256] {
            group.bench_with_input(
                BenchmarkId::new(format!("pgm_eps{}", epsilon), n),
                &(&data, epsilon),
                |b, (data, eps)| {
                    b.iter(|| Static::new(black_box(*data), *eps, 4).unwrap());
                },
            );
        }

        group.bench_with_input(BenchmarkId::new("one_level_eps64", n), &data, |b, data| {
            b.iter(|| OneLevel::new(black_box(data), 64).unwrap());
        });

        // BTreeSet construction
        group.bench_with_input(BenchmarkId::new("btreeset", n), &data, |b, data| {
            b.iter(|| {
                let set: BTreeSet<u64> = data.iter().copied().collect();
                black_box(set)
            });
        });

        // BTreeMap construction (with dummy values)
        group.bench_with_input(BenchmarkId::new("btreemap", n), &data, |b, data| {
            b.iter(|| {
                let map: BTreeMap<u64, u64> = data.iter().map(|&k| (k, k)).collect();
                black_box(map)
            });
        });

        // HashMap construction
        group.bench_with_input(BenchmarkId::new("hashmap", n), &data, |b, data| {
            b.iter(|| {
                let map: HashMap<u64, u64> = data.iter().map(|&k| (k, k)).collect();
                black_box(map)
            });
        });

        // Vec sort (baseline - data is already sorted, but we clone and sort)
        group.bench_with_input(BenchmarkId::new("vec_sort", n), &data, |b, data| {
            b.iter(|| {
                let mut v = data.clone();
                v.sort();
                black_box(v)
            });
        });
    }

    group.finish();
}

fn bench_datetime_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("datetime_queries");

    for &n in &[100_000usize, 1_000_000] {
        let now = std::time::SystemTime::now();
        let offsets = generate_uniform_data(n, 42);

        let data_u64: Vec<u64> = (offsets.into_iter())
            .map(std::time::Duration::from_nanos)
            .map(|offset| now.checked_add(offset).unwrap_or(now))
            .filter_map(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .collect();

        let queries_u64 = generate_queries(&data_u64, 10_000, 0.5, 123);

        // Pre-build all indexes
        let btree_set: BTreeSet<u64> = data_u64.iter().copied().collect();
        let btree_map: BTreeMap<u64, usize> =
            data_u64.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let hash_map: HashMap<u64, usize> =
            data_u64.iter().enumerate().map(|(i, &k)| (k, i)).collect();

        group.throughput(Throughput::Elements(queries_u64.len() as u64));

        // PGM-Index queries
        for &epsilon in &[16, 64, 256] {
            let index = Static::new(&data_u64, epsilon, 4).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("pgm_eps{}", epsilon), n),
                &(&data_u64, &queries_u64, &index),
                |b, (data, queries, index)| {
                    b.iter(|| {
                        let mut sum = 0usize;
                        for q in *queries {
                            sum += index.lower_bound(data, q);
                        }
                        black_box(sum)
                    });
                },
            );
        }

        // Slice binary search (baseline)
        group.bench_with_input(
            BenchmarkId::new("binary_search", n),
            &(&data_u64, &queries_u64),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += data.partition_point(|x| x < q);
                    }
                    black_box(sum)
                });
            },
        );

        // BTreeSet contains
        group.bench_with_input(
            BenchmarkId::new("btreeset", n),
            &(&btree_set, &queries_u64),
            |b, (set, queries)| {
                b.iter(|| {
                    let mut count = 0usize;
                    for q in *queries {
                        if set.contains(q) {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );

        // BTreeMap get
        group.bench_with_input(
            BenchmarkId::new("btreemap", n),
            &(&btree_map, &queries_u64),
            |b, (map, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        if let Some(&idx) = map.get(q) {
                            sum += idx;
                        }
                    }
                    black_box(sum)
                });
            },
        );

        // HashMap get (expected fastest for exact lookups)
        group.bench_with_input(
            BenchmarkId::new("hashmap", n),
            &(&hash_map, &queries_u64),
            |b, (map, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        if let Some(&idx) = map.get(q) {
                            sum += idx;
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_point_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_queries");

    for &n in &[100_000usize, 1_000_000] {
        let data = generate_uniform_data(n, 42);
        let queries = generate_queries(&data, 10_000, 0.5, 123);

        // Pre-build all indexes
        let btree_set: BTreeSet<u64> = data.iter().copied().collect();
        let btree_map: BTreeMap<u64, usize> =
            data.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let hash_map: HashMap<u64, usize> = data.iter().enumerate().map(|(i, &k)| (k, i)).collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        // PGM-Index queries
        for &epsilon in &[16, 64, 256] {
            let index = Static::new(&data, epsilon, 4).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("pgm_eps{}", epsilon), n),
                &(&data, &queries, &index),
                |b, (data, queries, index)| {
                    b.iter(|| {
                        let mut sum = 0usize;
                        for q in *queries {
                            sum += index.lower_bound(data, q);
                        }
                        black_box(sum)
                    });
                },
            );
        }

        // Slice binary search (baseline)
        group.bench_with_input(
            BenchmarkId::new("binary_search", n),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += data.partition_point(|x| x < q);
                    }
                    black_box(sum)
                });
            },
        );

        // BTreeSet contains
        group.bench_with_input(
            BenchmarkId::new("btreeset", n),
            &(&btree_set, &queries),
            |b, (set, queries)| {
                b.iter(|| {
                    let mut count = 0usize;
                    for q in *queries {
                        if set.contains(q) {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );

        // BTreeMap get
        group.bench_with_input(
            BenchmarkId::new("btreemap", n),
            &(&btree_map, &queries),
            |b, (map, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        if let Some(&idx) = map.get(q) {
                            sum += idx;
                        }
                    }
                    black_box(sum)
                });
            },
        );

        // HashMap get (expected fastest for exact lookups)
        group.bench_with_input(
            BenchmarkId::new("hashmap", n),
            &(&hash_map, &queries),
            |b, (map, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        if let Some(&idx) = map.get(q) {
                            sum += idx;
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_range_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_queries");

    let n = 1_000_000;
    let data = generate_uniform_data(n, 42);
    let queries = generate_queries(&data, 1_000, 0.5, 789);

    let btree_set: BTreeSet<u64> = data.iter().copied().collect();
    let index = Static::new(&data, 64, 4).unwrap();

    group.throughput(Throughput::Elements(queries.len() as u64));

    // PGM-Index range scan (find + scan 100 elements)
    group.bench_function("pgm_range_100", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &queries {
                let start = index.lower_bound(&data, q);
                for &val in data.iter().skip(start).take(100) {
                    sum += val as usize;
                }
            }
            black_box(sum)
        });
    });

    // Binary search + range scan
    group.bench_function("binary_search_range_100", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &queries {
                let start = data.partition_point(|x| x < q);
                for &val in data.iter().skip(start).take(100) {
                    sum += val as usize;
                }
            }
            black_box(sum)
        });
    });

    // BTreeSet range iterator
    group.bench_function("btreeset_range_100", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for q in &queries {
                for &val in btree_set.range(q..).take(100) {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");

    let n = 1_000_000;
    let data = generate_uniform_data(n, 42);
    let queries = generate_queries(&data, 10_000, 0.5, 456);

    let index = Static::new(&data, 64, 4).unwrap();
    let cached_index = Cached::new(&data, 64, 4).unwrap();
    let btree_set: BTreeSet<u64> = data.iter().copied().collect();
    let hash_set: std::collections::HashSet<u64> = data.iter().copied().collect();

    group.throughput(Throughput::Elements(queries.len() as u64));

    group.bench_function("pgm_contains", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for q in &queries {
                if index.contains(&data, q) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("pgm_cached_contains", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for q in &queries {
                if cached_index.contains(&data, q) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("binary_search_contains", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for q in &queries {
                if data.binary_search(q).is_ok() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("btreeset_contains", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for q in &queries {
                if btree_set.contains(q) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function("hashset_contains", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for q in &queries {
                if hash_set.contains(q) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

fn bench_memory_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        let data = generate_uniform_data(n, 42);

        // Measure actual memory usage using stats_alloc
        let reg = Region::new(GLOBAL);
        let index_16 = Static::new(&data, 16, 4).unwrap();
        let index_16_stats = reg.change();

        let reg = Region::new(GLOBAL);
        let index_64 = Static::new(&data, 64, 4).unwrap();
        let index_64_stats = reg.change();

        let reg = Region::new(GLOBAL);
        let index_256 = Static::new(&data, 256, 4).unwrap();
        let index_256_stats = reg.change();

        let reg = Region::new(GLOBAL);
        let btreeset: BTreeSet<u64> = data.iter().copied().collect();
        let btreeset_stats = reg.change();

        let reg = Region::new(GLOBAL);
        let btreemap: BTreeMap<u64, usize> =
            data.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let btreemap_stats = reg.change();

        let reg = Region::new(GLOBAL);
        let hashmap: HashMap<u64, usize> = data.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let hashmap_stats = reg.change();

        let index_16_bytes = index_16_stats.bytes_allocated;
        let index_64_bytes = index_64_stats.bytes_allocated;
        let index_256_bytes = index_256_stats.bytes_allocated;
        let btreeset_bytes = btreeset_stats.bytes_allocated;
        let btreemap_bytes = btreemap_stats.bytes_allocated;
        let hashmap_bytes = hashmap_stats.bytes_allocated;

        println!("\n=== Memory usage for n={} ===", n);
        println!(
            "Data (Vec<u64>):     {:>10} ({:.2} B/elem)",
            format_bytes(data.len() * 8),
            8.0
        );
        println!(
            "PGM (eps=16):        {:>10} ({:.4} B/elem, {} segments)",
            format_bytes(index_16_bytes),
            index_16_bytes as f64 / n as f64,
            index_16.segments_count()
        );
        println!(
            "PGM (eps=64):        {:>10} ({:.4} B/elem, {} segments)",
            format_bytes(index_64_bytes),
            index_64_bytes as f64 / n as f64,
            index_64.segments_count()
        );
        println!(
            "PGM (eps=256):       {:>10} ({:.4} B/elem, {} segments)",
            format_bytes(index_256_bytes),
            index_256_bytes as f64 / n as f64,
            index_256.segments_count()
        );
        println!(
            "BTreeSet:            {:>10} ({:.2} B/elem)",
            format_bytes(btreeset_bytes),
            btreeset_bytes as f64 / n as f64
        );
        println!(
            "BTreeMap:            {:>10} ({:.2} B/elem)",
            format_bytes(btreemap_bytes),
            btreemap_bytes as f64 / n as f64
        );
        println!(
            "HashMap:             {:>10} ({:.2} B/elem)",
            format_bytes(hashmap_bytes),
            hashmap_bytes as f64 / n as f64
        );

        // Keep variables alive
        black_box(&btreeset);
        black_box(&btreemap);
        black_box(&hashmap);

        // Dummyh benchmark to include in group
        group.bench_with_input(BenchmarkId::new("pgm_eps64_build", n), &data, |b, data| {
            b.iter(|| Static::new(black_box(data), 64, 4).unwrap());
        });
    }

    group.finish();
}

fn bench_data_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_distributions");

    let n = 500_000;

    let dense_data = generate_dense_data(n);
    let sparse_data = generate_sparse_data(n);
    let uniform_data = generate_uniform_data(n, 42);

    let datasets = [
        ("dense", &dense_data),
        ("sparse", &sparse_data),
        ("uniform", &uniform_data),
    ];

    for (name, data) in datasets {
        let queries = generate_queries(data, 10_000, 0.5, 789);
        let index = Static::new(data, 64, 4).unwrap();
        let btree: BTreeSet<u64> = data.iter().copied().collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("pgm", name),
            &(data, &queries, &index),
            |b, (data, queries, index)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += index.lower_bound(data, q);
                    }
                    black_box(sum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_search", name),
            &(data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += data.partition_point(|x| x < q);
                    }
                    black_box(sum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("btreeset", name),
            &(&btree, &queries),
            |b, (set, queries)| {
                b.iter(|| {
                    let mut count = 0usize;
                    for q in *queries {
                        if set.contains(q) {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

fn bench_segment_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_prediction");

    let float_seg: Segment<u64> = Segment::new(1000, 0.7, 500.0);

    let queries: Vec<u64> = (0..10_000).map(|i| 1000 + i * 100).collect();

    group.throughput(Throughput::Elements(queries.len() as u64));

    group.bench_function("float_segment", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for &q in &queries {
                sum += float_seg.predict(q);
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effectiveness");

    let n = 1_000_000;
    let data = generate_uniform_data(n, 42);

    let hot_keys: Vec<u64> = data.iter().step_by(10000).copied().collect();
    let hot_queries: Vec<u64> = (0..10_000).map(|i| hot_keys[i % hot_keys.len()]).collect();

    let random_queries = generate_queries(&data, 10_000, 0.5, 789);

    let index = Static::new(&data, 64, 4).unwrap();
    let cached_index = Cached::new(&data, 64, 4).unwrap();
    let hash_map: HashMap<u64, usize> = data.iter().enumerate().map(|(i, &k)| (k, i)).collect();

    group.throughput(Throughput::Elements(10_000));

    group.bench_function("pgm_hot_keys", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &hot_queries {
                sum += index.lower_bound(&data, q);
            }
            black_box(sum)
        });
    });

    group.bench_function("pgm_cached_hot_keys", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &hot_queries {
                sum += cached_index.lower_bound(&data, q);
            }
            black_box(sum)
        });
    });

    group.bench_function("hashmap_hot_keys", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &hot_queries {
                if let Some(&idx) = hash_map.get(q) {
                    sum += idx;
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("pgm_random_keys", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &random_queries {
                sum += index.lower_bound(&data, q);
            }
            black_box(sum)
        });
    });

    group.bench_function("pgm_cached_random_keys", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &random_queries {
                sum += cached_index.lower_bound(&data, q);
            }
            black_box(sum)
        });
    });

    group.finish();
}

#[cfg(feature = "simd")]
fn bench_simd_search(c: &mut Criterion) {
    use pgm_extra::util::search::{adaptive_search, linear_search, linear_search_simd};

    let mut group = c.benchmark_group("simd_search");

    for &range_size in &[16usize, 32, 64, 128] {
        let data: Vec<u64> = (0..range_size as u64).collect();
        let queries: Vec<u64> = (0..10_000).map(|i| (i % range_size) as u64).collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("linear_scalar", range_size),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += linear_search(data, q, 0, data.len());
                    }
                    black_box(sum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear_simd", range_size),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += linear_search_simd(data, q, 0, data.len());
                    }
                    black_box(sum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adaptive", range_size),
            &(&data, &queries),
            |b, (data, queries)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += adaptive_search(data, q, 0, data.len());
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

fn bench_epsilon_tradeoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("epsilon_tradeoff");

    let n = 1_000_000;
    let data = generate_uniform_data(n, 42);
    let queries = generate_queries(&data, 10_000, 0.5, 999);

    for &epsilon in &[4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let index = Static::new(&data, epsilon, epsilon.min(64)).unwrap();

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("query", epsilon),
            &(&data, &queries, &index),
            |b, (data, queries, index)| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for q in *queries {
                        sum += index.lower_bound(data, q);
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_parallel_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_construction");

    for &n in &[1_000_000usize, 10_000_000] {
        let data = generate_uniform_data(n, 42);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("sequential", n), &data, |b, data| {
            b.iter(|| Static::new(black_box(data), 64, 4).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("parallel", n), &data, |b, data| {
            b.iter(|| Static::new_parallel(black_box(data), 64, 4).unwrap());
        });
    }

    group.finish();
}

#[cfg(all(feature = "parallel", feature = "simd"))]
criterion_group!(
    benches,
    bench_construction,
    bench_point_queries,
    bench_datetime_queries,
    bench_range_queries,
    bench_contains,
    bench_memory_size,
    bench_data_distributions,
    bench_epsilon_tradeoff,
    bench_segment_prediction,
    bench_cache_effectiveness,
    bench_parallel_construction,
    bench_simd_search,
);

#[cfg(all(feature = "parallel", not(feature = "simd")))]
criterion_group!(
    benches,
    bench_construction,
    bench_point_queries,
    bench_datetime_queries,
    bench_range_queries,
    bench_contains,
    bench_memory_size,
    bench_data_distributions,
    bench_epsilon_tradeoff,
    bench_segment_prediction,
    bench_cache_effectiveness,
    bench_parallel_construction,
);

#[cfg(all(not(feature = "parallel"), feature = "simd"))]
criterion_group!(
    benches,
    bench_construction,
    bench_point_queries,
    bench_datetime_queries,
    bench_range_queries,
    bench_contains,
    bench_memory_size,
    bench_data_distributions,
    bench_epsilon_tradeoff,
    bench_segment_prediction,
    bench_cache_effectiveness,
    bench_simd_search,
);

#[cfg(all(not(feature = "parallel"), not(feature = "simd")))]
criterion_group!(
    benches,
    bench_construction,
    bench_point_queries,
    bench_datetime_queries,
    bench_range_queries,
    bench_contains,
    bench_memory_size,
    bench_data_distributions,
    bench_epsilon_tradeoff,
    bench_segment_prediction,
    bench_cache_effectiveness,
);

criterion_main!(benches);