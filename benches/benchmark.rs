#![allow(clippy::float_cmp)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kd_tree::*;
use nalgebra::{Scalar, UnitVector3};
use rand::Rng;
use std::sync::LazyLock;

static POINTS: LazyLock<Vec<TestItem<f64>>> = LazyLock::new(|| {
    const N: usize = 1_000_000;

    let mut rng = rand::thread_rng();
    std::iter::repeat_with(move || TestItem {
        coord: UnitVector3::new_normalize(rng.gen()),
    })
    .take(N)
    .collect()
});

static KD_TREE: LazyLock<KdTree<TestItem<f64>, Vec<TestItem<f64>>>> =
    LazyLock::new(|| KdTree::build(POINTS.clone()));

fn bench_kdtree_construction(c: &mut Criterion) {
    c.bench_function("construct", |b| {
        let points = &*POINTS;

        b.iter_with_setup(move || points.clone(), KdTree::build);
    });
}

fn bench_kdtree_nearest_search(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    c.bench_function("nearest", |b| {
        let kdtree = &*KD_TREE;

        b.iter_with_setup(
            || rng.gen_range(0..kdtree.len()),
            move |i| kdtree.nearest(&kdtree[i]).unwrap(),
        );
    });
}

fn bench_kdtree_k_nearest_search(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    c.bench_function("nearests", |b| {
        let kd_tree = &*KD_TREE;

        b.iter_with_setup(
            || rng.gen_range(0..kd_tree.len()),
            move |i| kd_tree.nearests_arr::<5>(&kd_tree[i]),
        );
    });
}

fn bench_kdtree_within_radius(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    for radius in &[0.05, 0.1, 0.2, 0.4] {
        c.bench_with_input(
            BenchmarkId::new("within_radius", radius),
            radius,
            |b, radius| {
                let kd_tree = &*KD_TREE;

                b.iter_with_setup(
                    || rng.gen_range(0..kd_tree.len()),
                    move |i| kd_tree.within_radius(&kd_tree[i], *radius),
                );
            },
        );
    }
}

criterion_group!(
    benches,
    bench_kdtree_construction,
    bench_kdtree_nearest_search,
    bench_kdtree_k_nearest_search,
    bench_kdtree_within_radius
);
criterion_main!(benches);

#[derive(Debug, Clone, Copy, PartialEq)]
struct TestItem<T: Scalar> {
    coord: UnitVector3<T>,
}
impl KdPoint for TestItem<f64> {
    type Scalar = f64;
    const DIM: usize = 3;
    fn at(&self, k: usize) -> f64 {
        self.coord[k]
    }
    fn distance_metric(&self, other: &Self) -> Self::Scalar {
        let diff = *self.coord - *other.coord;
        diff.dot(&diff)
    }
}
