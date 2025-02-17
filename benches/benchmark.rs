#![allow(clippy::float_cmp)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kd_tree::*;
use nalgebra::{Scalar, UnitVector3};

fn bench_kdtree_construction(c: &mut Criterion) {
    for log10n in &[2, 3, 4] {
        c.bench_with_input(
            BenchmarkId::new("construct", log10n),
            log10n,
            |b, log10n| {
                let points = gen_points3d(10usize.pow(*log10n));
                b.iter_with_setup(move || points.clone(), KdTree::build);
            },
        );
    }
}

fn bench_kdtree_nearest_search(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for log10n in &[2, 3, 4] {
        c.bench_with_input(BenchmarkId::new("nearest", log10n), log10n, |b, log10n| {
            let kdtree = KdTree::build(gen_points3d(10usize.pow(*log10n)));
            b.iter_with_setup(
                || rng.gen::<usize>() % kdtree.len(),
                |i| kdtree.nearest(&kdtree[i]).unwrap(),
            );
        });
    }
}

fn bench_kdtree_k_nearest_search(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    const N: usize = 1000000;
    let points = gen_points3d(N);
    let kd_tree = KdTree::build(points);
    c.bench_with_input(BenchmarkId::new("nearests", 4), &4, |b, _k| {
        b.iter_with_setup(
            || rng.gen::<usize>() % kd_tree.len(),
            |i| kd_tree.nearests_arr::<5>(&kd_tree[i]),
        );
    });
}

fn bench_kdtree_within_radius(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("within_radius");
    const N: usize = 100000;
    let points = gen_points3d(N);
    let kd_tree = KdTree::build(points);
    for radius in &[0.05, 0.1, 0.2, 0.4] {
        group.bench_with_input(BenchmarkId::new("kd_tree", radius), radius, |b, radius| {
            b.iter_with_setup(
                || rng.gen::<usize>() % N,
                |i| kd_tree.within_radius(&kd_tree[i], *radius),
            );
        });
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

fn gen_points3d(count: usize) -> Vec<TestItem<f64>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    std::iter::repeat_with(move || TestItem {
        coord: UnitVector3::new_normalize(rng.gen()),
    })
    .take(count)
    .collect()
}
