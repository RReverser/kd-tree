use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use kd_tree::KdTree;
use rayon::prelude::*;
use std::hint::black_box;

type XYZCoord = nalgebra::Point3<f64>;

fn gen_points3d(n: usize) -> Vec<XYZCoord> {
    std::iter::repeat_with(|| nalgebra::Vector3::new_random().into())
        .take(n)
        .collect()
}

#[library_benchmark(setup = gen_points3d)]
#[bench::small(1_000)]
#[bench::large(1_000_000)]
fn build(points: Vec<XYZCoord>) {
    black_box(KdTree::build(points));
}

fn build_kdtree(n: usize) -> KdTree<XYZCoord, Vec<XYZCoord>> {
    KdTree::build(gen_points3d(n))
}

#[library_benchmark(setup = build_kdtree)]
#[bench::small(1_000)]
#[bench::large(1_000_000)]
fn knn_graph(points: KdTree<XYZCoord, Vec<XYZCoord>>) {
    points.par_iter().for_each(|point| {
        black_box(points.nearests::<4>(point));
    });
}

library_benchmark_group!(
    name = benches;
    benchmarks = build, knn_graph
);

main!(library_benchmark_groups = benches);
