#![cfg(test)]
#![allow(clippy::float_cmp)]
use kd_tree::*;

use nalgebra::Const;
use nalgebra::{proptest::vector, Point3};
use prop::array::uniform2;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::prop_assert_eq;
use test_strategy::proptest;

fn point_strategy() -> impl Strategy<Value = Point3<f64>> {
    vector(-1.0..=1.0, Const).prop_map(Point3::from)
}

#[proptest]
fn test_nearest(
    #[strategy(vec(point_strategy(), 0..1_000))] points: Vec<Point3<f64>>,
    #[strategy(point_strategy())] query: Point3<f64>,
) {
    let kdtree = KdTree::build(points);

    let found = kdtree.nearest(&query).unwrap().item;
    let expected = kdtree
        .iter()
        .min_by_key(|p| ordered_float::OrderedFloat(nalgebra::distance_squared(p, &query)))
        .unwrap();
    prop_assert_eq!(found, expected);
}

#[proptest]
fn test_nearests(
    #[strategy(vec(point_strategy(), 0..1_000))] points: Vec<Point3<f64>>,
    #[strategy(point_strategy())] query: Point3<f64>,
    #[strategy(0..5_usize)] num: usize,
) {
    const NUM: usize = 5;

    let kdtree = KdTree::build(points);

    let found = kdtree.nearests(&query, num);
    prop_assert_eq!(found.len(), NUM);
    for pair in found.windows(2) {
        assert!(pair[0].distance_metric <= pair[1].distance_metric);
    }
    let count = kdtree
        .iter()
        .filter(|p| nalgebra::distance_squared(p, &query) <= found[NUM - 1].distance_metric)
        .count();
    prop_assert_eq!(count, NUM);
}

#[proptest]
fn test_within(
    #[strategy(vec(point_strategy(), 0..1_000))] points: Vec<Point3<f64>>,
    #[strategy(uniform2(point_strategy()))] mut p: [Point3<f64>; 2],
) {
    let [ref mut p0, ref mut p1] = &mut p;

    p0.iter_mut()
        .zip(p1.iter_mut())
        .filter(|(a, b)| a > b)
        .for_each(|(a, b)| std::mem::swap(a, b));

    let kdtree = KdTree::build(points);
    let found = kdtree.within(p.each_ref());
    let count = kdtree
        .iter()
        .filter(|f| {
            f.iter()
                .zip(p[0].iter().zip(p[1].iter()))
                .all(|(f, (p1, p2))| (p1..=p2).contains(&f))
        })
        .count();
    prop_assert_eq!(found.len(), count);
}

#[proptest]
fn test_within_radius(
    #[strategy(vec(point_strategy(), 0..1_000))] points: Vec<Point3<f64>>,
    #[strategy(point_strategy())] query: Point3<f64>,
    radius: f64,
) {
    let kdtree = KdTree::build(points);

    let found = kdtree.within_radius(&query, radius);
    let count = kdtree
        .iter()
        .filter(|p| nalgebra::distance_squared(p, &query) < radius.powi(2))
        .count();
    prop_assert_eq!(found.len(), count);
}
