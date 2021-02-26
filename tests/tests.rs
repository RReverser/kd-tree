#![cfg(test)]
#![allow(clippy::float_cmp)]
use kd_tree::*;
use ordered_float::OrderedFloat;

#[test]
fn test_nearest() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build(vec(10000, |_| gen3d()));
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.nearest(&query).unwrap().item;
        let expected = kdtree
            .iter()
            .min_by_key(|p| ordered_float::OrderedFloat(squared_distance(p, &query)))
            .unwrap();
        assert_eq!(found, expected);
    }
}

#[test]
fn test_nearests() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build(vec(10000, |_| gen3d()));
    const NUM: usize = 5;
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.nearests(&query, NUM);
        assert_eq!(found.len(), NUM);
        for i in 1..found.len() {
            assert!(found[i - 1].distance_metric <= found[i].distance_metric);
        }
        let count = kdtree
            .iter()
            .filter(|p| squared_distance(p, &query) <= found[NUM - 1].distance_metric)
            .count();
        assert_eq!(count, NUM);
    }
}

#[test]
fn test_nearests_with_cond() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build(vec(10000, |_| gen3d()));
    const NUM: usize = 50;
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.nearests_with_cond(&query, NUM, |point| point[0] < 0.1);
        for point in &found {
            assert!(point.item[0] < 0.5);
        }
        for i in 1..found.len() {
            assert!(found[i - 1].distance_metric <= found[i].distance_metric);
        }
        let mut all = kdtree
            .iter()
            .map(|item| ItemAndDistance { item, distance_metric: squared_distance(item, &query) })
            .filter(|i_d| i_d.item[0] < 0.1 && i_d.distance_metric <= found.last().unwrap().distance_metric)
            .collect::<Vec<_>>();
        all.sort_by_key(|i_d| OrderedFloat::from(i_d.distance_metric));
        assert_eq!(found.len(), all.len().min(NUM), "Found: {:.3?}\nAll:  {:.3?}", found, all);
    }
}

#[test]
fn test_within() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build(vec(10000, |_| gen3d()));
    for _ in 0..100 {
        let mut p1 = gen3d();
        let mut p2 = gen3d();
        for k in 0..3 {
            if p1[k] > p2[k] {
                std::mem::swap(&mut p1[k], &mut p2[k]);
            }
        }
        let found = kdtree.within([&p1, &p2]);
        let count = kdtree
            .iter()
            .filter(|p| (0..3).all(|k| p1[k] <= p[k] && p[k] <= p2[k]))
            .count();
        assert_eq!(found.len(), count);
    }
}

#[test]
fn test_within_radius() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build(vec(10000, |_| gen3d()));
    const RADIUS: f64 = 0.1;
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.within_radius(&query, RADIUS);
        let count = kdtree
            .iter()
            .filter(|p| squared_distance(p, &query) < RADIUS * RADIUS)
            .count();
        assert_eq!(found.len(), count);
    }
}

fn squared_distance<T: num_traits::Num + Copy>(p1: &[T; 3], p2: &[T; 3]) -> T {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

fn random3d_generator() -> impl FnMut() -> [f64; 3] {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    move || [rng.gen(), rng.gen(), rng.gen()]
}

fn vec<T>(count: usize, mut f: impl FnMut(usize) -> T) -> Vec<T> {
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(f(i));
    }
    items
}
