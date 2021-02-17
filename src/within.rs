use std::cmp::Ordering;

use crate::split_at_mid::split_at_mid;
use crate::KdPoint;

pub fn kd_within_by_cmp<T: KdPoint>(
    kdtree: &[T],
    compare: impl Fn(T::Scalar, usize) -> Ordering + Copy,
    final_check: impl Fn(&T) -> bool + Copy,
) -> Vec<&T> {
    fn recurse<'a, T: KdPoint>(
        results: &mut Vec<&'a T>,
        kdtree: &'a [T],
        axis: usize,
        compare: impl Fn(T::Scalar, usize) -> Ordering + Copy,
        final_check: impl Fn(&T) -> bool + Copy,
    ) {
        let (lower, item, upper) = split_at_mid(kdtree);
        let item = match item {
            Some(item) => item,
            None => return,
        };
        let next_axis = (axis + 1) % T::dim();
        match compare(item.at(axis), axis) {
            Ordering::Equal => {
                if (1..T::dim())
                    .map(|i| (axis + i) % T::dim())
                    .all(|i| compare(item.at(i), i) == Ordering::Equal)
                    && final_check(item)
                {
                    results.push(item);
                }
                recurse(results, lower, next_axis, compare, final_check);
                recurse(results, upper, next_axis, compare, final_check);
            }
            Ordering::Less => {
                recurse(results, upper, next_axis, compare, final_check);
            }
            Ordering::Greater => {
                recurse(results, lower, next_axis, compare, final_check);
            }
        }
    }
    let mut results = Vec::new();
    recurse(&mut results, kdtree, 0, compare, final_check);
    results
}
