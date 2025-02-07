use std::cmp::Ordering;

use crate::KdPoint;

pub fn kd_within_by_cmp<'a, T: KdPoint>(
    mut on_item: impl FnMut(&'a T),
    kdtree: &'a [T],
    compare: impl Fn(T::Scalar, usize) -> Ordering + Copy,
) {
    fn recurse<'a, T: KdPoint>(
        on_item: &mut impl FnMut(&'a T),
        kdtree: &'a [T],
        axis: usize,
        k: usize,
        compare: impl Fn(T::Scalar, usize) -> Ordering + Copy,
    ) {
        let item = match kdtree.get(k) {
            Some(item) => item,
            None => return,
        };
        let next_axis = (axis + 1) % T::DIM;
        match compare(item.at(axis), axis) {
            Ordering::Equal => {
                if (1..T::DIM)
                    .map(move |i| (axis + i) % T::DIM)
                    .all(move |i| compare(item.at(i), i) == Ordering::Equal)
                {
                    on_item(item);
                }
                recurse(on_item, kdtree, next_axis, 2 * k + 1, compare);
                recurse(on_item, kdtree, next_axis, 2 * k + 2, compare);
            }
            Ordering::Less => {
                recurse(on_item, kdtree, next_axis, 2 * k + 2, compare);
            }
            Ordering::Greater => {
                recurse(on_item, kdtree, next_axis, 2 * k + 1, compare);
            }
        }
    }
    recurse(&mut on_item, kdtree, 0, 0, compare);
}
