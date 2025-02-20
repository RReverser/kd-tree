use std::cmp::Ordering;

use crate::split_at_mid::split_at_mid;
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
        compare: impl Fn(T::Scalar, usize) -> Ordering + Copy,
    ) {
        let Some((lower, item, upper)) = split_at_mid(kdtree) else {
            for item in kdtree {
                if (0..T::DIM).all(|i| compare(item.at(i), i) == Ordering::Equal) {
                    on_item(item);
                }
            }
            return;
        };
        let mut next_axis = axis + 1;
        if next_axis == T::DIM {
            next_axis = 0;
        }
        match compare(item.at(axis), axis) {
            Ordering::Equal => {
                if (1..T::DIM)
                    .map(move |i| (axis + i) % T::DIM)
                    .all(move |i| compare(item.at(i), i) == Ordering::Equal)
                {
                    on_item(item);
                }
                recurse(on_item, lower, next_axis, compare);
                recurse(on_item, upper, next_axis, compare);
            }
            Ordering::Less => {
                recurse(on_item, upper, next_axis, compare);
            }
            Ordering::Greater => {
                recurse(on_item, lower, next_axis, compare);
            }
        }
    }
    recurse(&mut on_item, kdtree, 0, compare);
}
