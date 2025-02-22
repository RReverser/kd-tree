use std::cmp::Ordering;

use crate::split_at_mid::split_at_mid;
use crate::{KdPoint, KdScalar};

pub fn kd_within_by_cmp<'a, T: KdPoint>(
    mut on_item: impl FnMut(&'a T),
    kdtree: &'a [T],
    compare: impl Fn(KdScalar<T>, usize) -> Ordering + Copy,
) {
    fn recurse<'a, T: KdPoint>(
        on_item: &mut impl FnMut(&'a T),
        kdtree: &'a [T],
        axis: usize,
        compare: impl Fn(KdScalar<T>, usize) -> Ordering + Copy,
    ) {
        let Some((lower, item, upper)) = split_at_mid(kdtree) else {
            for item in kdtree {
                add_item_if_within(item, compare, on_item);
            }
            return;
        };
        let next_axis = T::next_axis(axis);
        match compare(item.at(axis), axis) {
            Ordering::Equal => {
                add_item_if_within(item, compare, on_item);
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

fn add_item_if_within<'a, T: KdPoint>(
    item: &'a T,
    compare: impl Fn(KdScalar<T>, usize) -> Ordering + Copy,
    on_item: &mut impl FnMut(&'a T),
) {
    if item
        .as_point()
        .into_iter()
        .enumerate()
        .all(|(i, x)| compare(x, i).is_eq())
    {
        on_item(item);
    }
}
