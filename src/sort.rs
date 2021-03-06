use crate::KdPoint;
use std::cmp::Ordering;

#[allow(dead_code)]
pub fn kd_sort<P: KdPoint>(points: &mut [P])
where
    P::Scalar: Ord,
{
    kd_sort_by_key(points, P::dim(), |item, k| item.at(k))
}

#[allow(dead_code)]
pub fn kd_sort_by_ordered_float<P: KdPoint>(points: &mut [P])
where
    P::Scalar: num_traits::Float,
{
    kd_sort_by_key(points, P::dim(), |item, k| {
        ordered_float::OrderedFloat(item.at(k))
    })
}

#[allow(dead_code)]
pub fn kd_sort_by_key<T, Key: Ord>(
    items: &mut [T],
    dim: usize,
    kd_key: impl Fn(&T, usize) -> Key + Copy,
) {
    kd_sort_by(items, dim, |item1, item2, k| {
        kd_key(item1, k).cmp(&kd_key(item2, k))
    })
}

pub fn kd_sort_by<T>(
    items: &mut [T],
    dim: usize,
    kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy,
) {
    fn recurse<T>(
        items: &mut [T],
        axis: usize,
        dim: usize,
        kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy,
    ) {
        if items.len() >= 2 {
            pdqselect::select_by(items, items.len() / 2, |x, y| kd_compare(x, y, axis));
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            recurse(&mut items[..mid], axis, dim, kd_compare);
            recurse(&mut items[mid + 1..], axis, dim, kd_compare);
        }
    }
    recurse(items, 0, dim, kd_compare);
}
