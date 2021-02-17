use std::cmp::Ordering;

use crate::split_at_mid::split_at_mid_mut;
use crate::KdPoint;

// A wrapper similar to OrderedFloat but for generic types.
// Moves any incomparable values to the end and treats them as equal.
struct OrdHelper<T: PartialOrd>(T);

impl<T: PartialOrd> Ord for OrdHelper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            // Couldn't compare values.
            // One of them is NaN-like - find out which.
            #[allow(clippy::eq_op)]
            match (self.0 != self.0, other.0 != other.0) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => Ordering::Equal,
            }
        })
    }
}

impl<T: PartialOrd> PartialOrd for OrdHelper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> PartialEq for OrdHelper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: PartialOrd> Eq for OrdHelper<T> {}

pub fn kd_sort_by<T: KdPoint>(items: &mut [T]) {
    fn recurse<T: KdPoint>(items: &mut [T], mut axis: usize) {
        if items.len() >= 2 {
            pdqselect::select_by_key(items, items.len() / 2, |item| OrdHelper(item.at(axis)));
            axis = (axis + 1) % T::dim();
            let (before, _, after) = split_at_mid_mut(items);
            recurse(before, axis);
            recurse(after, axis);
        }
    }
    recurse(items, 0);
}
