use std::cmp::Ordering;
use crate::KdPoint;

// A wrapper similar to OrderedFloat but for generic types.
// Moves any incomparable values to the end and treats them as equal.
pub struct OrdHelper<T: PartialOrd>(pub T);

impl<T: PartialOrd> Ord for OrdHelper<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(
            #[cold]
            move || {
                // Couldn't compare values.
                // One of them is NaN-like and should go to the end.
                #[allow(clippy::eq_op)]
                match (self.0 != self.0, other.0 != other.0) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    _ => Ordering::Equal,
                }
            },
        )
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
            let (before, _, after) = items.select_nth_unstable_by_key(items.len() / 2, move |item| OrdHelper(item.at(axis)));
            axis = (axis + 1) % T::dim();
            rayon::join(
                move || recurse(before, axis),
                move || recurse(after, axis),
            );
        }
    }
    recurse(items, 0);
}
