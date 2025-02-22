use crate::split_at_mid::split_at_mid_index;
use crate::KdPoint;
use std::cmp::Ordering;

// A wrapper similar to OrderedFloat but for generic types.
// Moves any incomparable values to the end and treats them as equal.
pub struct OrdHelper<T: PartialOrd>(pub T);

impl<T: PartialOrd> Ord for OrdHelper<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        unsafe { self.0.partial_cmp(&other.0).unwrap_unchecked() }
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
        let Some(index) = split_at_mid_index(items.len()) else {
            return;
        };
        let (before, _, after) =
            items.select_nth_unstable_by_key(index, move |item| OrdHelper(item.at(axis)));
        axis = T::next_axis(axis);
        rayon::join(move || recurse(before, axis), move || recurse(after, axis));
    }
    recurse(items, 0);
}
