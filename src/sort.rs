use crate::KdPoint;
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::mem::MaybeUninit;

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

#[repr(transparent)]
struct YoloCell<T> {
    value: UnsafeCell<MaybeUninit<T>>,
}

// TODO: replace implementation with SyncUnsafeCell once it stabilizes.
unsafe impl<T: Sync> Sync for YoloCell<T> {}

impl<T> YoloCell<T> {
    fn get(&self) -> *mut T {
        self.value.get().cast()
    }
}

pub fn kd_sort_by<T: KdPoint>(points: &mut [T]) {
    fn build_eytzinger_kdtree<T: KdPoint>(
        output: &[YoloCell<T>],
        points: &mut [T],
        mut k: usize,    // Logical Eytzinger index (root starts at 1)
        mut axis: usize, // Depth in the tree, to choose the axis
    ) {
        let Some(output_point) = output.get(k) else {
            return;
        };

        if points.is_empty() {
            return;
        }

        let (left, median, right) =
            points.select_nth_unstable_by_key(points.len() / 2, |p| OrdHelper(p.at(axis)));

        unsafe {
            // SAFETY: in Eytzingerization, each k points to a unique item that no other thread should override.
            // As long as we mutably access only the item pointed to by k, we are safe.
            output_point.get().copy_from_nonoverlapping(median, 1);
        }

        k *= 2;

        axis += 1;
        if axis == T::DIM {
            axis = 0;
        }

        rayon::join(
            move || build_eytzinger_kdtree(output, left, k + 1, axis),
            move || build_eytzinger_kdtree(output, right, k + 2, axis),
        );
    }

    let mut output = Vec::<YoloCell<T>>::with_capacity(points.len());
    unsafe {
        output.set_len(output.capacity());
    }
    build_eytzinger_kdtree(&output, points, 0, 0);
    unsafe {
        std::ptr::copy_nonoverlapping(output.as_ptr().cast(), points.as_mut_ptr(), points.len());
    }
}
