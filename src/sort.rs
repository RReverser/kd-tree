use crate::KdPoint;
use std::cell::UnsafeCell;
use std::cmp::{min, Ordering};
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
        let Some(output_point) = output.get(k - 1) else {
            return;
        };

        let left_count = match points.len() {
            1 => 0,
            // this is some complicated math to avoid sorting the array before Eytzingerization
            // (besides, for multi-dimensional data it's not really that much simpler to presort
            // and then do BFS)
            n => {
                // height of the fully-filled levels of the tree
                let h = n.ilog2();
                // left subtree cannot have more than 2^h - 1 nodes
                let left_subtree_capacity = (1 << h) - 1;
                // right subtree must have at least 2^(h-1) nodes
                let right_subtree_min_len = 1 << (h - 1);
                // max number of nodes in the left subtree if last level is partially filled
                let number_of_extra_nodes_in_left_subtree = n - right_subtree_min_len;
                min(number_of_extra_nodes_in_left_subtree, left_subtree_capacity)
            }
        };

        let (left, split, right) =
            points.select_nth_unstable_by_key(left_count, |p| OrdHelper(p.at(axis)));

        unsafe {
            // SAFETY: in Eytzingerization, each k points to a unique item that no other thread should override.
            // As long as we mutably access only the item pointed to by k, we are safe.
            output_point.get().copy_from_nonoverlapping(split, 1);
        }

        k *= 2;

        axis += 1;
        if axis == T::DIM {
            axis = 0;
        }

        rayon::join(
            move || build_eytzinger_kdtree(output, left, k, axis),
            move || build_eytzinger_kdtree(output, right, k + 1, axis),
        );
    }

    let mut output = Vec::<YoloCell<T>>::with_capacity(points.len());
    unsafe {
        output.set_len(output.capacity());
    }
    build_eytzinger_kdtree(&output, points, 1, 0);
    unsafe {
        std::ptr::copy_nonoverlapping(output.as_ptr().cast(), points.as_mut_ptr(), points.len());
    }
}

#[test]
fn check_single_dimensional_sort() {
    let mut points = (1..=5).map(|x| [x]).collect::<Vec<_>>();
    kd_sort_by(&mut points);
    let points = points.iter().map(|p| p[0]).collect::<Vec<_>>();
    assert_eq!(points, vec![4, 2, 5, 1, 3]);

    let mut points = (1..=4).map(|x| [x]).collect::<Vec<_>>();
    kd_sort_by(&mut points);
    let points = points.iter().map(|p| p[0]).collect::<Vec<_>>();
    assert_eq!(points, vec![3, 2, 4, 1]);
}
