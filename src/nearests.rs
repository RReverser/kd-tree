use crate::{ItemAndDistance, KdPoint};
use arrayvec::ArrayVec;
use num_traits::Signed;
use std::ops::DerefMut;

pub trait VecLike: DerefMut<Target = [<Self as VecLike>::Item]> {
    type Item;

    fn insert(&mut self, index: usize, value: Self::Item);
    fn capacity(&self) -> usize;
    fn truncate(&mut self, new_size: usize);
}

macro_rules! impl_vec_like {
    () => {
        type Item = T;

        fn insert(&mut self, index: usize, value: Self::Item) {
            Self::insert(self, index, value)
        }

        fn capacity(&self) -> usize {
            Self::capacity(self)
        }

        fn truncate(&mut self, new_size: usize) {
            Self::truncate(self, new_size)
        }
    };
}

impl<T> VecLike for Vec<T> {
    impl_vec_like!();
}

impl<T, const N: usize> VecLike for ArrayVec<T, N> {
    impl_vec_like!();
}

pub fn kd_nearests<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
    nearests: &mut V,
    kdtree: &'a [T],
    query: &T,
) {
    fn add_maybe_nearest<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
        nearests: &mut V,
        new_item: ItemAndDistance<'a, T>,
    ) {
        // note: for small K in KNN a linear search is noticeably faster than binary one
        let i = nearests
            .iter()
            .position(|item| item.distance_metric > new_item.distance_metric)
            .unwrap_or(nearests.len());
        if i < nearests.capacity() {
            nearests.truncate(nearests.capacity() - 1);
            nearests.insert(i, new_item);
        }
    }

    fn maybe_check_after<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
        nearests: &mut V,
        kdtree: &'a [T],
        after: &'a T,
        diff: T::Scalar,
        axis: usize,
        query: &T,
    ) {
        // Check the N-1 item - this covers both if nearests is not full yet and if it is, but the new item is closer.
        if nearests.get(nearests.capacity() - 1).map_or(true, |max| {
            T::from_distance_to_metric(diff) < max.distance_metric
        }) {
            recurse(nearests, kdtree, after, axis, query);
        }
    }

    fn recurse<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
        nearests: &mut V,
        kdtree: &'a [T],
        item: &'a T,
        mut axis: usize,
        query: &T,
    ) {
        let new_item = ItemAndDistance {
            item,
            distance_metric: item.distance_metric(query),
        };

        let after_and_diff = kdtree
            .get(
                unsafe { std::ptr::from_ref::<T>(item).offset_from(kdtree.as_ptr()) as usize }
                    * 2
                    + 1..,
            )
            .and_then(|slice| slice.split_first());

        if let Some((before, rest)) = after_and_diff {
            unsafe {
                std::hint::assert_unchecked(axis < T::DIM);
            }
            let diff = query.at(axis) - item.at(axis);
            axis += 1;
            if axis == T::DIM {
                axis = 0;
            }
            if diff.is_positive() {
                if let Some(after) = rest.first() {
                    recurse(nearests, kdtree, after, axis, query);
                }
                add_maybe_nearest(nearests, new_item);
                maybe_check_after(nearests, kdtree, before, diff, axis, query);
            } else {
                recurse(nearests, kdtree, before, axis, query);
                add_maybe_nearest(nearests, new_item);
                if let Some(after) = rest.first() {
                    maybe_check_after(nearests, kdtree, after, diff, axis, query);
                }
            }
        } else {
            add_maybe_nearest(nearests, new_item);
        }
    }
    if let Some(first) = kdtree.first() {
        recurse(nearests, kdtree, first, 0, query);
    }
}
