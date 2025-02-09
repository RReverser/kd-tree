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
    fn recurse<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
        nearests: &mut V,
        kdtree: &'a [T],
        mut k: usize,
        mut axis: usize,
        query: &T,
    ) {
        let item = match kdtree.get(k) {
            Some(item) => item,
            None => return,
        };
        let distance_metric = item.distance_metric(query);
        unsafe {
            std::hint::assert_unchecked(axis < T::DIM);
        }
        k = 2 * k + 1;
        let after_and_diff = if k < kdtree.len() {
            let diff = query.at(axis) - item.at(axis);
            axis += 1;
            if axis == T::DIM {
                axis = 0;
            }
            let (mut before, mut after) = (0, 1);
            if diff.is_positive() {
                std::mem::swap(&mut before, &mut after);
            }
            before += k;
            after += k;
            recurse(nearests, kdtree, before, axis, query);
            (after < kdtree.len()).then_some((after, diff))
        } else {
            None
        };
        // note: for small K in KNN a linear search is noticeably faster than binary one
        let i = nearests
            .iter()
            .position(|item| item.distance_metric > distance_metric)
            .unwrap_or(nearests.len());
        if i < nearests.capacity() {
            nearests.truncate(nearests.capacity() - 1);
            nearests.insert(
                i,
                ItemAndDistance {
                    item,
                    distance_metric,
                },
            );
        }
        if let Some((after, diff)) = after_and_diff {
            if nearests.last().map_or(true, |max| {
                T::from_distance_to_metric(diff) < max.distance_metric
            }) {
                recurse(nearests, kdtree, after, axis, query);
            }
        }
    }
    recurse(nearests, kdtree, 0, 0, query);
}
