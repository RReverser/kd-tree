use crate::{ItemAndDistance, KdPoint};
use arrayvec::ArrayVec;
use num_traits::Signed;
use prefetch::prefetch::{prefetch, Data, High, Read};
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
        let diff = {
            let item = match kdtree.get(k) {
                Some(item) => item,
                None => return,
            };
            let distance_metric = item.distance_metric(query);
            let i = nearests.partition_point(|item| item.distance_metric < distance_metric);
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
            query.at(axis) - item.at(axis)
        };
        k = 2 * k + 1;
        if let Some(next_start) = kdtree.get(k) {
            axis += 1;
            if axis == T::DIM {
                axis = 0;
            }
            let (mut before, mut after) = (0, 1);
            if diff.is_positive() {
                std::mem::swap(&mut before, &mut after);
            }
            prefetch::<Read, High, Data, T>(next_start);
            recurse(nearests, kdtree, k + before, axis, query);
            if nearests.last().map_or(true, |max| {
                T::from_distance_to_metric(diff) < max.distance_metric
            }) {
                recurse(nearests, kdtree, k + after, axis, query);
            }
        }
    }
    recurse(nearests, kdtree, 0, 0, query);
}
