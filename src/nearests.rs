use crate::split_at_mid::split_at_mid;
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
        query: &T,
        axis: usize,
    ) {
        let Some((mut before, item, mut after)) = split_at_mid(kdtree) else {
            return;
        };
        let diff = query.at(axis) - item.at(axis);
        if diff.is_positive() {
            std::mem::swap(&mut before, &mut after);
        }
        let mut next_axis = axis + 1;
        if next_axis == T::DIM {
            next_axis = 0;
        }
        recurse(nearests, before, query, next_axis);
        let distance_metric = item.distance_metric(query);
        let i = nearests
            .iter()
            .rposition(move |item| item.distance_metric <= distance_metric)
            .map_or(0, |i| i + 1);
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
        if !after.is_empty()
            && nearests.get(nearests.capacity() - 1).map_or(true, |max| {
                T::from_distance_to_metric(diff) < max.distance_metric
            })
        {
            recurse(nearests, after, query, next_axis);
        }
    }
    recurse(nearests, kdtree, query, 0);
}
