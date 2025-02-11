use crate::sort::OrdHelper;
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
        let (mut before, item, mut after) = split_at_mid(kdtree);
        let item = match item {
            Some(item) => item,
            None => return,
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
        if nearests.len() < nearests.capacity()
            || nearests.last().map_or(
                /* unreachable */ false,
                |max| distance_metric < max.distance_metric,
            )
        {
            nearests.truncate(nearests.capacity() - 1);
            let (Ok(i) | Err(i)) = nearests
                .binary_search_by_key(&OrdHelper(distance_metric), move |item| {
                    OrdHelper(item.distance_metric)
                });
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
