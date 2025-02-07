use crate::sort::OrdHelper;
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
        k: usize,
        query: &T,
        mut axis: usize,
    ) {
        let item = match kdtree.get(k) {
            Some(item) => item,
            None => return,
        };
        let (mut before, mut after) = (2 * k + 1, 2 * k + 2);
        let diff = query.at(axis) - item.at(axis);
        if diff.is_positive() {
            std::mem::swap(&mut before, &mut after);
        }
        axis += 1;
        if axis == T::DIM {
            axis = 0;
        }
        recurse(nearests, kdtree, before, query, axis);
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
        if after < kdtree.len()
            && nearests.last().map_or(true, |max| {
                T::from_distance_to_metric(diff) < max.distance_metric
            })
        {
            recurse(nearests, kdtree, after, query, axis);
        }
    }
    recurse(nearests, kdtree, 0, query, 0);
}
