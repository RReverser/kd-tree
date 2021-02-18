use crate::sort::OrdHelper;
use crate::split_at_mid::split_at_mid;
use crate::{ItemAndDistance, KdPoint};
use arrayvec::{Array, ArrayVec};
use num_traits::Signed;
use std::ops::DerefMut;

pub trait VecLike: DerefMut<Target = [<Self as VecLike>::Item]> {
    type Item;

    fn insert(&mut self, index: usize, value: Self::Item);
    fn push(&mut self, value: Self::Item);
    fn pop(&mut self) -> Option<Self::Item>;
    fn capacity(&self) -> usize;
}

macro_rules! impl_vec_like {
    () => {
        fn insert(&mut self, index: usize, value: Self::Item) {
            Self::insert(self, index, value)
        }

        fn push(&mut self, value: Self::Item) {
            Self::push(self, value)
        }

        fn pop(&mut self) -> Option<Self::Item> {
            Self::pop(self)
        }

        fn capacity(&self) -> usize {
            Self::capacity(self)
        }
    };
}

impl<T> VecLike for Vec<T> {
    type Item = T;

    impl_vec_like!();
}

impl<A: Array> VecLike for ArrayVec<A> {
    type Item = A::Item;

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
        let (before, item, after) = split_at_mid(kdtree);
        let item = match item {
            Some(item) => item,
            None => return,
        };
        let distance_metric = item.distance_metric(query);
        if nearests.len() < nearests.capacity()
            || distance_metric < nearests.last().unwrap().distance_metric
        {
            if nearests.len() == nearests.capacity() {
                nearests.pop();
            }
            let i = nearests
                .binary_search_by_key(&OrdHelper(distance_metric), move |item| {
                    OrdHelper(item.distance_metric)
                })
                .unwrap_or_else(|i| i);
            nearests.insert(
                i,
                ItemAndDistance {
                    item,
                    distance_metric,
                },
            );
        }
        let diff = query.at(axis) - item.at(axis);
        let (branch1, branch2) = if diff.is_negative() {
            (before, after)
        } else {
            (after, before)
        };
        recurse(nearests, branch1, query, (axis + 1) % T::dim());
        if !branch2.is_empty()
            && T::from_distance_to_metric(diff) < nearests.last().unwrap().distance_metric
        {
            recurse(nearests, branch2, query, (axis + 1) % T::dim());
        }
    }
    recurse(nearests, kdtree, query, 0);
}
