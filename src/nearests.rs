use crate::split_at_mid::split_at_mid;
use crate::{ItemAndDistance, KdPoint};
use arrayvec::ArrayVec;
use num_traits::Signed;
use num_traits::Zero;
use rayon::prelude::*;
use std::hint::assert_unchecked;
use std::iter::zip;
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
    mut axis: usize,
) {
    match split_at_mid(kdtree) {
        None => {}
        Some(([], item, [])) => {
            insert_nearests(
                nearests,
                ItemAndDistance {
                    item,
                    distance_metric: query.distance_metric(item),
                },
            );
        }
        Some((mut before, item, mut after)) => {
            unsafe {
                assert_unchecked(axis < T::DIM);
            }
            let diff = query.at(axis) - item.at(axis);
            if diff.is_positive() {
                std::mem::swap(&mut before, &mut after);
            }
            axis += 1;
            if axis == T::DIM {
                axis = 0;
            }
            kd_nearests(nearests, before, query, axis);
            insert_nearests(
                nearests,
                ItemAndDistance {
                    item,
                    distance_metric: query.distance_metric(item),
                },
            );
            if !after.is_empty()
                && nearests.get(nearests.capacity() - 1).map_or(true, |max| {
                    T::from_distance_to_metric(diff) < max.distance_metric
                })
            {
                kd_nearests(nearests, after, query, axis);
            }
        }
    }
}

pub fn kd_nearests_all<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>> + Send>(
    nearests: &mut [V],
    kdtree: &'a [T],
    axis: usize,
) {
    debug_assert_eq!(kdtree.len(), nearests.len());
    let Some((before, mid, after)) = split_at_mid(kdtree) else {
        return;
    };
    let mut next_axis = axis + 1;
    if next_axis == T::DIM {
        next_axis = 0;
    }
    let (nearests_before, nearests_mid_and_after) = nearests.split_at_mut(before.len());
    rayon::join(
        move || kd_nearests_all(nearests_before, before, next_axis),
        move || {
            // search the midpoint in "after" too; the loop below will take care of inserting itself and searching in "before"
            let (nearests_mid, nearests_after) = nearests_mid_and_after.split_first_mut().unwrap();
            kd_nearests(nearests_mid, after, mid, next_axis);
            kd_nearests_all(nearests_after, after, next_axis);
        },
    );
    kdtree
        .par_iter()
        .zip(&mut nearests[..])
        .for_each(|(query, nearests)| {
            insert_nearests(
                nearests,
                ItemAndDistance {
                    item: mid,
                    distance_metric: query.distance_metric(mid),
                },
            );
        });
    // Now search opposite sides, but only where it might be closer than the farthest nearest of current item.
    let (nearests_before, nearests_mid_and_after) = nearests.split_at_mut(before.len());
    let handle_side = |side: &'a [T], nearests: &mut [V], other_kd: &'a [T]| {
        if !other_kd.is_empty() {
            for (query, nearests) in zip(side, nearests) {
                if nearests.get(nearests.capacity() - 1).map_or(true, |max| {
                    T::from_distance_to_metric(query.at(axis) - mid.at(axis)) < max.distance_metric
                }) {
                    kd_nearests(nearests, other_kd, query, next_axis);
                }
            }
        }
    };
    rayon::join(
        move || handle_side(before, nearests_before, after),
        move || handle_side(&kdtree[before.len()..], nearests_mid_and_after, before),
    );
}

fn insert_nearests<'a, T: KdPoint, V: VecLike<Item = ItemAndDistance<'a, T>>>(
    nearests: &mut V,
    new_item: ItemAndDistance<'a, T>,
) {
    let i = nearests
        .iter()
        .rposition(|item| item.distance_metric <= new_item.distance_metric)
        .map_or(0, |i| i + 1);
    if i < nearests.capacity() {
        nearests.truncate(nearests.capacity() - 1);
        nearests.insert(i, new_item);
    }
}
