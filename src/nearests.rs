use crate::split_at_mid::split_at_mid;
use crate::{ItemAndDistance, KdPoint};
use arrayvec::ArrayVec;
use std::hint::assert_unchecked;
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
                let query_coord = query.at(axis);
                let item_coord = item.at(axis);
                if query_coord > item_coord {
                    std::mem::swap(&mut before, &mut after);
                }
                axis += 1;
                if axis == T::DIM {
                    axis = 0;
                }
                recurse(nearests, before, query, axis);
                insert_nearests(
                    nearests,
                    ItemAndDistance {
                        item,
                        distance_metric: query.distance_metric(item),
                    },
                );
                if !after.is_empty()
                    && nearests.get(nearests.capacity() - 1).map_or(true, |max| {
                        T::distance_metric_between(query_coord, item_coord) <= max.distance_metric
                    })
                {
                    recurse(nearests, after, query, axis);
                }
            }
        }
    }

    if nearests.capacity() != 0 {
        recurse(nearests, kdtree, query, 0);
    }
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
