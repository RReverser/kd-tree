use crate::split_at_mid::split_at_mid;
use crate::KdPoint;
use num_traits::bounds::UpperBounded;
use std::hint::assert_unchecked;

pub struct ItemsAndDistances<'a, T: KdPoint, const MAX: usize> {
    pub items: [Option<&'a T>; MAX],
    pub distances: [T::Scalar; MAX],
}

impl<'a, T: KdPoint, const N: usize> ItemsAndDistances<'a, T, N> {
    pub fn new() -> Self {
        Self {
            items: [None; N],
            distances: [T::Scalar::max_value(); N],
        }
    }

    pub fn insert(&mut self, item: &'a T, distance_metric: T::Scalar) {
        let i = self
            .distances
            .iter()
            .rposition(|other_distance_metric| *other_distance_metric <= distance_metric)
            .map_or(0, |i| i + 1);

        if i < N {
            self.items.copy_within(i..N - 1, i + 1);
            self.items[i] = Some(item);

            self.distances.copy_within(i..N - 1, i + 1);
            self.distances[i] = distance_metric;
        }
    }

    pub fn items(&self) -> impl Iterator<Item = &'a T> {
        self.items.into_iter().map_while(|item| item)
    }

    pub fn into_iter(&self) -> impl Iterator<Item = (&'a T, T::Scalar)> {
        self.items().zip(self.distances)
    }
}

pub fn kd_nearests<'a, T: KdPoint, const N: usize>(
    kdtree: &'a [T],
    query: &T,
) -> ItemsAndDistances<'a, T, N> {
    fn recurse<'a, T: KdPoint, const N: usize>(
        nearests: &mut ItemsAndDistances<'a, T, N>,
        kdtree: &'a [T],
        query: &T,
        mut axis: usize,
    ) {
        match split_at_mid(kdtree) {
            None => {}
            Some(([], item, [])) => {
                nearests.insert(item, query.distance_metric(item));
            }
            Some((before, item, after)) => {
                unsafe {
                    assert_unchecked(axis < T::DIM);
                }
                let halves = [before, after];
                let query_coord = query.at(axis);
                let item_coord = item.at(axis);
                // Use branchless half selection.
                let first_half = if query_coord > item_coord { 1 } else { 0 };
                axis += 1;
                if axis == T::DIM {
                    axis = 0;
                }
                recurse(nearests, halves[first_half], query, axis);
                if T::distance_metric_between(query_coord, item_coord)
                    > *unsafe { nearests.distances.last().unwrap_unchecked() }
                {
                    return;
                }
                nearests.insert(item, query.distance_metric(item));
                recurse(nearests, halves[1 - first_half], query, axis);
            }
        }
    }

    let mut nearests = ItemsAndDistances::new();
    if N > 0 {
        recurse(&mut nearests, kdtree, query, 0);
    }
    nearests
}
