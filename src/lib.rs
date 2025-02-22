//! k-dimensional tree.
//!
//! # Usage
//! ```
//! // construct kd-tree
//! let kdtree = kd_tree::KdTree::build(vec![
//!     [1.0, 2.0, 3.0],
//!     [3.0, 1.0, 2.0],
//!     [2.0, 3.0, 1.0],
//! ]);
//!
//! // search k-nearest neighbors
//! let mut found = kdtree.nearests::<2>(&[1.5, 2.5, 1.8]).into_iter();
//! assert_eq!(found.next().unwrap().0, &[2.0, 3.0, 1.0]);
//! assert_eq!(found.next().unwrap().0, &[1.0, 2.0, 3.0]);
//!
//! // search points within a sphere
//! let found = kdtree.within_radius(&[2.0, 1.5, 2.5], 1.5);
//! assert_eq!(found.len(), 2);
//! assert!(found.iter().any(|&&p| p == [1.0, 2.0, 3.0]));
//! assert!(found.iter().any(|&&p| p == [3.0, 1.0, 2.0]));
//! ```
mod nearests;
mod sort;
mod split_at_mid;
mod within;
use nearests::*;
use num_traits::bounds::UpperBounded;
use num_traits::{zero, Signed};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sort::*;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Index;
use within::*;

/// A trait to represent k-dimensional point.
///
/// # Example
/// ```
/// #[derive(Debug, PartialEq)]
/// struct Point3D {
///     pub x: f64,
///     pub y: f64,
///     pub z: f64,
/// }
/// impl kd_tree::KdPoint for Point3D {
///     type Scalar = f64;
///     const DIM: usize = 3;
///     fn at(&self, k: usize) -> f64 {
///         match k {
///             0 => self.x,
///             1 => self.y,
///             _ => self.z,
///         }
///     }
/// }
/// let kdtree = kd_tree::KdTree::build(vec![
///     Point3D { x: 1.0, y: 2.0, z: 3.0 },
///     Point3D { x: 3.0, y: 1.0, z: 2.0 },
///     Point3D { x: 2.0, y: 3.0, z: 1.0 },
/// ]);
/// assert_eq!(*kdtree.nearests::<1>(&Point3D { x: 3.1, y: 0.1, z: 2.2 }).into_iter().next().unwrap().0, Point3D { x: 3.0, y: 1.0, z: 2.0 });
/// ```
pub trait KdPoint: Send + Sync {
    type Point: IntoIterator<Item: Signed + Copy + PartialOrd + Send + Sync + UpperBounded>
        + Index<usize, Output = KdScalar<Self>>
        + Send
        + Sync
        + Copy;

    const DIM: usize;

    fn next_axis(mut axis: usize) -> usize {
        axis += 1;
        if axis == Self::DIM {
            0
        } else {
            axis
        }
    }

    fn as_point(&self) -> &Self::Point;

    fn at(&self, k: usize) -> KdScalar<Self> {
        self.as_point()[k]
    }

    // Distance metric between given hyperplane coordinates.
    fn distance_metric_between(coord1: KdScalar<Self>, coord2: KdScalar<Self>) -> KdScalar<Self> {
        let diff = coord1 - coord2;
        diff * diff
    }

    // Distance metric - doesn't need to be an actual distance, as long
    // as it preserves the order.
    // By default returns a squared distance.
    fn distance_metric(&self, other: &Self) -> KdScalar<Self> {
        std::iter::zip(*self.as_point(), *other.as_point())
            .map(|(a, b)| a - b)
            .map(|diff| diff * diff)
            .fold(zero(), |sum, x| sum + x)
    }
}

pub type KdScalar<T> = <<T as KdPoint>::Point as IntoIterator>::Item;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KdTree<T, V>(V, PhantomData<T>);

impl<T, V: Borrow<[T]> + BorrowMut<[T]>> std::ops::Deref for KdTree<T, V> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.0.borrow()
    }
}

impl<T: KdPoint, V: Borrow<[T]> + BorrowMut<[T]> + Sync> KdTree<T, V> {
    pub fn into_inner(self) -> V {
        self.0
    }

    /// # Example
    /// ```
    /// use kd_tree::KdTree;
    /// let kdtree = KdTree::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
    /// ```
    pub fn build(mut points: V) -> Self {
        kd_sort_by(points.borrow_mut());
        Self(points, PhantomData)
    }

    /// Same as [`Self::nearests`], but returns an ArrayVec.
    /// Will be faster for small number of points.
    pub fn nearests<'a, const N: usize>(&'a self, query: &T) -> ItemsAndDistances<'a, T, N> {
        kd_nearests(self, query)
    }

    pub fn nearests_all<'a, const N: usize>(&'a self) -> Vec<ItemsAndDistances<'a, T, N>> {
        self.par_iter()
            .map(|item| self.nearests::<N>(item))
            .collect()
    }

    /// search points within a rectangular region
    pub fn within(&self, query: [&T; 2]) -> Vec<&T> {
        let mut results = Vec::new();
        kd_within_by_cmp(
            |item| results.push(item),
            self,
            move |value, k| {
                if value < query[0].at(k) {
                    Ordering::Less
                } else if value > query[1].at(k) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            },
        );
        results
    }

    /// search points within k-dimensional sphere
    pub fn within_radius(&self, query: &T, radius: <T::Point as IntoIterator>::Item) -> Vec<&T> {
        let radius_metric = T::distance_metric_between(zero(), radius);
        let mut results = Vec::new();
        let results_mut = &mut results;
        kd_within_by_cmp(
            move |item| {
                if item.distance_metric(query) < radius_metric {
                    results_mut.push(item)
                }
            },
            self,
            move |value, k| {
                if value < query.at(k) - radius {
                    Ordering::Less
                } else if value > query.at(k) + radius {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            },
        );
        results
    }
}

impl<T: Signed + Copy + PartialOrd + Send + Sync + UpperBounded, const D: usize> KdPoint
    for [T; D]
{
    type Point = Self;
    const DIM: usize = D;

    fn as_point(&self) -> &Self::Point {
        self
    }
}

impl<
        N: Signed
            + PartialOrd
            + nalgebra::Scalar
            + Copy
            + Send
            + Sync
            + nalgebra::ComplexField<RealField = N>
            + UpperBounded,
        const D: usize,
    > KdPoint for nalgebra::Point<N, D>
{
    type Point = [N; D];
    const DIM: usize = D;

    fn as_point(&self) -> &Self::Point {
        &self.coords.data.0[0]
    }

    fn distance_metric(&self, other: &Self) -> N {
        nalgebra::distance_squared(self, other)
    }
}
