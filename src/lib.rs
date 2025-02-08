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
//! // search the nearest neighbor
//! let found = kdtree.nearest(&[3.1, 0.9, 2.1]).unwrap();
//! assert_eq!(found.item, &[3.0, 1.0, 2.0]);
//!
//! // search k-nearest neighbors
//! let found = kdtree.nearests(&[1.5, 2.5, 1.8], 2);
//! assert_eq!(found[0].item, &[2.0, 3.0, 1.0]);
//! assert_eq!(found[1].item, &[1.0, 2.0, 3.0]);
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
use arrayvec::ArrayVec;
use nearests::*;
use num_traits::{zero, Signed};
use sort::*;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::marker::PhantomData;
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
/// assert_eq!(*kdtree.nearest(&Point3D { x: 3.1, y: 0.1, z: 2.2 }).unwrap().item, Point3D { x: 3.0, y: 1.0, z: 2.0 });
/// ```
pub trait KdPoint: Send + Sync {
    type Scalar: Signed + Copy + PartialOrd + Send + Sync;
    const DIM: usize;
    fn at(&self, i: usize) -> Self::Scalar;
    // Conversion from actual distance to the metric used for comparisons.
    // By default a squared distance.
    fn from_distance_to_metric(distance: Self::Scalar) -> Self::Scalar {
        distance * distance
    }
    fn distance_to_hyperplane(&self, other: &Self, axis: usize) -> Self::Scalar {
        Self::from_distance_to_metric(self.at(axis) - other.at(axis))
    }
    // Distance metric - doesn't need to be an actual distance, as long
    // as it preserves the order.
    // By default returns a squared distance.
    fn distance_metric(&self, other: &Self) -> Self::Scalar {
        (0..Self::DIM)
            .map(move |i| self.distance_to_hyperplane(other, i))
            .fold(zero(), |sum, x| sum + x)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemAndDistance<'a, T: KdPoint> {
    pub item: &'a T,
    pub distance_metric: T::Scalar,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KdTree<T, V>(V, PhantomData<T>);

impl<T, V: Borrow<[T]> + BorrowMut<[T]>> std::ops::Deref for KdTree<T, V> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.0.borrow()
    }
}

impl<T: KdPoint, V: Borrow<[T]> + BorrowMut<[T]>> KdTree<T, V> {
    pub fn into_inner(self) -> V {
        self.0
    }

    /// # Example
    /// ```
    /// use kd_tree::KdTree;
    /// let kdtree = KdTree::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).unwrap().item, &[3, 1, 2]);
    /// ```
    pub fn build(mut points: V) -> Self {
        kd_sort_by(points.borrow_mut());
        Self(points, PhantomData)
    }

    /// Returns kNN(k nearest neighbors) from the input point.
    /// # Example
    /// ```
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1], [3, 2, 2]];
    /// let kdtree = kd_tree::KdTree::build(&mut items[..]);
    /// let nearests = kdtree.nearests(&[3, 1, 2], 2);
    /// assert_eq!(nearests.len(), 2);
    /// assert_eq!(nearests[0].item, &[3, 1, 2]);
    /// assert_eq!(nearests[1].item, &[3, 2, 2]);
    /// ```
    pub fn nearests(&self, query: &T, num: usize) -> Vec<ItemAndDistance<T>> {
        let mut nearests = Vec::with_capacity(num);
        kd_nearests(&mut nearests, self, query);
        nearests
    }

    /// Same as [`Self::nearests`], but returns an ArrayVec.
    /// Will be faster for small number of points.
    pub fn nearests_arr<'a, const N: usize>(
        &'a self,
        query: &T,
    ) -> ArrayVec<ItemAndDistance<'a, T>, N> {
        let mut nearests = ArrayVec::new();
        kd_nearests(&mut nearests, self, query);
        nearests
    }

    /// Returns the nearest item from the input point. Returns `None` if `self.is_empty()`.
    /// # Example
    /// ```
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree = kd_tree::KdTree::build(&mut items[..]);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).unwrap().item, &[3, 1, 2]);
    /// ```
    pub fn nearest(&self, query: &T) -> Option<ItemAndDistance<T>> {
        self.nearests_arr::<1>(query).pop()
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
    pub fn within_radius(&self, query: &T, radius: T::Scalar) -> Vec<&T> {
        let radius_metric = T::from_distance_to_metric(radius);
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

impl<T: Signed + Copy + PartialOrd + Send + Sync, const D: usize> KdPoint for [T; D] {
    type Scalar = T;
    const DIM: usize = D;
    fn at(&self, i: usize) -> T {
        self[i]
    }
}

impl<
        N: Signed
            + PartialOrd
            + nalgebra::Scalar
            + Copy
            + Send
            + Sync
            + nalgebra::ComplexField<RealField = N>,
        const D: usize,
    > KdPoint for nalgebra::Point<N, D>
{
    type Scalar = N;
    const DIM: usize = D;

    fn at(&self, i: usize) -> Self::Scalar {
        self[i]
    }
    fn distance_metric(&self, other: &Self) -> Self::Scalar {
        nalgebra::distance_squared(self, other)
    }
}
