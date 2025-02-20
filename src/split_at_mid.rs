// Heuristical size beyond which binary split gets worse than simple binary search.
const FLAT_SIZE: usize = 8;

pub fn split_at_mid_index(len: usize) -> Option<usize> {
    if len < FLAT_SIZE {
        return None;
    }
    Some(len / 2)
}

pub fn split_at_mid<T>(items: &[T]) -> Option<(&[T], &T, &[T])> {
    let index = split_at_mid_index(items.len())?;
    // Safety: this is only safe if slice is non-empty.
    // That invariant holds as long as `FLAT_SIZE` is at least 1.
    Some(unsafe {
        (
            items.get_unchecked(..index),
            items.get_unchecked(index),
            items.get_unchecked(index + 1..),
        )
    })
}
