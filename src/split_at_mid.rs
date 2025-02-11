pub fn split_at_mid<T>(items: &[T]) -> Option<(&[T], &T, &[T])> {
    let index = items.len() / 2;
    let item = items.get(index)?;
    Some(unsafe {
        (
            items.get_unchecked(..index),
            item,
            items.get_unchecked(index + 1..),
        )
    })
}
