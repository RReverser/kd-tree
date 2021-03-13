pub fn split_at_mid<T>(items: &[T]) -> (&[T], Option<&T>, &[T]) {
    if items.is_empty() {
        return (&[], None, &[]);
    }
    let index = items.len() / 2;
    unsafe {
        (
            items.get_unchecked(..index),
            Some(items.get_unchecked(index)),
            items.get_unchecked(index + 1..),
        )
    }
}
