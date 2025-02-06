pub fn split_at_mid<T>(items: &[T]) -> (&[T], Option<&T>, &[T]) {
    let index = items.len() / 2;
    let item = items.get(index);
    if item.is_none() {
        return (&[], item, &[]);
    }
    unsafe {
        (
            items.get_unchecked(..index),
            item,
            items.get_unchecked(index + 1..),
        )
    }
}
