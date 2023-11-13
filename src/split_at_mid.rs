pub fn split_at_mid<T>(items: &[T]) -> (&[T], Option<&T>, &[T]) {
    let Some((item, rest)) = items.split_first() else {
        return (&[], None, &[]);
    };
    let index = items.len() / 2;
    unsafe {
        (
            // before part for Eyztiner layout
            rest.get_unchecked(..index),
            Some(item),
            // after part for Eyztiner layout
            rest.get_unchecked(index..),
        )
    }
}
