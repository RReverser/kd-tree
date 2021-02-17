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

pub fn split_at_mid_mut<T>(items: &mut [T]) -> (&mut [T], Option<&mut T>, &mut [T]) {
    if items.is_empty() {
        return (&mut [], None, &mut []);
    }
    let index = items.len() / 2;
    unsafe {
        (
            &mut *(items.get_unchecked_mut(..index) as *mut _),
            Some(&mut *(items.get_unchecked_mut(index) as *mut _)),
            &mut *(items.get_unchecked_mut(index + 1..)),
        )
    }
}
