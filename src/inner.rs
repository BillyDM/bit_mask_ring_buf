use core::ptr::{copy_nonoverlapping, slice_from_raw_parts, slice_from_raw_parts_mut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mask(pub isize);

impl Mask {
    pub const fn new(ring_buf_len: usize) -> Self {
        Self(ring_buf_len as isize - 1)
    }
}

#[inline(always)]
pub fn constrain(i: isize, mask: Mask) -> isize {
    i & mask.0
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_slices<T>(start: isize, mask: Mask, ring_buf: &[T]) -> (&[T], &[T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let start = constrain(start, mask) as usize;

    let ring_buf_ptr = ring_buf.as_ptr();
    (
        &*slice_from_raw_parts(ring_buf_ptr.add(start), ring_buf.len() - start),
        &*slice_from_raw_parts(ring_buf_ptr, start),
    )
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_mut_slices<T>(
    start: isize,
    mask: Mask,
    ring_buf: &mut [T],
) -> (&mut [T], &mut [T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let start = constrain(start, mask) as usize;

    let ring_buf_ptr = ring_buf.as_mut_ptr();
    (
        &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), ring_buf.len() - start),
        &mut *slice_from_raw_parts_mut(ring_buf_ptr, start),
    )
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_slices_len<T>(
    start: isize,
    len: usize,
    mask: Mask,
    ring_buf: &[T],
) -> (&[T], &[T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let start = constrain(start, mask) as usize;

    let ring_buf_ptr = ring_buf.as_ptr();

    let first_portion_len = ring_buf.len() - start;
    if len > first_portion_len {
        let second_portion_len = core::cmp::min(len - first_portion_len, start);
        (
            &*slice_from_raw_parts(ring_buf_ptr.add(start), first_portion_len),
            &*slice_from_raw_parts(ring_buf_ptr, second_portion_len),
        )
    } else {
        (&*slice_from_raw_parts(ring_buf_ptr.add(start), len), &[])
    }
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_mut_slices_len<T>(
    start: isize,
    len: usize,
    mask: Mask,
    ring_buf: &mut [T],
) -> (&mut [T], &mut [T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let start = constrain(start, mask) as usize;

    let ring_buf_ptr = ring_buf.as_mut_ptr();

    let first_portion_len = ring_buf.len() - start;
    if len > first_portion_len {
        let second_portion_len = core::cmp::min(len - first_portion_len, start);
        (
            &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), first_portion_len),
            &mut *slice_from_raw_parts_mut(ring_buf_ptr, second_portion_len),
        )
    } else {
        (
            &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), len),
            &mut [],
        )
    }
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_slices_latest<T>(
    start: isize,
    len: usize,
    mask: Mask,
    ring_buf: &[T],
) -> (&[T], &[T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let ring_buf_ptr = ring_buf.as_ptr();

    if len > ring_buf.len() {
        let end_index = start + len as isize;
        let start = constrain(end_index - ring_buf.len() as isize, mask) as usize;

        (
            &*slice_from_raw_parts(ring_buf_ptr.add(start), ring_buf.len() - start),
            &*slice_from_raw_parts(ring_buf_ptr, start),
        )
    } else {
        let start = constrain(start, mask) as usize;
        let first_portion_len = ring_buf.len() - start;
        if len > first_portion_len {
            let second_portion_len = core::cmp::min(len - first_portion_len, start);
            (
                &*slice_from_raw_parts(ring_buf_ptr.add(start), first_portion_len),
                &*slice_from_raw_parts(ring_buf_ptr, second_portion_len),
            )
        } else {
            (&*slice_from_raw_parts(ring_buf_ptr.add(start), len), &[])
        }
    }
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn as_mut_slices_latest<T>(
    start: isize,
    len: usize,
    mask: Mask,
    ring_buf: &mut [T],
) -> (&mut [T], &mut [T]) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let ring_buf_ptr = ring_buf.as_mut_ptr();

    if len > ring_buf.len() {
        let end_index = start + len as isize;
        let start = constrain(end_index - ring_buf.len() as isize, mask) as usize;

        (
            &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), ring_buf.len() - start),
            &mut *slice_from_raw_parts_mut(ring_buf_ptr, start),
        )
    } else {
        let start = constrain(start, mask) as usize;
        let first_portion_len = ring_buf.len() - start;
        if len > first_portion_len {
            let second_portion_len = core::cmp::min(len - first_portion_len, start);
            (
                &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), first_portion_len),
                &mut *slice_from_raw_parts_mut(ring_buf_ptr, second_portion_len),
            )
        } else {
            (
                &mut *slice_from_raw_parts_mut(ring_buf_ptr.add(start), len),
                &mut [],
            )
        }
    }
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
#[inline(always)]
pub unsafe fn get<T>(mut i: isize, mask: Mask, ring_buf: &[T]) -> &T {
    i = constrain(i, mask);

    &*ring_buf.as_ptr().offset(i)
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
#[inline(always)]
pub unsafe fn get_mut<T>(mut i: isize, mask: Mask, ring_buf: &mut [T]) -> &mut T {
    i = constrain(i, mask);

    &mut *ring_buf.as_mut_ptr().offset(i)
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
#[inline(always)]
pub unsafe fn constrain_and_get<'a, T>(i: &mut isize, mask: Mask, ring_buf: &'a [T]) -> &'a T {
    *i = constrain(*i, mask);

    &*ring_buf.as_ptr().offset(*i)
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
#[inline(always)]
pub unsafe fn constrain_and_get_mut<'a, T>(
    i: &mut isize,
    mask: Mask,
    ring_buf: &'a mut [T],
) -> &'a mut T {
    *i = constrain(*i, mask);

    &mut *ring_buf.as_mut_ptr().offset(*i)
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn read_into<T: Clone + Copy>(
    slice: &mut [T],
    start: isize,
    mask: Mask,
    ring_buf: &[T],
) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    let start = constrain(start, mask) as usize;

    // SAFETY:
    // * By design, `constrain()` is always in range.
    // * The constructors ensures that the length of `ring_buf` is greater than `0`.
    // * Memory cannot overlap because a mutable and immutable reference do not
    // alias.
    // * The type `T` is constrained to implement `Clone + Copy`.

    let ring_buf_ptr = ring_buf.as_ptr();
    let mut slice_ptr = slice.as_mut_ptr();
    let mut slice_len = slice.len();

    // While slice is longer than from start to the end of ring_buf,
    // copy that first portion, then wrap to the beginning and copy the
    // second portion up to start.
    let first_portion_len = ring_buf.len() - start;
    while slice_len > first_portion_len {
        // Copy first portion
        copy_nonoverlapping(ring_buf_ptr.add(start), slice_ptr, first_portion_len);
        slice_ptr = slice_ptr.add(first_portion_len);
        slice_len -= first_portion_len;

        // Copy second portion
        let second_portion_len = core::cmp::min(slice_len, start);
        copy_nonoverlapping(ring_buf_ptr, slice_ptr, second_portion_len);
        slice_ptr = slice_ptr.add(second_portion_len);
        slice_len -= second_portion_len;
    }

    // Copy any elements remaining from start up to the end of ring_buf
    copy_nonoverlapping(ring_buf_ptr.add(start), slice_ptr, slice_len);
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn write_latest<T: Clone + Copy>(
    slice: &[T],
    start: isize,
    mask: Mask,
    ring_buf: &mut [T],
) {
    debug_assert_ne!(ring_buf.len(), 0);
    debug_assert_eq!(mask.0, ring_buf.len() as isize - 1);

    // If slice is longer than ring_buf, retreive only the latest portion
    let (slice, start_i) = if slice.len() > ring_buf.len() {
        let end_i = start + slice.len() as isize;
        (
            &slice[slice.len() - ring_buf.len()..],
            // Find new starting point if slice length has changed
            constrain(end_i - ring_buf.len() as isize, mask) as usize,
        )
    } else {
        (&slice[..], constrain(start, mask) as usize)
    };

    // SAFETY:
    // * By design, `self.constrain()` is always in range.
    // * The constructors ensures that the length of `ring_buf` is greater than `0`.
    // * Memory cannot overlap because a mutable and immutable reference do not
    // alias.
    // * The type `T` is constrained to implement `Clone + Copy`.
    unsafe {
        let slice_ptr = slice.as_ptr();
        let ring_buf_ptr = ring_buf.as_mut_ptr();

        // If the slice is longer than from start_i to the end of ring_buf, copy that
        // first portion, then wrap to the beginning and copy the remaining second portion.
        if start_i + slice.len() > ring_buf.len() {
            let first_portion_len = ring_buf.len() - start_i;
            copy_nonoverlapping(slice_ptr, ring_buf_ptr.add(start_i), first_portion_len);

            let second_portion_len = slice.len() - first_portion_len;
            copy_nonoverlapping(
                slice_ptr.add(first_portion_len),
                ring_buf_ptr,
                second_portion_len,
            );
        } else {
            // Otherwise, ring_buf fits so just copy it
            copy_nonoverlapping(slice_ptr, ring_buf_ptr.add(start_i), slice.len());
        }
    }
}

/// # SAFETY
/// * `ring_buf.len()` must be greater than `0` and equal a power of 2.
/// * `mask` must be equal to `ring_buf.len() - 1`.
pub unsafe fn write_latest_2<T: Clone + Copy>(
    first: &[T],
    second: &[T],
    start: isize,
    mask: Mask,
    ring_buf: &mut [T],
) {
    if first.len() + second.len() <= ring_buf.len() {
        // All ring_buf from both slices need to be copied.
        write_latest(first, start, mask, ring_buf);
    } else if second.len() < ring_buf.len() {
        // Only ring_buf from the end part of first and all of second needs to be copied.
        let first_end_part_len = ring_buf.len() - second.len();
        let first_end_part_start = first.len() - first_end_part_len;
        let first_end_part = &first[first_end_part_start..];

        write_latest(
            first_end_part,
            start + first_end_part_start as isize,
            mask,
            ring_buf,
        );
    }
    // else - Only ring_buf from second needs to be copied

    write_latest(second, start + first.len() as isize, mask, ring_buf);
}
