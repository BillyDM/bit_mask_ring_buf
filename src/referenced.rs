/// A fast ring buffer implementation with cheap and safe indexing. It works by bit-masking
/// an integer index to get the corresponding index in an array/vec whose length
/// is a power of 2. This is best used when indexing the buffer with an `isize` value.
/// Copies/reads with slices are implemented with memcpy. This works the same as
/// [`BMRingBuf`] except it uses a reference as its data source instead of an internal Vec.
///
/// [`BMRingBuf`]: struct.BMRingBuf.html
#[allow(missing_debug_implementations)]
pub struct BMRingBufRef<'a, T: Copy + Clone + Default> {
    data: &'a mut [T],
    mask: isize,
}

impl<'a, T: Copy + Clone + Default> BMRingBufRef<'a, T> {
    /// Creates a new [`BMRingBufRef`] with the given data.
    ///
    /// # Safety
    ///
    /// * The data in `slice` must be valid and properly aligned.
    /// See [`std::slice::from_raw_parts`] for more details.
    /// * The size in bytes of the data in `slice` should be no larger than `isize::MAX`.
    /// See [`std::ptr::offset`] for more information when indexing very large buffers
    /// on 32-bit and 16-bit platforms.
    ///
    /// # Panics
    ///
    /// * This will panic if the length of the given slice is not a power of 2
    /// * This will panic if the length of the slice is less than 2
    /// * This will panic if the length of the slice is greater than `(std::usize::MAX/2)+1`
    ///
    /// # Undefined Behavior
    ///
    /// * Using this struct may cause undefined behavior if the given data in `slice`
    /// was not initialized first
    ///
    /// [`BMRingBufRef`]: struct.BMRingBufRef.html
    /// [`std::slice::from_raw_parts`]: https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html
    /// [`std::ptr::offset`]: https://doc.rust-lang.org/std/primitive.pointer.html#method.offset
    pub fn new(slice: &'a mut [T]) -> Self {
        assert_eq!(slice.len(), crate::next_pow_of_2(slice.len()));

        let mask = (slice.len() as isize) - 1;

        Self { data: slice, mask }
    }

    /// Clears all values in the ring buffer to the default value.
    pub fn clear(&mut self) {
        for n in self.data.iter_mut() {
            *n = Default::default();
        }
    }

    /// Returns two slices that contain the all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Undefined Behavior
    ///
    /// * Using this may cause undefined behavior if the given data in `slice`
    /// in `BMRingBufRef::new()` was not initialized first.
    pub fn as_slices(&self, start: isize) -> (&[T], &[T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        unsafe {
            let self_vec_ptr = self.data.as_ptr();
            (
                &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), self.data.len() - start),
                &*std::ptr::slice_from_raw_parts(self_vec_ptr, start),
            )
        }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// capacity of the ring buffer, then that capacity will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Undefined Behavior
    ///
    /// * Using this may cause undefined behavior if the given data in `slice`
    /// in `BMRingBufRef::new()` was not initialized first.
    pub fn as_slices_len(&self, start: isize, len: usize) -> (&[T], &[T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        unsafe {
            let self_vec_ptr = self.data.as_ptr();

            let first_portion_len = self.data.len() - start;
            if len > first_portion_len {
                let second_portion_len = std::cmp::min(len - first_portion_len, start);
                (
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), first_portion_len),
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr, second_portion_len),
                )
            } else {
                (
                    &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), len),
                    &[],
                )
            }
        }
    }

    /// Returns two mutable slices that contain the all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Undefined Behavior
    ///
    /// * Using this may cause undefined behavior if the given data in `slice`
    /// in `BMRingBufRef::new()` was not initialized first.
    pub fn as_mut_slices(&mut self, start: isize) -> (&mut [T], &mut [T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        unsafe {
            let self_vec_ptr = self.data.as_mut_ptr();
            (
                &mut *std::ptr::slice_from_raw_parts_mut(
                    self_vec_ptr.add(start),
                    self.data.len() - start,
                ),
                &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, start),
            )
        }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// capacity of the ring buffer, then that capacity will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Undefined Behavior
    ///
    /// * Using this may cause undefined behavior if the given data in `slice`
    /// in `BMRingBufRef::new()` was not initialized first.
    pub fn as_mut_slices_len(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        unsafe {
            let self_vec_ptr = self.data.as_mut_ptr();

            let first_portion_len = self.data.len() - start;
            if len > first_portion_len {
                let second_portion_len = std::cmp::min(len - first_portion_len, start);
                (
                    &mut *std::ptr::slice_from_raw_parts_mut(
                        self_vec_ptr.add(start),
                        first_portion_len,
                    ),
                    &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr, second_portion_len),
                )
            } else {
                (
                    &mut *std::ptr::slice_from_raw_parts_mut(self_vec_ptr.add(start), len),
                    &mut [],
                )
            }
        }
    }

    /// Copies the data from the ring buffer starting from the index `start`
    /// into the given slice. If the length of `slice` is larger than the
    /// capacity of the ring buffer, then the data will be reapeated until
    /// the given slice is filled.
    ///
    /// * `slice` - This slice to copy the data into.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Undefined Behavior
    ///
    /// * Using this may cause undefined behavior if the given data in `slice`
    /// in `BMRingBufRef::new()` was not initialized first.
    pub fn read_into(&self, slice: &mut [T], start: isize) {
        let start = self.constrain(start) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let self_vec_ptr = self.data.as_ptr();
            let mut slice_ptr = slice.as_mut_ptr();
            let mut slice_len = slice.len();

            // While slice is longer than from start to the end of self.data,
            // copy that first portion, then wrap to the beginning and copy the
            // second portion up to start.
            let first_portion_len = self.data.len() - start;
            while slice_len > first_portion_len {
                // Copy first portion
                std::ptr::copy_nonoverlapping(
                    self_vec_ptr.add(start),
                    slice_ptr,
                    first_portion_len,
                );
                slice_ptr = slice_ptr.add(first_portion_len);
                slice_len -= first_portion_len;

                // Copy second portion
                let second_portion_len = std::cmp::min(slice_len, start);
                std::ptr::copy_nonoverlapping(self_vec_ptr, slice_ptr, second_portion_len);
                slice_ptr = slice_ptr.add(second_portion_len);
                slice_len -= second_portion_len;
            }

            // Copy any elements remaining from start up to the end of self.data
            std::ptr::copy_nonoverlapping(self_vec_ptr.add(start), slice_ptr, slice_len);
        }
    }

    /// Copies data from the given slice into the ring buffer starting from
    /// the index `start`. If the length of `slice` is larger than the
    /// capacity of the ring buffer, then only the latest data will be copied.
    ///
    /// * `slice` - This slice to copy data from.
    /// * `start` - The index of the ring buffer to start copying from.
    pub fn write_latest(&mut self, slice: &[T], start: isize) {
        let end_i = start + slice.len() as isize;

        // If slice is longer than self.data, retreive only the latest portion
        let slice = if slice.len() > self.data.len() {
            &slice[slice.len() - self.data.len()..]
        } else {
            &slice[..]
        };

        // Find new starting point if slice length has changed
        let start_i = self.constrain(end_i - slice.len() as isize) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let slice_ptr = slice.as_ptr();
            let self_vec_ptr = self.data.as_mut_ptr();

            // If the slice is longer than from start_i to the end of self.data, copy that
            // first portion, then wrap to the beginning and copy the remaining second portion.
            if start_i + slice.len() > self.data.len() {
                let first_portion_len = self.data.len() - start_i;
                std::ptr::copy_nonoverlapping(
                    slice_ptr,
                    self_vec_ptr.add(start_i),
                    first_portion_len,
                );

                let second_portion_len = slice.len() - first_portion_len;
                std::ptr::copy_nonoverlapping(
                    slice_ptr.add(first_portion_len),
                    self_vec_ptr,
                    second_portion_len,
                );
            } else {
                // Otherwise, data fits so just copy it
                std::ptr::copy_nonoverlapping(slice_ptr, self_vec_ptr.add(start_i), slice.len());
            }
        }
    }

    /// Returns the capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns the actual index of the ring buffer from the given
    /// `i` index. This is cheap due to the ring buffer's bit-masking
    /// algorithm.
    pub fn constrain(&self, i: isize) -> isize {
        i & self.mask
    }
}

impl<'a, T: Copy + Clone + Default> std::ops::Index<isize> for BMRingBufRef<'a, T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        // The constructors also correctly call this function.
        unsafe { &*self.data.as_ptr().offset(i & self.mask) }
    }
}

impl<'a, T: Copy + Clone + Default> std::ops::IndexMut<isize> for BMRingBufRef<'a, T> {
    fn index_mut(&mut self, i: isize) -> &mut T {
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.data and the value of self.mask are only modified
        // in the constructor, which makes sure these values are valid.
        // The constructors also correctly call this function.
        unsafe { &mut *self.data.as_mut_ptr().offset(i & self.mask) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_mask_ring_buf_ref_initialize() {
        let mut data = [0.0; 4];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert_eq!(&ring_buf.data[..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_constrain() {
        let mut data = [0.0; 4];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert_eq!(&ring_buf.data[..], &[0.0, 0.0, 0.0, 0.0]);

        assert_eq!(ring_buf.constrain(-8), 0);
        assert_eq!(ring_buf.constrain(-7), 1);
        assert_eq!(ring_buf.constrain(-6), 2);
        assert_eq!(ring_buf.constrain(-5), 3);
        assert_eq!(ring_buf.constrain(-4), 0);
        assert_eq!(ring_buf.constrain(-3), 1);
        assert_eq!(ring_buf.constrain(-2), 2);
        assert_eq!(ring_buf.constrain(-1), 3);
        assert_eq!(ring_buf.constrain(0), 0);
        assert_eq!(ring_buf.constrain(1), 1);
        assert_eq!(ring_buf.constrain(2), 2);
        assert_eq!(ring_buf.constrain(3), 3);
        assert_eq!(ring_buf.constrain(4), 0);
        assert_eq!(ring_buf.constrain(5), 1);
        assert_eq!(ring_buf.constrain(6), 2);
        assert_eq!(ring_buf.constrain(7), 3);
        assert_eq!(ring_buf.constrain(8), 0);
    }

    #[test]
    fn bit_mask_ring_buf_ref_clear() {
        let mut data = [0.0; 4];
        let mut ring_buf = BMRingBufRef::new(&mut data);

        assert_eq!(&ring_buf.data[..], &[0.0, 0.0, 0.0, 0.0]);

        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 3.0, 4.0]);

        ring_buf.clear();
        assert_eq!(ring_buf.data, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_index() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        let ring_buf = &ring_buf;

        assert_eq!(ring_buf[-8], 0.0);
        assert_eq!(ring_buf[-7], 1.0);
        assert_eq!(ring_buf[-6], 2.0);
        assert_eq!(ring_buf[-5], 3.0);
        assert_eq!(ring_buf[-4], 0.0);
        assert_eq!(ring_buf[-3], 1.0);
        assert_eq!(ring_buf[-2], 2.0);
        assert_eq!(ring_buf[-1], 3.0);
        assert_eq!(ring_buf[0], 0.0);
        assert_eq!(ring_buf[1], 1.0);
        assert_eq!(ring_buf[2], 2.0);
        assert_eq!(ring_buf[3], 3.0);
        assert_eq!(ring_buf[4], 0.0);
        assert_eq!(ring_buf[5], 1.0);
        assert_eq!(ring_buf[6], 2.0);
        assert_eq!(ring_buf[7], 3.0);
        assert_eq!(ring_buf[8], 0.0);
    }

    #[test]
    fn bit_mask_ring_buf_ref_index_mut() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let mut ring_buf = BMRingBufRef::new(&mut data);

        assert_eq!(&mut ring_buf[-8], &mut 0.0);
        assert_eq!(&mut ring_buf[-7], &mut 1.0);
        assert_eq!(&mut ring_buf[-6], &mut 2.0);
        assert_eq!(&mut ring_buf[-5], &mut 3.0);
        assert_eq!(&mut ring_buf[-4], &mut 0.0);
        assert_eq!(&mut ring_buf[-3], &mut 1.0);
        assert_eq!(&mut ring_buf[-2], &mut 2.0);
        assert_eq!(&mut ring_buf[-1], &mut 3.0);
        assert_eq!(&mut ring_buf[0], &mut 0.0);
        assert_eq!(&mut ring_buf[1], &mut 1.0);
        assert_eq!(&mut ring_buf[2], &mut 2.0);
        assert_eq!(&mut ring_buf[3], &mut 3.0);
        assert_eq!(&mut ring_buf[4], &mut 0.0);
        assert_eq!(&mut ring_buf[5], &mut 1.0);
        assert_eq!(&mut ring_buf[6], &mut 2.0);
        assert_eq!(&mut ring_buf[7], &mut 3.0);
        assert_eq!(&mut ring_buf[8], &mut 0.0);
    }

    #[test]
    fn bit_mask_ring_buf_ref_as_slices() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        let (s1, s2) = ring_buf.as_slices(0);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices(1);
        assert_eq!(s1, &[2.0, 3.0, 4.0]);
        assert_eq!(s2, &[1.0]);

        let (s1, s2) = ring_buf.as_slices(2);
        assert_eq!(s1, &[3.0, 4.0]);
        assert_eq!(s2, &[1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices(3);
        assert_eq!(s1, &[4.0]);
        assert_eq!(s2, &[1.0, 2.0, 3.0]);

        let (s1, s2) = ring_buf.as_slices(4);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_as_mut_slices() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ring_buf = BMRingBufRef::new(&mut data);

        let (s1, s2) = ring_buf.as_mut_slices(0);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_mut_slices(1);
        assert_eq!(s1, &[2.0, 3.0, 4.0]);
        assert_eq!(s2, &[1.0]);

        let (s1, s2) = ring_buf.as_mut_slices(2);
        assert_eq!(s1, &[3.0, 4.0]);
        assert_eq!(s2, &[1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices(3);
        assert_eq!(s1, &[4.0]);
        assert_eq!(s2, &[1.0, 2.0, 3.0]);

        let (s1, s2) = ring_buf.as_mut_slices(4);
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s2, &[]);
    }

    #[repr(C, align(1))]
    struct Aligned1([f32; 8]);

    #[repr(C, align(2))]
    struct Aligned2([f32; 8]);

    #[repr(C, align(4))]
    struct Aligned4([f32; 8]);

    #[repr(C, align(8))]
    struct Aligned8([f32; 8]);

    #[repr(C, align(16))]
    struct Aligned16([f32; 8]);

    #[repr(C, align(32))]
    struct Aligned32([f32; 8]);

    #[repr(C, align(64))]
    struct Aligned64([f32; 8]);

    #[repr(C, align(32))]
    struct Aligned324([f32; 4]);

    #[test]
    fn bit_mask_ring_buf_ref_write_latest() {
        let mut data = Aligned324([0.0f32; 4]);
        let mut ring_buf = BMRingBufRef::new(&mut data.0);

        let input = [0.0f32, 1.0, 2.0, 3.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 2.0, 3.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[3.0, 0.0, 1.0, 2.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[2.0, 3.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[1.0, 2.0, 3.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 2.0, 3.0]);

        let input = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[4.0, 5.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[6.0, 7.0, 4.0, 5.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[5.0, 6.0, 7.0, 4.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[4.0, 5.0, 6.0, 7.0]);

        let input = [0.0f32, 1.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.data, &[0.0, 0.0, 1.0, 7.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.data, &[0.0, 0.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.data, &[1.0, 0.0, 0.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.data, &[0.0, 1.0, 0.0, 0.0]);

        let aligned_input = Aligned1([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned2([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned4([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned8([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned16([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned32([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned64([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.data, &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_as_slices_len() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        let (s1, s2) = ring_buf.as_slices_len(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(0, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices_len(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(1, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_slices_len(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_len(2, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_slices_len(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_len(3, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices_len(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_len(4, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_as_mut_slices_len() {
        let mut data = [0.0f32, 1.0, 2.0, 3.0];
        let mut ring_buf = BMRingBufRef::new(&mut data);

        let (s1, s2) = ring_buf.as_mut_slices_len(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(0, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_mut_slices_len(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(1, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(2, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_len(3, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices_len(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_mut_slices_len(4, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
    }

    #[test]
    fn bit_mask_ring_buf_ref_read_into() {
        let mut data = Aligned324([0.0f32, 1.0, 2.0, 3.0]);
        let ring_buf = BMRingBufRef::new(&mut data.0);

        let mut output = [0.0f32; 4];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0]);

        let mut output = [0.0f32; 3];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0]);

        let mut output = [0.0f32; 5];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0]);
        ring_buf.read_into(&mut output, 1);
        assert_eq!(output, [1.0, 2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 2);
        assert_eq!(output, [2.0, 3.0, 0.0, 1.0, 2.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0, 3.0]);
        ring_buf.read_into(&mut output, 4);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0]);

        let mut output = [0.0f32; 10];

        ring_buf.read_into(&mut output, 0);
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);
        ring_buf.read_into(&mut output, 3);
        assert_eq!(output, [3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0]);

        let mut aligned_output = Aligned1([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned2([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned4([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned8([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned16([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned32([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let mut aligned_output = Aligned64([0.0; 8]);
        ring_buf.read_into(&mut aligned_output.0, 0);
        assert_eq!(aligned_output.0, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    }
}
