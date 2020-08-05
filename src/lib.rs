//! A fast ring buffer implementation with cheap and safe indexing written in Rust. It works
//! by bit-masking an integer index to get the corresponding index in an array/vec whose length
//! is a power of 2. This is best used when indexing the buffer with an `isize` value.
//! Copies/reads with slices are implemented with memcpy. This is most useful for high
//! performance algorithms such as audio DSP.
//!
//! This crate has no consumer/producer logic, and is meant to be used as a raw data structure
//! or a base for other data structures.
//!
//! Note, this crate has not been tested in a production environment yet. If you find any bugs,
//! especially ones pertaining to memory safety, please let me know!
//!
//! ```rust
//! use bit_mask_ring_buf::{BitMaskRingBuf, BitMaskRingBufRef};
//!
//! // Create a ring buffer with type u32. The data will be initialized with the default
//! // value (0 in this case). The actual capacity will be set to the next highest power
//! // of 2 if the given capacity is not already a power of 2.
//! let mut rb = BitMaskRingBuf::<u32>::from_capacity(3);
//! assert_eq!(rb.capacity(), 4);
//!
//! // read/write to buffer by indexing
//! rb[0] = 0;
//! rb[1] = 1;
//! rb[2] = 2;
//! rb[3] = 3;
//!
//! // cheaply wrap when reading/writing outside of bounds
//! assert_eq!(rb[-1], 3);
//! assert_eq!(rb[10], 2);
//!
//! // memcpy into slices at arbitrary points and length
//! let mut read_buffer = [0u32; 7];
//! rb.read_into(&mut read_buffer, 2);
//! assert_eq!(read_buffer, [2, 3, 0, 1, 2, 3, 0]);
//!
//! // memcpy data from a slice into the ring buffer. Only
//! // the latest data will be copied.
//! rb.write_latest(&[0, 2, 3, 4, 1], 0);
//! assert_eq!(rb[0], 1);
//! assert_eq!(rb[1], 2);
//! assert_eq!(rb[2], 3);
//! assert_eq!(rb[3], 4);
//!
//! // read/write by retreiving slices directly
//! let (s1, s2) = rb.as_slices_len(1, 4);
//! assert_eq!(s1, &[2, 3, 4]);
//! assert_eq!(s2, &[1]);
//!
//! // aligned/stack data may also be used
//! let mut stack_data = [0u32, 1, 2, 3];
//! let mut rb_ref = BitMaskRingBufRef::new(&mut stack_data);
//! rb_ref[-4] = 5;
//! assert_eq!(rb_ref[0], 5);
//! assert_eq!(rb_ref[1], 1);
//! assert_eq!(rb_ref[2], 2);
//! assert_eq!(rb_ref[3], 3);
//! ```

mod referenced;
pub use referenced::BitMaskRingBufRef;

/// Returns the next highest power of 2 if `n` is not already a power of 2.
/// This will return `2` if `n < 2`.
pub fn next_pow_of_2(n: usize) -> usize {
    if n < 2 {
        return 2;
    }

    // algorithm by wrl#0828 on Discord
    let shift = usize::MAX
        .count_ones()
        .saturating_sub((n - 1).leading_zeros());
    1usize << shift
}

static MS_TO_SEC_RATIO: f64 = 1.0 / 1000.0;

/// A fast ring buffer implementation with cheap and safe indexing. It works by bit-masking
/// an integer index to get the corresponding index in an array/vec whose length
/// is a power of 2. This is best used when indexing the buffer with an `isize` value.
/// Copies/reads with slices are implemented with memcpy.
#[allow(missing_debug_implementations)]
pub struct BitMaskRingBuf<T: Copy + Clone + Default> {
    vec: Vec<T>,
    mask: isize,
}

impl<T: Copy + Clone + Default> BitMaskRingBuf<T> {
    /// Creates a new [`BitMaskRingBuf`] with a capacity that is at least the given
    /// capacity. The buffer will be initialized with the default value.
    ///
    /// * `capacity` - The capacity of the ring buffer. The actual capacity will be set
    /// to the next highest power of 2 if `capacity` is not already a power of 2.
    ///
    /// [`BitMaskRingBuf`]: struct.BitMaskRingBuf.html
    pub fn from_capacity(capacity: usize) -> Self {
        let mut new_self = Self {
            vec: Vec::new(),
            mask: 0,
        };

        new_self.set_capacity(capacity);

        new_self
    }

    /// Creates a new [`BitMaskRingBuf`] with a capacity that holds at least a number of
    /// frames/samples in a given time peroid. The actual capacity will be set to the
    /// lowest power of 2 that can hold that many values.
    /// The buffer will be initialized with the default value.
    ///
    /// * `milliseconds` - The time period in milliseconds.
    /// * `sample_rate` - The sample rate in samples per second.
    ///
    /// ## Panics
    /// * This will panic if either `milliseconds` or `sample_rate` is less than 0.
    ///
    /// [`BitMaskRingBuf`]: struct.BitMaskRingBuf.html
    pub fn from_ms(milliseconds: f64, sample_rate: f64) -> Self {
        let mut new_self = Self {
            vec: Vec::new(),
            mask: 0,
        };

        new_self.set_capacity_from_ms(milliseconds, sample_rate);

        new_self
    }

    /// Sets the capacity of the ring buffer. The actual capacity will be set
    /// to the next highest power of 2 if `capacity` is not already a power of 2.
    /// This will also clear all values to the default value.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.vec.clear();
        self.vec.resize(next_pow_of_2(capacity), Default::default());
        self.mask = (self.vec.len() as isize) - 1;
    }

    /// Creates a new [`BitMaskRingBuf`] with a capacity that holds at least a number of
    /// frames/samples in a given time peroid. The actual capacity will be set to the
    /// lowest power of 2 that can hold that many values.
    /// This will also clear all values to the default value.
    ///
    /// * `milliseconds` - The time period in milliseconds.
    /// * `sample_rate` - The sample rate in samples per second.
    ///
    /// ## Panics
    /// * This will panic if either `milliseconds` or `sample_rate` is less than 0.
    ///
    /// [`BitMaskRingBuf`]: struct.BitMaskRingBuf.html
    pub fn set_capacity_from_ms(&mut self, milliseconds: f64, sample_rate: f64) {
        assert!(milliseconds >= 0.0);
        assert!(sample_rate >= 0.0);

        self.set_capacity((milliseconds * MS_TO_SEC_RATIO * sample_rate).ceil() as usize);
    }

    /// Clears all values in the ring buffer to the default value.
    pub fn clear(&mut self) {
        let len = self.vec.len();
        self.vec.clear();
        self.vec.resize(len, Default::default());
    }

    /// Returns two slices that contain the all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// ## Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    pub fn as_slices(&self, start: isize) -> (&[T], &[T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();
            (
                &*std::ptr::slice_from_raw_parts(self_vec_ptr.add(start), self.vec.len() - start),
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
    /// ## Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    pub fn as_slices_len(&self, start: isize, len: usize) -> (&[T], &[T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();

            let first_portion_len = self.vec.len() - start;
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
    /// ## Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    pub fn as_mut_slices(&mut self, start: isize) -> (&mut [T], &mut [T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_mut_ptr();
            (
                &mut *std::ptr::slice_from_raw_parts_mut(
                    self_vec_ptr.add(start),
                    self.vec.len() - start,
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
    /// ## Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    pub fn as_mut_slices_len(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        let start = (start & self.mask) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe {
            let self_vec_ptr = self.vec.as_mut_ptr();

            let first_portion_len = self.vec.len() - start;
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
    pub fn read_into(&self, slice: &mut [T], start: isize) {
        let start = self.constrain(start) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let self_vec_ptr = self.vec.as_ptr();
            let mut slice_ptr = slice.as_mut_ptr();
            let mut slice_len = slice.len();

            // While slice is longer than from start to the end of self.vec,
            // copy that first portion, then wrap to the beginning and copy the
            // second portion up to start.
            let first_portion_len = self.vec.len() - start;
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

            // Copy any elements remaining from start up to the end of self.vec
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

        // If slice is longer than self.vec, retreive only the latest portion
        let slice = if slice.len() > self.vec.len() {
            &slice[slice.len() - self.vec.len()..]
        } else {
            &slice[..]
        };

        // Find new starting point if slice length has changed
        let start_i = self.constrain(end_i - slice.len() as isize) as usize;

        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly calls this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        //
        // Memory cannot overlap because a mutable and immutable reference do not
        // alias.
        unsafe {
            let slice_ptr = slice.as_ptr();
            let self_vec_ptr = self.vec.as_mut_ptr();

            // If the slice is longer than from start_i to the end of self.vec, copy that
            // first portion, then wrap to the beginning and copy the remaining second portion.
            if start_i + slice.len() > self.vec.len() {
                let first_portion_len = self.vec.len() - start_i;
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
        self.vec.len()
    }

    /// Returns the actual index of the ring buffer from the given
    /// `i` index. This is cheap due to the ring buffer's bit-masking
    /// algorithm.
    pub fn constrain(&self, i: isize) -> isize {
        i & self.mask
    }
}

impl<T: Copy + Clone + Default> std::ops::Index<isize> for BitMaskRingBuf<T> {
    type Output = T;
    fn index(&self, i: isize) -> &T {
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &*self.vec.as_ptr().offset(i & self.mask) }
    }
}

impl<T: Copy + Clone + Default> std::ops::IndexMut<isize> for BitMaskRingBuf<T> {
    fn index_mut(&mut self, i: isize) -> &mut T {
        // Safe because of the algorithm of bit-masking the index on an array/vec
        // whose length is a power of 2.
        //
        // Both the length of self.vec and the value of self.mask are only modified
        // in self.set_capacity(). This function makes sure these values are valid.
        // The constructors also correctly call this function.
        //
        // Memory is created and initialized by a Vec, so it is always valid.
        unsafe { &mut *self.vec.as_mut_ptr().offset(i & self.mask) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn next_pow_of_2_test() {
        assert_eq!(next_pow_of_2(0), 2);
        assert_eq!(next_pow_of_2(1), 2);
        assert_eq!(next_pow_of_2(2), 2);
        assert_eq!(next_pow_of_2(30), 32);
        assert_eq!(next_pow_of_2(127), 128);
        assert_eq!(next_pow_of_2(128), 128);
        assert_eq!(next_pow_of_2(129), 256);
        assert_eq!(next_pow_of_2(4000), 4096);
        assert_eq!(next_pow_of_2(5000), 8192);
    }

    #[test]
    fn bit_mask_ring_buf_initialize() {
        let ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);

        assert_eq!(&ring_buf.vec[..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_constrain() {
        let ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);

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
    fn bit_mask_ring_buf_clear() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);

        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        ring_buf.clear();
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_index() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[0.0f32, 1.0, 2.0, 3.0], 0);

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
    fn bit_mask_ring_buf_index_mut() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[0.0f32, 1.0, 2.0, 3.0], 0);

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
    fn bit_mask_ring_buf_as_slices() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);

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
    fn bit_mask_ring_buf_as_mut_slices() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[1.0f32, 2.0, 3.0, 4.0], 0);

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

    #[test]
    fn bit_mask_ring_buf_write_latest() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);

        let input = [0.0f32, 1.0, 2.0, 3.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 0.0, 1.0, 2.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[2.0, 3.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

        let input = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[6.0, 7.0, 4.0, 5.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[5.0, 6.0, 7.0, 4.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 6.0, 7.0]);

        let input = [0.0f32, 1.0];

        ring_buf.write_latest(&input, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 6.0, 7.0]);
        ring_buf.write_latest(&input, 1);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 1.0, 7.0]);
        ring_buf.write_latest(&input, 2);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 0.0, 0.0, 1.0]);
        ring_buf.write_latest(&input, 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 0.0, 0.0, 0.0]);
        ring_buf.write_latest(&input, 4);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0, 1.0, 0.0, 0.0]);

        let aligned_input = Aligned1([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned2([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned4([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned8([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned16([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned32([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);

        let aligned_input = Aligned64([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        ring_buf.write_latest(&aligned_input.0, 0);
        assert_eq!(ring_buf.vec.as_slice(), &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn bit_mask_ring_buf_as_slices_len() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

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
    fn bit_mask_ring_buf_as_mut_slices_len() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

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
    fn bit_mask_ring_buf_read_into() {
        let mut ring_buf = BitMaskRingBuf::<f32>::from_capacity(4);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

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
