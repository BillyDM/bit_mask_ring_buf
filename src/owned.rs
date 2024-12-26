use alloc::vec::Vec;
use core::fmt::Debug;
use core::num::NonZeroUsize;

use crate::{
    inner::{self, Mask},
    next_pow_of_2,
};

/// A fast ring buffer implementation with cheap and safe indexing. It works by bit-masking
/// an integer index to get the corresponding index in an array/vec whose length
/// is a power of 2. This is best used when indexing the buffer with an `isize` value.
/// Copies/reads with slices are implemented with memcpy.
///
/// This struct has no consumer/producer logic, and is meant to be used for DSP or as
/// a base for other data structures.
///
/// The length of this ring buffer cannot be `0`.
///
/// //! ## Example
/// ```rust
/// # use bit_mask_ring_buf::BitMaskRB;
/// // Create a ring buffer with type u32. The data will be
/// // initialized with the given value (0 in this case).
/// // The actual length will be set to the next highest
/// // power of 2 if the given length is not already
/// // a power of 2.
/// let mut rb = BitMaskRB::<u32>::new(3, 0);
/// assert_eq!(rb.len().get(), 4);
///
/// // Read/write to buffer by indexing with an `isize`.
/// rb[0] = 0;
/// rb[1] = 1;
/// rb[2] = 2;
/// rb[3] = 3;
///
/// // Cheaply wrap when reading/writing outside of bounds.
/// assert_eq!(rb[-1], 3);
/// assert_eq!(rb[10], 2);
///
/// // Memcpy into slices at arbitrary `isize` indexes
/// // and length.
/// let mut read_buffer = [0u32; 7];
/// rb.read_into(&mut read_buffer, 2);
/// assert_eq!(read_buffer, [2, 3, 0, 1, 2, 3, 0]);
///
/// // Memcpy data from a slice into the ring buffer at
/// // arbitrary `isize` indexes. Earlier data will not be
/// // copied if it will be overwritten by newer data,
/// // avoiding unecessary memcpy's. The correct placement
/// // of the newer data will still be preserved.
/// rb.write_latest(&[0, 2, 3, 4, 1], 0);
/// assert_eq!(rb[0], 1);
/// assert_eq!(rb[1], 2);
/// assert_eq!(rb[2], 3);
/// assert_eq!(rb[3], 4);
///
/// // Read/write by retrieving slices directly.
/// let (s1, s2) = rb.as_slices_len(1, 4);
/// assert_eq!(s1, &[2, 3, 4]);
/// assert_eq!(s2, &[1]);
///
/// # #[cfg(feature = "interpolation")] {
/// // Linear interpolation is also provided (requires the
/// // `interpolation` feature which requires the standard
/// // library.)
/// let rb = BitMaskRB::<f32>::from_vec(vec![0.0, 2.0, 4.0, 6.0]);
/// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f32::EPSILON);
/// # }
/// ```
pub struct BitMaskRB<T> {
    vec: Vec<T>,
    mask: Mask,
}

impl<T> BitMaskRB<T> {
    /// Creates a new [`BitMaskRB`] with the given vec as its data source.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len()` is less than two or is not a power of two.
    ///
    /// # Example
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::from_vec(vec![0, 1, 2, 3]);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// assert_eq!(rb[-3], 1);
    /// ```
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = next_pow_of_2(vec.len());
        assert_eq!(vec.len(), len);

        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Creates a new [`BitMaskRB`] with a length that is at least the given
    /// length. The data in the buffer will not be initialized.
    ///
    /// * `len` - The length of the ring buffer. The actual length will be set
    /// to the next highest power of 2 if `len` is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// unsafe {
    ///     let rb = BitMaskRB::<u32>::new_uninit(3);
    ///     assert_eq!(rb.len().get(), 4);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if `len > (core::usize::MAX/2)+1`.
    ///
    /// [`BitMaskRB`]: struct.BitMaskRB.html
    pub unsafe fn new_uninit(len: usize) -> Self {
        let len = next_pow_of_2(len);

        let mut vec = Vec::<T>::with_capacity(len);
        vec.set_len(len);

        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Creates a new [`BitMaskRB`] with an allocated capacity equal to exactly the
    /// given length. No data will be initialized.
    ///
    /// * `len` - The length of the ring buffer. The actual length will be set
    /// to the next highest power of 2 if `len` is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// unsafe {
    ///     let rb = BitMaskRB::<u32>::new_exact_uninit(3);
    ///     assert_eq!(rb.len().get(), 4);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if `len > (core::usize::MAX/2)+1`.
    pub unsafe fn new_exact_uninit(len: usize) -> Self {
        let len = next_pow_of_2(len);

        let mut vec = Vec::new();
        vec.reserve_exact(len);
        vec.set_len(len);

        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Creates a new [`BitMaskRB`] with a length that is at least the given
    /// length, while reserving extra capacity for future changes to `len`.
    /// The data in the buffer will not be initialized.
    ///
    /// * `len` - The length of the ring buffer. The actual length will be set
    /// to the next highest power of 2 if `len` is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    /// * `capacity` - The allocated capacity of the ring buffer. The actual capacity
    /// will be set to the next highest power of 2 if `capacity` is not already a power of 2.
    /// The capacity will be set to 2 if `capacity < 2`. If this is less than `len`, then it
    /// will be ignored.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// unsafe {
    ///     let rb = BitMaskRB::<u32>::with_capacity_uninit(3, 15);
    ///     assert_eq!(rb.len().get(), 4);
    ///     assert!(rb.capacity().get() >= 16);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if `len > (core::usize::MAX/2)+1`.
    ///
    /// [`BitMaskRB`]: struct.BitMaskRB.html
    pub unsafe fn with_capacity_uninit(len: usize, capacity: usize) -> Self {
        let len = next_pow_of_2(len);
        let capacity = next_pow_of_2(capacity);

        let mut vec = Vec::<T>::with_capacity(core::cmp::max(len, capacity));
        vec.set_len(len);

        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Sets the length of the ring buffer without initializing any newly allocated data.
    ///
    /// * If the resulting length is less than the current length, then the data
    /// will be truncated.
    /// * If the resulting length is larger than the current length, then all newly
    /// allocated elements appended to the end will be unitialized.
    ///
    /// # Safety
    ///
    /// * Undefined behavior may occur if uninitialized data is read from. By using
    /// this you assume the responsibility of making sure any data is initialized
    /// before it is read.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// unsafe {
    ///     rb.set_len_uninit(3);
    ///
    ///     assert_eq!(rb.len().get(), 4);
    ///
    ///     assert_eq!(rb[0], 1);
    ///     assert_eq!(rb[1], 2);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if `len > (core::usize::MAX/2)+1`.
    pub unsafe fn set_len_uninit(&mut self, len: usize) {
        let len = next_pow_of_2(len);

        if len != self.vec.len() {
            if len > self.vec.len() {
                // Extend without initializing.
                self.vec.reserve(len - self.vec.len());
            }
            self.vec.set_len(len);

            self.mask = Mask::new(self.vec.len());
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the internal `Vec`. This is equivalant to `Vec::reserve()`.
    ///
    /// The actual capacity will be set to the next highest power of 2 if the
    /// resulting capacity is not already a power of 2.
    /// The capacity will be set to 2 if the resulting capacity is less than 2.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore,
    /// capacity can not be relied upon to be precisely minimal.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    ///
    /// rb.reserve(8);
    ///
    /// // next_pow_of_2(2 + 8) == 16
    /// assert!(rb.capacity().get() >= 16);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if the new capacity overflows `usize`.
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if the resulting length is greater than `(core::usize::MAX/2)+1`.
    pub fn reserve(&mut self, additional: usize) {
        let total_len = next_pow_of_2(self.vec.len() + additional);
        if total_len > self.vec.len() {
            self.vec.reserve(total_len - self.vec.len());
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the internal `Vec`. This is equivalant to `Vec::reserve_exact()`.
    ///
    /// The actual capacity will be set to the next highest power of 2 if the
    /// resulting capacity is not already a power of 2.
    /// The capacity will be set to 2 if the resulting capacity is less than 2.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore,
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    ///
    /// rb.reserve_exact(8);
    ///
    /// // next_pow_of_2(2 + 8) == 16
    /// assert!(rb.capacity().get() >= 16);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if the new capacity overflows `usize`.
    /// * This will panic if allocation failed due to out of memory.
    /// * This will panic if the resulting length is greater than `(core::usize::MAX/2)+1`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let total_len = next_pow_of_2(self.vec.len() + additional);
        if total_len > self.vec.len() {
            self.vec.reserve_exact(total_len - self.vec.len());
        }
    }

    /// Shrinks the capacity of the internal `Vec` as much as possible. This is equivalant to
    /// `Vec::shrink_to_fit`.
    ///
    /// It will drop down as close as possible to the length but the allocator may still inform
    /// the vector that there is space for a few more elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    ///
    /// rb.reserve(8);
    /// // next_pow_of_2(2 + 8) == 16
    /// assert!(rb.capacity().get() >= 16);
    ///
    /// rb.shrink_to_fit();
    /// assert!(rb.capacity().get() >= 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit();
    }

    /// Returns two slices that contain all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices(-4);
    /// assert_eq!(s1, &[1, 2, 3, 4]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices(3);
    /// assert_eq!(s1, &[4]);
    /// assert_eq!(s2, &[1, 2, 3]);
    /// ```
    pub fn as_slices(&self, start: isize) -> (&[T], &[T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_slices(start, self.mask, &self.vec) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices_len(-4, 3);
    /// assert_eq!(s1, &[1, 2, 3]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices_len(3, 5);
    /// assert_eq!(s1, &[4]);
    /// assert_eq!(s2, &[1, 2, 3]);
    /// ```
    pub fn as_slices_len(&self, start: isize, len: usize) -> (&[T], &[T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_slices_len(start, len, self.mask, &self.vec) }
    }

    /// Returns two slices of data in the ring buffer
    /// starting at the index `start` and with length `len`. If `len` is greater
    /// than the length of the ring buffer, then the buffer's length will be used
    /// instead, while still preserving the position of the last element.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead, while
    /// still preserving the position of the last element.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_slices_latest(-4, 3);
    /// assert_eq!(s1, &[1, 2, 3]);
    /// assert_eq!(s2, &[]);
    ///
    /// let (s1, s2) = rb.as_slices_latest(0, 5);
    /// assert_eq!(s1, &[2, 3, 4]);
    /// assert_eq!(s2, &[1]);
    /// ```
    pub fn as_slices_latest(&self, start: isize, len: usize) -> (&[T], &[T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_slices_latest(start, len, self.mask, &self.vec) }
    }

    /// Returns two mutable slices that contain all the data in the ring buffer
    /// starting at the index `start`.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data. This will never be empty.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices(-4);
    /// assert_eq!(s1, &mut [1, 2, 3, 4]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices(3);
    /// assert_eq!(s1, &mut [4]);
    /// assert_eq!(s2, &mut [1, 2, 3]);
    /// ```
    pub fn as_mut_slices(&mut self, start: isize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_mut_slices(start, self.mask, &mut self.vec) }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices_len(-4, 3);
    /// assert_eq!(s1, &mut [1, 2, 3]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices_len(3, 5);
    /// assert_eq!(s1, &mut [4]);
    /// assert_eq!(s2, &mut [1, 2, 3]);
    /// ```
    pub fn as_mut_slices_len(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_mut_slices_len(start, len, self.mask, &mut self.vec) }
    }

    /// Returns two mutable slices of data in the ring buffer
    /// starting at the index `start` and with length `len`. If `len` is greater
    /// than the length of the ring buffer, then the buffer's length will be used
    /// instead, while still preserving the position of the last element.
    ///
    /// * `start` - The starting index
    /// * `len` - The length of data to read. If `len` is greater than the
    /// length of the ring buffer, then the buffer's length will be used instead, while
    /// still preserving the position of the last element.
    ///
    /// # Returns
    ///
    /// * The first slice is the starting chunk of data.
    /// * The second slice is the second contiguous chunk of data. This may
    /// or may not be empty depending if the buffer needed to wrap around to the beginning of
    /// its internal memory layout.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let (s1, s2) = rb.as_mut_slices_latest(-4, 3);
    /// assert_eq!(s1, &mut [1, 2, 3]);
    /// assert_eq!(s2, &mut []);
    ///
    /// let (s1, s2) = rb.as_mut_slices_latest(0, 5);
    /// assert_eq!(s1, &mut [2, 3, 4]);
    /// assert_eq!(s2, &mut [1]);
    /// ```
    pub fn as_mut_slices_latest(&mut self, start: isize, len: usize) -> (&mut [T], &mut [T]) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::as_mut_slices_latest(start, len, self.mask, &mut self.vec) }
    }

    /// Returns the length of the ring buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::new(4, 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        // * All constructors ensure that the length is greater than `0`.
        unsafe { NonZeroUsize::new_unchecked(self.vec.len()) }
    }

    /// Returns the allocated capacity of the ring buffer.
    ///
    /// Please note this is not the same as the length of the buffer.
    /// For that use BitMaskRB::len().
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::new(4, 0);
    ///
    /// assert!(rb.capacity().get() >= 4);
    /// ```
    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        // * All constructors ensure that the length is greater than `0`.
        unsafe { NonZeroUsize::new_unchecked(self.vec.capacity()) }
    }

    /// Returns the actual index of the ring buffer from the given
    /// `i` index. This is cheap due to the ring buffer's bit-masking
    /// algorithm. This is useful to keep indexes from growing indefinitely.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::new(4, 0);
    ///
    /// assert_eq!(rb.constrain(2), 2);
    /// assert_eq!(rb.constrain(4), 0);
    /// assert_eq!(rb.constrain(-3), 1);
    /// assert_eq!(rb.constrain(7), 3);
    /// ```
    #[inline(always)]
    pub fn constrain(&self, i: isize) -> isize {
        inner::constrain(i, self.mask)
    }

    /// Returns all the data in the buffer. The starting index will
    /// always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let raw_data = rb.raw_data();
    /// assert_eq!(raw_data, &[1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data(&self) -> &[T] {
        &self.vec[..]
    }

    /// Returns all the data in the buffer as mutable. The starting
    /// index will always be `0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let raw_data = rb.raw_data_mut();
    /// assert_eq!(raw_data, &mut [1u32, 2, 3, 4]);
    /// ```
    pub fn raw_data_mut(&mut self) -> &mut [T] {
        &mut self.vec[..]
    }

    /// Returns an immutable reference the element at the index of type `isize`. This
    /// is cheap due to the ring buffer's bit-masking algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// assert_eq!(*rb.get(-3), 2);
    /// ```
    #[inline(always)]
    pub fn get(&self, i: isize) -> &T {
        // SAFETY:
        // * The constructors ensure that `self.data.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.data.len() - 1`.
        unsafe { inner::get(i, self.mask, &self.vec) }
    }

    /// Returns a mutable reference the element at the index of type `isize`. This
    /// is cheap due to the ring buffer's bit-masking algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// *rb.get_mut(-3) = 5;
    ///
    /// assert_eq!(rb[-3], 5);
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, i: isize) -> &mut T {
        // SAFETY:
        // * The constructors ensure that `self.data.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.data.len() - 1`.
        unsafe { inner::get_mut(i, self.mask, &mut self.vec) }
    }

    /// Returns an immutable reference to the element at the index of type `isize`
    /// while also constraining the index `i`. This is more efficient than calling
    /// both methods individually.
    ///
    /// This is cheap due to the ring buffer's bit-masking algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut i = -3;
    /// assert_eq!(*rb.constrain_and_get(&mut i), 2);
    /// assert_eq!(i, 1);
    /// ```
    #[inline(always)]
    pub fn constrain_and_get(&self, i: &mut isize) -> &T {
        // SAFETY:
        // * The constructors ensure that `self.data.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.data.len() - 1`.
        unsafe { inner::constrain_and_get(i, self.mask, &self.vec) }
    }

    /// Returns a mutable reference to the element at the index of type `isize`
    /// while also constraining the index `i`. This is more efficient than calling
    /// both methods individually.
    ///
    /// This is cheap due to the ring buffer's bit-masking algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut i = -3;
    /// *rb.constrain_and_get_mut(&mut i) = 5;
    ///
    /// assert_eq!(rb[i], 5);
    /// assert_eq!(i, 1);
    /// ```
    #[inline(always)]
    pub fn constrain_and_get_mut(&mut self, i: &mut isize) -> &mut T {
        // SAFETY:
        // * The constructors ensure that `self.data.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.data.len() - 1`.
        unsafe { inner::constrain_and_get_mut(i, self.mask, &mut self.vec) }
    }
}

impl<T: Clone> BitMaskRB<T> {
    /// Creates a new [`BitMaskRB`] with a length that is at least the given
    /// length. The buffer will be initialized with the given value.
    ///
    /// * `len` - The length of the ring buffer. The actual length will be set
    /// to the next highest power of 2 if `len` is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::new(3, 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory
    /// * This will panic if `len > (core::usize::MAX/2)+1`
    pub fn new(len: usize, value: T) -> Self {
        let len = next_pow_of_2(len);

        let vec: Vec<T> = alloc::vec![value; len];
        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Creates a new [`BitMaskRB`] with a length that is at least the given
    /// length, while reserving extra capacity for future changes to `len`.
    /// All data from `[0..len)` will be initialized with the given value.
    ///
    /// * `len` - The length of the ring buffer. The actual length will be set
    /// to the next highest power of 2 if `len` is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    /// * `capacity` - The allocated capacity of the ring buffer. The actual capacity
    /// will be set to the next highest power of 2 if `capacity` is not already a power of 2.
    /// The capacity will be set to 2 if `capacity < 2`. If this is less than `len`, then it
    /// will be ignored.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let rb = BitMaskRB::<u32>::with_capacity(3, 15, 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    /// assert!(rb.capacity().get() >= 16);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory
    /// * This will panic if `len > (core::usize::MAX/2)+1`
    /// * This will panic if `capacity > (core::usize::MAX/2)+1`
    pub fn with_capacity(len: usize, capacity: usize, value: T) -> Self {
        let len = next_pow_of_2(len);
        let capacity = next_pow_of_2(capacity);

        let mut vec = Vec::<T>::with_capacity(core::cmp::max(len, capacity));
        vec.resize(len, value);

        let mask = Mask::new(vec.len());

        Self { vec, mask }
    }

    /// Sets the length of the ring buffer while clearing all values to the default value.
    ///
    /// The actual length will be set to the next highest power of 2 if `len`
    /// is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.clear_set_len(3, 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 0);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory
    /// * This will panic if `len > (core::usize::MAX/2)+1`
    pub fn clear_set_len(&mut self, len: usize, value: T) {
        self.vec.clear();
        self.vec.resize(next_pow_of_2(len), value);
        self.mask = Mask::new(self.vec.len());
    }

    /// Sets the length of the ring buffer.
    ///
    /// * If the resulting length is less than the current length, then the data
    /// will be truncated.
    /// * If the resulting length is larger than the current length, then all newly
    /// allocated elements appended to the end will be initialized with the given value.
    ///
    /// The actual length will be set to the next highest power of 2 if `len`
    /// is not already a power of 2.
    /// The length will be set to 2 if `len < 2`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(2, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    ///
    /// rb.set_len(3, 0);
    ///
    /// assert_eq!(rb.len().get(), 4);
    ///
    /// assert_eq!(rb[0], 1);
    /// assert_eq!(rb[1], 2);
    /// assert_eq!(rb[2], 0);
    /// assert_eq!(rb[3], 0);
    /// ```
    ///
    /// # Panics
    ///
    /// * This will panic if allocation failed due to out of memory
    /// * This will panic if `len > (core::usize::MAX/2)+1`
    pub fn set_len(&mut self, len: usize, value: T) {
        let len = next_pow_of_2(len);

        if len != self.vec.len() {
            self.vec.resize(len, value);
            self.mask = Mask::new(self.vec.len());
        }
    }
}

impl<T: Clone + Copy> BitMaskRB<T> {
    /// Copies the data from the ring buffer starting from the index `start`
    /// into the given slice. If the length of `slice` is larger than the
    /// length of the ring buffer, then the data will be reapeated until
    /// the given slice is filled.
    ///
    /// * `slice` - This slice to copy the data into.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    /// rb[0] = 1;
    /// rb[1] = 2;
    /// rb[2] = 3;
    /// rb[3] = 4;
    ///
    /// let mut read_buf = [0u32; 3];
    /// rb.read_into(&mut read_buf[..], -3);
    /// assert_eq!(read_buf, [2, 3, 4]);
    ///
    /// let mut read_buf = [0u32; 9];
    /// rb.read_into(&mut read_buf[..], 2);
    /// assert_eq!(read_buf, [3, 4, 1, 2, 3, 4, 1, 2, 3]);
    /// ```
    pub fn read_into(&self, slice: &mut [T], start: isize) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::read_into(slice, start, self.mask, &self.vec) }
    }

    /// Copies data from the given slice into the ring buffer starting from
    /// the index `start`.
    ///
    /// Earlier data will not be copied if it will be
    /// overwritten by newer data, avoiding unecessary memcpy's. The correct
    /// placement of the newer data will still be preserved.
    ///
    /// * `slice` - This slice to copy data from.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<u32>::new(4, 0);
    ///
    /// let input = [1u32, 2, 3];
    /// rb.write_latest(&input[..], -3);
    /// assert_eq!(rb[0], 0);
    /// assert_eq!(rb[1], 1);
    /// assert_eq!(rb[2], 2);
    /// assert_eq!(rb[3], 3);
    ///
    /// let input = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];
    /// rb.write_latest(&input[..], 2);
    /// assert_eq!(rb[0], 7);
    /// assert_eq!(rb[1], 8);
    /// assert_eq!(rb[2], 9);
    /// assert_eq!(rb[3], 6);
    /// ```
    pub fn write_latest(&mut self, slice: &[T], start: isize) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::write_latest(slice, start, self.mask, &mut self.vec) }
    }

    /// Copies data from two given slices into the ring buffer starting from
    /// the index `start`. The `first` slice will be copied first then `second`
    /// will be copied next.
    ///
    /// Earlier data will not be copied if it will be
    /// overwritten by newer data, avoiding unecessary memcpy's. The correct
    /// placement of the newer data will still be preserved.
    ///
    /// * `first` - This first slice to copy data from.
    /// * `second` - This second slice to copy data from.
    /// * `start` - The index of the ring buffer to start copying from.
    ///
    /// # Performance
    ///
    /// Prefer to use this to manipulate data in bulk over indexing one element at a time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut input_rb = BitMaskRB::<u32>::new(4, 0);
    /// input_rb[0] = 1;
    /// input_rb[1] = 2;
    /// input_rb[2] = 3;
    /// input_rb[3] = 4;
    ///
    /// let mut output_rb = BitMaskRB::<u32>::new(4, 0);
    /// // s1 == &[1, 2], s2 == &[]
    /// let (s1, s2) = input_rb.as_slices_len(0, 2);
    /// output_rb.write_latest_2(s1, s2, -3);
    /// assert_eq!(output_rb[0], 0);
    /// assert_eq!(output_rb[1], 1);
    /// assert_eq!(output_rb[2], 2);
    /// assert_eq!(output_rb[3], 0);
    ///
    /// let mut output_rb = BitMaskRB::<u32>::new(2, 0);
    /// // s1 == &[4],  s2 == &[1, 2, 3]
    /// let (s1, s2) = input_rb.as_slices_len(3, 4);
    /// // rb[1] = 4  ->  rb[0] = 1  ->  rb[1] = 2  ->  rb[0] = 3
    /// output_rb.write_latest_2(s1, s2, 1);
    /// assert_eq!(output_rb[0], 3);
    /// assert_eq!(output_rb[1], 2);
    /// ```
    pub fn write_latest_2(&mut self, first: &[T], second: &[T], start: isize) {
        // SAFETY:
        // * The constructors ensure that `self.vec.len()` is greater than `0` and
        // equal to a power of 2, and they ensure that `self.mask == self.vec.len() - 1`.
        unsafe { inner::write_latest_2(first, second, start, self.mask, &mut self.vec) }
    }
}

impl<T> core::ops::Index<isize> for BitMaskRB<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: isize) -> &T {
        self.get(i)
    }
}

impl<T> core::ops::IndexMut<isize> for BitMaskRB<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: isize) -> &mut T {
        self.get_mut(i)
    }
}

impl<T: Clone> Clone for BitMaskRB<T> {
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
            mask: self.mask,
        }
    }
}

impl<T: Debug> Debug for BitMaskRB<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut f = f.debug_struct("BitMaskRB");
        f.field("vec", &self.vec);
        f.field("mask", &self.mask);
        f.finish()
    }
}

impl<T> Into<Vec<T>> for BitMaskRB<T> {
    fn into(self) -> Vec<T> {
        self.vec
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
        assert_eq!(
            next_pow_of_2(core::usize::MAX / 2),
            (core::usize::MAX / 2) + 1
        );
        assert_eq!(
            next_pow_of_2((core::usize::MAX / 2) + 1),
            (core::usize::MAX / 2) + 1
        );
    }

    #[test]
    #[should_panic]
    fn next_pow_of_2_panic_test() {
        assert_eq!(next_pow_of_2((core::usize::MAX / 2) + 2), core::usize::MAX);
    }

    #[test]
    fn bit_mask_ring_buf_initialize() {
        let ring_buf = BitMaskRB::<f32>::new(3, 0.0);

        assert_eq!(&ring_buf.vec[..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_initialize_uninit() {
        unsafe {
            let ring_buf = BitMaskRB::<f32>::new_uninit(3);

            assert_eq!(ring_buf.vec.len(), 4);
        }
    }

    #[test]
    fn bit_mask_ring_buf_clear_set_len() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.clear_set_len(8, 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[0.0; 8]);
    }

    #[test]
    fn bit_mask_ring_buf_set_len() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        ring_buf.set_len(1, 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0]);

        ring_buf.set_len(4, 0.0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn bit_mask_ring_buf_set_len_uninit() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
        ring_buf[0] = 1.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 3.0;
        ring_buf[3] = 4.0;

        unsafe {
            ring_buf.set_len_uninit(1);
        }

        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0]);
        assert_eq!(ring_buf.vec.len(), 2);

        unsafe {
            ring_buf.set_len_uninit(4);
        }

        assert_eq!(ring_buf.vec.len(), 4);
    }

    #[test]
    fn bit_mask_ring_buf_constrain() {
        let ring_buf = BitMaskRB::<f32>::new(4, 0.0);

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
    fn bit_mask_ring_buf_index() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
    fn bit_mask_ring_buf_write_latest_2() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);

        ring_buf.write_latest_2(&[], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[-1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[2.0, 3.0, 4.0, 1.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[-2.0, -1.0], &[0.0, 1.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[-1.0, 0.0, 1.0, -2.0]);
        ring_buf.write_latest_2(&[0.0, 1.0], &[2.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 0.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0], &[], 0);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        ring_buf.write_latest_2(&[1.0, 2.0], &[], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[], &[], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[1.0, 2.0, 1.0, 2.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[], 1);
        assert_eq!(ring_buf.vec.as_slice(), &[4.0, 5.0, 2.0, 3.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[3.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0], 2);
        assert_eq!(ring_buf.vec.as_slice(), &[7.0, 4.0, 5.0, 6.0]);
        ring_buf.write_latest_2(&[1.0, 2.0, 3.0, 4.0, 5.0], &[6.0, 7.0, 8.0, 9.0, 10.0], 3);
        assert_eq!(ring_buf.vec.as_slice(), &[10.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn bit_mask_ring_buf_write_latest() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);

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
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
    fn bit_mask_ring_buf_as_slices_latest() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_slices_latest(0, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 6);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 7);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(0, 10);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);

        let (s1, s2) = ring_buf.as_slices_latest(1, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 1);
        assert_eq!(s1, &[1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 2);
        assert_eq!(s1, &[1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 3);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 4);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 5);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 6);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 7);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(1, 10);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_slices_latest(2, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 1);
        assert_eq!(s1, &[2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 2);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 3);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 4);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 5);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 6);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 7);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(2, 10);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);

        let (s1, s2) = ring_buf.as_slices_latest(3, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 1);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 2);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 3);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 4);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 5);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 6);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 7);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(3, 10);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);

        let (s1, s2) = ring_buf.as_slices_latest(4, 0);
        assert_eq!(s1, &[]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 1);
        assert_eq!(s1, &[0.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 2);
        assert_eq!(s1, &[0.0, 1.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 3);
        assert_eq!(s1, &[0.0, 1.0, 2.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 4);
        assert_eq!(s1, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &[]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 5);
        assert_eq!(s1, &[1.0, 2.0, 3.0]);
        assert_eq!(s2, &[0.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 6);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 7);
        assert_eq!(s1, &[3.0]);
        assert_eq!(s2, &[0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_slices_latest(4, 10);
        assert_eq!(s1, &[2.0, 3.0]);
        assert_eq!(s2, &[0.0, 1.0]);
    }

    #[test]
    fn bit_mask_ring_buf_as_mut_slices_len() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
    fn bit_mask_ring_buf_as_mut_slices_latest() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
        ring_buf.write_latest(&[0.0, 1.0, 2.0, 3.0], 0);

        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 1);
        assert_eq!(s1, &mut [0.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 2);
        assert_eq!(s1, &mut [0.0, 1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 3);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 4);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 5);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 6);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 7);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(0, 10);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 1);
        assert_eq!(s1, &mut [1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 2);
        assert_eq!(s1, &mut [1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 3);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 4);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 5);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 6);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 7);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(1, 10);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 1);
        assert_eq!(s1, &mut [2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 2);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 3);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 4);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 5);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 6);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 7);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(2, 10);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);

        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 1);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 2);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 3);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 4);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 5);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 6);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 7);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(3, 10);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);

        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 0);
        assert_eq!(s1, &mut []);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 1);
        assert_eq!(s1, &mut [0.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 2);
        assert_eq!(s1, &mut [0.0, 1.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 3);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 4);
        assert_eq!(s1, &mut [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut []);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 5);
        assert_eq!(s1, &mut [1.0, 2.0, 3.0]);
        assert_eq!(s2, &mut [0.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 6);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 7);
        assert_eq!(s1, &mut [3.0]);
        assert_eq!(s2, &mut [0.0, 1.0, 2.0]);
        let (s1, s2) = ring_buf.as_mut_slices_latest(4, 10);
        assert_eq!(s1, &mut [2.0, 3.0]);
        assert_eq!(s2, &mut [0.0, 1.0]);
    }

    #[test]
    fn bit_mask_ring_buf_read_into() {
        let mut ring_buf = BitMaskRB::<f32>::new(4, 0.0);
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
