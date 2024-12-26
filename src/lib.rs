//! A ring buffer implementation with cheap indexing written in Rust.
//!
//! Note, this crate is only beneficial if your algorithm indexes elements one at a time
//! and has buffer sizes that are always a power of two. If your algorithm instead reads
//! chunks of data as slices or requires buffer sizes that are not a power of two, then
//! check out my crate [`slice_ring_buf`].
//!
//! This crate has no consumer/producer logic, and is meant to be used for DSP or as
//! a base for other data structures.
//!
//! This crate can also be used without the standard library (`#![no_std]`).
//!
//! ## Example
//! ```rust
//! use bit_mask_ring_buf::{BitMaskRB, BitMaskRbRefMut};
//!
//! // Create a ring buffer with type u32. The data will be
//! // initialized with the given value (0 in this case).
//! // The actual length will be set to the next highest
//! // power of 2 if the given length is not already
//! // a power of 2.
//! let mut rb = BitMaskRB::<u32>::new(3, 0);
//! assert_eq!(rb.len().get(), 4);
//!
//! // Read/write to buffer by indexing with an `isize`.
//! rb[0] = 0;
//! rb[1] = 1;
//! rb[2] = 2;
//! rb[3] = 3;
//!
//! // Cheaply wrap when reading/writing outside of bounds.
//! assert_eq!(rb[-1], 3);
//! assert_eq!(rb[10], 2);
//!
//! // Memcpy into slices at arbitrary `isize` indexes
//! // and length.
//! let mut read_buffer = [0u32; 7];
//! rb.read_into(&mut read_buffer, 2);
//! assert_eq!(read_buffer, [2, 3, 0, 1, 2, 3, 0]);
//!
//! // Memcpy data from a slice into the ring buffer at
//! // arbitrary `isize` indexes. Earlier data will not be
//! // copied if it will be overwritten by newer data,
//! // avoiding unecessary memcpy's. The correct placement
//! // of the newer data will still be preserved.
//! rb.write_latest(&[0, 2, 3, 4, 1], 0);
//! assert_eq!(rb[0], 1);
//! assert_eq!(rb[1], 2);
//! assert_eq!(rb[2], 3);
//! assert_eq!(rb[3], 4);
//!
//! // Read/write by retrieving slices directly.
//! let (s1, s2) = rb.as_slices_len(1, 4);
//! assert_eq!(s1, &[2, 3, 4]);
//! assert_eq!(s2, &[1]);
//!
//! // Aligned/stack data may also be used.
//! let mut stack_data = [0u32, 1, 2, 3];
//! let mut rb_ref = BitMaskRbRefMut::new(&mut stack_data);
//! rb_ref[-4] = 5;
//! assert_eq!(rb_ref[0], 5);
//! assert_eq!(rb_ref[1], 1);
//! assert_eq!(rb_ref[2], 2);
//! assert_eq!(rb_ref[3], 3);
//!
//! # #[cfg(feature = "interpolation")] {
//! // Linear interpolation is also provided (requires the
//! // `interpolation` feature which requires the standard
//! // library.)
//! let rb = BitMaskRB::<f32>::from_vec(vec![0.0, 2.0, 4.0, 6.0]);
//! assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f32::EPSILON);
//! # }
//! ```
//!
//! [`slice_ring_buf`]: https://crates.io/crates/slice_ring_buf

#![cfg_attr(not(feature = "interpolation"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod inner;
mod referenced;

pub use referenced::{BitMaskRbRef, BitMaskRbRefMut};

#[cfg(feature = "alloc")]
mod owned;
#[cfg(feature = "alloc")]
pub use owned::BitMaskRB;

#[cfg(feature = "interpolation")]
mod interpolation;

/// Returns the next highest power of 2 if `n` is not already a power of 2.
/// This will return `2` if `n < 2`.
///
/// # Example
///
/// ```
/// # use bit_mask_ring_buf::next_pow_of_2;
/// assert_eq!(next_pow_of_2(3), 4);
/// assert_eq!(next_pow_of_2(8), 8);
/// assert_eq!(next_pow_of_2(0), 2);
/// ```
///
/// # Panics
///
/// * This will panic if `n > (std::usize::MAX/2)+1`
pub const fn next_pow_of_2(n: usize) -> usize {
    if n < 2 {
        return 2;
    }

    n.checked_next_power_of_two().unwrap()
}
