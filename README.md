# Bit-Masking Ring Buffer
![Test](https://github.com/BillyDM/bit_mask_ring_buf/workflows/Test/badge.svg)
[![Documentation](https://docs.rs/bit_mask_ring_buf/badge.svg)][documentation]
[![Crates.io](https://img.shields.io/crates/v/bit_mask_ring_buf.svg)](https://crates.io/crates/bit_mask_ring_buf)
[![License](https://img.shields.io/crates/l/bit_mask_ring_buf.svg)](https://github.com/BillyDM/bit_mask_ring_buf/blob/master/LICENSE)

A ring buffer implementation with cheap indexing written in Rust.

Note, this crate is only beneficial if your algorithm indexes elements one at a time and has buffer sizes that are always a power of two. If your algorithm instead reads chunks of data as slices or requires buffer sizes that are not a power of two, then check out my crate [`slice_ring_buf`].

This crate has no consumer/producer logic, and is meant to be used as a raw data structure or a base for other data structures.

## Installation
Add `bit_mask_ring_buf` as a dependency in your `Cargo.toml`:
```toml
bit_mask_ring_buf = 0.5
```

## Example
```rust
use bit_mask_ring_buf::{BMRingBuf, BMRingBufRef};

// Create a ring buffer with type u32. The data will be
// initialized with the default value (0 in this case).
// The actual length will be set to the next highest
// power of 2 if the given length is not already
// a power of 2.
let mut rb = BMRingBuf::<u32>::from_len(3);
assert_eq!(rb.len(), 4);

// Read/write to buffer by indexing with an `isize`.
rb[0] = 0;
rb[1] = 1;
rb[2] = 2;
rb[3] = 3;

// Cheaply wrap when reading/writing outside of bounds.
assert_eq!(rb[-1], 3);
assert_eq!(rb[10], 2);

// Memcpy into slices at arbitrary `isize` indexes
// and length.
let mut read_buffer = [0u32; 7];
rb.read_into(&mut read_buffer, 2);
assert_eq!(read_buffer, [2, 3, 0, 1, 2, 3, 0]);

// Memcpy data from a slice into the ring buffer at
// arbitrary `isize` indexes. Earlier data will not be
// copied if it will be overwritten by newer data,
// avoiding unecessary memcpy's. The correct placement
// of the newer data will still be preserved.
rb.write_latest(&[0, 2, 3, 4, 1], 0);
assert_eq!(rb[0], 1);
assert_eq!(rb[1], 2);
assert_eq!(rb[2], 3);
assert_eq!(rb[3], 4);

// Read/write by retrieving slices directly.
let (s1, s2) = rb.as_slices_len(1, 4);
assert_eq!(s1, &[2, 3, 4]);
assert_eq!(s2, &[1]);

// Aligned/stack data may also be used.
let mut stack_data = [0u32, 1, 2, 3];
let mut rb_ref = BMRingBufRef::new(&mut stack_data);
rb_ref[-4] = 5;
assert_eq!(rb_ref[0], 5);
assert_eq!(rb_ref[1], 1);
assert_eq!(rb_ref[2], 2);
assert_eq!(rb_ref[3], 3);

// Get linear interpolation on floating point buffers.
let mut rb = BMRingBuf::<f64>::from_len(4);
rb[0] = 0.0;
rb[1] = 2.0;
rb[2] = 4.0;
rb[3] = 6.0;
assert!((rb.lin_interp_f64(1.0) - 2.0).abs() <= f64::EPSILON);
assert!((rb.lin_interp_f64(1.25) - 2.5).abs() <= f64::EPSILON);
assert!((rb.lin_interp_f64(3.75) - 1.5).abs() <= f64::EPSILON);
```

[documentation]: https://docs.rs/bit_mask_ring_buf/
[`slice_ring_buf`]: https://crates.io/crates/slice_ring_buf/