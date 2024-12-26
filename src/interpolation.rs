//! Extra functions for buffers of floating point numbers

use crate::referenced::{BitMaskRbRef, BitMaskRbRefMut};

#[cfg(feature = "alloc")]
use crate::owned::BitMaskRB;

#[cfg(feature = "alloc")]
impl BitMaskRB<f32> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<f32>::new(4, 0.0);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f32) -> f32 {
        lin_interp_f32(self, index)
    }
}

#[cfg(feature = "alloc")]
impl BitMaskRB<f64> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRB;
    /// let mut rb = BitMaskRB::<f64>::new(4, 0.0);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f64) -> f64 {
        lin_interp_f64(self, index)
    }
}

impl<'a> BitMaskRbRef<'a, f32> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRbRef;
    /// let data: [f32; 4] = [0.0, 2.0, 4.0, 6.0];
    /// let rb = BitMaskRbRef::new(&data);
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f32) -> f32 {
        lin_interp_f32(self, index)
    }
}

impl<'a> BitMaskRbRef<'a, f64> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRbRef;
    /// let data: [f64; 4] = [0.0, 2.0, 4.0, 6.0];
    /// let rb = BitMaskRbRef::new(&data);
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f64) -> f64 {
        lin_interp_f64(self, index)
    }
}

impl<'a> BitMaskRbRefMut<'a, f32> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRbRefMut;
    /// let mut data: [f32; 4] = [0.0, 2.0, 4.0, 6.0];
    /// let rb = BitMaskRbRefMut::new(&mut data);
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f32) -> f32 {
        lin_interp_f32(self, index)
    }
}

impl<'a> BitMaskRbRefMut<'a, f64> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.ceil()`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bit_mask_ring_buf::BitMaskRbRefMut;
    /// let mut data: [f64; 4] = [0.0, 2.0, 4.0, 6.0];
    /// let rb = BitMaskRbRefMut::new(&mut data);
    ///
    /// assert!((rb.lin_interp(1.0) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(1.25) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(7.75) - 1.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-0.5) - 3.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp(-1.75) - 4.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp(&self, index: f64) -> f64 {
        lin_interp_f64(self, index)
    }
}

#[inline]
fn lin_interp_f32<B: core::ops::Index<isize, Output = f32>>(buffer: &B, index: f32) -> f32 {
    let index_floor = index.floor();
    let fract = index - index_floor;
    let index_isize = index_floor as isize;

    let val_1 = buffer[index_isize];
    let val_2 = buffer[index_isize + 1];

    val_1 + ((val_2 - val_1) * fract)
}

#[inline]
fn lin_interp_f64<B: core::ops::Index<isize, Output = f64>>(buffer: &B, index: f64) -> f64 {
    let index_floor = index.floor();
    let fract = index - index_floor;
    let index_isize = index_floor as isize;

    let val_1 = buffer[index_isize];
    let val_2 = buffer[index_isize + 1];

    val_1 + ((val_2 - val_1) * fract)
}
