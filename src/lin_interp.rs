//! Extra functions for buffers of floating point numbers

use crate::{BMRingBuf, BMRingBufRef};

impl BMRingBuf<f32> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBuf;
    /// let mut rb = BMRingBuf::<f32>::from_len(4);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp_f32(1.0) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f32(1.25) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f32(3.75) - 1.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f32(&self, index: f32) -> f32 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract)
    }

    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBuf;
    /// let mut rb = BMRingBuf::<f32>::from_len(4);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp_f64(1.0f64) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f64(1.25f64) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f64(3.75f64) - 1.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f64(&self, index: f64) -> f32 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract as f32)
    }
}

impl<'a> BMRingBufRef<'a, f32> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBufRef;
    /// let mut data = [0.0f32, 2.0, 4.0, 6.0];
    /// let rb = BMRingBufRef::new(&mut data[..]);
    ///
    /// assert!((rb.lin_interp_f32(1.0) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f32(1.25) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f32(3.75) - 1.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f32(&self, index: f32) -> f32 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract)
    }

    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBufRef;
    /// let mut data = [0.0f32, 2.0, 4.0, 6.0];
    /// let rb = BMRingBufRef::new(&mut data[..]);
    ///
    /// assert!((rb.lin_interp_f64(1.0f64) - 2.0).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f64(1.25f64) - 2.5).abs() <= f32::EPSILON);
    /// assert!((rb.lin_interp_f64(3.75f64) - 1.5).abs() <= f32::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f64(&self, index: f64) -> f32 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract as f32)
    }
}

impl BMRingBuf<f64> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBuf;
    /// let mut rb = BMRingBuf::<f64>::from_len(4);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp_f32(1.0f32) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f32(1.25f32) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f32(3.75f32) - 1.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f32(&self, index: f32) -> f64 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract as f64)
    }

    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBuf;
    /// let mut rb = BMRingBuf::<f64>::from_len(4);
    /// rb[0] = 0.0;
    /// rb[1] = 2.0;
    /// rb[2] = 4.0;
    /// rb[3] = 6.0;
    ///
    /// assert!((rb.lin_interp_f64(1.0) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f64(1.25) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f64(3.75) - 1.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f64(&self, index: f64) -> f64 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract)
    }
}

impl<'a> BMRingBufRef<'a, f64> {
    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f32`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBufRef;
    /// let mut data = [0.0f64, 2.0, 4.0, 6.0];
    /// let rb = BMRingBufRef::new(&mut data[..]);
    ///
    /// assert!((rb.lin_interp_f32(1.0f32) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f32(1.25f32) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f32(3.75f32) - 1.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f32(&self, index: f32) -> f64 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract as f64)
    }

    /// Gets the linearly interpolated value between the two values
    /// at `index.floor()` and `index.floor() + 1`, where `index`
    /// is an `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use bit_mask_ring_buf::BMRingBufRef;
    /// let mut data = [0.0f64, 2.0, 4.0, 6.0];
    /// let rb = BMRingBufRef::new(&mut data[..]);
    ///
    /// assert!((rb.lin_interp_f64(1.0) - 2.0).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f64(1.25) - 2.5).abs() <= f64::EPSILON);
    /// assert!((rb.lin_interp_f64(3.75) - 1.5).abs() <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lin_interp_f64(&self, index: f64) -> f64 {
        let index_floor = index.floor();
        let fract = index - index_floor;
        let index_isize = index_floor as isize;

        let val_1 = self[index_isize];
        let val_2 = self[index_isize + 1];

        val_1 + ((val_2 - val_1) * fract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bit_mask_ring_buf_f32_lin_interp_f32() {
        let mut ring_buf = BMRingBuf::<f32>::from_len(4);
        ring_buf[0] = 0.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 4.0;
        ring_buf[3] = 6.0;

        assert!((ring_buf.lin_interp_f32(-2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(4.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(5.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(63.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-63.0) - 2.0).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(0.5) - 1.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.25) - 0.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.75) - 1.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.25) - 2.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(3.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.25) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.75) - (6.0 * 0.25)).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(-1.5) - 5.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.25) - 5.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.75) - 4.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(-0.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.75) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.25) - (6.0 * 0.25)).abs() <= f32::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_f32_lin_interp_f64() {
        let mut ring_buf = BMRingBuf::<f32>::from_len(4);
        ring_buf[0] = 0.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 4.0;
        ring_buf[3] = 6.0;

        assert!((ring_buf.lin_interp_f64(-2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(4.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(5.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(63.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-63.0) - 2.0).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(0.5) - 1.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.25) - 0.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.75) - 1.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.25) - 2.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(3.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.25) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.75) - (6.0 * 0.25)).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(-1.5) - 5.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.25) - 5.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.75) - 4.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(-0.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.75) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.25) - (6.0 * 0.25)).abs() <= f32::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_ref_f32_lin_interp_f32() {
        let mut data = [0.0f32, 2.0, 4.0, 6.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert!((ring_buf.lin_interp_f32(-2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(4.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(5.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(63.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-63.0) - 2.0).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(0.5) - 1.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.25) - 0.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.75) - 1.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.25) - 2.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(3.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.25) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.75) - (6.0 * 0.25)).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(-1.5) - 5.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.25) - 5.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.75) - 4.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f32(-0.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.75) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.25) - (6.0 * 0.25)).abs() <= f32::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_ref_f32_lin_interp_f64() {
        let mut data = [0.0f32, 2.0, 4.0, 6.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert!((ring_buf.lin_interp_f64(-2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(2.0) - 4.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(4.0) - 0.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(5.0) - 2.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(63.0) - 6.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-63.0) - 2.0).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(0.5) - 1.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.25) - 0.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.75) - 1.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.25) - 2.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(3.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.25) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.75) - (6.0 * 0.25)).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(-1.5) - 5.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.25) - 5.5).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.75) - 4.5).abs() <= f32::EPSILON);

        assert!((ring_buf.lin_interp_f64(-0.5) - 3.0).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.75) - (6.0 * 0.75)).abs() <= f32::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.25) - (6.0 * 0.25)).abs() <= f32::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_f64_lin_interp_f32() {
        let mut ring_buf = BMRingBuf::<f64>::from_len(4);
        ring_buf[0] = 0.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 4.0;
        ring_buf[3] = 6.0;

        assert!((ring_buf.lin_interp_f32(-2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(4.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(5.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(63.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-63.0) - 2.0).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(0.5) - 1.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.25) - 0.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.75) - 1.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.25) - 2.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(3.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.25) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.75) - (6.0 * 0.25)).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(-1.5) - 5.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.25) - 5.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.75) - 4.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(-0.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.75) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.25) - (6.0 * 0.25)).abs() <= f64::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_f64_lin_interp_f64() {
        let mut ring_buf = BMRingBuf::<f64>::from_len(4);
        ring_buf[0] = 0.0;
        ring_buf[1] = 2.0;
        ring_buf[2] = 4.0;
        ring_buf[3] = 6.0;

        assert!((ring_buf.lin_interp_f64(-2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(4.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(5.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(63.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-63.0) - 2.0).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(0.5) - 1.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.25) - 0.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.75) - 1.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.25) - 2.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(3.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.25) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.75) - (6.0 * 0.25)).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(-1.5) - 5.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.25) - 5.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.75) - 4.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(-0.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.75) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.25) - (6.0 * 0.25)).abs() <= f64::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_ref_f64_lin_interp_f32() {
        let mut data = [0.0f64, 2.0, 4.0, 6.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert!((ring_buf.lin_interp_f32(-2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(4.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(5.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(63.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-63.0) - 2.0).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(0.5) - 1.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.25) - 0.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(0.75) - 1.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(1.25) - 2.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(3.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.25) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(3.75) - (6.0 * 0.25)).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(-1.5) - 5.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.25) - 5.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-1.75) - 4.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f32(-0.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.75) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f32(-0.25) - (6.0 * 0.25)).abs() <= f64::EPSILON);
    }

    #[test]
    fn bit_mask_ring_buf_ref_f64_lin_interp_f64() {
        let mut data = [0.0f64, 2.0, 4.0, 6.0];
        let ring_buf = BMRingBufRef::new(&mut data);

        assert!((ring_buf.lin_interp_f64(-2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(2.0) - 4.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(4.0) - 0.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(5.0) - 2.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(63.0) - 6.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-63.0) - 2.0).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(0.5) - 1.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.25) - 0.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(0.75) - 1.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(1.25) - 2.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(3.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.25) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(3.75) - (6.0 * 0.25)).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(-1.5) - 5.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.25) - 5.5).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-1.75) - 4.5).abs() <= f64::EPSILON);

        assert!((ring_buf.lin_interp_f64(-0.5) - 3.0).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.75) - (6.0 * 0.75)).abs() <= f64::EPSILON);
        assert!((ring_buf.lin_interp_f64(-0.25) - (6.0 * 0.25)).abs() <= f64::EPSILON);
    }
}
