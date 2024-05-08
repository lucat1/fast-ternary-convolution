#include "impl/baseline_nhwc/quantize.hpp"
#include "common.hpp"

// Quantize the floats in data to {+1 (01), 0 (00), -1 (11)}
// Input:
//   data: data to be quantized, using the (N, H, W, C) format
//   thresholds: quantization threshold values for each N
//   padding_h: padding around height
//   padding_w: padding around width
// Output:
//   quantized_data: the quantized data, using the (N, H, W, C, BITS) format
Tensor5D<int64_t> ternarize(const Tensor4D<float>& data,
			    const Tensor1D<float>& thresholds,
			    const size_t padding_h, const size_t padding_w) {
  // NOTE Can we precompute this once somewhere?
  // the i-th element of onebit contains an int64_t where exactly the i-th bit
  // is set to 1.
  int64_t onebit[CNTBITS];
  for (size_t i = 0; i < CNTBITS; i++) {
    onebit[i] = 1 << i;
  }

  // sizes for our data
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;

  // sizes for the quantized data
  // formulas follow from definition of padding
  const size_t packed_h = height + 2 * padding_h;
  const size_t packed_w = width + 2 * padding_w;
  // equivalent to ceil(channels/64)
  const size_t full_blocks_c = channels / 64;
  const size_t packed_c = (channels % 64) ? (full_blocks_c + 1) : (full_blocks_c);

  // We initialize by zero so we don't need to deal with quantizing the padding
  // "Why is this not const?" - If it was, then returning this would require us
  // to copy, as moving may change an object.
  Tensor5D<int64_t> quantized_data (n, packed_h, packed_w, packed_c,
				    BITS, true);

  for (size_t in = 0; in < n; in++) {
    for (size_t ih = 0; ih < height; ih++) {
      for (size_t iw = 0; iw < width; iw++) {
	// Process 64 channels at a time
	for (size_t ic = 0; ic < full_blocks_c; ic++) {
	  // We will now take 64 elements across the channels, ternarize them
	  // (turn them into two bits) and store the first and second bit separately
	  // (packing).
	  int64_t first_bits = 0;
	  int64_t second_bits = 0;

	  // Ternarize and pack the data
	  for (size_t bit = 0; bit < CNTBITS; bit++) {
	    // NOTE I wonder whether we can apply strength reduction here
	    float current_value = data.get(in, ih, iw, ic * CNTBITS + bit);

	    // NOTE Do scalar replacement on thresholds
	    if (current_value > thresholds.get(in)) {
	      // Pack 1: 01 => only need to set second bit
	      second_bits = second_bits | onebit[bit];
	    } else if (current_value < -thresholds.get(in)) {
	      // Pack -1: 11 => need to set both bits
	      first_bits = first_bits | onebit[bit];
	      second_bits = second_bits | onebit[bit];
	    }
	    // else: Pack 0: 00 => no bits need to be set
	  }

	  // Store the ternarized and packed data
	  quantized_data.set(first_bits, in, ih + padding_h,
			     iw + padding_w, ic, 0);
	  quantized_data.set(second_bits, in, ih + padding_h,
			     iw + padding_w, ic, 1);
	}

	// Process rest of the channels (< 64)
	if (channels % 64) {
	  int64_t first_bits = 0;
	  int64_t second_bits = 0;

	  // Ternarize and pack the data
	  for (size_t bit = 0; bit < (channels % 64); bit++) {
	    // NOTE I wonder whether we can apply strength reduction here
	    float current_value = data.get(in, ih, iw, full_blocks_c * CNTBITS + bit);

	    // NOTE Do scalar replacement on thresholds
	    if (current_value > thresholds.get(in)) {
	      // Pack 1: 01 => only need to set second bit
	      second_bits = second_bits | onebit[bit];
	    } else if (current_value < -thresholds.get(in)) {
	      // Pack -1: 11 => need to set both bits
	      first_bits = first_bits | onebit[bit];
	      second_bits = second_bits | onebit[bit];
	    }
	    // else: Pack 0: 00 => no bits need to be set
	  }

	  // Store the ternarized and packed data
	  quantized_data.set(first_bits, in, ih + padding_h,
			     iw + padding_w, full_blocks_c, 0);
	  quantized_data.set(second_bits, in, ih + padding_h,
			     iw + padding_w, full_blocks_c, 1);
	}
      }
    }
  }

  return quantized_data;
}

void test(Tensor4D<float> data, Tensor1D<float> thresholds,
	  size_t padding_h, size_t padding_w) {
  Tensor5D<int64_t> foo = ternarize(data, thresholds, padding_h, padding_w);
}
