#include "common.hpp"
#include "tensor.hpp"

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
  const size_t packed_c = (channels % 64) ? (channels / 64 + 1) : (channels / 64);

  // We initialize by zero so we don't need to deal with quantizing the padding
  // "Why is this not const?" - If it was, then returning this would require us
  // to copy, as moving may change an object.
  Tensor5D<int64_t> quantized_data (n, packed_h, packed_w, packed_c,
				    BITS, true);

  return quantized_data;
}

void test(Tensor4D<float> data, Tensor1D<float> thresholds,
	  size_t padding_h, size_t padding_w) {
  Tensor5D<int64_t> foo = ternarize(data, thresholds, padding_h, padding_w);
}
