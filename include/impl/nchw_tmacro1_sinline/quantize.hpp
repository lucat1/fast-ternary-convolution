#pragma once
#include "tensor.hpp"
#include "tensor_macros1.hpp"

namespace nchw_tmacro1_sinline {
inline Tensor5D<int64_t> ternarize(const Tensor4D<float> &data,
                            const Tensor1D<float> &thresholds,
                            const size_t padding_h, const size_t padding_w) {
  int64_t onebit[CNTBITS];
  for (size_t i = 0; i < CNTBITS; i++) {
    onebit[i] = (int64_t)1 << i;
  }

  const float *const thresholds_data = thresholds.data;

  const float *const data_data = data.data;
  const size_t n = data.dim1;
  const size_t channels = data.dim2;
  const size_t height = data.dim3;
  const size_t width = data.dim4;

  const size_t packed_h = height + 2 * padding_h;
  const size_t packed_w = width + 2 * padding_w;
  const size_t full_blocks_c = channels / 64;
  const size_t packed_c =
      (channels % 64) ? (full_blocks_c + 1) : (full_blocks_c);

  Tensor5D<int64_t> quantized_data(n, packed_h, packed_w, packed_c, BITS, true);
  int64_t *const quantized_data_data = quantized_data.data;

  for (size_t in = 0; in < n; in++) {
    for (size_t ih = 0; ih < height; ih++) {
      for (size_t iw = 0; iw < width; iw++) {
        for (size_t ic = 0; ic < full_blocks_c; ic++) {
          int64_t first_bits = 0;
          int64_t second_bits = 0;

          for (size_t bit = 0; bit < CNTBITS; bit++) {
            const float current_value =
                tensor4d_get(data_data, channels, height, width, in,
                              ic * CNTBITS + bit, ih, iw);

            if (current_value > tensor1d_get(thresholds_data, in)) {
              second_bits |= onebit[bit];
            } else if (current_value < -tensor1d_get(thresholds_data, in)) {
              first_bits |= onebit[bit];
              second_bits |= onebit[bit];
            }
          }

          tensor5d_set(quantized_data_data, packed_h, packed_w, packed_c, BITS,
                        first_bits, in, ih + padding_h, iw + padding_w, ic, 0);
          tensor5d_set(quantized_data_data, packed_h, packed_w, packed_c, BITS,
                        second_bits, in, ih + padding_h, iw + padding_w, ic, 1);
        }

        if (channels % 64) {
          int64_t first_bits = 0;
          int64_t second_bits = 0;

          for (size_t bit = 0; bit < (channels % 64); bit++) {
            const float current_value =
                tensor4d_get(data_data, channels, height, width, in,
                              full_blocks_c * CNTBITS + bit, ih, iw);

            if (current_value > tensor1d_get(thresholds_data, in)) {
              second_bits |= onebit[bit];
            } else if (current_value < -tensor1d_get(thresholds_data, in)) {
              first_bits |= onebit[bit];
              second_bits |= onebit[bit];
            }
          }

          tensor5d_set(quantized_data_data, packed_h, packed_w, packed_c, BITS,
                        first_bits, in, ih + padding_h, iw + padding_w,
                        full_blocks_c, 0);
          tensor5d_set(quantized_data_data, packed_h, packed_w, packed_c, BITS,
                        second_bits, in, ih + padding_h, iw + padding_w,
                        full_blocks_c, 1);
        }
      }
    }
  }

  return quantized_data;
}
} // namespace nchw_inlined
