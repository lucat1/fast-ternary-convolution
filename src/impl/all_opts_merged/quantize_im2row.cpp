#include "impl/all_opts_merged/quantize_im2row.hpp"
#include "common.hpp"
#include <immintrin.h>

// inner loop
#define iil(data, thresholds_data, onebit, in, padded_data_h, padding_h,       \
            padded_data_w, padding_w, ic, bits, first_bits, second_bits)       \
  {                                                                            \
    const float cur_thres = thresholds_data[in];                               \
    for (size_t bit = 0; bit < bits; bit++) {                                  \
      float current_value =                                                    \
          tensor4d_get(data, in, padded_data_h - padding_h,                    \
                       padded_data_w - padding_w, ic * CNTBITS + bit);         \
      if (current_value > cur_thres) {                                         \
        second_bits |= onebit[bit];                                            \
      } else if (current_value < -cur_thres) {                                 \
        first_bits |= onebit[bit];                                             \
        second_bits |= onebit[bit];                                            \
    }                                                                          \
  }

// vectorized version of inner loop
#define iil_vectorized(data, thresholds_data, onebit, in, padded_data_h,       \
                       padding_h, padded_data_w, padding_w, ic, bits,          \
                       first_bits, second_bits)                                \
  {                                                                            \
    size_t bit = 0;                                                            \
    const float cur_thres = thresholds_data[in];                               \
    /* (t.data[((i) * (t.dim2 * t.dim3 * t.dim4)) + ((j) * (t.dim3 * t.dim4))  \
     * + ((k) * t.dim4) + (l)]) */                                             \
    __m512i first_bits_vec = _mm512_setzero_si512(),                           \
            second_bits_vec = _mm512_setzero_si512(), cur_value_vec, one_bits; \
    __m512 threshold = _mm512_set1_ps(cur_thres),                              \
           neg_threshold = _mm512_set1_ps(-cur_thres);                         \
    __m512 cur_value_vec;                                                      \
    for (; bit < 25; bit += 8) {                                               \
      one_bits = _mm512_loadu_epi64(onebit + bit);                             \
                                                                               \
      cur_value_vec = _mm512_loadu_ps(data);                                   \
      __mmask16 first_mask =                                                   \
          _mm512_cmp_ps_mask(cur_value_vec, threshold, _CMP_GT_OQ);            \
      __mmask16 second_mask =                                                  \
          _mm512_cmp_ps_mask(cur_value_vec, neg_threshold, _CMP_LT_OQ);        \
                                                                               \
      if (current_value > cur_thres) {                                         \
        second_bits |= onebit[bit];                                            \
      } else if (current_value < -cur_thres) {                                 \
        first_bits |= onebit[bit];                                             \
        second_bits |= onebit[bit];                                            \
      }                                                                        \
    }                                                                          \
    _mm512_reduce_or_epi64() for (; bit < bits; bit++) {                       \
      float current_value =                                                    \
          tensor4d_get(data, in, padded_data_h - padding_h,                    \
                       padded_data_w - padding_w, ic * CNTBITS + bit);         \
      if (current_value > tensor1d_get(thresholds, in)) {                      \
        second_bits |= onebit[bit];                                            \
      } else if (current_value < -tensor1d_get(thresholds, in)) {              \
        first_bits |= onebit[bit];                                             \
    }                                                                          \
  }

#define copy_all_channels(quantized_reshaped, in, io_h, io_w, ik_h, ik_w,      \
                          offst_io_h, offst_io_w, offst_ik_h, offst_ik_w,      \
                          full_blocks_c, channels)                             \
  {                                                                            \
    size_t fbcpp = full_blocks_c + (channels % 64 ? 1 : 0);                    \
    size_t bytes = 2 * sizeof(int64_t) * fbcpp;                                \
    int64_t *src = tensor7d_addr(quantized_reshaped, in, io_h - offst_io_h,    \
                                 io_w - offst_io_w, ik_h + offst_ik_h,         \
                                 ik_w + offst_ik_w, 0, 0);                     \
    int64_t *dest =                                                            \
        tensor7d_addr(quantized_reshaped, in, io_h, io_w, ik_h, ik_w, 0, 0);   \
    memcpy(dest, src, bytes);                                                  \
  }

#define compute(data, thresholds_data, onebit, quantized_reshaped, in, io_h,   \
                io_w, ik_h, ik_w, padding_h, padding_w, stride_h, stride_w,    \
                packed_h, packed_w, full_blocks_c, channels)                   \
  {                                                                            \
    const size_t padded_data_h = io_h * stride_h + ik_h;                       \
    const size_t padded_data_w = io_w * stride_w + ik_w;                       \
    for (size_t ic = 0; ic < full_blocks_c; ic++) {                            \
      if (!((padded_data_h < padding_h) ||                                     \
            (padded_data_h >= (packed_h - padding_h)) ||                       \
            (padded_data_w < padding_w) ||                                     \
            (padded_data_w >= (packed_w - padding_w)))) {                      \
        int64_t first_bits = 0;                                                \
        int64_t second_bits = 0;                                               \
        iil(data, thresholds_data, onebit, in, padded_data_h, padding_h,       \
            padded_data_w, padding_w, ic, CNTBITS, first_bits, second_bits);   \
        tensor7d_set(quantized_reshaped, first_bits, in, io_h, io_w, ik_h,     \
                     ik_w, ic, 0);                                             \
        tensor7d_set(quantized_reshaped, second_bits, in, io_h, io_w, ik_h,    \
                     ik_w, ic, 1);                                             \
      }                                                                        \
    }                                                                          \
    if (channels % 64) {                                                       \
      if (!((padded_data_h < padding_h) ||                                     \
            (padded_data_h >= (packed_h - padding_h)) ||                       \
            (padded_data_w < padding_w) ||                                     \
            (padded_data_w >= (packed_w - padding_w)))) {                      \
        int64_t first_bits = 0;                                                \
        int64_t second_bits = 0;                                               \
        iil(data, thresholds_data, onebit, in, padded_data_h, padding_h,       \
            padded_data_w, padding_w, full_blocks_c, (channels % 64),          \
            first_bits, second_bits);                                          \
        tensor7d_set(quantized_reshaped, first_bits, in, io_h, io_w, ik_h,     \
                     ik_w, full_blocks_c, 0);                                  \
        tensor7d_set(quantized_reshaped, second_bits, in, io_h, io_w, ik_h,    \
                     ik_w, full_blocks_c, 1);                                  \
      }                                                                        \
    }                                                                          \
  }

namespace all_opts_merged {
Tensor7D<int64_t>
ternarize_im2row(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w) {
  int64_t onebit[CNTBITS];
  const float *const thresholds_data = thresholds.data;
  for (size_t i = 0; i < CNTBITS; i++) {
    // cast is important - otherwise we get wrong results
    onebit[i] = (int64_t)1 << i;
  }

  // sizes for our data
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;

  // sizes for the quantized and reshaped data
  // formulas follow from definition of padding
  const size_t packed_h = height + 2 * padding_h;
  const size_t packed_w = width + 2 * padding_w;
  // equivalent to ceil(channels/64)
  const size_t full_blocks_c = channels / 64;
  const size_t packed_c =
      (channels % 64) ? (full_blocks_c + 1) : (full_blocks_c);
  const size_t out_h = (packed_h - kernel_h) / stride_h + 1;
  const size_t out_w = (packed_w - kernel_w) / stride_w + 1;

  Tensor7D<int64_t> quantized_reshaped(n, out_h, out_w, kernel_h, kernel_w,
                                       packed_c, 2, true);

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      size_t start_ik_h =
          kernel_h > stride_h && io_h > 0 ? kernel_h - stride_h : 0;
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        size_t start_ik_w =
            kernel_w > stride_w && io_w > 0 ? kernel_w - stride_w : 0;

        for (size_t ik_h = 0; ik_h < start_ik_h; ik_h++) {
          // rectangle (kernel_h-stride_h, kernel_w) that needs to be
          // copied
          for (size_t ik_w = 0; ik_w < kernel_w; ik_w++) {
            copy_all_channels(quantized_reshaped, in, io_h, io_w, ik_h, ik_w, 1,
                              0, stride_h, 0, full_blocks_c, channels);
          }
        }

        for (size_t ik_h = start_ik_h; ik_h < kernel_h; ik_h++) {
          // kernel_w-stride_w cells that can be copied from the previous iter
          for (size_t ik_w = 0; ik_w < start_ik_w; ik_w++) {
            copy_all_channels(quantized_reshaped, in, io_h, io_w, ik_h, ik_w, 0,
                              1, 0, stride_w, full_blocks_c, channels);
          }

          for (size_t ik_w = start_ik_w; ik_w < kernel_w; ik_w++) {
            compute(data, thresholds_data, onebit, quantized_reshaped, in, io_h,
                    io_w, ik_h, ik_w, padding_h, padding_w, stride_h, stride_w,
                    packed_h, packed_w, full_blocks_c, channels);
          }
        }
      }
    }
  }
  return quantized_reshaped;
}
} // namespace all_opts_merged
