#include "main_impls/best_impl_avx2/t2r_avx2.hpp"
#include "common.hpp"
#include "tensor.hpp"
#include "tensor_macros1.hpp"
#include <cstddef>
#include <immintrin.h>

// inner loop
#define iil_avx2(data_data, height, width, channels, thresholds_data, onebit,  \
                 in, padded_data_h, padding_h, padded_data_w, padding_w, ic,   \
                 bits, first_bits, second_bits)                                \
  {                                                                            \
    size_t bit = 0;                                                            \
    __m256i first_bits3 = _mm256_setzero_si256();                              \
    __m256i first_bits2 = _mm256_setzero_si256();                              \
    __m256i first_bits1 = _mm256_setzero_si256();                              \
    __m256i first_bits0 = _mm256_setzero_si256();                              \
    __m256i second_bits3 = _mm256_setzero_si256();                             \
    __m256i second_bits2 = _mm256_setzero_si256();                             \
    __m256i second_bits1 = _mm256_setzero_si256();                             \
    __m256i second_bits0 = _mm256_setzero_si256();                             \
    for (; (int)bit <= (int)bits - 16; bit += 16) {                            \
      const float *const data_index01 = tensor4d_addr(                         \
          data_data, height, width, channels, in, padded_data_h - padding_h,   \
          padded_data_w - padding_w, ic * CNTBITS + bit);                      \
      const float *const data_index23 = tensor4d_addr(                         \
          data_data, height, width, channels, in, padded_data_h - padding_h,   \
          padded_data_w - padding_w, ic * CNTBITS + bit + 8);                  \
      __m256 current_values01 = _mm256_loadu_ps(data_index01);                 \
      __m256 current_values23 = _mm256_loadu_ps(data_index23);                 \
                                                                               \
      __m256i one_bits0 = _mm256_loadu_si256((__m256i const *)(&onebit[bit])); \
      __m256i one_bits1 =                                                      \
          _mm256_loadu_si256((__m256i const *)(&onebit[bit + 4]));             \
      __m256i one_bits2 =                                                      \
          _mm256_loadu_si256((__m256i const *)(&onebit[bit + 8]));             \
      __m256i one_bits3 =                                                      \
          _mm256_loadu_si256((__m256i const *)(&onebit[bit + 12]));            \
                                                                               \
      __m256i current_values01_gt_thresh = _mm256_castps_si256(                \
          _mm256_cmp_ps(current_values01, curr_thresh_avx, _CMP_GT_OS));       \
      __m256i current_values01_lt_neg_thresh = _mm256_castps_si256(            \
          _mm256_cmp_ps(current_values01, neg_curr_thresh_avx, _CMP_LT_OS));   \
      __m256i current_values01_gt_or_lt = _mm256_or_si256(                     \
          current_values01_gt_thresh, current_values01_lt_neg_thresh);         \
                                                                               \
      __m256i current_values23_gt_thresh = _mm256_castps_si256(                \
          _mm256_cmp_ps(current_values23, curr_thresh_avx, _CMP_GT_OS));       \
      __m256i current_values23_lt_neg_thresh = _mm256_castps_si256(            \
          _mm256_cmp_ps(current_values23, neg_curr_thresh_avx, _CMP_LT_OS));   \
      __m256i current_values23_gt_or_lt = _mm256_or_si256(                     \
          current_values23_gt_thresh, current_values23_lt_neg_thresh);         \
                                                                               \
      __m256i first_bits0_mask = _mm256_cvtepi32_epi64(                        \
          _mm256_extracti128_si256(current_values01_lt_neg_thresh, 0));        \
      __m256i first_bits1_mask = _mm256_cvtepi32_epi64(                        \
          _mm256_extracti128_si256(current_values01_lt_neg_thresh, 1));        \
      __m256i first_bits2_mask = _mm256_cvtepi32_epi64(                        \
          _mm256_extracti128_si256(current_values23_lt_neg_thresh, 0));        \
      __m256i first_bits3_mask = _mm256_cvtepi32_epi64(                        \
          _mm256_extracti128_si256(current_values23_lt_neg_thresh, 1));        \
                                                                               \
      first_bits0 = _mm256_blendv_epi8(                                        \
          first_bits0, _mm256_or_si256(first_bits0, one_bits0),                \
          first_bits0_mask);                                                   \
      first_bits1 = _mm256_blendv_epi8(                                        \
          first_bits1, _mm256_or_si256(first_bits1, one_bits1),                \
          first_bits1_mask);                                                   \
      first_bits2 = _mm256_blendv_epi8(                                        \
          first_bits2, _mm256_or_si256(first_bits2, one_bits2),                \
          first_bits2_mask);                                                   \
      first_bits3 = _mm256_blendv_epi8(                                        \
          first_bits3, _mm256_or_si256(first_bits3, one_bits3),                \
          first_bits3_mask);                                                   \
                                                                               \
      __m256i second_bits0_mask = _mm256_cvtepi32_epi64(                       \
          _mm256_extracti128_si256(current_values01_gt_or_lt, 0));             \
      __m256i second_bits1_mask = _mm256_cvtepi32_epi64(                       \
          _mm256_extracti128_si256(current_values01_gt_or_lt, 1));             \
      __m256i second_bits2_mask = _mm256_cvtepi32_epi64(                       \
          _mm256_extracti128_si256(current_values23_gt_or_lt, 0));             \
      __m256i second_bits3_mask = _mm256_cvtepi32_epi64(                       \
          _mm256_extracti128_si256(current_values23_gt_or_lt, 1));             \
                                                                               \
      second_bits0 = _mm256_blendv_epi8(                                       \
          second_bits0, _mm256_or_si256(second_bits0, one_bits0),              \
          second_bits0_mask);                                                  \
      second_bits1 = _mm256_blendv_epi8(                                       \
          second_bits1, _mm256_or_si256(second_bits1, one_bits1),              \
          second_bits1_mask);                                                  \
      second_bits2 = _mm256_blendv_epi8(                                       \
          second_bits2, _mm256_or_si256(second_bits2, one_bits2),              \
          second_bits2_mask);                                                  \
      second_bits3 = _mm256_blendv_epi8(                                       \
          second_bits3, _mm256_or_si256(second_bits3, one_bits3),              \
          second_bits3_mask);                                                  \
    }                                                                          \
                                                                               \
    __m256i first_bits_ored = _mm256_or_si256(                                 \
        first_bits0,                                                           \
        _mm256_or_si256(first_bits1,                                           \
                        _mm256_or_si256(first_bits2, first_bits3)));           \
    __m256i first_bits_ored_perm =                                             \
        _mm256_permute4x64_epi64(first_bits_ored, _MM_SHUFFLE(2, 3, 0, 1));    \
    __m256i first_bits_ored_perm_ored =                                        \
        _mm256_or_si256(first_bits_ored, first_bits_ored_perm);                \
                                                                               \
    __m256i second_bits_ored = _mm256_or_si256(                                \
        second_bits0,                                                          \
        _mm256_or_si256(second_bits1,                                          \
                        _mm256_or_si256(second_bits2, second_bits3)));         \
    __m256i second_bits_ored_perm =                                            \
        _mm256_permute4x64_epi64(second_bits_ored, _MM_SHUFFLE(2, 3, 0, 1));   \
    __m256i second_bits_ored_perm_ored =                                       \
        _mm256_or_si256(second_bits_ored, second_bits_ored_perm);              \
                                                                               \
    int64_t first_bits_0 = _mm256_extract_epi64(first_bits_ored_perm_ored, 0); \
    int64_t first_bits_1 = _mm256_extract_epi64(first_bits_ored_perm_ored, 2); \
    int64_t second_bits_0 =                                                    \
        _mm256_extract_epi64(second_bits_ored_perm_ored, 0);                   \
    int64_t second_bits_1 =                                                    \
        _mm256_extract_epi64(second_bits_ored_perm_ored, 2);                   \
    for (; (int)bit <= (int)bits - 2; bit += 2) {                              \
      float current_value0 = tensor4d_get(                                     \
          data_data, height, width, channels, in, padded_data_h - padding_h,   \
          padded_data_w - padding_w, ic * CNTBITS + bit);                      \
      float current_value1 = tensor4d_get(                                     \
          data_data, height, width, channels, in, padded_data_h - padding_h,   \
          padded_data_w - padding_w, ic * CNTBITS + bit + 1);                  \
      int64_t onebit0 = onebit[bit];                                           \
      int64_t onebit1 = onebit[bit + 1];                                       \
      if (current_value0 > current_threshold) {                                \
        second_bits_0 |= onebit0;                                              \
      } else if (current_value0 < neg_current_threshold) {                     \
        first_bits_0 |= onebit0;                                               \
        second_bits_0 |= onebit0;                                              \
      }                                                                        \
      if (current_value1 > current_threshold) {                                \
        second_bits_1 |= onebit1;                                              \
      } else if (current_value1 < neg_current_threshold) {                     \
        first_bits_1 |= onebit1;                                               \
        second_bits_1 |= onebit1;                                              \
      }                                                                        \
    }                                                                          \
    first_bits |= first_bits_0 | first_bits_1;                                 \
    second_bits |= second_bits_0 | second_bits_1;                              \
                                                                               \
    for (; bit < bits; bit++) {                                                \
      float current_value = tensor4d_get(                                      \
          data_data, height, width, channels, in, padded_data_h - padding_h,   \
          padded_data_w - padding_w, ic * CNTBITS + bit);                      \
      if (current_value > tensor1d_get(thresholds_data, in)) {                 \
        second_bits |= onebit[bit];                                            \
      } else if (current_value < -tensor1d_get(thresholds_data, in)) {         \
        first_bits |= onebit[bit];                                             \
        second_bits |= onebit[bit];                                            \
      }                                                                        \
    }                                                                          \
  }

#define copy_all_channels(quantized_reshaped_data, out_h, out_w, kernel_h,     \
                          kernel_w, packed_c, in, io_h, io_w, ik_h, ik_w,      \
                          offst_io_h, offst_io_w, offst_ik_h, offst_ik_w,      \
                          full_blocks_c, channels)                             \
  {                                                                            \
    size_t fbcpp = full_blocks_c + (channels % 64 ? 1 : 0);                    \
    for (size_t ic = 0; ic < fbcpp; ic++) {                                    \
      int64_t v0 = tensor7d_get(quantized_reshaped_data, out_h, out_w,         \
                                kernel_h, kernel_w, packed_c, 2, in,           \
                                io_h - offst_io_h, io_w - offst_io_w,          \
                                ik_h + offst_ik_h, ik_w + offst_ik_w, ic, 0);  \
      tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h, kernel_w,  \
                   packed_c, 2, v0, in, io_h, io_w, ik_h, ik_w, ic, 0);        \
      int64_t v1 = tensor7d_get(quantized_reshaped_data, out_h, out_w,         \
                                kernel_h, kernel_w, packed_c, 2, in,           \
                                io_h - offst_io_h, io_w - offst_io_w,          \
                                ik_h + offst_ik_h, ik_w + offst_ik_w, ic, 1);  \
      tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h, kernel_w,  \
                   packed_c, 2, v1, in, io_h, io_w, ik_h, ik_w, ic, 1);        \
    }                                                                          \
  }

#define compute(data_data, height, width, channels, thresholds_data, onebit,   \
                quantized_reshaped_data, out_h, out_w, kernel_h, kernel_w,     \
                packed_c, in, io_h, io_w, ik_h, ik_w, padding_h, padding_w,    \
                stride_h, stride_w, packed_h, packed_w, full_blocks_c)         \
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
        iil_avx2(data_data, height, width, channels, thresholds_data, onebit,  \
                 in, padded_data_h, padding_h, padded_data_w, padding_w, ic,   \
                 CNTBITS, first_bits, second_bits);                            \
        tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h,          \
                     kernel_w, packed_c, 2, first_bits, in, io_h, io_w, ik_h,  \
                     ik_w, ic, 0);                                             \
        tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h,          \
                     kernel_w, packed_c, 2, second_bits, in, io_h, io_w, ik_h, \
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
        iil_avx2(data_data, height, width, channels, thresholds_data, onebit,  \
                 in, padded_data_h, padding_h, padded_data_w, padding_w,       \
                 full_blocks_c, (channels % 64), first_bits, second_bits);     \
        tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h,          \
                     kernel_w, packed_c, 2, first_bits, in, io_h, io_w, ik_h,  \
                     ik_w, full_blocks_c, 0);                                  \
        tensor7d_set(quantized_reshaped_data, out_h, out_w, kernel_h,          \
                     kernel_w, packed_c, 2, second_bits, in, io_h, io_w, ik_h, \
                     ik_w, full_blocks_c, 1);                                  \
      }                                                                        \
    }                                                                          \
  }

namespace best_impl_avx2 {
Tensor7D<int64_t> t2r_avx2(const Tensor4D<float> &data,
                           const Tensor1D<float> &thresholds,
                           const size_t padding_h, const size_t padding_w,
                           const size_t kernel_h, const size_t kernel_w,
                           const size_t stride_h, const size_t stride_w) {
  __m256 neg_one_avx = _mm256_set1_ps(-1.0);
  int64_t onebit[CNTBITS];
  for (size_t i = 0; i < CNTBITS; i++) {
    onebit[i] = (int64_t)1 << i;
  }

  const float *const thresholds_data = thresholds.data;

  const float *const data_data = data.data;
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;

  const size_t packed_h = height + 2 * padding_h;
  const size_t packed_w = width + 2 * padding_w;
  const size_t full_blocks_c = channels / 64;
  const size_t packed_c =
      (channels % 64) ? (full_blocks_c + 1) : (full_blocks_c);
  const size_t out_h = (packed_h - kernel_h) / stride_h + 1;
  const size_t out_w = (packed_w - kernel_w) / stride_w + 1;

  Tensor7D<int64_t> quantized_reshaped(n, out_h, out_w, kernel_h, kernel_w,
                                       packed_c, 2, true);
  int64_t *const quantized_reshaped_data = quantized_reshaped.data;

  for (size_t in = 0; in < n; in++) {
    float current_threshold = tensor1d_get(thresholds_data, in);
    float neg_current_threshold = -current_threshold;
    __m256 curr_thresh_avx = _mm256_set1_ps(current_threshold);
    __m256 neg_curr_thresh_avx = _mm256_mul_ps(neg_one_avx, curr_thresh_avx);
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
            copy_all_channels(quantized_reshaped_data, out_h, out_w, kernel_h,
                              kernel_w, packed_c, in, io_h, io_w, ik_h, ik_w, 1,
                              0, stride_h, 0, full_blocks_c, channels);
          }
        }

        for (size_t ik_h = start_ik_h; ik_h < kernel_h; ik_h++) {
          // kernel_w-stride_w cells that can be copied from the previous iter
          for (size_t ik_w = 0; ik_w < start_ik_w; ik_w++) {
            copy_all_channels(quantized_reshaped_data, out_h, out_w, kernel_h,
                              kernel_w, packed_c, in, io_h, io_w, ik_h, ik_w, 0,
                              1, 0, stride_w, full_blocks_c, channels);
          }

          for (size_t ik_w = start_ik_w; ik_w < kernel_w; ik_w++) {
            compute(data_data, height, width, channels, thresholds_data, onebit,
                    quantized_reshaped_data, out_h, out_w, kernel_h, kernel_w,
                    packed_c, in, io_h, io_w, ik_h, ik_w, padding_h, padding_w,
                    stride_h, stride_w, packed_h, packed_w, full_blocks_c);
          }
        }
      }
    }
  }
  return quantized_reshaped;
}
} // namespace best_impl_avx2
