#include "impl/all_opts_merged/gemm.hpp"
#include "common.hpp"
#include <cstdint>
#include <immintrin.h>

// Based of off nhwc.
namespace all_opts_merged {

int array_equals(int64_t *a, int64_t *b, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (a[i] != b[i]) {
      std::cout << "a is " << a[i] << " while b is " << b[i] << " at position "
                << i << std::endl;
      return 0;
    }
  }
  return 1;
}

void print_array(int64_t *a, size_t len) {
  for (size_t i = 0; i < len; i++) {
    std::cout << a[i];
  }
  std::cout << std::endl;
}

#define inner_loop_vectorized(activation, kernel, K, im, in, output, alpha)    \
  {                                                                            \
    size_t ik = 0;                                                             \
    __m512i cntp1_vec1 = _mm512_setzero_si512(),                               \
            cntp1_vec2 = _mm512_setzero_si512();                               \
    __m512i cntp2_vec1 = _mm512_setzero_si512(),                               \
            cntp2_vec2 = _mm512_setzero_si512();                               \
    __m512i p11, p21, p12, p22;                                                \
    __m512i activation11, activation21, activation_lo1, activation_hi1,        \
        kernel11, kernel21, kernel_lo1, kernel_hi1, activation12,              \
        activation22, activation_lo2, activation_hi2, kernel12, kernel22,      \
        kernel_lo2, kernel_hi2;                                                \
    for (; ik + 15 * BITS < K; ik += 16 * BITS) {                              \
      activation11 = _mm512_loadu_epi64(activation.data + K * im + ik);        \
      activation21 = _mm512_loadu_epi64(activation.data + K * im + ik + 8);    \
      kernel11 = _mm512_loadu_epi64(kernel.data + K * in + ik);                \
      kernel21 = _mm512_loadu_epi64(kernel.data + K * in + ik + 8);            \
                                                                               \
      activation12 =                                                           \
          _mm512_loadu_epi64(activation.data + K * im + ik + 2 * 8);           \
      activation22 =                                                           \
          _mm512_loadu_epi64(activation.data + K * im + ik + 3 * 8);           \
      kernel12 = _mm512_loadu_epi64(kernel.data + K * in + ik + 2 * 8);        \
      kernel22 = _mm512_loadu_epi64(kernel.data + K * in + ik + 3 * 8);        \
                                                                               \
      activation_lo1 = _mm512_unpacklo_epi64(activation11, activation21);      \
      activation_hi1 = _mm512_unpackhi_epi64(activation11, activation21);      \
      activation_lo2 = _mm512_unpacklo_epi64(activation12, activation22);      \
      activation_hi2 = _mm512_unpackhi_epi64(activation12, activation22);      \
                                                                               \
      kernel_lo1 = _mm512_unpacklo_epi64(kernel11, kernel21);                  \
      kernel_hi1 = _mm512_unpackhi_epi64(kernel11, kernel21);                  \
      kernel_lo2 = _mm512_unpacklo_epi64(kernel12, kernel22);                  \
      kernel_hi2 = _mm512_unpackhi_epi64(kernel12, kernel22);                  \
                                                                               \
      p11 = _mm512_xor_epi64(activation_lo1, kernel_lo1);                      \
      p21 = _mm512_and_epi64(activation_hi1, kernel_hi1);                      \
      p12 = _mm512_xor_epi64(activation_lo2, kernel_lo2);                      \
      p22 = _mm512_and_epi64(activation_hi2, kernel_hi2);                      \
                                                                               \
      cntp1_vec1 = _mm512_add_epi64(cntp1_vec1, _mm512_popcnt_epi64(p21));     \
      cntp2_vec1 = _mm512_add_epi64(                                           \
          cntp2_vec1, _mm512_popcnt_epi64(_mm512_and_epi64(p11, p21)));        \
      cntp1_vec2 = _mm512_add_epi64(cntp1_vec2, _mm512_popcnt_epi64(p22));     \
      cntp2_vec2 = _mm512_add_epi64(                                           \
          cntp2_vec2, _mm512_popcnt_epi64(_mm512_and_epi64(p12, p22)));        \
    }                                                                          \
    int cntp1 = _mm512_reduce_add_epi64(cntp1_vec1) +                          \
                _mm512_reduce_add_epi64(cntp1_vec2);                           \
    int cntp2 = _mm512_reduce_add_epi64(cntp2_vec1) +                          \
                _mm512_reduce_add_epi64(cntp2_vec2);                           \
    for (; ik < K; ik += BITS) {                                               \
      int64_t p1 =                                                             \
          activation.get_123_4567(im, ik) ^ kernel.get_1_2345(in, ik);         \
      int64_t p2 =                                                             \
          activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1); \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const float current = cntp1 - cntp2 - cntp2;                               \
    const float value = current > 0 ? current : current * alpha;               \
    output.set_123_4(value, im, in);                                           \
  }

#define inner_loop_unrolled(activation, kernel, K, im, in, output, alpha)      \
  {                                                                            \
    int cntp11 = 0, cntp12 = 0, cntp13 = 0, cntp14 = 0;                        \
    int cntp15 = 0, cntp16 = 0, cntp17 = 0, cntp18 = 0;                        \
    int cntp21 = 0, cntp22 = 0, cntp23 = 0, cntp24 = 0;                        \
    int cntp25 = 0, cntp26 = 0, cntp27 = 0, cntp28 = 0;                        \
    size_t ik = 0;                                                             \
    int64_t p11, p12, p13, p14, p15, p16, p17, p18;                            \
    int64_t p21, p22, p23, p24, p25, p26, p27, p28;                            \
    for (; ik + 7 * BITS < K; ik += 8 * BITS) {                                \
      p11 = activation.get_123_4567(im, ik) ^ kernel.get_1_2345(in, ik);       \
      p21 =                                                                    \
          activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1); \
      p12 = activation.get_123_4567(im, ik + BITS) ^                           \
            kernel.get_1_2345(in, ik + BITS);                                  \
      p22 = activation.get_123_4567(im, ik + BITS + 1) &                       \
            kernel.get_1_2345(in, ik + BITS + 1);                              \
      p13 = activation.get_123_4567(im, ik + 2 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 2 * BITS);                              \
      p23 = activation.get_123_4567(im, ik + 2 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 2 * BITS + 1);                          \
      p14 = activation.get_123_4567(im, ik + 3 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 3 * BITS);                              \
      p24 = activation.get_123_4567(im, ik + 3 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 3 * BITS + 1);                          \
      p15 = activation.get_123_4567(im, ik + 4 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 4 * BITS);                              \
      p25 = activation.get_123_4567(im, ik + 4 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 4 * BITS + 1);                          \
      p16 = activation.get_123_4567(im, ik + 5 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 5 * BITS);                              \
      p26 = activation.get_123_4567(im, ik + 5 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 5 * BITS + 1);                          \
      p17 = activation.get_123_4567(im, ik + 6 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 6 * BITS);                              \
      p27 = activation.get_123_4567(im, ik + 6 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 6 * BITS + 1);                          \
      p18 = activation.get_123_4567(im, ik + 7 * BITS) ^                       \
            kernel.get_1_2345(in, ik + 7 * BITS);                              \
      p28 = activation.get_123_4567(im, ik + 7 * BITS + 1) &                   \
            kernel.get_1_2345(in, ik + 7 * BITS + 1);                          \
      cntp11 += popcnt64(p21);                                                 \
      cntp21 += popcnt64(p11 & p21);                                           \
      cntp12 += popcnt64(p22);                                                 \
      cntp22 += popcnt64(p12 & p22);                                           \
      cntp13 += popcnt64(p23);                                                 \
      cntp23 += popcnt64(p13 & p23);                                           \
      cntp14 += popcnt64(p24);                                                 \
      cntp24 += popcnt64(p14 & p24);                                           \
      cntp15 += popcnt64(p25);                                                 \
      cntp25 += popcnt64(p15 & p25);                                           \
      cntp16 += popcnt64(p26);                                                 \
      cntp26 += popcnt64(p16 & p26);                                           \
      cntp17 += popcnt64(p27);                                                 \
      cntp27 += popcnt64(p17 & p27);                                           \
      cntp18 += popcnt64(p28);                                                 \
      cntp28 += popcnt64(p18 & p28);                                           \
    }                                                                          \
    int cntp1 =                                                                \
        cntp11 + cntp12 + cntp13 + cntp14 + cntp15 + cntp16 + cntp17 + cntp18; \
    int cntp2 =                                                                \
        cntp21 + cntp22 + cntp23 + cntp24 + cntp25 + cntp26 + cntp27 + cntp28; \
    for (; ik < K; ik += BITS) {                                               \
      int64_t p1 =                                                             \
          activation.get_123_4567(im, ik) ^ kernel.get_1_2345(in, ik);         \
      int64_t p2 =                                                             \
          activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1); \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const float current = cntp1 - cntp2 - cntp2;                               \
    const float value = current > 0 ? current : current * alpha;               \
    output.set_123_4(value, im, in);                                           \
  }

#define inner_loop(activation, kernel, K, im, in, output, alpha)               \
  {                                                                            \
    int cntp1 = 0, cntp2 = 0;                                                  \
    size_t ik = 0;                                                             \
    for (; ik < K; ik += BITS) {                                               \
      int64_t p1 =                                                             \
          activation.get_123_4567(im, ik) ^ kernel.get_1_2345(in, ik);         \
      int64_t p2 =                                                             \
          activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1); \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const float current = cntp1 - cntp2 - cntp2;                               \
    const float value = current > 0 ? current : current * alpha;               \
    output.set_123_4(value, im, in);                                           \
  }

// Multiply two matrices containing ternary values together (Algorithm 3).
Tensor4D<float> ternary_gemm(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel, float alpha) {
  // our sizes
  const size_t batch_size = activation.dim1;
  const size_t output_height = activation.dim2;
  const size_t output_width = activation.dim3;
  const size_t block_size = 32;

  const size_t kernel_number = kernel.dim1;
  // We essentially reinterpret the tensors as 2D tensors and do
  // Matrix-Matrix multiplication.
  const size_t M = batch_size * output_height * output_width;
  // KH * KW * C * BITS
  const size_t K =
      activation.dim4 * activation.dim5 * activation.dim6 * activation.dim7;
  const size_t N = kernel_number;
  // sanity check: K (from activation) == KH * KW * C * BITS (from weights)
  // assert(K == kernel.dim2 * kernel.dim3 * kernel.dim4 * kernel.dim5);

  // NOTE In the original code he initializes this to 0. Why?
  Tensor4D<float> output(batch_size, output_height, output_width, kernel_number,
                         false);

  size_t im_blk = 0;

  for (; im_blk + block_size < M + 1; im_blk += block_size) {
    size_t in_blk = 0;
    for (; in_blk + block_size < N + 1; in_blk += block_size) {
      for (size_t im = im_blk; im < im_blk + block_size; im++) {
        for (size_t in = in_blk; in < in_blk + block_size; in++) {
          inner_loop_unrolled(activation, kernel, K, im, in, output, alpha);
        }
      }
    }
    for (size_t im = im_blk; im < im_blk + block_size; im++) {
      for (size_t in = in_blk; in < N; in++) {
        inner_loop_unrolled(activation, kernel, K, im, in, output, alpha);
      }
    }
  }
  for (; im_blk < M; im_blk++) {
    for (size_t in = 0; in < N; in++) {
      inner_loop_unrolled(activation, kernel, K, im_blk, in, output, alpha);
    }
  }
  return output;
}
} // namespace all_opts_merged
