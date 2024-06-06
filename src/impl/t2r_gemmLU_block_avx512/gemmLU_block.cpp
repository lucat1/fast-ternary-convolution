#include "impl/t2r_gemmLU_block_avx512/gemmLU_block.hpp"
#include "common.hpp"
#include "tensor_macros1.hpp"

#define gemm_kernel_512(activation, kernel, output, N, K, BITS, iM, iN, alpha) \
  {                                                                            \
                                                                               \
    __m512i load1 = (_mm512_setzero_si512());                                  \
    __m512i load2 = (_mm512_setzero_si512());                                  \
    __m512i load7 = (_mm512_setzero_si512());                                  \
    __m512i load8 = (_mm512_setzero_si512());                                  \
    int32_t load13 = (0);                                                      \
    int32_t load14 = (0);                                                      \
    int32_t load19 = (0);                                                      \
    int32_t load20 = (0);                                                      \
    size_t iK = (0);                                                           \
    for (; (int)iK <= (int)K - ((2 * (8 * (int)BITS)));                        \
         iK += (2 * (8 * BITS))) {                                             \
                                                                               \
      __m512i load3 = (_mm512_loadu_si512(                                     \
          (__m512i *)(activation +                                             \
                      (((iM * K) + ((iK + (0 * (8 * BITS))) + 0))))));         \
      __m512i load4 = (_mm512_loadu_si512((                                    \
          __m512i *)(kernel + (((iN * K) + ((iK + (0 * (8 * BITS))) + 0)))))); \
      __m512i load5 = (_mm512_loadu_si512(                                     \
          (__m512i *)(activation +                                             \
                      (((iM * K) + ((iK + (0 * (8 * BITS))) + 8))))));         \
      __m512i load6 = (_mm512_loadu_si512((                                    \
          __m512i *)(kernel + (((iN * K) + ((iK + (0 * (8 * BITS))) + 8)))))); \
      __m512i load9 = (_mm512_loadu_si512(                                     \
          (__m512i *)(activation +                                             \
                      (((iM * K) + ((iK + (1 * (8 * BITS))) + 0))))));         \
      __m512i load10 = (_mm512_loadu_si512((                                   \
          __m512i *)(kernel + (((iN * K) + ((iK + (1 * (8 * BITS))) + 0)))))); \
      __m512i load11 = (_mm512_loadu_si512(                                    \
          (__m512i *)(activation +                                             \
                      (((iM * K) + ((iK + (1 * (8 * BITS))) + 8))))));         \
      __m512i load12 = (_mm512_loadu_si512((                                   \
          __m512i *)(kernel + (((iN * K) + ((iK + (1 * (8 * BITS))) + 8)))))); \
                                                                               \
      __m512i comp1 = (_mm512_unpacklo_epi64(load3, load5));                   \
      __m512i comp2 = (_mm512_unpackhi_epi64(load3, load5));                   \
      __m512i comp3 = (_mm512_unpacklo_epi64(load4, load6));                   \
      __m512i comp4 = (_mm512_unpackhi_epi64(load4, load6));                   \
      __m512i comp5 = (_mm512_xor_epi64(comp1, comp3));                        \
      __m512i comp6 = (_mm512_and_epi64(comp2, comp4));                        \
      __m512i comp8 = (_mm512_popcnt_epi64(comp6));                            \
      __m512i comp7 = (_mm512_and_epi64(comp5, comp6));                        \
      __m512i comp9 = (_mm512_popcnt_epi64(comp7));                            \
      __m512i comp10 = (_mm512_add_epi64(load1, comp8));                       \
      __m512i comp11 = (_mm512_add_epi64(load2, comp9));                       \
      __m512i comp12 = (_mm512_unpacklo_epi64(load9, load11));                 \
      __m512i comp13 = (_mm512_unpackhi_epi64(load9, load11));                 \
      __m512i comp14 = (_mm512_unpacklo_epi64(load10, load12));                \
      __m512i comp15 = (_mm512_unpackhi_epi64(load10, load12));                \
      __m512i comp16 = (_mm512_xor_epi64(comp12, comp14));                     \
      __m512i comp17 = (_mm512_and_epi64(comp13, comp15));                     \
      __m512i comp19 = (_mm512_popcnt_epi64(comp17));                          \
      __m512i comp18 = (_mm512_and_epi64(comp16, comp17));                     \
      __m512i comp20 = (_mm512_popcnt_epi64(comp18));                          \
      __m512i comp21 = (_mm512_add_epi64(load7, comp19));                      \
      __m512i comp22 = (_mm512_add_epi64(load8, comp20));                      \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
      load7 = (comp21);                                                        \
      load8 = (comp22);                                                        \
    }                                                                          \
                                                                               \
    for (; (iK + (1 * BITS)) < K; iK += (BITS * 2)) {                          \
                                                                               \
      int64_t load15 = (activation[((iM * K) + ((iK + (0 * BITS)) + 0))]);     \
      int64_t load16 = (kernel[((iN * K) + ((iK + (0 * BITS)) + 0))]);         \
      int64_t load17 = (activation[((iM * K) + ((iK + (0 * BITS)) + 1))]);     \
      int64_t load18 = (kernel[((iN * K) + ((iK + (0 * BITS)) + 1))]);         \
      int64_t load21 = (activation[((iM * K) + ((iK + (1 * BITS)) + 0))]);     \
      int64_t load22 = (kernel[((iN * K) + ((iK + (1 * BITS)) + 0))]);         \
      int64_t load23 = (activation[((iM * K) + ((iK + (1 * BITS)) + 1))]);     \
      int64_t load24 = (kernel[((iN * K) + ((iK + (1 * BITS)) + 1))]);         \
                                                                               \
      int64_t comp27 = ((load15 ^ load16));                                    \
      int64_t comp28 = ((load17 & load18));                                    \
      int32_t comp29 = (popcnt64(comp28));                                     \
      int64_t comp30 = ((comp27 & comp28));                                    \
      int32_t comp31 = (popcnt64(comp30));                                     \
      int32_t comp32 = ((load13 + comp29));                                    \
      int32_t comp33 = ((load14 + comp31));                                    \
      int64_t comp34 = ((load21 ^ load22));                                    \
      int64_t comp35 = ((load23 & load24));                                    \
      int32_t comp36 = (popcnt64(comp35));                                     \
      int64_t comp37 = ((comp34 & comp35));                                    \
      int32_t comp38 = (popcnt64(comp37));                                     \
      int32_t comp39 = ((load19 + comp36));                                    \
      int32_t comp40 = ((load20 + comp38));                                    \
      load13 = (comp32);                                                       \
      load14 = (comp33);                                                       \
      load19 = (comp39);                                                       \
      load20 = (comp40);                                                       \
    }                                                                          \
                                                                               \
    __m512i comp23 = (_mm512_add_epi64(load7, load1));                         \
    __m512i comp24 = (_mm512_add_epi64(load8, load2));                         \
    int32_t comp41 = ((load19 + load13));                                      \
    int32_t comp42 = ((load20 + load14));                                      \
    int32_t comp25 = (_mm512_reduce_add_epi64(comp23));                        \
    int32_t comp26 = (_mm512_reduce_add_epi64(comp24));                        \
    int32_t comp43 = ((comp25 + comp41));                                      \
    int32_t comp44 = ((comp26 + comp42));                                      \
    for (; iK < K; iK += BITS) {                                               \
                                                                               \
      int64_t load25 = (activation[((iM * K) + (iK + 0))]);                    \
      int64_t load26 = (kernel[((iN * K) + (iK + 0))]);                        \
      int64_t load27 = (activation[((iM * K) + (iK + 1))]);                    \
      int64_t load28 = (kernel[((iN * K) + (iK + 1))]);                        \
                                                                               \
      int64_t comp48 = ((load25 ^ load26));                                    \
      int64_t comp49 = ((load27 & load28));                                    \
      int32_t comp50 = (popcnt64(comp49));                                     \
      int64_t comp51 = ((comp48 & comp49));                                    \
      int32_t comp52 = (popcnt64(comp51));                                     \
      int32_t comp53 = ((comp43 + comp50));                                    \
      int32_t comp54 = ((comp44 + comp52));                                    \
      comp43 = (comp53);                                                       \
      comp44 = (comp54);                                                       \
    }                                                                          \
                                                                               \
    int64_t comp45 = ((comp43 - comp44));                                      \
    int64_t comp46 = ((comp45 - comp44));                                      \
    float comp47 = ((((comp46 > 0)) ? (comp46) : ((comp46 * alpha))));         \
    output[((iM * N) + iN)] = (comp47);                                        \
  }                                                                            \
  // vi: ft=c

namespace t2r_gemmLU_block_avx512 {
Tensor4D<float> gemmLU_block(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel, float alpha) {

  // block sizes; ideally parameterized
  // const size_t K_block_size = 8 * BITS;
  const size_t N_block_size = 16;
  const size_t M_block_size = 16;

  const int64_t *const activation_data = activation.data;
  const size_t batch_size = activation.dim1;
  const size_t output_height = activation.dim2;
  const size_t output_width = activation.dim3;
  const size_t kernel_height = activation.dim4;
  const size_t kernel_width = activation.dim5;
  const size_t channels = activation.dim6;
  const size_t bits = activation.dim7;

  const int64_t *const kernel_data = kernel.data;
  const size_t kernel_number = kernel.dim1;

  const size_t M = batch_size * output_height * output_width;
  const size_t K = kernel_height * kernel_width * channels * bits;
  const size_t N = kernel_number;

  Tensor4D<float> output(batch_size, output_height, output_width, kernel_number,
                         false);
  float *const output_data = output.data;

  size_t im = 0;
  // handle full blocks of M
  for (; (int)im <= (int)M - (int)M_block_size; im += M_block_size) {
    for (size_t imb = 0; imb < M_block_size; imb++) {

      size_t in = 0;
      // handle full blocks of N
      for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
        for (size_t inb = 0; inb < N_block_size; inb++) {
          // Use the PROCESS_BLOCKS macro here
          // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
          //                kernel_width, channels, bits, im, imb, in, inb, K,
          //                K_block_size, output_data, kernel_number, alpha);
          gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS,
                          (im + imb), (in + inb), alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        // Use the PROCESS_BLOCKS macro for leftover N processing
        // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
        //                kernel_width, channels, bits, im, imb, in, 0, K,
        //                K_block_size, output_data, kernel_number, alpha);
        gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS,
                        (im + imb), in, alpha);
      }
    }
  }
  // handle leftovers of M
  for (; im < M; im++) {
    size_t in = 0;
    for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
      for (size_t inb = 0; inb < N_block_size; inb++) {
        // Use the PROCESS_BLOCKS macro here
        // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
        //                kernel_width, channels, bits, im, 0, in, inb, K,
        //                K_block_size, output_data, kernel_number, alpha);
        gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS,
                        im, (in + inb), alpha);
      }
    }
    for (; in < N; in++) {
      // Use the PROCESS_BLOCKS macro for final leftover processing
      // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
      // kernel_width,
      //                channels, bits, im, 0, in, 0, K, K_block_size,
      //                output_data, kernel_number, alpha);
      gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS, im,
                      in, alpha);
    }
  }

  return output;
}
} // namespace t2r_gemmLU_block_avx512
