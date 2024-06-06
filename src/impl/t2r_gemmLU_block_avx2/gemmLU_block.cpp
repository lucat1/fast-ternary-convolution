#include "impl/t2r_gemmLU_block_avx2/gemmLU_block.hpp"
#include "common.hpp"
#include "libpopcnt.h"
#include "tensor_macros1.hpp"

#define gemm_kernel_256(activation, kernel, output, N, K, BITS, iM, iN, alpha) \
  do {                                                                         \
                                                                               \
    size_t iK = (0);                                                           \
    int load1 = (0);                                                           \
    int load2 = (0);                                                           \
    for (; (((int)(K))) <= (((((int)(K))) - (((int)(((4) * (BITS)))))));       \
         iK += ((4) * (BITS))) {                                               \
                                                                               \
      __m256i load3 = (_mm256_loadu_si256(                                     \
          (__m256i *)(activation + (((((iM) * (K))) + (((iK) + (0))))))));     \
      __m256i load4 = (_mm256_loadu_si256(                                     \
          (__m256i *)(kernel + (((((iN) * (K))) + (((iK) + (0))))))));         \
      __m256i load5 = (_mm256_loadu_si256(                                     \
          (__m256i *)(activation + (((((iM) * (K))) + (((iK) + (4))))))));     \
      __m256i load6 = (_mm256_loadu_si256(                                     \
          (__m256i *)(kernel + (((((iN) * (K))) + (((iK) + (4))))))));         \
                                                                               \
      __m256i comp1 = (_mm256_unpacklo_epi64(load3, load5));                   \
      __m256i comp2 = (_mm256_unpackhi_epi64(load3, load5));                   \
      __m256i comp3 = (_mm256_unpacklo_epi64(load4, load6));                   \
      __m256i comp4 = (_mm256_unpackhi_epi64(load4, load6));                   \
      __m256i comp5 = (_mm256_xor_si256(comp1, comp3));                        \
      __m256i comp6 = (_mm256_and_si256(comp2, comp4));                        \
      int comp8 = (popcnt(&comp6, sizeof(comp5)));                             \
      __m256i comp7 = (_mm256_and_si256(comp5, comp6));                        \
      int comp9 = (popcnt(&comp7, sizeof(comp7)));                             \
      int comp10 = (((load1) + (comp8)));                                      \
      int comp11 = (((load2) + (comp9)));                                      \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
    }                                                                          \
                                                                               \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      int64_t load7 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);       \
      int64_t load8 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);           \
      int64_t load9 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);       \
      int64_t load10 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);          \
                                                                               \
      int64_t comp15 = (((load7) ^ (load8)));                                  \
      int64_t comp16 = (((load9) & (load10)));                                 \
      int comp17 = (popcnt64(comp16));                                         \
      int64_t comp18 = (((comp15) & (comp16)));                                \
      int comp19 = (popcnt64(comp18));                                         \
      int comp20 = (((load1) + (comp17)));                                     \
      int comp21 = (((load2) + (comp19)));                                     \
      load1 = (comp20);                                                        \
      load2 = (comp21);                                                        \
    }                                                                          \
                                                                               \
    int64_t comp12 = (((load1) - (load2)));                                    \
    int64_t comp13 = (((comp12) - (load2)));                                   \
    float comp14 = (((((comp13) > (0))) ? (comp13) : (((comp13) * (alpha))))); \
    (output)[((((iM) * (N))) + (iN))] = (comp14);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace t2r_gemmLU_block_avx2 {
Tensor4D<float> gemmLU_block(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel, float alpha) {

  // block sizes; ideally parameterized
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
          gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
                          (im + imb), (in + inb), alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        // Use the PROCESS_BLOCKS macro for leftover N processing
        // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
        //                kernel_width, channels, bits, im, imb, in, 0, K,
        //                K_block_size, output_data, kernel_number, alpha);
        gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
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
        gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
                        im, (in + inb), alpha);
      }
    }
    for (; in < N; in++) {
      // Use the PROCESS_BLOCKS macro for final leftover processing
      // PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
      // kernel_width,
      //                channels, bits, im, 0, in, 0, K, K_block_size,
      //                output_data, kernel_number, alpha);
      gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS, im,
                      in, alpha);
    }
  }

  return output;
}
} // namespace t2r_gemmLU_block_avx2
