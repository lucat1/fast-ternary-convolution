#include "impl/avx2_test/gemm_avx512_autogen.hpp"
#include "common.hpp"
#include <immintrin.h>

/*gemm_kernel_512(activation_data, kernel_data, K, im, in, output_data,
                      alpha, N);*/
#define gemm_kernel_512(activation, kernel, K, iM, iN, output, alpha, N)       \
  do {                                                                         \
    __m256i comp2;                                                             \
    int64_t comp15;                                                            \
    int64_t comp12;                                                            \
    int comp28;                                                                \
    int comp16;                                                                \
    int comp10;                                                                \
    int64_t comp20;                                                            \
    float comp21;                                                              \
    int64_t load9;                                                             \
    __m256i comp7;                                                             \
    int64_t comp23;                                                            \
    int comp26;                                                                \
    __m256i comp3;                                                             \
    int comp11;                                                                \
    size_t iK;                                                                 \
    int comp17;                                                                \
    int64_t load10;                                                            \
    int comp18;                                                                \
    int64_t comp19;                                                            \
    int comp24;                                                                \
    int load1;                                                                 \
    __m256i load4;                                                             \
    int64_t load7;                                                             \
    __m256i comp5;                                                             \
    int comp8;                                                                 \
    int load2;                                                                 \
    int comp14;                                                                \
    __m256i comp6;                                                             \
    int64_t load13;                                                            \
    int64_t comp22;                                                            \
    int64_t comp13;                                                            \
    __m256i comp1;                                                             \
    __m256i load3;                                                             \
    int comp27;                                                                \
    int64_t load12;                                                            \
    int64_t load11;                                                            \
    int64_t comp25;                                                            \
    int64_t load14;                                                            \
    __m256i comp4;                                                             \
    __m256i load5;                                                             \
    __m256i load6;                                                             \
    int comp9;                                                                 \
    int64_t load8;                                                             \
                                                                               \
    load1 = (0);                                                               \
    load2 = (0);                                                               \
    iK = (0);                                                                  \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((1) * (((4) * (BITS)))))))));             \
         iK += ((1) * (((4) * (BITS))))) {                                     \
                                                                               \
      load3 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (0))))))));  \
      load4 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (0))))))));  \
      load5 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (4))))))));  \
      load6 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (4))))))));  \
                                                                               \
      comp1 = (_mm256_xor_si256(load3, load4));                                \
      comp3 = (_mm256_xor_si256(load5, load6));                                \
      comp2 = (_mm256_and_si256(load3, load4));                                \
      comp4 = (_mm256_and_si256(load5, load6));                                \
      comp5 = (_mm256_unpacklo_epi64(comp1, comp3));                           \
      comp6 = (_mm256_unpackhi_epi64(comp2, comp4));                           \
      int v11 = popcnt64(comp6[0]);                                            \
      int v12 = popcnt64(comp6[1]);                                            \
      int v13 = popcnt64(comp6[2]);                                            \
      int v14 = popcnt64(comp6[3]);                                            \
      int v15 = v11 + v12;                                                     \
      int v16 = v13 + v14;                                                     \
      comp8 = v15 + v16;                                                       \
      comp7 = (_mm256_and_si256(comp5, comp6));                                \
      int v21 = popcnt64(comp7[0]);                                            \
      int v22 = popcnt64(comp7[1]);                                            \
      int v23 = popcnt64(comp6[2]);                                            \
      int v24 = popcnt64(comp7[3]);                                            \
      int v25 = v21 + v22;                                                     \
      int v26 = v23 + v24;                                                     \
      comp9 = v25 + v26;                                                       \
      comp10 = (((load1) + (comp8)));                                          \
      comp11 = (((load2) + (comp9)));                                          \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
    }                                                                          \
    for (; (((int)(iK))) <= (((((int)(K))) - (((int)(((1) * (BITS)))))));      \
         iK += ((1) * (BITS))) {                                               \
                                                                               \
      load7 = ((activation)[((((iM) * (K))) +                                  \
                             (((((iK) + (((0) * (BITS))))) + (0))))]);         \
      load8 = ((                                                               \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load9 = ((activation)[((((iM) * (K))) +                                  \
                             (((((iK) + (((0) * (BITS))))) + (1))))]);         \
      load10 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
                                                                               \
      comp12 = (((load7) ^ (load8)));                                          \
      comp13 = (((load9) & (load10)));                                         \
      comp14 = (popcnt64(comp13));                                             \
      comp15 = (((comp12) & (comp13)));                                        \
      comp16 = (popcnt64(comp15));                                             \
      comp17 = (((load1) + (comp14)));                                         \
      comp18 = (((load2) + (comp16)));                                         \
      load1 = (comp17);                                                        \
      load2 = (comp18);                                                        \
    }                                                                          \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load11 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load12 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load13 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load14 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp22 = (((load11) ^ (load12)));                                        \
      comp23 = (((load13) & (load14)));                                        \
      comp24 = (popcnt64(comp23));                                             \
      comp25 = (((comp22) & (comp23)));                                        \
      comp26 = (popcnt64(comp25));                                             \
      comp27 = (((load1) + (comp24)));                                         \
      comp28 = (((load2) + (comp26)));                                         \
      load1 = (comp27);                                                        \
      load2 = (comp28);                                                        \
    }                                                                          \
                                                                               \
    comp19 = (((load1) - (load2)));                                            \
    comp20 = (((comp19) - (load2)));                                           \
    comp21 = (((((comp20) > (0))) ? (comp20) : (((comp20) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp21);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace avx2_test {
Tensor4D<float> gemm_avx512_autogen(const Tensor7D<int64_t> &activation,
                                    const Tensor5D<int64_t> &kernel,
                                    float alpha) {

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
          // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
          //             im + imb, in + inb, alpha);
          gemm_kernel_512(activation_data, kernel_data, K, im + imb, in + inb,
                          output_data, alpha, N);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        // Use the PROCESS_BLOCKS macro for leftover N processing
        // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
        //             im + imb, in, alpha);
        gemm_kernel_512(activation_data, kernel_data, K, im + imb, in,
                        output_data, alpha, N);
      }
    }
  }
  // handle leftovers of M
  for (; im < M; im++) {
    size_t in = 0;
    for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
      for (size_t inb = 0; inb < N_block_size; inb++) {
        // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
        // im,
        //             in + inb, alpha);
        gemm_kernel_512(activation_data, kernel_data, K, im, in + inb,
                        output_data, alpha, N);
      }
    }
    for (; in < N; in++) {
      // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS, im,
      // in,
      //             alpha);
      gemm_kernel_512(activation_data, kernel_data, K, im, in, output_data,
                      alpha, N);
    }
  }

  return output;
}
} // namespace avx2_test
