#include "impl/avx2_lessunpack/gemm_avx512_autogen.hpp"
#include "common.hpp"
#include <immintrin.h>

/*gemm_kernel_512(activation_data, kernel_data, K, im, in, output_data,
                      alpha, N);*/
#define gemm_kernel_512(activation, kernel, K, iM, iN, output, alpha, N)       \
  do {                                                                         \
    __m256i load5;                                                             \
    __m256i comp18;                                                            \
    int64_t load16;                                                            \
    __m256i load6;                                                             \
    __m256i comp12;                                                            \
    int comp27;                                                                \
    int comp29;                                                                \
    int64_t comp31;                                                            \
    int64_t load13;                                                            \
    int64_t comp45;                                                            \
    int comp46;                                                                \
    int comp47;                                                                \
    int64_t comp26;                                                            \
    int64_t comp30;                                                            \
    int comp35;                                                                \
    int comp10;                                                                \
    __m256i comp7;                                                             \
    int comp37;                                                                \
    __m256i comp14;                                                            \
    int64_t load24;                                                            \
    __m256i load10;                                                            \
    int64_t comp24;                                                            \
    __m256i comp16;                                                            \
    int comp8;                                                                 \
    int comp21;                                                                \
    int64_t load23;                                                            \
    __m256i load11;                                                            \
    __m256i comp6;                                                             \
    __m256i load4;                                                             \
    int64_t load19;                                                            \
    int64_t load22;                                                            \
    int64_t load14;                                                            \
    int comp11;                                                                \
    __m256i comp13;                                                            \
    int64_t load15;                                                            \
    __m256i comp17;                                                            \
    int comp36;                                                                \
    int comp44;                                                                \
    __m256i comp3;                                                             \
    int64_t comp23;                                                            \
    int comp34;                                                                \
    __m256i comp2;                                                             \
    int64_t load21;                                                            \
    __m256i load3;                                                             \
    int64_t comp43;                                                            \
    int64_t comp39;                                                            \
    int load7;                                                                 \
    int load2;                                                                 \
    int comp38;                                                                \
    int64_t comp40;                                                            \
    int comp28;                                                                \
    int64_t load20;                                                            \
    __m256i comp5;                                                             \
    int64_t comp33;                                                            \
    int load1;                                                                 \
    int64_t load17;                                                            \
    __m256i load9;                                                             \
    float comp41;                                                              \
    size_t iK;                                                                 \
    int comp25;                                                                \
    __m256i load12;                                                            \
    __m256i comp4;                                                             \
    int64_t load18;                                                            \
    int comp32;                                                                \
    int load8;                                                                 \
    __m256i comp1;                                                             \
    int comp48;                                                                \
    int comp22;                                                                \
    int64_t comp42;                                                            \
    int comp20;                                                                \
    int comp19;                                                                \
    __m256i comp15;                                                            \
    int comp9;                                                                 \
                                                                               \
    load1 = (0);                                                               \
    load2 = (0);                                                               \
    load7 = (0);                                                               \
    load8 = (0);                                                               \
    iK = (0);                                                                  \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((2) * (((4) * (BITS)))))))));             \
         iK += ((2) * (((4) * (BITS))))) {                                     \
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
      load9 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (0))))))));  \
      load10 = (_mm256_loadu_si256(                                            \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (0))))))));  \
      load11 = (_mm256_loadu_si256(                                            \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (4))))))));  \
      load12 = (_mm256_loadu_si256(                                            \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (4))))))));  \
                                                                               \
      comp1 = (_mm256_xor_si256(load3, load4));                                \
      comp3 = (_mm256_xor_si256(load5, load6));                                \
      comp2 = (_mm256_and_si256(load3, load4));                                \
      comp4 = (_mm256_and_si256(load5, load6));                                \
      comp5 = (_mm256_unpacklo_epi64(comp1, comp3));                           \
      comp6 = (_mm256_unpackhi_epi64(comp2, comp4));                           \
      comp8 = (popcnt(&comp6, sizeof(comp5)));                                 \
      comp7 = (_mm256_and_si256(comp5, comp6));                                \
      comp9 = (popcnt(&comp7, sizeof(comp7)));                                 \
      comp10 = (((load1) + (comp8)));                                          \
      comp11 = (((load2) + (comp9)));                                          \
      comp12 = (_mm256_xor_si256(load9, load10));                              \
      comp14 = (_mm256_xor_si256(load11, load12));                             \
      comp13 = (_mm256_and_si256(load9, load10));                              \
      comp15 = (_mm256_and_si256(load11, load12));                             \
      comp16 = (_mm256_unpacklo_epi64(comp12, comp14));                        \
      comp17 = (_mm256_unpackhi_epi64(comp13, comp15));                        \
      comp19 = (popcnt(&comp17, sizeof(comp16)));                              \
      comp18 = (_mm256_and_si256(comp16, comp17));                             \
      comp20 = (popcnt(&comp18, sizeof(comp18)));                              \
      comp21 = (((load7) + (comp19)));                                         \
      comp22 = (((load8) + (comp20)));                                         \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
      load7 = (comp21);                                                        \
      load8 = (comp22);                                                        \
    }                                                                          \
    for (; (((int)(iK))) <= (((((int)(K))) - (((int)(((2) * (BITS)))))));      \
         iK += ((2) * (BITS))) {                                               \
                                                                               \
      load13 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (0))))]);        \
      load14 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load15 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (1))))]);        \
      load16 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
      load17 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (0))))]);        \
      load18 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (0))))]);  \
      load19 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (1))))]);        \
      load20 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);  \
                                                                               \
      comp23 = (((load13) ^ (load14)));                                        \
      comp24 = (((load15) & (load16)));                                        \
      comp25 = (popcnt64(comp24));                                             \
      comp26 = (((comp23) & (comp24)));                                        \
      comp27 = (popcnt64(comp26));                                             \
      comp28 = (((load1) + (comp25)));                                         \
      comp29 = (((load2) + (comp27)));                                         \
      comp30 = (((load17) ^ (load18)));                                        \
      comp31 = (((load19) & (load20)));                                        \
      comp32 = (popcnt64(comp31));                                             \
      comp33 = (((comp30) & (comp31)));                                        \
      comp34 = (popcnt64(comp33));                                             \
      comp35 = (((load7) + (comp32)));                                         \
      comp36 = (((load8) + (comp34)));                                         \
      load1 = (comp28);                                                        \
      load2 = (comp29);                                                        \
      load7 = (comp35);                                                        \
      load8 = (comp36);                                                        \
    }                                                                          \
                                                                               \
    comp37 = (((load7) + (load1)));                                            \
    comp38 = (((load8) + (load2)));                                            \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load21 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load22 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load23 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load24 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp42 = (((load21) ^ (load22)));                                        \
      comp43 = (((load23) & (load24)));                                        \
      comp44 = (popcnt64(comp43));                                             \
      comp45 = (((comp42) & (comp43)));                                        \
      comp46 = (popcnt64(comp45));                                             \
      comp47 = (((comp37) + (comp44)));                                        \
      comp48 = (((comp38) + (comp46)));                                        \
      comp37 = (comp47);                                                       \
      comp38 = (comp48);                                                       \
    }                                                                          \
                                                                               \
    comp39 = (((comp37) - (comp38)));                                          \
    comp40 = (((comp39) - (comp38)));                                          \
    comp41 = (((((comp40) > (0))) ? (comp40) : (((comp40) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp41);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace avx2_lessunpack {
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
} // namespace avx2_lessunpack
