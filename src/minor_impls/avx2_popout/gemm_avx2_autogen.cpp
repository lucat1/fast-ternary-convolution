#include "minor_impls/avx2_popout/gemm_avx2_autogen.hpp"
#include "common.hpp"
#include <immintrin.h>

#define gemm_kernel_256(activation, kernel, K, iM, iN, output, alpha, N)       \
  do {                                                                         \
    int64_t comp30;                                                            \
    int load11;                                                                \
    int64_t comp11;                                                            \
    int comp40;                                                                \
    __m256i declr2[((((K) / (((((4) * (BITS))) * (1))))) * (1))];              \
    __m256i comp3;                                                             \
    int64_t load10;                                                            \
    int comp14;                                                                \
    int comp34;                                                                \
    __m256i comp2;                                                             \
    int64_t comp52;                                                            \
    int64_t comp23;                                                            \
    int64_t comp46;                                                            \
    __m256i comp1;                                                             \
    int64_t load26;                                                            \
    int load24;                                                                \
    int64_t comp16;                                                            \
    int comp55;                                                                \
    int comp38;                                                                \
    int comp41;                                                                \
    int comp17;                                                                \
    int64_t comp22;                                                            \
    int comp28;                                                                \
    int64_t comp15;                                                            \
    int64_t comp49;                                                            \
    int64_t load16;                                                            \
    int64_t load19;                                                            \
    int64_t load29;                                                            \
    int comp31;                                                                \
    __m256i comp6;                                                             \
    int64_t load14;                                                            \
    int64_t load31;                                                            \
    int comp24;                                                                \
    int load6;                                                                 \
    int64_t comp32;                                                            \
    int64_t load22;                                                            \
    int64_t comp25;                                                            \
    int64_t load28;                                                            \
    int64_t comp47;                                                            \
    int load5;                                                                 \
    int comp43;                                                                \
    int comp51;                                                                \
    int64_t load15;                                                            \
    int64_t comp8;                                                             \
    int load17;                                                                \
    int comp20;                                                                \
    __m256i load1;                                                             \
    int64_t load8;                                                             \
    int comp19;                                                                \
    int comp33;                                                                \
    float comp48;                                                              \
    int comp12;                                                                \
    int load12;                                                                \
    int64_t load30;                                                            \
    int64_t comp9;                                                             \
    __m256i load4;                                                             \
    int64_t load21;                                                            \
    __m256i comp7;                                                             \
    int comp54;                                                                \
    int64_t load7;                                                             \
    int64_t comp18;                                                            \
    int comp21;                                                                \
    __m256i declr1[((((K) / (((((4) * (BITS))) * (1))))) * (1))];              \
    int load18;                                                                \
    int64_t load32;                                                            \
    int64_t load27;                                                            \
    int64_t comp29;                                                            \
    int comp53;                                                                \
    int comp27;                                                                \
    int comp35;                                                                \
    int64_t load20;                                                            \
    int64_t load25;                                                            \
    size_t iK;                                                                 \
    __m256i load3;                                                             \
    int comp44;                                                                \
    __m256i load2;                                                             \
    int64_t load13;                                                            \
    int comp36;                                                                \
    int64_t comp50;                                                            \
    int comp13;                                                                \
    int comp10;                                                                \
    int comp39;                                                                \
    int comp37;                                                                \
    int comp45;                                                                \
    int load23;                                                                \
    int comp42;                                                                \
    int64_t load9;                                                             \
    __m256i comp5;                                                             \
    __m256i comp4;                                                             \
    int comp26;                                                                \
                                                                               \
    iK = (0);                                                                  \
                                                                               \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((1) * (((4) * (BITS)))))))));             \
         iK += ((1) * (((4) * (BITS))))) {                                     \
                                                                               \
      load1 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (0))))))));  \
      load2 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (0))))))));  \
      load3 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (4))))))));  \
      load4 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((4) * (BITS))))))) + (4))))))));  \
                                                                               \
      comp1 = (_mm256_unpacklo_epi64(load1, load3));                           \
      comp2 = (_mm256_unpackhi_epi64(load1, load3));                           \
      comp3 = (_mm256_unpacklo_epi64(load2, load4));                           \
      comp4 = (_mm256_unpackhi_epi64(load2, load4));                           \
      comp5 = (_mm256_xor_si256(comp1, comp3));                                \
      comp6 = (_mm256_and_si256(comp2, comp4));                                \
      comp7 = (_mm256_and_si256(comp5, comp6));                                \
      (declr1)[((((iK) + (((0) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp6);                                                             \
      (declr2)[((((iK) + (((0) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp7);                                                             \
    }                                                                          \
                                                                               \
    load5 = (0);                                                               \
    load6 = (0);                                                               \
    load11 = (0);                                                              \
    load12 = (0);                                                              \
    load17 = (0);                                                              \
    load18 = (0);                                                              \
    load23 = (0);                                                              \
    load24 = (0);                                                              \
                                                                               \
    comp36 = (popcnt(declr1, sizeof(declr1)));                                 \
    comp37 = (popcnt(declr2, sizeof(declr2)));                                 \
    for (; (((int)(iK))) <= (((((int)(K))) - (((int)(((4) * (BITS)))))));      \
         iK += ((4) * (BITS))) {                                               \
                                                                               \
      load7 = ((activation)[((((iM) * (K))) +                                  \
                             (((((iK) + (((0) * (BITS))))) + (0))))]);         \
      load8 = ((                                                               \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load9 = ((activation)[((((iM) * (K))) +                                  \
                             (((((iK) + (((0) * (BITS))))) + (1))))]);         \
      load10 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
      load13 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (0))))]);        \
      load14 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (0))))]);  \
      load15 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (1))))]);        \
      load16 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);  \
      load19 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((2) * (BITS))))) + (0))))]);        \
      load20 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (0))))]);  \
      load21 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((2) * (BITS))))) + (1))))]);        \
      load22 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (1))))]);  \
      load25 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((3) * (BITS))))) + (0))))]);        \
      load26 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (0))))]);  \
      load27 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((3) * (BITS))))) + (1))))]);        \
      load28 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (1))))]);  \
                                                                               \
      comp8 = (((load7) ^ (load8)));                                           \
      comp9 = (((load9) & (load10)));                                          \
      comp10 = (popcnt64(comp9));                                              \
      comp11 = (((comp8) & (comp9)));                                          \
      comp12 = (popcnt64(comp11));                                             \
      comp13 = (((load5) + (comp10)));                                         \
      comp14 = (((load6) + (comp12)));                                         \
      comp15 = (((load13) ^ (load14)));                                        \
      comp16 = (((load15) & (load16)));                                        \
      comp17 = (popcnt64(comp16));                                             \
      comp18 = (((comp15) & (comp16)));                                        \
      comp19 = (popcnt64(comp18));                                             \
      comp20 = (((load11) + (comp17)));                                        \
      comp21 = (((load12) + (comp19)));                                        \
      comp22 = (((load19) ^ (load20)));                                        \
      comp23 = (((load21) & (load22)));                                        \
      comp24 = (popcnt64(comp23));                                             \
      comp25 = (((comp22) & (comp23)));                                        \
      comp26 = (popcnt64(comp25));                                             \
      comp27 = (((load17) + (comp24)));                                        \
      comp28 = (((load18) + (comp26)));                                        \
      comp29 = (((load25) ^ (load26)));                                        \
      comp30 = (((load27) & (load28)));                                        \
      comp31 = (popcnt64(comp30));                                             \
      comp32 = (((comp29) & (comp30)));                                        \
      comp33 = (popcnt64(comp32));                                             \
      comp34 = (((load23) + (comp31)));                                        \
      comp35 = (((load24) + (comp33)));                                        \
      load5 = (comp13);                                                        \
      load6 = (comp14);                                                        \
      load11 = (comp20);                                                       \
      load12 = (comp21);                                                       \
      load17 = (comp27);                                                       \
      load18 = (comp28);                                                       \
      load23 = (comp34);                                                       \
      load24 = (comp35);                                                       \
    }                                                                          \
                                                                               \
    comp38 = (((comp36) + (load23)));                                          \
    comp39 = (((load17) + (load11)));                                          \
    comp40 = (((comp39) + (comp38)));                                          \
    comp41 = (((comp40) + (load5)));                                           \
    comp42 = (((comp37) + (load24)));                                          \
    comp43 = (((load18) + (load12)));                                          \
    comp44 = (((comp43) + (comp42)));                                          \
    comp45 = (((comp44) + (load6)));                                           \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load29 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load30 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load31 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load32 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp49 = (((load29) ^ (load30)));                                        \
      comp50 = (((load31) & (load32)));                                        \
      comp51 = (popcnt64(comp50));                                             \
      comp52 = (((comp49) & (comp50)));                                        \
      comp53 = (popcnt64(comp52));                                             \
      comp54 = (((comp41) + (comp51)));                                        \
      comp55 = (((comp45) + (comp53)));                                        \
      comp41 = (comp54);                                                       \
      comp45 = (comp55);                                                       \
    }                                                                          \
                                                                               \
    comp46 = (((comp41) - (comp45)));                                          \
    comp47 = (((comp46) - (comp45)));                                          \
    comp48 = (((((comp47) > (0))) ? (comp47) : (((comp47) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp48);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace avx2_popout {
Tensor4D<float> gemm_avx2_autogen(const Tensor7D<int64_t> &activation,
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
          gemm_kernel_256(activation_data, kernel_data, K, im + imb, in + inb,
                          output_data, alpha, N);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        gemm_kernel_256(activation_data, kernel_data, K, im + imb, in,
                        output_data, alpha, N);
      }
    }
  }
  // handle leftovers of M
  for (; im < M; im++) {
    size_t in = 0;
    for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
      for (size_t inb = 0; inb < N_block_size; inb++) {
        gemm_kernel_256(activation_data, kernel_data, K, im, in + inb,
                        output_data, alpha, N);
      }
    }
    for (; in < N; in++) {
      gemm_kernel_256(activation_data, kernel_data, K, im, in, output_data,
                      alpha, N);
    }
  }

  return output;
}
} // namespace avx2_popout
