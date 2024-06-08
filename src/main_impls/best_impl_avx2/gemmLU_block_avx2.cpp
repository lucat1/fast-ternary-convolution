#include "common.hpp"
#include "libpopcnt.h"
#include "main_impls/best_impl_avx2/gemmLU_block.hpp"
#include "tensor_macros1.hpp"

/*gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
                        im, (in + inb), alpha);*/
#define gemm_kernel_256(activation, kernel, output, N, K, BITS, iM, iN, alpha) \
  do {                                                                         \
    __m256i declr2[((((K) / (((((4) * (BITS))) * (2))))) * (2))];              \
    int comp17;                                                                \
    int64_t comp22;                                                            \
    int64_t load34;                                                            \
    int load28;                                                                \
    int load22;                                                                \
    int comp47;                                                                \
    __m256i load6;                                                             \
    int64_t load31;                                                            \
    int comp33;                                                                \
    int64_t comp57;                                                            \
    int load16;                                                                \
    int comp50;                                                                \
    int64_t comp18;                                                            \
    int comp48;                                                                \
    __m256i comp8;                                                             \
    int64_t comp23;                                                            \
    int64_t comp25;                                                            \
    int comp49;                                                                \
    int64_t load23;                                                            \
    int64_t load33;                                                            \
    int64_t load11;                                                            \
    int64_t load32;                                                            \
    int comp31;                                                                \
    int64_t comp37;                                                            \
    int64_t load20;                                                            \
    int64_t load24;                                                            \
    int comp20;                                                                \
    __m256i comp9;                                                             \
    int comp62;                                                                \
    int comp60;                                                                \
    int64_t comp16;                                                            \
    int comp45;                                                                \
    __m256i comp2;                                                             \
    int64_t load26;                                                            \
    __m256i comp4;                                                             \
    __m256i load8;                                                             \
    int64_t load12;                                                            \
    int comp46;                                                                \
    int64_t comp54;                                                            \
    __m256i comp7;                                                             \
    __m256i load7;                                                             \
    int comp28;                                                                \
    int comp42;                                                                \
    __m256i comp11;                                                            \
    __m256i load2;                                                             \
    int comp52;                                                                \
    int64_t comp39;                                                            \
    int comp58;                                                                \
    __m256i load3;                                                             \
    int comp34;                                                                \
    int load21;                                                                \
    int comp40;                                                                \
    int64_t comp56;                                                            \
    int64_t comp36;                                                            \
    int64_t comp15;                                                            \
    int64_t comp29;                                                            \
    int comp27;                                                                \
    int64_t comp30;                                                            \
    int comp51;                                                                \
    int64_t load35;                                                            \
    __m256i load1;                                                             \
    __m256i comp13;                                                            \
    size_t iK;                                                                 \
    int64_t load19;                                                            \
    __m256i comp5;                                                             \
    int load15;                                                                \
    int64_t load18;                                                            \
    int comp35;                                                                \
    __m256i comp6;                                                             \
    int64_t load14;                                                            \
    int comp19;                                                                \
    int comp24;                                                                \
    int comp21;                                                                \
    __m256i comp3;                                                             \
    __m256i comp12;                                                            \
    int64_t load25;                                                            \
    float comp55;                                                              \
    __m256i load4;                                                             \
    int comp26;                                                                \
    __m256i comp10;                                                            \
    __m256i declr1[((((K) / (((((4) * (BITS))) * (2))))) * (2))];              \
    int load27;                                                                \
    int64_t load13;                                                            \
    int comp61;                                                                \
    int64_t load36;                                                            \
    int64_t comp59;                                                            \
    int load10;                                                                \
    int comp44;                                                                \
    int64_t load29;                                                            \
    int comp43;                                                                \
    int load9;                                                                 \
    int64_t load30;                                                            \
    int64_t comp32;                                                            \
    int comp41;                                                                \
    int64_t comp53;                                                            \
    __m256i load5;                                                             \
    __m256i comp14;                                                            \
    int64_t load17;                                                            \
    __m256i comp1;                                                             \
    int comp38;                                                                \
                                                                               \
    iK = (0);                                                                  \
                                                                               \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((2) * (((4) * (BITS)))))))));             \
         iK += ((2) * (((4) * (BITS))))) {                                     \
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
      load5 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (0))))))));  \
      load6 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (0))))))));  \
      load7 = (_mm256_loadu_si256(                                             \
          (__m256i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (4))))))));  \
      load8 = (_mm256_loadu_si256(                                             \
          (__m256i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((4) * (BITS))))))) + (4))))))));  \
                                                                               \
      comp1 = (_mm256_xor_si256(load1, load2));                                \
      comp3 = (_mm256_xor_si256(load3, load4));                                \
      comp2 = (_mm256_and_si256(load1, load2));                                \
      comp4 = (_mm256_and_si256(load3, load4));                                \
      comp5 = (_mm256_unpacklo_epi64(comp1, comp3));                           \
      comp6 = (_mm256_unpackhi_epi64(comp2, comp4));                           \
      comp7 = (_mm256_and_si256(comp5, comp6));                                \
      comp8 = (_mm256_xor_si256(load5, load6));                                \
      comp10 = (_mm256_xor_si256(load7, load8));                               \
      comp9 = (_mm256_and_si256(load5, load6));                                \
      comp11 = (_mm256_and_si256(load7, load8));                               \
      comp12 = (_mm256_unpacklo_epi64(comp8, comp10));                         \
      comp13 = (_mm256_unpackhi_epi64(comp9, comp11));                         \
      comp14 = (_mm256_and_si256(comp12, comp13));                             \
      (declr1)[((((iK) + (((0) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp6);                                                             \
      (declr2)[((((iK) + (((0) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp7);                                                             \
      (declr1)[((((iK) + (((1) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp13);                                                            \
      (declr2)[((((iK) + (((1) * (((4) * (BITS))))))) / (((4) * (BITS))))] =   \
          (comp14);                                                            \
    }                                                                          \
                                                                               \
    load9 = (0);                                                               \
    load10 = (0);                                                              \
    load15 = (0);                                                              \
    load16 = (0);                                                              \
    load21 = (0);                                                              \
    load22 = (0);                                                              \
    load27 = (0);                                                              \
    load28 = (0);                                                              \
                                                                               \
    comp43 = (popcnt(declr1, sizeof(declr1)));                                 \
    comp44 = (popcnt(declr2, sizeof(declr2)));                                 \
    for (; (((int)(iK))) <= (((((int)(K))) - (((int)(((4) * (BITS)))))));      \
         iK += ((4) * (BITS))) {                                               \
                                                                               \
      load11 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (0))))]);        \
      load12 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load13 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (1))))]);        \
      load14 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
      load17 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (0))))]);        \
      load18 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (0))))]);  \
      load19 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (1))))]);        \
      load20 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);  \
      load23 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((2) * (BITS))))) + (0))))]);        \
      load24 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (0))))]);  \
      load25 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((2) * (BITS))))) + (1))))]);        \
      load26 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (1))))]);  \
      load29 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((3) * (BITS))))) + (0))))]);        \
      load30 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (0))))]);  \
      load31 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((3) * (BITS))))) + (1))))]);        \
      load32 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (1))))]);  \
                                                                               \
      comp15 = (((load11) ^ (load12)));                                        \
      comp16 = (((load13) & (load14)));                                        \
      comp17 = (popcnt64(comp16));                                             \
      comp18 = (((comp15) & (comp16)));                                        \
      comp19 = (popcnt64(comp18));                                             \
      comp20 = (((load9) + (comp17)));                                         \
      comp21 = (((load10) + (comp19)));                                        \
      comp22 = (((load17) ^ (load18)));                                        \
      comp23 = (((load19) & (load20)));                                        \
      comp24 = (popcnt64(comp23));                                             \
      comp25 = (((comp22) & (comp23)));                                        \
      comp26 = (popcnt64(comp25));                                             \
      comp27 = (((load15) + (comp24)));                                        \
      comp28 = (((load16) + (comp26)));                                        \
      comp29 = (((load23) ^ (load24)));                                        \
      comp30 = (((load25) & (load26)));                                        \
      comp31 = (popcnt64(comp30));                                             \
      comp32 = (((comp29) & (comp30)));                                        \
      comp33 = (popcnt64(comp32));                                             \
      comp34 = (((load21) + (comp31)));                                        \
      comp35 = (((load22) + (comp33)));                                        \
      comp36 = (((load29) ^ (load30)));                                        \
      comp37 = (((load31) & (load32)));                                        \
      comp38 = (popcnt64(comp37));                                             \
      comp39 = (((comp36) & (comp37)));                                        \
      comp40 = (popcnt64(comp39));                                             \
      comp41 = (((load27) + (comp38)));                                        \
      comp42 = (((load28) + (comp40)));                                        \
      load9 = (comp20);                                                        \
      load10 = (comp21);                                                       \
      load15 = (comp27);                                                       \
      load16 = (comp28);                                                       \
      load21 = (comp34);                                                       \
      load22 = (comp35);                                                       \
      load27 = (comp41);                                                       \
      load28 = (comp42);                                                       \
    }                                                                          \
                                                                               \
    comp45 = (((comp43) + (load27)));                                          \
    comp46 = (((load21) + (load15)));                                          \
    comp47 = (((comp46) + (comp45)));                                          \
    comp48 = (((comp47) + (load9)));                                           \
    comp49 = (((comp44) + (load28)));                                          \
    comp50 = (((load22) + (load16)));                                          \
    comp51 = (((comp50) + (comp49)));                                          \
    comp52 = (((comp51) + (load10)));                                          \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load33 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load34 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load35 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load36 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp56 = (((load33) ^ (load34)));                                        \
      comp57 = (((load35) & (load36)));                                        \
      comp58 = (popcnt64(comp57));                                             \
      comp59 = (((comp56) & (comp57)));                                        \
      comp60 = (popcnt64(comp59));                                             \
      comp61 = (((comp48) + (comp58)));                                        \
      comp62 = (((comp52) + (comp60)));                                        \
      comp48 = (comp61);                                                       \
      comp52 = (comp62);                                                       \
    }                                                                          \
                                                                               \
    comp53 = (((comp48) - (comp52)));                                          \
    comp54 = (((comp53) - (comp52)));                                          \
    comp55 = (((((comp54) > (0))) ? (comp54) : (((comp54) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp55);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace best_impl_avx2 {
Tensor4D<float> gemmLU_block_avx2(const Tensor7D<int64_t> &activation,
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
          gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
                          (im + imb), (in + inb), alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
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
        gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS,
                        im, (in + inb), alpha);
      }
    }
    for (; in < N; in++) {
      gemm_kernel_256(activation_data, kernel_data, output_data, N, K, BITS, im,
                      in, alpha);
    }
  }

  return output;
}
} // namespace best_impl_avx2
