#include "minor_impls/t2r_gemmLU_block_avx512/gemmLU_block.hpp"
#include "common.hpp"
#include "tensor_macros1.hpp"

#define gemm_kernel_512(activation, kernel, output, N, K, BITS, iM, iN, alpha) \
  do {                                                                         \
    int64_t comp15;                                                            \
    __m512i comp3;                                                             \
    __m512i load1;                                                             \
    __m512i comp9;                                                             \
    int64_t comp29;                                                            \
    __m512i comp6;                                                             \
    __m512i comp2;                                                             \
    __m512i comp1;                                                             \
    int64_t load16;                                                            \
    float comp25;                                                              \
    int comp16;                                                                \
    __m512i comp11;                                                            \
    __m512i comp7;                                                             \
    int64_t load9;                                                             \
    int64_t load12;                                                            \
    int comp13;                                                                \
    __m512i load3;                                                             \
    int64_t load14;                                                            \
    int64_t comp27;                                                            \
    int comp30;                                                                \
    int comp32;                                                                \
    __m512i comp10;                                                            \
    int comp19;                                                                \
    int64_t comp14;                                                            \
    int64_t comp23;                                                            \
    int load8;                                                                 \
    __m512i load4;                                                             \
    __m512i comp8;                                                             \
    __m512i comp4;                                                             \
    int comp31;                                                                \
    int64_t comp17;                                                            \
    int load7;                                                                 \
    int64_t load10;                                                            \
    __m512i comp5;                                                             \
    __m512i load6;                                                             \
    __m512i load5;                                                             \
    int comp22;                                                                \
    size_t iK;                                                                 \
    int64_t load11;                                                            \
    int comp20;                                                                \
    int64_t comp26;                                                            \
    int comp21;                                                                \
    __m512i load2;                                                             \
    int comp12;                                                                \
    int64_t comp24;                                                            \
    int comp28;                                                                \
    int64_t load15;                                                            \
    int64_t load13;                                                            \
    int comp18;                                                                \
                                                                               \
    load1 = (_mm512_setzero_si512());                                          \
    load2 = (_mm512_setzero_si512());                                          \
    load7 = (0);                                                               \
    load8 = (0);                                                               \
    iK = (0);                                                                  \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((1) * (((8) * (BITS)))))))));             \
         iK += ((1) * (((8) * (BITS))))) {                                     \
                                                                               \
      load3 = (_mm512_loadu_si512(                                             \
          (__m512i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((8) * (BITS))))))) + (0))))))));  \
      load4 = (_mm512_loadu_si512(                                             \
          (__m512i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((8) * (BITS))))))) + (0))))))));  \
      load5 = (_mm512_loadu_si512(                                             \
          (__m512i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((0) * (((8) * (BITS))))))) + (8))))))));  \
      load6 = (_mm512_loadu_si512(                                             \
          (__m512i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((0) * (((8) * (BITS))))))) + (8))))))));  \
                                                                               \
      comp1 = (_mm512_xor_epi64(load3, load4));                                \
      comp3 = (_mm512_xor_epi64(load5, load6));                                \
      comp2 = (_mm512_and_epi64(load3, load4));                                \
      comp4 = (_mm512_and_epi64(load5, load6));                                \
      comp5 = (_mm512_unpacklo_epi64(comp1, comp3));                           \
      comp6 = (_mm512_unpackhi_epi64(comp2, comp4));                           \
      comp8 = (_mm512_popcnt_epi64(comp6));                                    \
      comp7 = (_mm512_and_epi64(comp5, comp6));                                \
      comp9 = (_mm512_popcnt_epi64(comp7));                                    \
      comp10 = (_mm512_add_epi64(load1, comp8));                               \
      comp11 = (_mm512_add_epi64(load2, comp9));                               \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
    }                                                                          \
    for (; (((int)(iK))) < (((((int)(K))) - (((int)(((4) * (BITS)))))));       \
         iK += ((4) * (BITS))) {                                               \
                                                                               \
      load9 = ((activation)[((((iM) * (K))) +                                  \
                             (((((iK) + (((0) * (BITS))))) + (0))))]);         \
      load10 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load11 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (1))))]);        \
      load12 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
                                                                               \
      comp14 = (((load9) ^ (load10)));                                         \
      comp15 = (((load11) & (load12)));                                        \
      comp16 = (popcnt64(comp15));                                             \
      comp17 = (((comp14) & (comp15)));                                        \
      comp18 = (popcnt64(comp17));                                             \
      comp19 = (((load7) + (comp16)));                                         \
      comp20 = (((load8) + (comp18)));                                         \
      load7 = (comp19);                                                        \
      load8 = (comp20);                                                        \
    }                                                                          \
                                                                               \
    comp12 = (_mm512_reduce_add_epi64(load1));                                 \
    comp13 = (_mm512_reduce_add_epi64(load2));                                 \
    comp21 = (((comp12) + (load7)));                                           \
    comp22 = (((comp13) + (load8)));                                           \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load13 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load14 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load15 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load16 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp26 = (((load13) ^ (load14)));                                        \
      comp27 = (((load15) & (load16)));                                        \
      comp28 = (popcnt64(comp27));                                             \
      comp29 = (((comp26) & (comp27)));                                        \
      comp30 = (popcnt64(comp29));                                             \
      comp31 = (((comp21) + (comp28)));                                        \
      comp32 = (((comp22) + (comp30)));                                        \
      comp21 = (comp31);                                                       \
      comp22 = (comp32);                                                       \
    }                                                                          \
                                                                               \
    comp23 = (((comp21) - (comp22)));                                          \
    comp24 = (((comp23) - (comp22)));                                          \
    comp25 = (((((comp24) > (0))) ? (comp24) : (((comp24) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp25);                              \
  } while (0);                                                                 \
  // vi: ft=c

namespace t2r_gemmLU_block_avx512 {
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
          gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS,
                          (im + imb), (in + inb), alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
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
        gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS,
                        im, (in + inb), alpha);
      }
    }
    for (; in < N; in++) {
      gemm_kernel_512(activation_data, kernel_data, output_data, N, K, BITS, im,
                      in, alpha);
    }
  }

  return output;
}
} // namespace t2r_gemmLU_block_avx512
