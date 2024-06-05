#include "impl/gemm_avx512_manual/gemm_avx512_manual.hpp"
#include "common.hpp"
#include <immintrin.h>

#define inner_loop_vectorized(activation_data, kernel_data, K, im, in, output, \
                              alpha, N)                                        \
  {                                                                            \
    size_t ik = 0;                                                             \
    __m512i cntp1_vec1 = _mm512_setzero_si512(),                               \
            cntp2_vec1 = _mm512_setzero_si512(),                               \
            cntp1_vec2 = _mm512_setzero_si512(),                               \
            cntp2_vec2 = _mm512_setzero_si512();                               \
    __m512i p11, p21, p12, p22;                                                \
    __m512i activation11, activation21, activation_lo1, activation_hi1,        \
        kernel11, kernel21, kernel_lo1, kernel_hi1, activation12,              \
        activation22, activation_lo2, activation_hi2, kernel12, kernel22,      \
        kernel_lo2, kernel_hi2;                                                \
    for (; ik + 15 * BITS < K; ik += 16 * BITS) {                              \
      activation11 = _mm512_loadu_epi64(activation_data + K * im + ik);        \
      activation21 = _mm512_loadu_epi64(activation_data + K * im + ik + 8);    \
      activation12 =                                                           \
          _mm512_loadu_epi64(activation_data + K * im + ik + 2 * 8);           \
      activation22 =                                                           \
          _mm512_loadu_epi64(activation_data + K * im + ik + 3 * 8);           \
      kernel11 = _mm512_loadu_epi64(kernel_data + K * in + ik);                \
      kernel21 = _mm512_loadu_epi64(kernel_data + K * in + ik + 8);            \
      kernel12 = _mm512_loadu_epi64(kernel_data + K * in + ik + 2 * 8);        \
      kernel22 = _mm512_loadu_epi64(kernel_data + K * in + ik + 3 * 8);        \
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
      int64_t p1 = activation_data[K * im + ik] ^ kernel_data[K * in + ik];    \
      int64_t p2 =                                                             \
          activation_data[K * im + ik + 1] & kernel_data[K * in + ik + 1];     \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const float current = cntp1 - cntp2 - cntp2;                               \
    const float value = current > 0 ? current : current * alpha;               \
    output[im * N + in] = value;                                               \
  }

#define gemm_kernel(activation, kernel, output, N, K, BITS, iM, iN, alpha)     \
  do {                                                                         \
                                                                               \
    int load1 = (0);                                                           \
    int load2 = (0);                                                           \
    int load7 = (0);                                                           \
    int load8 = (0);                                                           \
    int load13 = (0);                                                          \
    int load14 = (0);                                                          \
    int load19 = (0);                                                          \
    int load20 = (0);                                                          \
    int load25 = (0);                                                          \
    int load26 = (0);                                                          \
    int load31 = (0);                                                          \
    int load32 = (0);                                                          \
    int load37 = (0);                                                          \
    int load38 = (0);                                                          \
    int load43 = (0);                                                          \
    int load44 = (0);                                                          \
    size_t iK = (0);                                                           \
    for (; (((int)(K))) <= (((((int)(K))) - (((int)(((8) * (BITS)))))));       \
         iK += ((8) * (BITS))) {                                               \
                                                                               \
      int64_t load3 = ((activation)[((((iM) * (K))) +                          \
                                     (((((iK) + (((0) * (BITS))))) + (0))))]); \
      int64_t load4 = ((                                                       \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      int64_t load5 = ((activation)[((((iM) * (K))) +                          \
                                     (((((iK) + (((0) * (BITS))))) + (1))))]); \
      int64_t load6 = ((                                                       \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
      int64_t load9 = ((activation)[((((iM) * (K))) +                          \
                                     (((((iK) + (((1) * (BITS))))) + (0))))]); \
      int64_t load10 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (0))))]);  \
      int64_t load11 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);           \
      int64_t load12 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);  \
      int64_t load15 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((2) * (BITS))))) + (0))))]);           \
      int64_t load16 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (0))))]);  \
      int64_t load17 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((2) * (BITS))))) + (1))))]);           \
      int64_t load18 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((2) * (BITS))))) + (1))))]);  \
      int64_t load21 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((3) * (BITS))))) + (0))))]);           \
      int64_t load22 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (0))))]);  \
      int64_t load23 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((3) * (BITS))))) + (1))))]);           \
      int64_t load24 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((3) * (BITS))))) + (1))))]);  \
      int64_t load27 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((4) * (BITS))))) + (0))))]);           \
      int64_t load28 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((4) * (BITS))))) + (0))))]);  \
      int64_t load29 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((4) * (BITS))))) + (1))))]);           \
      int64_t load30 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((4) * (BITS))))) + (1))))]);  \
      int64_t load33 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((5) * (BITS))))) + (0))))]);           \
      int64_t load34 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((5) * (BITS))))) + (0))))]);  \
      int64_t load35 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((5) * (BITS))))) + (1))))]);           \
      int64_t load36 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((5) * (BITS))))) + (1))))]);  \
      int64_t load39 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((6) * (BITS))))) + (0))))]);           \
      int64_t load40 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((6) * (BITS))))) + (0))))]);  \
      int64_t load41 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((6) * (BITS))))) + (1))))]);           \
      int64_t load42 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((6) * (BITS))))) + (1))))]);  \
      int64_t load45 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((7) * (BITS))))) + (0))))]);           \
      int64_t load46 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((7) * (BITS))))) + (0))))]);  \
      int64_t load47 = ((activation)[(                                         \
          (((iM) * (K))) + (((((iK) + (((7) * (BITS))))) + (1))))]);           \
      int64_t load48 = ((                                                      \
          kernel)[((((iN) * (K))) + (((((iK) + (((7) * (BITS))))) + (1))))]);  \
                                                                               \
      int64_t comp1 = (((load3) ^ (load4)));                                   \
      int64_t comp2 = (((load5) & (load6)));                                   \
      int comp3 = (popcnt64(comp2));                                           \
      int64_t comp4 = (((comp1) & (comp2)));                                   \
      int comp5 = (popcnt64(comp4));                                           \
      int comp6 = (((load1) + (comp3)));                                       \
      int comp7 = (((load2) + (comp5)));                                       \
      int64_t comp8 = (((load9) ^ (load10)));                                  \
      int64_t comp9 = (((load11) & (load12)));                                 \
      int comp10 = (popcnt64(comp9));                                          \
      int64_t comp11 = (((comp8) & (comp9)));                                  \
      int comp12 = (popcnt64(comp11));                                         \
      int comp13 = (((load7) + (comp10)));                                     \
      int comp14 = (((load8) + (comp12)));                                     \
      int64_t comp15 = (((load15) ^ (load16)));                                \
      int64_t comp16 = (((load17) & (load18)));                                \
      int comp17 = (popcnt64(comp16));                                         \
      int64_t comp18 = (((comp15) & (comp16)));                                \
      int comp19 = (popcnt64(comp18));                                         \
      int comp20 = (((load13) + (comp17)));                                    \
      int comp21 = (((load14) + (comp19)));                                    \
      int64_t comp22 = (((load21) ^ (load22)));                                \
      int64_t comp23 = (((load23) & (load24)));                                \
      int comp24 = (popcnt64(comp23));                                         \
      int64_t comp25 = (((comp22) & (comp23)));                                \
      int comp26 = (popcnt64(comp25));                                         \
      int comp27 = (((load19) + (comp24)));                                    \
      int comp28 = (((load20) + (comp26)));                                    \
      int64_t comp29 = (((load27) ^ (load28)));                                \
      int64_t comp30 = (((load29) & (load30)));                                \
      int comp31 = (popcnt64(comp30));                                         \
      int64_t comp32 = (((comp29) & (comp30)));                                \
      int comp33 = (popcnt64(comp32));                                         \
      int comp34 = (((load25) + (comp31)));                                    \
      int comp35 = (((load26) + (comp33)));                                    \
      int64_t comp36 = (((load33) ^ (load34)));                                \
      int64_t comp37 = (((load35) & (load36)));                                \
      int comp38 = (popcnt64(comp37));                                         \
      int64_t comp39 = (((comp36) & (comp37)));                                \
      int comp40 = (popcnt64(comp39));                                         \
      int comp41 = (((load31) + (comp38)));                                    \
      int comp42 = (((load32) + (comp40)));                                    \
      int64_t comp43 = (((load39) ^ (load40)));                                \
      int64_t comp44 = (((load41) & (load42)));                                \
      int comp45 = (popcnt64(comp44));                                         \
      int64_t comp46 = (((comp43) & (comp44)));                                \
      int comp47 = (popcnt64(comp46));                                         \
      int comp48 = (((load37) + (comp45)));                                    \
      int comp49 = (((load38) + (comp47)));                                    \
      int64_t comp50 = (((load45) ^ (load46)));                                \
      int64_t comp51 = (((load47) & (load48)));                                \
      int comp52 = (popcnt64(comp51));                                         \
      int64_t comp53 = (((comp50) & (comp51)));                                \
      int comp54 = (popcnt64(comp53));                                         \
      int comp55 = (((load43) + (comp52)));                                    \
      int comp56 = (((load44) + (comp54)));                                    \
      load1 = (comp6);                                                         \
      load2 = (comp7);                                                         \
      load7 = (comp13);                                                        \
      load8 = (comp14);                                                        \
      load13 = (comp20);                                                       \
      load14 = (comp21);                                                       \
      load19 = (comp27);                                                       \
      load20 = (comp28);                                                       \
      load25 = (comp34);                                                       \
      load26 = (comp35);                                                       \
      load31 = (comp41);                                                       \
      load32 = (comp42);                                                       \
      load37 = (comp48);                                                       \
      load38 = (comp49);                                                       \
      load43 = (comp55);                                                       \
      load44 = (comp56);                                                       \
    }                                                                          \
                                                                               \
    int comp57 = (((load43) + (load37)));                                      \
    int comp58 = (((load31) + (load25)));                                      \
    int comp59 = (((load19) + (load13)));                                      \
    int comp60 = (((load7) + (load1)));                                        \
    int comp61 = (((comp60) + (comp59)));                                      \
    int comp62 = (((comp58) + (comp57)));                                      \
    int comp63 = (((comp62) + (comp61)));                                      \
    int comp64 = (((load44) + (load38)));                                      \
    int comp65 = (((load32) + (load26)));                                      \
    int comp66 = (((load20) + (load14)));                                      \
    int comp67 = (((load8) + (load2)));                                        \
    int comp68 = (((comp67) + (comp66)));                                      \
    int comp69 = (((comp65) + (comp64)));                                      \
    int comp70 = (((comp69) + (comp68)));                                      \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      int64_t load49 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);      \
      int64_t load50 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);          \
      int64_t load51 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);      \
      int64_t load52 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);          \
                                                                               \
      int64_t comp74 = (((load49) ^ (load50)));                                \
      int64_t comp75 = (((load51) & (load52)));                                \
      int comp76 = (popcnt64(comp75));                                         \
      int64_t comp77 = (((comp74) & (comp75)));                                \
      int comp78 = (popcnt64(comp77));                                         \
      int comp79 = (((comp63) + (comp76)));                                    \
      int comp80 = (((comp70) + (comp78)));                                    \
      comp63 = (comp79);                                                       \
      comp70 = (comp80);                                                       \
    }                                                                          \
                                                                               \
    int64_t comp71 = (((comp63) - (comp70)));                                  \
    int64_t comp72 = (((comp71) - (comp70)));                                  \
    float comp73 = (((((comp72) > (0))) ? (comp72) : (((comp72) * (alpha))))); \
    (output)[((((iM) * (N))) + (iN))] = (comp73);                              \
  } while (0);

#define inner_loop(activation, kernel, K, im, in, output, alpha, N)            \
  {                                                                            \
    int cntp1 = 0, cntp2 = 0;                                                  \
    size_t ik = 0;                                                             \
    for (; ik < K; ik += BITS) {                                               \
      int64_t p1 = activation[im * K + ik] ^ kernel[in * K + ik];              \
      int64_t p2 = activation[im * K + ik + 1] & kernel[in * K + ik + 1];      \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const float current = cntp1 - cntp2 - cntp2;                               \
    const float value = current > 0 ? current : current * alpha;               \
    output[im * N + in] = value;                                               \
  }

namespace gemm_avx512_manual {
Tensor4D<float> gemm_avx512_manual(const Tensor7D<int64_t> &activation,
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
          inner_loop_vectorized(activation_data, kernel_data, K, (im + imb),
                                (in + inb), output_data, alpha, N);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        // Use the PROCESS_BLOCKS macro for leftover N processing
        // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
        //             im + imb, in, alpha);
        inner_loop_vectorized(activation_data, kernel_data, K, (im + imb), in,
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
        inner_loop_vectorized(activation_data, kernel_data, K, im, (in + inb),
                              output_data, alpha, N);
      }
    }
    for (; in < N; in++) {
      // gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS, im,
      // in,
      //             alpha);
      inner_loop_vectorized(activation_data, kernel_data, K, im, in,
                            output_data, alpha, N);
    }
  }

  return output;
}
} // namespace gemm_avx512_manual
