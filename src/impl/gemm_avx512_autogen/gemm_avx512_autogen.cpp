#include "impl/gemm_avx512_autogen/gemm_avx512_autogen.hpp"
#include "common.hpp"
#include <immintrin.h>


#define gemm_kernel_512(activation, kernel, K, iM, iN, output,alpha,N) \
  do {                                                                         \
    int64_t comp28;                                                            \
    __m512i comp17;                                                            \
    int comp50;                                                                \
    __m512i comp24;                                                            \
    __m512i comp1;                                                             \
    __m512i comp10;                                                            \
    int64_t load27;                                                            \
    __m512i comp7;                                                             \
    int comp26;                                                                \
    int64_t comp27;                                                            \
    int comp36;                                                                \
    int comp38;                                                                \
    __m512i load1;                                                             \
    __m512i comp14;                                                            \
    __m512i comp2;                                                             \
    float comp47;                                                              \
    int comp54;                                                                \
    __m512i comp4;                                                             \
    __m512i comp13;                                                            \
    int64_t comp51;                                                            \
    __m512i comp20;                                                            \
    int comp40;                                                                \
    __m512i comp12;                                                            \
    __m512i comp22;                                                            \
    __m512i load11;                                                            \
    __m512i load4;                                                             \
    int64_t load18;                                                            \
    int load14;                                                                \
    __m512i comp3;                                                             \
    int64_t load17;                                                            \
    __m512i comp11;                                                            \
    __m512i load9;                                                             \
    __m512i comp8;                                                             \
    __m512i load10;                                                            \
    int64_t comp49;                                                            \
    __m512i comp16;                                                            \
    __m512i load12;                                                            \
    size_t iK;                                                                 \
    int64_t comp35;                                                            \
    int comp43;                                                                \
    int64_t load25;                                                            \
    int comp33;                                                                \
    int load13;                                                                \
    __m512i comp18;                                                            \
    int64_t comp46;                                                            \
    __m512i comp6;                                                             \
    int comp31;                                                                \
    __m512i comp21;                                                            \
    __m512i load7;                                                             \
    int comp29;                                                                \
    __m512i comp5;                                                             \
    int comp53;                                                                \
    int64_t load26;                                                            \
    int64_t load28;                                                            \
    __m512i comp9;                                                             \
    __m512i comp23;                                                            \
    int64_t comp45;                                                            \
    __m512i load2;                                                             \
    __m512i load5;                                                             \
    __m512i load3;                                                             \
    int64_t load16;                                                            \
    int64_t comp48;                                                            \
    __m512i comp19;                                                            \
    int comp25;                                                                \
    int comp41;                                                                \
    int64_t load15;                                                            \
    int comp52;                                                                \
    int64_t comp34;                                                            \
    int64_t load23;                                                            \
    int64_t load24;                                                            \
    int comp32;                                                                \
    int64_t comp37;                                                            \
    int load20;                                                                \
    int comp42;                                                                \
    int64_t comp30;                                                            \
    int64_t load21;                                                            \
    int comp44;                                                                \
    __m512i load6;                                                             \
    __m512i load8;                                                             \
    int comp39;                                                                \
    int load19;                                                                \
    int64_t load22;                                                            \
    __m512i comp15;                                                            \
                                                                               \
    load1 = (_mm512_setzero_si512());                                          \
    load2 = (_mm512_setzero_si512());                                          \
    load7 = (_mm512_setzero_si512());                                          \
    load8 = (_mm512_setzero_si512());                                          \
    load13 = (0);                                                              \
    load14 = (0);                                                              \
    load19 = (0);                                                              \
    load20 = (0);                                                              \
    iK = (0);                                                                  \
    for (; (((int)(iK))) <=                                                    \
           (((((int)(K))) - (((int)(((2) * (((8) * (BITS)))))))));             \
         iK += ((2) * (((8) * (BITS))))) {                                     \
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
      load9 = (_mm512_loadu_si512(                                             \
          (__m512i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((8) * (BITS))))))) + (0))))))));  \
      load10 = (_mm512_loadu_si512(                                            \
          (__m512i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((8) * (BITS))))))) + (0))))))));  \
      load11 = (_mm512_loadu_si512(                                            \
          (__m512i *)(activation +                                             \
                      (((((iM) * (K))) +                                       \
                        (((((iK) + (((1) * (((8) * (BITS))))))) + (8))))))));  \
      load12 = (_mm512_loadu_si512(                                            \
          (__m512i *)(kernel +                                                 \
                      (((((iN) * (K))) +                                       \
                        (((((iK) + (((1) * (((8) * (BITS))))))) + (8))))))));  \
                                                                               \
      comp1 = (_mm512_unpacklo_epi64(load3, load5));                           \
      comp2 = (_mm512_unpackhi_epi64(load3, load5));                           \
      comp3 = (_mm512_unpacklo_epi64(load4, load6));                           \
      comp4 = (_mm512_unpackhi_epi64(load4, load6));                           \
      comp5 = (_mm512_xor_epi64(comp1, comp3));                                \
      comp6 = (_mm512_and_epi64(comp2, comp4));                                \
      comp8 = (_mm512_popcnt_epi64(comp6));                                    \
      comp7 = (_mm512_and_epi64(comp5, comp6));                                \
      comp9 = (_mm512_popcnt_epi64(comp7));                                    \
      comp10 = (_mm512_add_epi64(load1, comp8));                               \
      comp11 = (_mm512_add_epi64(load2, comp9));                               \
      comp12 = (_mm512_unpacklo_epi64(load9, load11));                         \
      comp13 = (_mm512_unpackhi_epi64(load9, load11));                         \
      comp14 = (_mm512_unpacklo_epi64(load10, load12));                        \
      comp15 = (_mm512_unpackhi_epi64(load10, load12));                        \
      comp16 = (_mm512_xor_epi64(comp12, comp14));                             \
      comp17 = (_mm512_and_epi64(comp13, comp15));                             \
      comp19 = (_mm512_popcnt_epi64(comp17));                                  \
      comp18 = (_mm512_and_epi64(comp16, comp17));                             \
      comp20 = (_mm512_popcnt_epi64(comp18));                                  \
      comp21 = (_mm512_add_epi64(load7, comp19));                              \
      comp22 = (_mm512_add_epi64(load8, comp20));                              \
      load1 = (comp10);                                                        \
      load2 = (comp11);                                                        \
      load7 = (comp21);                                                        \
      load8 = (comp22);                                                        \
    }                                                                          \
    for (; (((int)(iK))) < (((((int)(K))) - (((int)(((2) * (BITS)))))));       \
         iK += ((2) * (BITS))) {                                               \
                                                                               \
      load15 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (0))))]);        \
      load16 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (0))))]);  \
      load17 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((0) * (BITS))))) + (1))))]);        \
      load18 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((0) * (BITS))))) + (1))))]);  \
      load21 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (0))))]);        \
      load22 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (0))))]);  \
      load23 = ((activation)[((((iM) * (K))) +                                 \
                              (((((iK) + (((1) * (BITS))))) + (1))))]);        \
      load24 = ((                                                              \
          kernel)[((((iN) * (K))) + (((((iK) + (((1) * (BITS))))) + (1))))]);  \
                                                                               \
      comp27 = (((load15) ^ (load16)));                                        \
      comp28 = (((load17) & (load18)));                                        \
      comp29 = (popcnt64(comp28));                                             \
      comp30 = (((comp27) & (comp28)));                                        \
      comp31 = (popcnt64(comp30));                                             \
      comp32 = (((load13) + (comp29)));                                        \
      comp33 = (((load14) + (comp31)));                                        \
      comp34 = (((load21) ^ (load22)));                                        \
      comp35 = (((load23) & (load24)));                                        \
      comp36 = (popcnt64(comp35));                                             \
      comp37 = (((comp34) & (comp35)));                                        \
      comp38 = (popcnt64(comp37));                                             \
      comp39 = (((load19) + (comp36)));                                        \
      comp40 = (((load20) + (comp38)));                                        \
      load13 = (comp32);                                                       \
      load14 = (comp33);                                                       \
      load19 = (comp39);                                                       \
      load20 = (comp40);                                                       \
    }                                                                          \
                                                                               \
    comp23 = (_mm512_add_epi64(load7, load1));                                 \
    comp24 = (_mm512_add_epi64(load8, load2));                                 \
    comp41 = (((load19) + (load13)));                                          \
    comp42 = (((load20) + (load14)));                                          \
    comp25 = (_mm512_reduce_add_epi64(comp23));                                \
    comp26 = (_mm512_reduce_add_epi64(comp24));                                \
    comp43 = (((comp25) + (comp41)));                                          \
    comp44 = (((comp26) + (comp42)));                                          \
    for (; (iK) < (K); iK += BITS) {                                           \
                                                                               \
      load25 = ((activation)[((((iM) * (K))) + (((iK) + (0))))]);              \
      load26 = ((kernel)[((((iN) * (K))) + (((iK) + (0))))]);                  \
      load27 = ((activation)[((((iM) * (K))) + (((iK) + (1))))]);              \
      load28 = ((kernel)[((((iN) * (K))) + (((iK) + (1))))]);                  \
                                                                               \
      comp48 = (((load25) ^ (load26)));                                        \
      comp49 = (((load27) & (load28)));                                        \
      comp50 = (popcnt64(comp49));                                             \
      comp51 = (((comp48) & (comp49)));                                        \
      comp52 = (popcnt64(comp51));                                             \
      comp53 = (((comp43) + (comp50)));                                        \
      comp54 = (((comp44) + (comp52)));                                        \
      comp43 = (comp53);                                                       \
      comp44 = (comp54);                                                       \
    }                                                                          \
                                                                               \
    comp45 = (((comp43) - (comp44)));                                          \
    comp46 = (((comp45) - (comp44)));                                          \
    comp47 = (((((comp46) > (0))) ? (comp46) : (((comp46) * (alpha)))));       \
    (output)[((((iM) * (N))) + (iN))] = (comp47);                              \
  } while (0);                                                                 \
  // vi: ft=c

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

namespace gemm_avx512_autogen {
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
} // namespace gemm_avx512_autogen
