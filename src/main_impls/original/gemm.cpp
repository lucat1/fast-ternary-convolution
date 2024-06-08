#include "main_impls/original/gemm.hpp"
#include "common.hpp"

namespace original {
// TABGEMM: TNN
// In M-K, N-K order, Output is M-N,
// K is the absolute K, it should *BITS to get the real memory boundary
// a is activation  in MK: N * OH * OW, KH * kW * C * BITS. (This C has been
// quantized) b is weights     in NK: KN,          KH * KW * C * BITS y is conv
// result in MN: N * OH * OW, KN (the same as N, OH, OW, KN)
std::vector<int> TNNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K) {
  std::vector<int> y = std::vector<int>(M * N, 0);
  const int KB = K * BITS;
  for (int oh = 0; oh < M; oh++) {
    for (int ow = 0; ow < N; ow++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (int ik = 0; ik < KB; ik += BITS) {
        // Use H_W_B format
        int64_t p1 = a[oh * KB + ik + 0] ^ b[ow * KB + ik + 0];
        int64_t p2 = a[oh * KB + ik + 1] & b[ow * KB + ik + 1];
        cntp1 = cntp1 + popcnt64(p2);
        cntp2 = cntp2 + popcnt64(p1 & p2);
      }
      y[oh * N + ow] = cntp1 - cntp2 - cntp2;
    }
  }
  return y;
}

// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> TBNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K) {
  std::vector<int> y = std::vector<int>(M * N);
  for (int oh = 0; oh < M; oh++) {
    for (int ow = 0; ow < N; ow++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (int ik = 0; ik < K; ik++) {
        // Use H_W_B format
        int64_t p1 = a[(oh * K + ik) * BITS + 0] ^ b[ow * K + ik];
        int64_t p2 = a[(oh * K + ik) * BITS + 1];
        cntp1 = cntp1 + popcnt64(p2);
        cntp2 = cntp2 + popcnt64(p1 & p2);
      }
      y[oh * N + ow] = cntp1 - cntp2 - cntp2;
    }
  }
  return y;
}

// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> BTNGEMM_baseline(int64_t *a, int64_t *b, int *cnt1, int M,
                                  int N, int K) {
  std::vector<int> y = std::vector<int>(M * N);
  for (int oh = 0; oh < M; oh++) {
    for (int ow = 0; ow < N; ow++) {
      int cntp2 = 0;
      for (int ik = 0; ik < K; ik++) {
        // Use H_W_B format
        int64_t p1 = a[oh * K + ik] ^ b[(ow * K + ik) * BITS + 0];
        cntp2 = cntp2 + popcnt64(p1 & b[(ow * K + ik) * BITS + 1]);
      }
      y[oh * N + ow] = cnt1[ow] - cntp2 - cntp2;
    }
  }
  return y;
}

// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> BNNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K,
                                  int NUM) {
  std::vector<int> y = std::vector<int>(M * N);
  for (int oh = 0; oh < M; oh++) {
    for (int ow = 0; ow < N; ow++) {
      int cntp1 = 0;
      for (int ik = 0; ik < K; ik++) {
        // Use H_W_B format
        cntp1 = cntp1 + popcnt64(a[oh * K + ik] ^ b[ow * K + ik]);
      }
      y[oh * N + ow] = NUM - cntp1 - cntp1;
    }
  }
  return y;
}
} // namespace original
