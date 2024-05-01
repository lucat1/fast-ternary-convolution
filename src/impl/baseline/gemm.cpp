#include "impl/baseline/gemm.hpp"

#include "common.hpp"
#include "measure.hpp"

// TABGEMM: TNN
// In M-K, N-K order, Output is M-N,
// K is the absolute K, it should *BITS to get the real memory boundary
// a is activation  in MK: N * OH * OW, KH * kW * C * BITS. (This C has been
// quantized) b is weights     in NK: KN,          KH * KW * C * BITS y is conv
// result in MN: N * OH * OW, KN (the same as N, OH, OW, KN)
// TODO: Verify whether it really needs to be initialized to 0
// y: pointer to m * n ints initialized to 0
// result is stored in y
void tnn_gemm_baseline(int64_t *a, int64_t *b, int m, int n, int k, int *y) {
  measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::START);
  const int k_bits = k * BITS;

  for (int output_height = 0; output_height < m; output_height++) {
    for (int output_width = 0; output_width < n; output_width++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (int ik = 0; ik < k_bits; ik += BITS) {
        // Use H_W_B format
        int64_t p1 = a[output_height * k_bits + ik + 0] ^
                     b[output_width * k_bits + ik + 0];
        int64_t p2 = a[output_height * k_bits + ik + 1] &
                     b[output_width * k_bits + ik + 1];
        cntp1 = cntp1 + popcnt64(p2);
        cntp2 = cntp2 + popcnt64(p1 & p2);
      }
      y[output_height * n + output_width] = cntp1 - cntp2 - cntp2;
    }
  }
  measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::END);
}

// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
// y: pointer to m * n ints
// result is stored in y
void tbn_gemm_baseline(int64_t *a, int64_t *b, int m, int n, int k, int *y) {
  measure_point(MeasurementFunction::TBN_GEMM, MeasurementEvent::START);
  for (int output_height = 0; output_height < m; output_height++) {
    for (int output_width = 0; output_width < n; output_width++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (int ik = 0; ik < k; ik++) {
        // Use H_W_B format
        int64_t p1 =
            a[(output_height * k + ik) * BITS + 0] ^ b[output_width * k + ik];
        int64_t p2 = a[(output_height * k + ik) * BITS + 1];
        cntp1 = cntp1 + popcnt64(p2);
        cntp2 = cntp2 + popcnt64(p1 & p2);
      }
      y[output_height * n + output_width] = cntp1 - cntp2 - cntp2;
    }
  }
  measure_point(MeasurementFunction::TBN_GEMM, MeasurementEvent::END);
}

// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
// y: pointer to m * n ints
// result is stored in y
void btn_gemm_baseline(int64_t *a, int64_t *b, int *cnt1, int m, int n, int k,
                       int *y) {
  measure_point(MeasurementFunction::BTN_GEMM, MeasurementEvent::START);
  for (int output_height = 0; output_height < m; output_height++) {
    for (int output_width = 0; output_width < n; output_width++) {
      int cntp2 = 0;
      for (int ik = 0; ik < k; ik++) {
        // Use H_W_B format
        int64_t p1 =
            a[output_height * k + ik] ^ b[(output_width * k + ik) * BITS + 0];
        cntp2 = cntp2 + popcnt64(p1 & b[(output_width * k + ik) * BITS + 1]);
      }
      y[output_height * n + output_width] = cnt1[output_width] - cntp2 - cntp2;
    }
  }
  measure_point(MeasurementFunction::BTN_GEMM, MeasurementEvent::END);
}

// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
// y: pointer to m * n ints
// result is stored in y
void bnn_gemm_baseline(int64_t *a, int64_t *b, int m, int n, int k, int NUM,
                       int *y) {
  measure_point(MeasurementFunction::BNN_GEMM, MeasurementEvent::START);
  for (int output_height = 0; output_height < m; output_height++) {
    for (int output_width = 0; output_width < n; output_width++) {
      int cntp1 = 0;
      for (int ik = 0; ik < k; ik++) {
        // Use H_W_B format
        cntp1 = cntp1 +
                popcnt64(a[output_height * k + ik] ^ b[output_width * k + ik]);
      }
      y[output_height * n + output_width] = NUM - cntp1 - cntp1;
    }
  }
  measure_point(MeasurementFunction::BNN_GEMM, MeasurementEvent::END);
}
