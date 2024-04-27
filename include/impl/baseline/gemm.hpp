#ifndef _BASELINE_GEMM_HPP
#define _BASELINE_GEMM_HPP

#include <cstddef>
#include <cstdlib>

// TABGEMM: TNN
// In M-K, N-K order, M-N,
// K is the absolute K, it should *BITS to get the real memory boundary
void tnn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K, int *y);

// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
void tbn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K, int *y);

// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
void btn_gemm_baseline(int64_t *a, int64_t *b, int *cnt1, int M, int N, int K, int *y);

// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
void bnn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K, int NUM, int *y);

#endif // _BASELINE_GEMM_HPP
