#ifndef _BASELINE_GEMM_HPP
#define _BASELINE_GEMM_HPP

#include <cstddef>
#include <cstdlib>
#include <vector>

// TABGEMM: TNN
// In M-K, N-K order, M-N,
// K is the absolute K, it should *BITS to get the real memory boundary
std::vector<int> tnn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K);

// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> tbn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K);

// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> btn_gemm_baseline(int64_t *a, int64_t *b, int *cnt1, int M,
                                   int N, int K);

// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> bnn_gemm_baseline(int64_t *a, int64_t *b, int M, int N, int K,
                                   int NUM);

#endif // _BASELINE_GEMM_HPP
