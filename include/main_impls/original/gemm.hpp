#pragma once
#include <cstdint>
#include <vector>

namespace original {
// TABGEMM: TNN
// In M-K, N-K order, M-N,
// K is the absolute K, it should *BITS to get the real memory boundary
std::vector<int> TNNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K);

// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> TBNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K);

// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> BTNGEMM_baseline(int64_t *a, int64_t *b, int *cnt1, int M,
                                  int N, int K);

// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> BNNGEMM_baseline(int64_t *a, int64_t *b, int M, int N, int K,
                                  int NUM);
}
