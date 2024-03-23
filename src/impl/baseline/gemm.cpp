#include "common.h"
#include "GEMM.h"

// TABGEMM: TNN
// In M-K, N-K order, Output is M-N, 
// K is the absolute K, it should *BITS to get the real memory boundary
// a is activation  in MK: N * OH * OW, KH * kW * C * BITS. (This C has been quantized)
// b is weights     in NK: KN,          KH * KW * C * BITS 
// y is conv result in MN: N * OH * OW, KN (the same as N, OH, OW, KN)
std::vector<int> tnn_gemm_baseline(int64_t* a, int64_t* b, int m, int n, int k) {
    std::vector<int> y = std::vector<int>(m * n, 0);
    const int k_bits = k * BITS;
    for (int output_height = 0; output_height < m; output_height++) {
        for (int output_width = 0; output_width < n; output_width++) {
            int cntp1 = 0;
            int cntp2 = 0;
            for (int ik = 0; ik < k_bits; ik += BITS) {
                // Use H_W_B format
                int64_t p1 = a[output_height * k_bits + ik + 0] ^ b[output_width * k_bits + ik + 0];
                int64_t p2 = a[output_height * k_bits + ik + 1] & b[output_width * k_bits + ik + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[output_height * n + output_width] = cntp1 - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> tbn_gemm_baseline(int64_t* a, int64_t* b, int m, int n, int k) {
    std::vector<int> y = std::vector<int>(m * n);
    for (int output_height = 0; output_height < m; output_height++) {
        for (int output_width = 0; output_width < n; output_width++) {
            int cntp1 = 0;
            int cntp2 = 0;
            for (int ik = 0; ik < k; ik++) {
                // Use H_W_B format
                int64_t p1 = a[(output_height * K + ik) * BITS + 0] ^ b[output_width * K + ik];
                int64_t p2 = a[(output_height * K + ik) * BITS + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[output_height * n + output_width] = cntp1 - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> btn_gemm_baseline(int64_t* a, int64_t* b, int* cnt1, int m, int n, int k) {
    std::vector<int> y = std::vector<int>(m * n);
    for (int output_height = 0; output_height < m; output_height++) {
        for (int output_width = 0; output_width < n; output_width++) {
            int cntp2 = 0;
            for (int ik = 0; ik < k; ik++) {
                // Use H_W_B format
                int64_t p1 = a[output_height * k + ik] ^ b[(output_width * k + ik) * BITS + 0];
                cntp2 = cntp2 + popcnt64(p1 & b[(output_width * k + ik) * BITS + 1]);
            }
            y[output_height * n + output_width] = cnt1[output_width] - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> bnn_gemm_baseline(int64_t* a, int64_t* b, int m, int n, int k, int NUM) {
    std::vector<int> y = std::vector<int>(m * n);
    for (int output_height = 0; output_height < m; output_height++) {
        for (int output_width = 0; output_width < n; output_width++) {
            int cntp1 = 0;
            for (int ik = 0; ik < k; ik++) {
                // Use H_W_B format
                cntp1 = cntp1 + popcnt64(a[output_height * K + ik] ^ b[output_width * k + ik]);
            }
            y[output_height * n + output_width] = NUM - cntp1 - cntp1;
        }
    }
    return y;
}

