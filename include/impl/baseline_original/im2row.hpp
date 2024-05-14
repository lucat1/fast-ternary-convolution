#pragma once
#include <vector>

namespace baseline_original {
template <typename T>
std::vector<T> Img2Row_NHWCB_to_N_OHOW_KHKWC(T* X, int N, int C, int H, int W, int KH, int KW, int StrideH, int StrideW) {

    const int OH = (H - KH) / StrideH + 1;
    const int OW = (W - KW) / StrideW + 1;
    const int H1 = OH * OW;      // Fused Height
    const int W1 = KH * KW * C;  // Fused Width
    std::vector<T> y = std::vector<T>(N * H1 * W1);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int c = 0; c < C; c++)
                            // y[N, OH, OW, KH, KW, C] = X[N, H+kh, W+kw, C]
                            y[(n * H1 + oh * OW + ow) * W1 + kh * KW * C + kw * C + c] = X[((n * H + oh * StrideH + kh) * W + ow * StrideW + kw) * C + c];
                    }
                }
            }
        }
    }

    return y;
}
}
