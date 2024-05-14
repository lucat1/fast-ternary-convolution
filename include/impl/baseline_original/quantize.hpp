#pragma once
#include <cstdint>
#include <vector>

namespace baseline_original {
std::vector<int64_t> Ternarize_NCHW_to_NHWCB(float* X, int PaddingH, int PaddingW, float* Q_Threshold, int N, int C, int H, int W);
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float* X, int PaddingH, int PaddingW, int N, int C, int H, int W);
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float* X, int PaddingH, int PaddingW, float* Q_Threshold, int N, int C, int H, int W);
std::vector<int> BTN_CNT_W2(int64_t* QW, int KN, int C, int KH, int KW);
}
