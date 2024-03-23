#ifndef _BASELINE_QUANTIZE_HPP
#define _BASELINE_QUANTIZE_HPP

#include <cstddef>
#include <cstdlib>
#include <vector>

std::vector<int64_t> ternarize_NCHW_to_NHWCB(float *X, int PaddingH,
                                             int PaddingW, float *Q_Threshold,
                                             int N, int C, int H, int W);
std::vector<int64_t> binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, int N, int C, int H,
                                           int W);
std::vector<int64_t> binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, float *Q_Threshold,
                                           int N, int C, int H, int W);
std::vector<int> btn_cnt_w2(int64_t *QW, int KN, int C, int KH, int KW);

#endif // _BASELINE_QUANTIZE_HPP
