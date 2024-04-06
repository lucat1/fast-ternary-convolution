#ifndef _BASELINE_QUANTIZE_HPP
#define _BASELINE_QUANTIZE_HPP

#include <cstddef>
#include <cstdlib>

void ternarize_NCHW_to_NHWCB(float *X, int PaddingH,
                                             int PaddingW, float *Q_Threshold,
                                             int N, int C, int H, int W, int64_t *qx);
void binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, int N, int C, int H,
                                           int W, int64_t *qx);
void binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, float *Q_Threshold,
                                           int N, int C, int H, int W, int64_t *qx);
void btn_cnt_w2(int64_t *QW, int KN, int C, int KH, int KW, int *y);

#endif // _BASELINE_QUANTIZE_HPP
