#ifndef _BASELINE_QUANTIZE_HPP
#define _BASELINE_QUANTIZE_HPP

#include <cstddef>
#include <cstdlib>

void ternarize_NCHW_to_NHWCB(float *input, int padding_height,
                             int padding_width, float *quant_threshold,
                             int batch_size, int C, int input_height,
                             int input_width, int64_t *qx);
void Binarize_NCHW_to_NHWC(const float *input, int padding_height,
                           int padding_width, float *quant_threshold,
                           int batch_size, int C, int input_height,
                           int input_width, int64_t *qx);
void binarize_NCHW_to_NHWC(const float *input, int padding_height,
                           int padding_width, int batch_size, int num_channels,
                           int input_height, int input_width, int64_t *qx);
void btn_cnt_w2(int64_t *quantized_weights, int num_channels, int kernel_number,
                int kernel_height, int kernel_width, int *y);

#endif // _BASELINE_QUANTIZE_HPP
