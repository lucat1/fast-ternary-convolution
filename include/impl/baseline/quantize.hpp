#ifndef _BASELINE_QUANTIZE_HPP
#define _BASELINE_QUANTIZE_HPP

#include <cstddef>
#include <cstdlib>

void ternarize_NCHW_to_NHWCB(float *input, size_t padding_height,
                             size_t padding_width, float *quant_threshold,
                             size_t batch_size, size_t C, size_t input_height,
                             size_t input_width, int64_t *qx);
void binarize_NCHW_to_NHWC(const float *input, size_t padding_height,
                           size_t padding_width, float *quant_threshold,
                           size_t batch_size, size_t C, size_t input_height,
                           size_t input_width, int64_t *qx);
void binarize_NCHW_to_NHWC(const float *input, size_t padding_height,
                           size_t padding_width, size_t batch_size,
                           size_t num_channels, size_t input_height,
                           size_t input_width, int64_t *qx);
void btn_cnt_w2(int64_t *quantized_weights, size_t num_channels,
                size_t kernel_number, size_t kernel_height, size_t kernel_width,
                int *y);

#endif // _BASELINE_QUANTIZE_HPP
