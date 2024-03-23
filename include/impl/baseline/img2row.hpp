#pragma once
#include "common.h"

template <typename T>
std::vector<T> img2row_NHWCB_to_N_OHOW_KHKWC(T* input, int batch_size, int num_channels, int input_height, int input_width, int kernel_height, int kernel_width, int stride_height, int stride_width) {

    const int output_height = (input_height - kernel_height + 1) / stride_height;
    const int output_width = (input_width - kernel_width + 1) / stride_width;
    const int fused_height = output_height * output_width;      // Fused Height
    const int fused_width = kernel_height * kernel_width * num_channels;  // Fused Width
    std::vector<T> y = std::vector<T>(batch_size * H1 * W1);

    for (int n = 0; n < batch_size; n++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int c = 0; c < num_channels; c++)
                            // y[N, OH, OW, KH, KW, C] = X[N, H+kh, W+kw, C]
                            y[(n * fused_height + oh * output_width + ow) * fused_width + kh * kernel_width * num_channels + kw * num_channels + c] = input[((n * input_height + oh * stride_height + kh) * input_width + ow * stride_width + kw) * num_channels + c];
                    }
                }
            }
        }
    }

    return y;
}