#pragma once

#include "common.hpp"

#include <cstddef>

namespace baseline_nhwc {

void conv(ConvolutionType type, int *btn_cnt1, float *input,
          uint32_t input_height, uint32_t input_width, uint32_t padding_height,
          uint32_t padding_width, float *quant_threshold, int c,
          int64_t *quant_wieghts, uint32_t batch_size, uint32_t stride_height,
          uint32_t string_width, uint32_t kernel_number, uint32_t kernel_height,
          uint32_t kernel_width, float relu_alpha, float *output);

}

