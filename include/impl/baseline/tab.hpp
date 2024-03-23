#ifndef _BASELINE_TAB_CONV_HPP
#define _BASELINE_TAB_CONV_HPP

#include "registry.hpp"

#include <cstddef>

namespace baseline {

void conv(registry::conv_type_t type, int *btn_cnt1, double *x,
          uint32_t padding_height, uint32_t padding_width, double *q_threshold,
          int c, int64_t *q_weights, uint32_t batch_size,
          uint32_t stride_height, uint32_t string_width, uint32_t kernel_number,
          uint32_t kernel_height, uint32_t kernel_width, float *output,
          uint32_t output_height, uint32_t output_width);


}

#endif // _BASELINE_TAB_CONV_HPP
