#ifndef _BASELINE_HPP
#define _BASELINE_HPP

#include "registry.hpp"

#include <cstddef>

namespace baseline {

float *tab_conv(registry::conv_type_t type, int *btn_cnt1, double *x,
                int padding_h, int padding_w, double *q_threshold, int c, int h,
                int w, int64_t *q_weights, int batch_size, int stride_h,
                int string_w, int kn, int kh, int kw);

}

#endif // _BASELINE_HPP
