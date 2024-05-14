#pragma once

#include "tensor.hpp"
#include <vector>

Tensor4D<float> reshape_nhwc_nchw(const Tensor4D<float> &src);
Tensor4D<float> direct_pad(const Tensor4D<float> &x, const size_t padding_h,
                           const size_t padding_w);
// Tensor4D<float> direct_conv(const Tensor4D<float> &input,
//                             const Tensor4D<float> &kernel,
//                             const size_t stride_height,
//                             const size_t stride_width);
std::vector<float> direct_conv(float *x, float *w, size_t stride_height,
                               size_t stride_width, size_t N, size_t C,
                               size_t H, size_t W, size_t KN, size_t KH,
                               size_t KW);
