#pragma once

#include "tensor.hpp"
#include "tensor_macros1.hpp"
#include <vector>

Tensor4D<float> direct_pad(const Tensor4D<float> &x, const size_t padding_h,
                           const size_t padding_w);

// Tensor equivalent:
// Tensor4D<float> direct_conv(const Tensor4D<float> &input,
//                             const Tensor4D<float> &kernel,
//                             const size_t stride_height,
//                             const size_t stride_width);
std::vector<float> direct_conv(float *x, float *w, int stride1, int stride2,
                               int N, int C, int H, int W, int KN, int KH,
                               int KW);
