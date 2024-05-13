#pragma once

#include "tensor.hpp"

Tensor4D<float> reshape_nchw_nhwc(const Tensor4D<float> &src);
Tensor4D<float> direct_pad(const Tensor4D<float> &x, const size_t padding_h,
                           const size_t padding_w);
Tensor4D<float> direct_conv(const Tensor4D<float> &input,
                            const Tensor4D<float> &kernel,
                            const size_t stride_height,
                            const size_t stride_width);
