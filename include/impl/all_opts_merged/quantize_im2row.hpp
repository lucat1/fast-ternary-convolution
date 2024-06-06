#pragma once
#include "tensor.hpp"
#include "tensor_macros0.hpp"

namespace all_opts_merged {
Tensor7D<int64_t>
ternarize_im2row(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w);
}