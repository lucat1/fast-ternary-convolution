#pragma once
#include "tensor.hpp"

namespace tern2row_memcpy {
Tensor7D<int64_t>
tern2row_memcpy(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w);
}
