#pragma once
#include "tensor.hpp"

namespace best_impl_avx512 {
Tensor7D<int64_t>
t2r_avx512(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w);
}
