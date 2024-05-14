#pragma once
#include "tensor.hpp"

namespace baseline_nhwc {
Tensor7D<int64_t> im2row(const Tensor5D<int64_t> &data, const size_t kernel_h,
                         const size_t kernel_w, const size_t stride_h,
                         const size_t stride_w);
}
