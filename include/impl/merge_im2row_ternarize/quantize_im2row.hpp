#pragma once
#include "tensor.hpp"

namespace merge_im2row_ternarize {
Tensor7D<int64_t> ternarize_im2row(const Tensor4D<float> &data,
                            const Tensor1D<float> &thresholds,
                            const size_t padding_h, const size_t padding_w,
                            const size_t kernel_h,
                            const size_t kernel_w, const size_t stride_h,
                            const size_t stride_w);
}
