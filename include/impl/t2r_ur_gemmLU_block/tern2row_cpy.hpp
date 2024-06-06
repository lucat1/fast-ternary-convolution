#pragma once
#include "tensor.hpp"

namespace t2r_ur_gemmLU_block {
Tensor7D<int64_t>
tern2row_cpy(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w);
}
