#pragma once
#include "tensor.hpp"

namespace merge_im2row_ternarize_prelu {
Tensor4D<float> ternary_gemm(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel,
                             const float relu_alpha);
}
