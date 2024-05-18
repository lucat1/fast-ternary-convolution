#pragma once
#include "tensor.hpp"

namespace more_indirect_nhwc {

Tensor4D<int64_t>
ternary_gemm(const Tensor3D<const int64_t *> &indirect_activation,
             const Tensor5D<int64_t> &kernel, const size_t input_width);
} // namespace more_indirect_nhwc
