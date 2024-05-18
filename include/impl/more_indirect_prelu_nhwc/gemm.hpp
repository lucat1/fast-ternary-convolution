#pragma once
#include "tensor.hpp"

namespace more_indirect_prelu_nhwc {

Tensor4D<float>
ternary_gemm(const Tensor3D<const int64_t *> &indirect_activation,
             const Tensor5D<int64_t> &kernel, const size_t input_width,
             const float relu_alpha);
} // namespace more_indirect_prelu_nhwc
