#pragma once
#include "tensor.hpp"

namespace indirect {

Tensor4D<int64_t>
indirect_gemm(const Tensor5D<const int64_t *> &indirect_activation,
             const Tensor5D<int64_t> &kernel);
} // namespace indirect_nhwc
