#pragma once
#include "tensor.hpp"

namespace data_order_nhwc {
Tensor4D<int64_t> ternary_gemm(const Tensor7D<int64_t> &activation,
                               const Tensor5D<int64_t> &kernel);
}
