#pragma once
#include "tensor.hpp"

namespace data_order_nhwc_tensor_macro1 {
Tensor5D<int64_t> ternarize(const Tensor4D<float> &data,
                            const Tensor1D<float> &thresholds,
				   const size_t padding_h, const size_t padding_w);
}
