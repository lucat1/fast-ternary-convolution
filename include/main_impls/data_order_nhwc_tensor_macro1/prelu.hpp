#pragma once
#include "tensor.hpp"

namespace data_order_nhwc_tensor_macro1 {
  Tensor4D<float> prelu(Tensor4D<int64_t> &pre_activation, float alpha);
}
