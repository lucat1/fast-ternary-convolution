#pragma once
#include "tensor.hpp"

namespace nhwc_tmacro2 {
  Tensor4D<float> prelu(Tensor4D<int64_t> &pre_activation, float alpha);
}
