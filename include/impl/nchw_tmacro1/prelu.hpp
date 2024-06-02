#pragma once
#include "tensor.hpp"

namespace nchw_tmacro1 {
Tensor4D<float> prelu(Tensor4D<int64_t>& pre_activation, float alpha);
}
