#pragma once
#include "tensor.hpp"

Tensor4D<float> prelu(Tensor4D<int64_t>& pre_activation, float alpha);
