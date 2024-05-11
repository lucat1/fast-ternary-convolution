#pragma once
#include "tensor.hpp"


Tensor2D<float> prelu(Tensor2D<int64_t>& pre_activation, float alpha);
