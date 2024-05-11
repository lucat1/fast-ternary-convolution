#pragma once
#include "tensor.hpp"

Tensor2D<int64_t> ternary_gemm(Tensor2D<int64_t>& activation, Tensor2D<int64_t>& weights);
