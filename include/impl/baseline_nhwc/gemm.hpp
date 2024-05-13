#pragma once
#include "tensor.hpp"

Tensor4D<int64_t> ternary_gemm(Tensor7D<int64_t>& activation, Tensor5D<int64_t>& weights);
