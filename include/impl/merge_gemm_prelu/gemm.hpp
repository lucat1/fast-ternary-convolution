#pragma once
#include "tensor.hpp"
#include <cstddef>

// Based of off baseline_nhwc.
namespace merge_gemm_prelu {
Tensor4D<float> ternary_gemm(const Tensor7D<int64_t> &activation,
			     const Tensor5D<int64_t> &kernel,
			     float alpha);
}
