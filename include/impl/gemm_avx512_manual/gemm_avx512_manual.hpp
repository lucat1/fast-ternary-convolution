#pragma once
#include "tensor.hpp"

namespace gemm_avx512_manual {
Tensor4D<float> gemm_avx512_manual(const Tensor7D<int64_t> &activation,
			     const Tensor5D<int64_t> &kernel,
			     float alpha);
}
