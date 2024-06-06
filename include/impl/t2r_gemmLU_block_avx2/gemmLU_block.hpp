#pragma once
#include "tensor.hpp"

namespace t2r_gemmLU_block_avx2 {
Tensor4D<float> gemmLU_block(const Tensor7D<int64_t> &activation,
			     const Tensor5D<int64_t> &kernel,
			     float alpha);
}
