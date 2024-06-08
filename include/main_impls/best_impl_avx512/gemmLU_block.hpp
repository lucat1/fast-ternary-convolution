#pragma once
#include "tensor.hpp"

namespace best_impl_avx512 {
Tensor4D<float> gemmLU_block(const Tensor7D<int64_t> &activation,
			     const Tensor5D<int64_t> &kernel,
			     float alpha);
}
