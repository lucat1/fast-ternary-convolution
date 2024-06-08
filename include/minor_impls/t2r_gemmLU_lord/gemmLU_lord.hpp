#pragma once
#include "tensor.hpp"

namespace t2r_gemmLU_lord {
Tensor4D<float> gemmLU_lord(const Tensor7D<int64_t> &activation,
			     const Tensor5D<int64_t> &kernel,
			     float alpha);
}
