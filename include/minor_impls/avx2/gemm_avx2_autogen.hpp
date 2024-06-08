#pragma once
#include "tensor.hpp"

namespace avx2 {
Tensor4D<float> gemm_avx2_autogen(const Tensor7D<int64_t> &activation,
                                    const Tensor5D<int64_t> &kernel,
                                    float alpha);
}
