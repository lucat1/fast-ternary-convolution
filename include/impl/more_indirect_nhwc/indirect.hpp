#pragma once
#include "tensor.hpp"

namespace more_indirect_nhwc {

Tensor3D<const int64_t *> indirection_buffer(const Tensor5D<int64_t> &data,
                                             const size_t kernel_h,
                                             const size_t kernel_w,
                                             const size_t stride_h,
                                             const size_t stride_w);

}
