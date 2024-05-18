#pragma once
#include "tensor.hpp"

namespace indirect_nhwc {

Tensor5D<const int64_t *> indirection_buffer(const Tensor5D<int64_t> &data,
                                             const size_t kernel_h,
                                             const size_t kernel_w,
                                             const size_t stride_h,
                                             const size_t stride_w);

}
