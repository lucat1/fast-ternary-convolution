#include "impl/indirect/indirect.hpp"
#include "tensor_macros1.hpp"

namespace indirect {
Tensor5D<const int64_t *> indirection_buffer(const Tensor5D<int64_t> &data,
                                             const size_t kernel_h,
                                             const size_t kernel_w,
                                             const size_t stride_h,
                                             const size_t stride_w) {
  const int64_t *const data_data = data.data;
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t d_dim4 = data.dim4;
  const size_t d_dim5 = data.dim5;

  const size_t out_h = (height - kernel_h) / stride_h + 1;
  const size_t out_w = (width - kernel_w) / stride_w + 1;

  Tensor5D<const int64_t *> reshaped_data(n, out_h, out_w, kernel_h, kernel_w,
                                          false);
  const int64_t **const reshaped_data_data = reshaped_data.data;

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        for (size_t ik_h = 0; ik_h < kernel_h; ik_h++) {
          for (size_t ik_w = 0; ik_w < kernel_w; ik_w++) {
            const int64_t *ptr = tensor5d_addr(
                data_data, height, width, d_dim4, d_dim5, in,
                io_h * stride_h + ik_h, io_w * stride_w + ik_w, 0, 0);
            tensor5d_set(reshaped_data_data, out_h, out_w, kernel_h, kernel_w,
                         ptr, in, io_h, io_w, ik_h, ik_w);
          }
        }
      }
    }
  }
  return reshaped_data;
}
} // namespace indirect