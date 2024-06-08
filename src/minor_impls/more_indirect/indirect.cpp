#include "minor_impls/more_indirect/indirect.hpp"
#include "tensor_macros1.hpp"

namespace more_indirect {

// Reshape (N, H, W, C, B) into (N, OH, OW, KH, KW, C, B) using im2row.
// Input:
//  data: data to be reshaped, in the (N, H, W, C, B) format and
//        with padding already applied
//  stride_h: stride in the height dimension
//  stride_w: stride in the width dimension
// Output:
//  reshaped_data: data reshaped into (N, OH, OW, KH, KW, C, B) using im2row
Tensor3D<const int64_t *> indirection_buffer(const Tensor5D<int64_t> &data,
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

  Tensor3D<const int64_t *> reshaped_data(n, out_h, out_w, false);
  const int64_t **const reshaped_data_data = reshaped_data.data;

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        const int64_t *ptr =
            tensor5d_addr(data_data, height, width, d_dim4, d_dim5, in,
                          io_h * stride_h, io_w * stride_w, 0, 0);
        tensor3d_set(reshaped_data_data, out_h, out_w, ptr, in, io_h, io_w);
      }
    }
  }
  return reshaped_data;
}
} // namespace more_indirect
