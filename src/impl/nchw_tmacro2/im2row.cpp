#include "impl/nchw_tmacro2/im2row.hpp"
#include "tensor_macros2.hpp"

namespace nchw_tmacro2 {
// NOTE Specialize this for ternary?

// Reshape (N, H, W, C, B) into (N, OH, OW, KH, KW, C, B) using im2row.
// Input:
//  data: data to be reshaped, in the (N, H, W, C, B) format and
//        with padding already applied
//  stride_h: stride in the height dimension
//  stride_w: stride in the width dimension
// Output:
//  reshaped_data: data reshaped into (N, OH, OW, KH, KW, C, B) using im2row
Tensor7D<int64_t> im2row(const Tensor5D<int64_t> &data, const size_t kernel_h,
                         const size_t kernel_w, const size_t stride_h,
                         const size_t stride_w) {
  const int64_t *const data_data = data.data;
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;
  const size_t bits = data.dim5;

  const size_t out_h = (height - kernel_h) / stride_h + 1;
  const size_t out_w = (width - kernel_w) / stride_w + 1;

  Tensor7D<int64_t> reshaped_data(n, out_h, out_w, kernel_h, kernel_w, channels,
                                  bits, false);
  int64_t *const reshaped_data_data = reshaped_data.data;

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        for (size_t ik_h = 0; ik_h < kernel_h; ik_h++) {
          for (size_t ik_w = 0; ik_w < kernel_w; ik_w++) {
            for (size_t ic = 0; ic < channels; ic++) {
              for (size_t ib = 0; ib < bits; ib++) {
                const int64_t current_value = tensor5d_get(
                    data_data, height, width, channels, bits, in,
                    io_h * stride_h + ik_h, io_w * stride_w + ik_w, ic, ib);
                tensor7d_set(reshaped_data_data, out_h, out_w, kernel_h,
                             kernel_w, channels, bits, current_value, in, io_h,
                             io_w, ik_h, ik_w, ic, ib);
              }
            }
          }
        }
      }
    }
  }
  return reshaped_data;
}
} // namespace nchw_tmacro2
