#include "main_impls/data_order_nhwc/im2row.hpp"

namespace data_order_nhwc {
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
  // sizes for our data
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;
  const size_t bits = data.dim5;

  // We essentially implement im2row for one H x W x C image, which we can then
  // simply extend to account for the extra N and B dimensions.

  // Check out
  // https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer
  // for more information on how the output size is calculated. Note that we
  // assume that data is already padded, hence we do not account for it here
  // anymore. NOTE Possible improvement: Do not copy padding; instead use calloc
  // and only copy
  //   relevant data.
  const size_t out_h = (height - kernel_h) / stride_h + 1;
  const size_t out_w = (width - kernel_w) / stride_w + 1;

  // Checkout
  // https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
  // for more information on the idea of im2col. im2row should follow directly
  // from that.
  Tensor7D<int64_t> reshaped_data(n, out_h, out_w, kernel_h, kernel_w, channels,
                                  bits, false);

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        for (size_t ik_h = 0; ik_h < kernel_h; ik_h++) {
          for (size_t ik_w = 0; ik_w < kernel_w; ik_w++) {
            for (size_t ic = 0; ic < channels; ic++) {
              for (size_t ib = 0; ib < bits; ib++) {
                const int64_t current_value = data.get(
                    in, io_h * stride_h + ik_h, io_w * stride_w + ik_w, ic, ib);
                reshaped_data.set(current_value, in, io_h, io_w, ik_h, ik_w, ic,
                                  ib);
              }
            }
          }
        }
      }
    }
  }
  return reshaped_data;
}
} // namespace data_order_nhwc
