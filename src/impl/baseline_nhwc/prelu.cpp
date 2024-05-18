#include "impl/baseline_nhwc/prelu.hpp"
#include "tensor.hpp"

namespace baseline_nhwc {
// NOTE We can probably merge this with GEMM
Tensor4D<float> prelu(Tensor4D<int64_t> &pre_activation, float alpha) {
  // our sizes
  const size_t n = pre_activation.dim1;
  const size_t output_h = pre_activation.dim2;
  const size_t output_w = pre_activation.dim3;
  const size_t kernel_number = pre_activation.dim4;

  Tensor4D<float> post_activation(n, output_h, output_w, kernel_number, false);

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < output_h; io_h++) {
      for (size_t io_w = 0; io_w < output_w; io_w++) {
        for (size_t ik_n = 0; ik_n < kernel_number; ik_n++) {
          const float current = pre_activation.get(in, io_h, io_w, ik_n);
          // NOTE If we cannot merge this with GEMM, make sure to apply scalar
          // replacement
          //  here.
          if (current > 0) {
            post_activation.set(current, in, io_h, io_w, ik_n);
          } else {
            post_activation.set(current * alpha, in, io_h, io_w, ik_n);
          }
        }
      }
    }
  }

  return post_activation;
}
} // namespace baseline_nhwc
