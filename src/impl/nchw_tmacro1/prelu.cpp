#include "impl/nchw_tmacro1/prelu.hpp"
#include "tensor_macros1.hpp"

namespace nchw_tmacro1 {
// NOTE We can probably merge this with GEMM
Tensor4D<float> prelu(Tensor4D<int64_t> &pre_activation, float alpha) {
  const int64_t *const pre_activation_data = pre_activation.data;
  const size_t n = pre_activation.dim1;
  const size_t output_h = pre_activation.dim2;
  const size_t output_w = pre_activation.dim3;
  const size_t kernel_number = pre_activation.dim4;

  Tensor4D<float> post_activation(n, output_h, output_w, kernel_number, false);
  float *const post_activation_data = post_activation.data;

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < output_h; io_h++) {
      for (size_t io_w = 0; io_w < output_w; io_w++) {
        for (size_t ik_n = 0; ik_n < kernel_number; ik_n++) {
          const float current =
              tensor4d_get(pre_activation_data, output_h, output_w,
                           kernel_number, in, io_h, io_w, ik_n);
          if (current > 0) {
            tensor4d_set(post_activation_data, output_h, output_w,
                         kernel_number, current, in, io_h, io_w, ik_n);
          } else {
            tensor4d_set(post_activation_data, output_h, output_w,
                         kernel_number, current * alpha, in, io_h, io_w, ik_n);
          }
        }
      }
    }
  }

  return post_activation;
}
} // namespace nchw_tmacro1
