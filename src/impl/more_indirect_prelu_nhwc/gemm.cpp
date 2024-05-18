#include "impl/more_indirect_prelu_nhwc/gemm.hpp"
#include "common.hpp"

namespace more_indirect_prelu_nhwc {
// Multiply two matrices containing ternary values together (Algorithm 3).
Tensor4D<float>
ternary_gemm(const Tensor3D<const int64_t *> &indirect_activation,
             const Tensor5D<int64_t> &kernel, const size_t input_width,
             const float relu_alpha) {
  // our sizes
  const size_t batch_size = indirect_activation.dim1;
  const size_t output_height = indirect_activation.dim2;
  const size_t output_width = indirect_activation.dim3;

  const size_t kernel_number = kernel.dim1;

  // We essentially reinterpret the tensors as 2D tensors and do
  // Matrix-Matrix multiplication.
  const size_t M = batch_size * output_height * output_width;
  const size_t N = kernel_number;
  const size_t KH = kernel.dim2;
  const size_t KW = kernel.dim3;
  const size_t C = kernel.dim4;

  // NOTE In the original code he initializes this to 0. Why?
  Tensor4D<float> output(batch_size, output_height, output_width, kernel_number,
                         false);

  for (size_t im = 0; im < M; im++) {
    for (size_t in = 0; in < N; in++) {
      int cntp1 = 0;
      int cntp2 = 0;
      const int64_t *no_kernel_addr = indirect_activation.get_123(im);
      for (size_t ikh = 0; ikh < KH; ikh++) {
        for (size_t ikw = 0; ikw < KW; ikw++) {
          const int64_t *base_addr =
              no_kernel_addr + ikh * (input_width * C * 2) + ikw * (C * 2);
          for (size_t ic = 0; ic < C; ic++) {
            const int64_t *bit0_addr = base_addr + ic * 2;
            size_t ik = ikh * (KW * C * 2) + ikw * (C * 2) + ic * 2;
            int64_t p1 = *bit0_addr ^ kernel.get_1_2345(in, ik + 0);
            int64_t p2 = *(bit0_addr + 1) & kernel.get_1_2345(in, ik + 1);
            cntp1 += popcnt64(p2);
            cntp2 += popcnt64(p1 & p2);
          }
        }
      }
      output.set_123_4((cntp1 - cntp2 - cntp2) * relu_alpha, im, in);
    }
  }

  return output;
}
} // namespace more_indirect_prelu_nhwc
