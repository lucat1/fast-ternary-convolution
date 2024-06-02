#include "impl/nhwc/gemm.hpp"
#include "common.hpp"

namespace nhwc {
// Multiply two matrices containing ternary values together (Algorithm 3).
Tensor4D<int64_t> ternary_gemm(const Tensor7D<int64_t> &activation,
                               const Tensor5D<int64_t> &kernel) {
  // our sizes
  const size_t batch_size = activation.dim1;
  const size_t output_height = activation.dim2;
  const size_t output_width = activation.dim3;

  const size_t kernel_number = kernel.dim1;

  // We essentially reinterpret the tensors as 2D tensors and do
  // Matrix-Matrix multiplication.
  const size_t M = batch_size * output_height * output_width;
  // KH * KW * C * BITS
  const size_t K =
      activation.dim4 * activation.dim5 * activation.dim6 * activation.dim7;
  const size_t N = kernel_number;
  // sanity check: K (from activation) == KH * KW * C * BITS (from weights)
  //assert(K == kernel.dim2 * kernel.dim3 * kernel.dim4 * kernel.dim5);

  // NOTE In the original code he initializes this to 0. Why?
  Tensor4D<int64_t> output(batch_size, output_height, output_width,
                           kernel_number, false);

  for (size_t im = 0; im < M; im++) {
    for (size_t in = 0; in < N; in++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (size_t ik = 0; ik < K; ik += BITS) {
        int64_t p1 =
            activation.get_123_4567(im, ik + 0) ^ kernel.get_1_2345(in, ik + 0);
        int64_t p2 =
            activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1);
        cntp1 += popcnt64(p2);
        cntp2 += popcnt64(p1 & p2);
      }
      output.set_123_4(cntp1 - cntp2 - cntp2, im, in);
    }
  }

  return output;
}
} // namespace nhwc
