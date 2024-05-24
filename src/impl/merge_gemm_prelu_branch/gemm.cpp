#include "impl/merge_gemm_prelu_branch/gemm.hpp"
#include "common.hpp"
#include <cstdint>

// Based of off baseline_nhwc.
namespace merge_gemm_prelu_branch {
// Multiply two matrices containing ternary values together (Algorithm 3).
Tensor4D<float> ternary_gemm(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel, float alpha) {
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
  assert(K == kernel.dim2 * kernel.dim3 * kernel.dim4 * kernel.dim5);

  // NOTE In the original code he initializes this to 0. Why?
  Tensor4D<float> output(batch_size, output_height, output_width, kernel_number,
                         false);

  for (size_t im = 0; im < M; im++) {
    for (size_t in = 0; in < N; in++) {
      int cntp11 = 0, cntp12 = 0, cntp13 = 0, cntp14 = 0;
      int cntp15 = 0, cntp16 = 0;
      int cntp21 = 0, cntp22 = 0, cntp23 = 0, cntp24 = 0;
      int cntp25 = 0, cntp26 = 0;
      size_t ik = 0;
      int64_t p11, p12, p13, p14, p15, p16;
      int64_t p21, p22, p23, p24, p25, p26;
      for (; ik + 5 * BITS < K; ik += 6 * BITS) {
        p11 =
            activation.get_123_4567(im, ik + 0) ^ kernel.get_1_2345(in, ik + 0);
        p21 =
            activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1);
        cntp11 += popcnt64(p21);
        cntp21 += popcnt64(p11 & p21);
        p12 = activation.get_123_4567(im, ik + BITS + 0) ^
              kernel.get_1_2345(in, ik + BITS + 0);
        p22 = activation.get_123_4567(im, ik + BITS + 1) &
              kernel.get_1_2345(in, ik + BITS + 1);
        cntp12 += popcnt64(p22);
        cntp22 += popcnt64(p12 & p22);
        p13 = activation.get_123_4567(im, ik + 2 * BITS + 0) ^
              kernel.get_1_2345(in, ik + 2 * BITS + 0);
        p23 = activation.get_123_4567(im, ik + 2 * BITS + 1) &
              kernel.get_1_2345(in, ik + 2 * BITS + 1);
        cntp13 += popcnt64(p23);
        cntp23 += popcnt64(p13 & p23);
        p14 = activation.get_123_4567(im, ik + 3 * BITS + 0) ^
              kernel.get_1_2345(in, ik + 3 * BITS + 0);
        p24 = activation.get_123_4567(im, ik + 3 * BITS + 1) &
              kernel.get_1_2345(in, ik + 3 * BITS + 1);
        cntp14 += popcnt64(p24);
        cntp24 += popcnt64(p14 & p24);
        p15 = activation.get_123_4567(im, ik + 4 * BITS + 0) ^
              kernel.get_1_2345(in, ik + 4 * BITS + 0);
        p25 = activation.get_123_4567(im, ik + 4 * BITS + 1) &
              kernel.get_1_2345(in, ik + 4 * BITS + 1);
        cntp15 += popcnt64(p25);
        cntp25 += popcnt64(p15 & p25);
        p16 = activation.get_123_4567(im, ik + 5 * BITS + 0) ^
              kernel.get_1_2345(in, ik + 5 * BITS + 0);
        p26 = activation.get_123_4567(im, ik + 5 * BITS + 1) &
              kernel.get_1_2345(in, ik + 5 * BITS + 1);
        cntp16 += popcnt64(p26);
        cntp26 += popcnt64(p16 & p26);
      }
      int cntp1 = cntp11 + cntp12 + cntp13 + cntp14 + cntp15 + cntp16;
      int cntp2 = cntp21 + cntp22 + cntp23 + cntp24 + cntp25 + cntp26;
      for (; ik < K; ik += BITS) {
        int64_t p1 =
            activation.get_123_4567(im, ik + 0) ^ kernel.get_1_2345(in, ik + 0);
        int64_t p2 =
            activation.get_123_4567(im, ik + 1) & kernel.get_1_2345(in, ik + 1);
        cntp1 += popcnt64(p2);
        cntp2 += popcnt64(p1 & p2);
      }
      const float current = cntp1 - cntp2 - cntp2;
      output.set_123_4(current > 0 ? current : current * alpha, im, in);
    }
  }

  return output;
}
} // namespace merge_gemm_prelu_branch
