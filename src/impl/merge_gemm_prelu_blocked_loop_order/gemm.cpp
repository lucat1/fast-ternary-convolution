#include "impl/merge_gemm_prelu_blocked_loop_order/gemm.hpp"
#include "common.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>

// Based of off baseline_nhwc.
namespace merge_gemm_prelu_blocked_loop_order {
// Multiply two matrices containing ternary values together (Algorithm 3).
Tensor4D<float> ternary_gemm(const Tensor7D<int64_t> &activation,
                             const Tensor5D<int64_t> &kernel, float alpha) {

  const size_t block_size = 32;

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

  size_t in_blk = 0;

  for (; in_blk + block_size < N + 1; in_blk += block_size) {
    size_t im_blk = 0;

    for (; im_blk + block_size < M + 1; im_blk += block_size) {
      for (size_t in = in_blk; in < in_blk + block_size; in++) {
        assert(in < N);
        for (size_t im = im_blk; im < im_blk + block_size; im++) {
          assert(in < N);
          int cntp1 = 0;
          int cntp2 = 0;
          for (size_t ik = 0; ik < K; ik += BITS) {
            int64_t p1 = activation.get_123_4567(im, ik + 0) ^
                         kernel.get_1_2345(in, ik + 0);
            int64_t p2 = activation.get_123_4567(im, ik + 1) &
                         kernel.get_1_2345(in, ik + 1);
            cntp1 += popcnt64(p2);
            cntp2 += popcnt64(p1 & p2);
          }
          const float current = cntp1 - cntp2 - cntp2;
          if (current > 0) {
            output.set_123_4(current, im, in);
          } else {
            output.set_123_4(current * alpha, im, in);
          }
        }
      }
    }

    for (size_t in = in_blk; in < in_blk + block_size; in++) {
      assert(in < N);
      for (size_t im = im_blk; im < M; im++) {
        int cntp1 = 0;
        int cntp2 = 0;
        for (size_t ik = 0; ik < K; ik += BITS) {
          int64_t p1 = activation.get_123_4567(im, ik + 0) ^
                       kernel.get_1_2345(in, ik + 0);
          int64_t p2 = activation.get_123_4567(im, ik + 1) &
                       kernel.get_1_2345(in, ik + 1);
          cntp1 += popcnt64(p2);
          cntp2 += popcnt64(p1 & p2);
        }
        const float current = cntp1 - cntp2 - cntp2;
        if (current > 0) {
          output.set_123_4(current, im, in);
        } else {
          output.set_123_4(current * alpha, im, in);
        }
      }
    }
  }

  for (; in_blk < N; in_blk++) {
    for (size_t im = 0; im < M; im++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (size_t ik = 0; ik < K; ik += BITS) {
        int64_t p1 = activation.get_123_4567(im, ik + 0) ^
                     kernel.get_1_2345(in_blk, ik + 0);
        int64_t p2 = activation.get_123_4567(im, ik + 1) &
                     kernel.get_1_2345(in_blk, ik + 1);
        cntp1 += popcnt64(p2);
        cntp2 += popcnt64(p1 & p2);
      }
      const float current = cntp1 - cntp2 - cntp2;
      if (current > 0) {
        output.set_123_4(current, im, in_blk);
      } else {
        output.set_123_4(current * alpha, im, in_blk);
      }
    }
  }

  return output;
}
} // namespace merge_gemm_prelu_blocked_loop_order