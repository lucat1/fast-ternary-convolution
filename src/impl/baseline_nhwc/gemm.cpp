#include "impl/baseline_nhwc/gemm.hpp"
#include "common.hpp"

// Multiply two matrices containing ternary values together (Algorithm 3).
// Input:
//  activation: matrix of shape (M, K)
//  weights: matrix of shape (N, K)
// Output:
//  output: output feature map with shape (M, N)
Tensor2D<int64_t> ternary_gemm(Tensor2D<int64_t>& activation, Tensor2D<int64_t>& weights) {
  // our data sizes
  const size_t M = activation.dim1;
  const size_t K = activation.dim2;
  const size_t N = weights.dim1;
  assert(K == weights.dim2);

  // NOTE In the original code he initializes this to 0. Why?
  Tensor2D<int64_t> output (M, N, false);

  for (size_t im = 0; im < M; im++) {
    for (size_t in = 0; in < N; in++) {
      int cntp1 = 0;
      int cntp2 = 0;
      for (size_t ik = 0; ik < K; ik += BITS) {
	int64_t p1 = activation.get(im, ik + 0) ^ weights.get(in, ik + 0);
	int64_t p2 = activation.get(im, ik + 1) & weights.get(in, ik + 1);
	cntp1 += popcnt64(p2);
	cntp2 += popcnt64(p1 & p2);
      }
      output.set(cntp1 - cntp2 - cntp2, im, in);
    }
  }

  return output;
}
