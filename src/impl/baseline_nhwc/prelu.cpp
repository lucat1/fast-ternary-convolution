#include "impl/baseline_nhwc/prelu.hpp"
#include "tensor.hpp"

// NOTE We can probably merge this with GEMM
Tensor2D<float> prelu(Tensor2D<int64_t>& pre_activation, float alpha) {
  // our sizes
  const size_t M = pre_activation.dim1;
  const size_t N = pre_activation.dim2;

  Tensor2D<float> post_activation (M, N, false);

  for (size_t im = 0; im < M; im++) {
    for (size_t in = 0; in < N; in++) {
      float current = pre_activation.get(im, in);
      // NOTE If we cannot merge this with GEMM, make sure to apply scalar replacement
      //  here.
      if (current > 0) {
	post_activation.set(current, im, in);
      } else {
	post_activation.set(current * alpha, im, in);
      }
    }
  }

  return post_activation;
}
