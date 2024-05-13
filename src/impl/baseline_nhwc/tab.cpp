#include "impl/baseline_nhwc/tab.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/baseline_nhwc/quantize.hpp"

namespace baseline_nhwc {

// input: NHWC
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // TODO @daniel: It may be better to return the largest possible tensor

  // quantization + packing
  Tensor5D<int64_t> quantized =
      ternarize(input, thresholds, padding_h, padding_w);

  // im2row
  Tensor7D<int64_t> reshaped =
      im2row(quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);

  // gemm
  auto gemm_result = ternary_gemm(reshaped, kernel);

  // activation
  return prelu(gemm_result, relu_alpha);
}

} // namespace baseline_nhwc
