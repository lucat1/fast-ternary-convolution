#include "impl/baseline_nchw/tab.hpp"
#include "impl/baseline_nchw/quantize.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"

namespace baseline_nchw {

// input: NCHW
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing
  Tensor5D<int64_t> quantized =
      ternarize(input, thresholds, padding_h, padding_w);

  // im2row
  Tensor7D<int64_t> reshaped =
    baseline_nhwc::im2row(quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);

  // gemm
  auto gemm_result = baseline_nhwc::ternary_gemm(reshaped, kernel);

  // activation
  return baseline_nhwc::prelu(gemm_result, relu_alpha);
}

} // namespace baseline_nchw
