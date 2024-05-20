#include "impl/optmerge_im2row_ternarize/tab.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/optmerge_im2row_ternarize/quantize_im2row.hpp"

namespace optmerge_im2row_ternarize {

// input: NCHW
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing + reshaping
  Tensor7D<int64_t> quantized_reshaped =
      ternarize_im2row(input, thresholds, padding_h, padding_w, kernel.dim2,
                       kernel.dim3, stride_h, stride_w);

  // gemm
  auto gemm_result = baseline_nhwc::ternary_gemm(quantized_reshaped, kernel);

  // activation
  return baseline_nhwc::prelu(gemm_result, relu_alpha);
}

} // namespace optmerge_im2row_ternarize
