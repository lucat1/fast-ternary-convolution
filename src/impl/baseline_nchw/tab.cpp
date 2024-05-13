#include "impl/baseline_nchw/tab.hpp"
#include "impl/baseline_nchw/quantize.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"

namespace baseline_nchw {

// input: NCHW
// kernel: NCHWB?
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
  // NOTE @daniel: kernel.dim3 and kernel.dim4 could be wrong depending on the
  // shape of kernel. This is if the kernel is NCHWB. Change to dim2,dim3 if
  // kernel is NHWCB
  Tensor7D<int64_t> reshaped =
      im2row(quantized, kernel.dim3, kernel.dim4, stride_h, stride_w);

  // gemm
  auto gemm_result = ternary_gemm(reshaped, kernel);

  // activation
  return prelu(gemm_result, relu_alpha);
}

} // namespace baseline_nchw
