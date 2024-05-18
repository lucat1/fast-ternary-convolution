#include "impl/merge_im2row_ternarize_prelu/tab.hpp"
#include "impl/merge_im2row_ternarize/quantize_im2row.hpp"
#include "impl/merge_im2row_ternarize_prelu/gemm.hpp"

namespace merge_im2row_ternarize_prelu {

// input: NCHW
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing + reshaping
  Tensor7D<int64_t> quantized_reshaped =
      merge_im2row_ternarize::ternarize_im2row(input, thresholds, padding_h,
                                               padding_w, kernel.dim2,
                                               kernel.dim3, stride_h, stride_w);

  // gemm + activation
  return ternary_gemm(quantized_reshaped, kernel, relu_alpha);
}

} // namespace merge_im2row_ternarize_prelu
