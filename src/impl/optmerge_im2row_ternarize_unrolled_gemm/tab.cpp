#include "impl/optmerge_im2row_ternarize_unrolled_gemm/tab.hpp"
#include "impl/merge_gemm_prelu_branch/gemm.hpp"
#include "impl/optmerge_im2row_ternarize/quantize_im2row.hpp"
#include "measure.hpp"

namespace optmerge_im2row_ternarize_unrolled_gemm {

// input: NCHW
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing + reshaping
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::START);
  Tensor7D<int64_t> quantized_reshaped =
      optmerge_im2row_ternarize::ternarize_im2row(
          input, thresholds, padding_h, padding_w, kernel.dim2, kernel.dim3,
          stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  auto result = merge_gemm_prelu_branch::ternary_gemm(quantized_reshaped,
                                                      kernel, relu_alpha);
  measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
  return result;
}

} // namespace optmerge_im2row_ternarize_unrolled_gemm
