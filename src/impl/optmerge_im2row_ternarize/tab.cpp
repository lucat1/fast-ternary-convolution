#include "impl/optmerge_im2row_ternarize/tab.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/optmerge_im2row_ternarize/quantize_im2row.hpp"
#include "measure.hpp"

namespace optmerge_im2row_ternarize {

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
      ternarize_im2row(input, thresholds, padding_h, padding_w, kernel.dim2,
                       kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemm, MeasurementEvent::START);
  auto gemm_result = baseline_nhwc::ternary_gemm(quantized_reshaped, kernel);
  measure_point(measurement_point::gemm, MeasurementEvent::END);

  // activation
  measure_point(measurement_point::prelu, MeasurementEvent::START);
  auto result = baseline_nhwc::prelu(gemm_result, relu_alpha);
  measure_point(measurement_point::prelu, MeasurementEvent::END);
  return result;
}

} // namespace optmerge_im2row_ternarize
