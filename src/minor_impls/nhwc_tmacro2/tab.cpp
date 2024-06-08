#include "minor_impls/nhwc_tmacro2/tab.hpp"
#include "measure.hpp"
#include "minor_impls/nhwc_tmacro2/gemm.hpp"
#include "minor_impls/nhwc_tmacro2/im2row.hpp"
#include "minor_impls/nhwc_tmacro2/prelu.hpp"
#include "minor_impls/nhwc_tmacro2/quantize.hpp"

namespace nhwc_tmacro2 {

// input: NCHW
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing
  measure_point(measurement_point::ternarize, MeasurementEvent::START);
  Tensor5D<int64_t> quantized =
      ternarize(input, thresholds, padding_h, padding_w);
  measure_point(measurement_point::ternarize, MeasurementEvent::END);

  // im2row
  measure_point(measurement_point::im2row, MeasurementEvent::START);
  Tensor7D<int64_t> reshaped =
      im2row(quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemm, MeasurementEvent::START);
  auto gemm_result = ternary_gemm(reshaped, kernel);
  measure_point(measurement_point::gemm, MeasurementEvent::END);

  // activation
  measure_point(measurement_point::prelu, MeasurementEvent::START);
  auto result = prelu(gemm_result, relu_alpha);
  measure_point(measurement_point::prelu, MeasurementEvent::END);
  return result;
}

} // namespace nhwc_tmacro2
