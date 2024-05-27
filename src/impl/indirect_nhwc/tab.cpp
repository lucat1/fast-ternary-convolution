#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "impl/indirect_nhwc/gemm.hpp"
#include "impl/indirect_nhwc/indirect.hpp"
#include "measure.hpp"

namespace indirect_nhwc {

// input: NHWC
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing
  measure_point(measurement_point::ternarize, MeasurementEvent::START);
  Tensor5D<int64_t> quantized =
      baseline_nhwc::ternarize(input, thresholds, padding_h, padding_w);
  measure_point(measurement_point::ternarize, MeasurementEvent::END);

  // im2row
  measure_point(measurement_point::im2row, MeasurementEvent::START);
  Tensor5D<const int64_t *> ib = indirection_buffer(
      quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemm, MeasurementEvent::START);
  auto gemm_result = ternary_gemm(ib, kernel);
  measure_point(measurement_point::gemm, MeasurementEvent::END);

  // activation
  measure_point(measurement_point::prelu, MeasurementEvent::START);
  auto result = baseline_nhwc::prelu(gemm_result, relu_alpha);
  measure_point(measurement_point::prelu, MeasurementEvent::END);
  return result;
}

} // namespace indirect_nhwc
