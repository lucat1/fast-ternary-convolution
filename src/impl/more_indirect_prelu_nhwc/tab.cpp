#include "impl/baseline_nhwc/quantize.hpp"
#include "impl/more_indirect_nhwc/indirect.hpp"
#include "impl/more_indirect_prelu_nhwc/gemm.hpp"
#include "measure.hpp"

namespace more_indirect_prelu_nhwc {

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

  measure_point(measurement_point::im2row, MeasurementEvent::START);
  Tensor3D<const int64_t *> ib = more_indirect_nhwc::indirection_buffer(
      quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::im2row, MeasurementEvent::END);

  // gemm + activation
  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  auto result = ternary_gemm(ib, kernel, quantized.dim3, relu_alpha);
  measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
  return result;
}

} // namespace more_indirect_prelu_nhwc
