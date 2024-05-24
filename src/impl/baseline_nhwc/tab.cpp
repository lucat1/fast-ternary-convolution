#include "impl/baseline_nhwc/tab.hpp"
#include "impl/baseline_nhwc/gemm.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "measure.hpp"

namespace baseline_nhwc {

// input: NHWC
// kernel: NHWCB
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  // quantization + packing
  measure_point(MeasurementFunction::TERNARIZE_IMG2ROW,
                MeasurementEvent::START);
  Tensor5D<int64_t> quantized =
      ternarize(input, thresholds, padding_h, padding_w);

  // im2row
  Tensor7D<int64_t> reshaped =
      im2row(quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(MeasurementFunction::TERNARIZE_IMG2ROW, MeasurementEvent::END);

  // gemm
  measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::START);
  auto gemm_result = ternary_gemm(reshaped, kernel);

  // activation
  auto result = prelu(gemm_result, relu_alpha);

  measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::END);

  return result;
}

} // namespace baseline_nhwc
