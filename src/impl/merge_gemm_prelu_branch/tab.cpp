#include "impl/merge_gemm_prelu_branch/tab.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "impl/merge_gemm_prelu_branch/gemm.hpp"
#include "measure.hpp"

// Based of off baseline_nhwc.
namespace merge_gemm_prelu_branch {

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
  Tensor7D<int64_t> reshaped = baseline_nhwc::im2row(
      quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  auto result = ternary_gemm(reshaped, kernel, relu_alpha);
  measure_point(measurement_point::gemmprelu, MeasurementEvent::END);

  return result;
}

} // namespace merge_gemm_prelu_branch
