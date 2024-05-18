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
  measure_point(MeasurementFunction::TERNARIZE, MeasurementEvent::START);
  Tensor5D<int64_t> quantized =
      baseline_nhwc::ternarize(input, thresholds, padding_h, padding_w);
  measure_point(MeasurementFunction::TERNARIZE, MeasurementEvent::END);

  // im2row
  Tensor5D<const int64_t *> ib = indirection_buffer(
      quantized, kernel.dim2, kernel.dim3, stride_h, stride_w);

  // gemm
  auto gemm_result = ternary_gemm(ib, kernel);

  // activation
  return baseline_nhwc::prelu(gemm_result, relu_alpha);
}

} // namespace indirect_nhwc
