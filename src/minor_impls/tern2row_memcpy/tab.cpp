#include "minor_impls/tern2row_memcpy/tab.hpp"
#include "main_impls/data_order_nhwc_tensor_macro1/gemm.hpp"
#include "main_impls/data_order_nhwc_tensor_macro1/prelu.hpp"
#include "measure.hpp"
#include "minor_impls/tern2row_memcpy/tern2row_memcpy.hpp"

namespace tern2row_memcpy {

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
      tern2row_memcpy(input, thresholds, padding_h, padding_w, kernel.dim2,
                      kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemm, MeasurementEvent::START);
  auto gemm_result =
      data_order_nhwc_tensor_macro1::ternary_gemm(quantized_reshaped, kernel);
  measure_point(measurement_point::gemm, MeasurementEvent::END);

  // activation
  measure_point(measurement_point::prelu, MeasurementEvent::START);
  auto result = data_order_nhwc_tensor_macro1::prelu(gemm_result, relu_alpha);
  measure_point(measurement_point::prelu, MeasurementEvent::END);
  return result;
}

} // namespace tern2row_memcpy
