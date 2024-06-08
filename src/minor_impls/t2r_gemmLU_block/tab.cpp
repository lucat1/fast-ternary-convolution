#include "minor_impls/t2r_gemmLU_block/tab.hpp"
#include "measure.hpp"
#include "minor_impls/t2r_gemmLU_block/gemmLU_block.hpp"
#include "minor_impls/tern2row_cpy/tern2row_cpy.hpp"

namespace t2r_gemmLU_block {
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  const size_t kernel_h = kernel.dim2;
  const size_t kernel_w = kernel.dim3;

  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::START);
  Tensor7D<int64_t> quantized_reshaped =
      tern2row_cpy::tern2row_cpy(input, thresholds, padding_h, padding_w,
                                 kernel_h, kernel_w, stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  Tensor4D<float> result = gemmLU_block(quantized_reshaped, kernel, relu_alpha);
  measure_point(measurement_point::gemmprelu, MeasurementEvent::END);

  return result;
}

} // namespace t2r_gemmLU_block
