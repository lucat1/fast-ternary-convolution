#include "minor_impls/t2r_avx512u_gemmLU_block/tab.hpp"
#include "measure.hpp"
#include "minor_impls/t2r_avx512u_gemmLU_block/t2r_avx512.hpp"
#include "minor_impls/t2r_gemmLU_block/gemmLU_block.hpp"

namespace t2r_avx512u_gemmLU_block {
Tensor4D<float> conv(const Tensor4D<float> &input,
                     const Tensor1D<float> &thresholds, const size_t padding_h,
                     const size_t padding_w, const Tensor5D<int64_t> &kernel,
                     const size_t stride_h, const size_t stride_w,
                     float relu_alpha) {
  const size_t kernel_h = kernel.dim2;
  const size_t kernel_w = kernel.dim3;

  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::START);
  Tensor7D<int64_t> quantized_reshaped =
      t2r_avx512(input, thresholds, padding_h, padding_w, kernel_h, kernel_w,
                 stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  Tensor4D<float> result =
      t2r_gemmLU_block::gemmLU_block(quantized_reshaped, kernel, relu_alpha);
  measure_point(measurement_point::gemmprelu, MeasurementEvent::END);

  return result;
}

} // namespace t2r_avx512u_gemmLU_block
