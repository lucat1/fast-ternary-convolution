#include "main_impls/best_impl_avx2/tab.hpp"
#include "main_impls/best_impl_avx2/gemmLU_block.hpp"
#include "main_impls/best_impl_avx2/gemmLU_block_avx2.hpp"
#include "main_impls/best_impl_avx2/t2r_avx2.hpp"
#include "measure.hpp"

namespace best_impl_avx2 {

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
      best_impl_avx2::t2r_avx2(input, thresholds, padding_h, padding_w,
                               kernel.dim2, kernel.dim3, stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  if (quantized_reshaped.dim6 >= 512) {
    auto result = best_impl_avx2::gemmLU_block_avx2(quantized_reshaped, kernel,
                                                    relu_alpha);
    measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
    return result;
  } else {
    auto result =
        best_impl_avx2::gemmLU_block(quantized_reshaped, kernel, relu_alpha);
    measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
    return result;
  }
}

} // namespace best_impl_avx2
