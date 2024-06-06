#include "impl/all_opts_merged_avx2/tab.hpp"
#include "impl/t2r_avx2u_permute_ur_gemmLU_block/t2r_avx2.hpp"
#include "impl/t2r_gemmLU_block/gemmLU_block.hpp"
#include "impl/t2r_gemmLU_block_avx2/gemmLU_block.hpp"
#include "measure.hpp"

namespace all_opts_merged_avx2 {

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
      t2r_avx2u_permute_ur_gemmLU_block::t2r_avx2(
          input, thresholds, padding_h, padding_w, kernel.dim2, kernel.dim3,
          stride_h, stride_w);
  measure_point(measurement_point::ternarize_im2row, MeasurementEvent::END);

  // gemm
  measure_point(measurement_point::gemmprelu, MeasurementEvent::START);
  if (quantized_reshaped.dim6 >= 512) {
    auto result = t2r_gemmLU_block_avx2::gemmLU_block(quantized_reshaped,
                                                      kernel, relu_alpha);
    measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
    return result;
  } else {
    auto result =
        t2r_gemmLU_block::gemmLU_block(quantized_reshaped, kernel, relu_alpha);
    measure_point(measurement_point::gemmprelu, MeasurementEvent::END);
    return result;
  }
}

} // namespace all_opts_merged_avx2