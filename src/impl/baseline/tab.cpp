#include "impl/baseline/tab.hpp"
#include "impl/baseline/activation.hpp"
#include "impl/baseline/gemm.hpp"
#include "impl/baseline/img2row.hpp"
#include "impl/baseline/quantize.hpp"

#include "alloc.hpp"
#include "common.hpp"
#include "measure.hpp"

#include <cstring>

namespace baseline {

// Input: (N, C, H, W)
// Kernel: (KN, KH, KW, C, B)
// Thresholds: (N)
Tensor4D<float> conv(const Tensor4D<float> &_input,
                     const Tensor1D<float> &_thresholds,
                     const size_t _padding_h, const size_t _padding_w,
                     const Tensor5D<int64_t> &_kernel, const size_t _stride_h,
                     const size_t _stride_w, float _relu_alpha) {
  ConvolutionType type = ConvolutionType::TNN;
  int *btn_cnt1 = nullptr;
  float *input = _input.data;
  uint32_t input_height = _input.dim3;
  uint32_t input_width = _input.dim4;
  uint32_t padding_height = _padding_h;
  uint32_t padding_width = _padding_w;
  float *quant_threshold = _thresholds.data;
  int num_channels = _input.dim2;
  int64_t *quant_weights = _kernel.data;
  uint32_t batch_size = _input.dim1;
  uint32_t stride_height = _stride_h;
  uint32_t stride_width = _stride_w;
  uint32_t kernel_number = _kernel.dim1;
  uint32_t kernel_height = _kernel.dim2;
  uint32_t kernel_width = _kernel.dim3;
  float relu_alpha = _relu_alpha;

  size_t packed_height, packed_width, packed_channels, output_height,
      output_width, fused_height, fused_width;
  packed_height = input_height + 2 * padding_height;
  packed_width = input_width + 2 * padding_width;
  packed_channels = (num_channels % CNTBITS) ? ((num_channels / CNTBITS) + 1)
                                             : (num_channels / CNTBITS);

  output_height = (packed_height - kernel_height) / stride_height + 1;
  output_width = (packed_width - kernel_width) / stride_width + 1;

  fused_height = output_height * output_width;
  fused_width = kernel_height * kernel_width * (packed_channels * BITS);

  size_t qx_size;
  int64_t *qx;
  size_t i2rqx_size;
  int64_t *i2rqx;
  int *y_intermediate;

  // Quantize and Img2Row/Img2Col
  if (has_ternary_input(type)) {
    measure_point(MeasurementFunction::ALLOC, MeasurementEvent::START);
    qx_size =
        batch_size * packed_height * packed_width * packed_channels * BITS;
    qx = alloc::calloc<int64_t>(qx_size);
    i2rqx_size = batch_size * fused_height * fused_width;
    i2rqx = alloc::calloc<int64_t>(i2rqx_size);
    measure_point(MeasurementFunction::ALLOC, MeasurementEvent::END);

    ternarize_NCHW_to_NHWCB(input, padding_height, padding_width,
                            quant_threshold, batch_size, num_channels,
                            input_height, input_width, qx);
    img2row_NHWCB_to_N_OHOW_KHKWC<int64_t>(
        qx, batch_size, packed_channels * BITS, packed_height, packed_width,
        kernel_height, kernel_width, stride_height, stride_width, i2rqx);
  } else {
    measure_point(MeasurementFunction::ALLOC, MeasurementEvent::START);
    qx_size = batch_size * packed_height * packed_width * packed_channels;
    qx = alloc::calloc<int64_t>(qx_size);
    i2rqx_size = batch_size * fused_height * fused_width;
    i2rqx = alloc::calloc<int64_t>(i2rqx_size);
    measure_point(MeasurementFunction::ALLOC, MeasurementEvent::END);

    binarize_NCHW_to_NHWC(input, padding_height, padding_width, quant_threshold,
                          batch_size, num_channels, input_height, input_width,
                          qx);
    img2row_NHWCB_to_N_OHOW_KHKWC<int64_t>(
        qx, batch_size, packed_channels, packed_height, packed_width,
        kernel_height, kernel_width, stride_height, stride_width, i2rqx);
  }

  // Bitwise GEMM

  measure_point(MeasurementFunction::ALLOC2, MeasurementEvent::START);
  size_t y_size = batch_size * output_height * output_width * kernel_number;
  y_intermediate = alloc::calloc<int>(y_size);
  measure_point(MeasurementFunction::ALLOC2, MeasurementEvent::END);
  switch (type) {
  case ConvolutionType::TNN: {
    tnn_gemm_baseline(i2rqx, quant_weights,
                      batch_size * output_height * output_width, kernel_number,
                      packed_channels * kernel_height * kernel_width,
                      y_intermediate);
    break;
  }
  case ConvolutionType::TBN: {
    tbn_gemm_baseline(i2rqx, quant_weights,
                      batch_size * output_height * output_width, kernel_number,
                      packed_channels * kernel_height * kernel_width,
                      y_intermediate);
    break;
  }
  case ConvolutionType::BTN: {
    btn_gemm_baseline(i2rqx, quant_weights, btn_cnt1,
                      batch_size * output_height * output_width, kernel_number,
                      packed_channels * kernel_height * kernel_width,
                      y_intermediate);
    break;
  }
  case ConvolutionType::BNN: {
    bnn_gemm_baseline(
        i2rqx, quant_weights, batch_size * output_height * output_width,
        kernel_number, packed_channels * kernel_height * kernel_width,
        num_channels * kernel_height * kernel_width, y_intermediate);
    break;
  }
  default:
    assert(0);
  } // switch

  // Activation function: PReLU

  Tensor4D<float> output = Tensor4D<float>(batch_size, output_height,
                                           output_width, kernel_number, false);
  PReLU(y_intermediate, batch_size, kernel_number, output_height, output_width,
        relu_alpha, output.data);

  measure_point(MeasurementFunction::FREE, MeasurementEvent::START);
  alloc::free(qx);
  alloc::free(i2rqx);
  alloc::free(y_intermediate);
  measure_point(MeasurementFunction::FREE, MeasurementEvent::END);
  return output;
}

} // namespace baseline
