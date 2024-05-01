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

// output: point to n * c * h * w floats (batch_size * kernel_number *
// output_height * output_width)
void conv(ConvolutionType type, int *btn_cnt1, float *input,
          uint32_t input_height, uint32_t input_width, uint32_t padding_height,
          uint32_t padding_width, float *quant_threshold, int num_channels,
          int64_t *quant_weights, uint32_t batch_size, uint32_t stride_height,
          uint32_t stride_width, uint32_t kernel_number, uint32_t kernel_height,
          uint32_t kernel_width, float relu_alpha, float *output) {
  int packed_height, packed_width, packed_channels, output_height, output_width,
      fused_height, fused_width;
  packed_height = input_height + 2 * padding_height;
  packed_width = input_width + 2 * padding_width;
  packed_channels = (num_channels % CNTBITS) ? ((num_channels / CNTBITS) + 1)
                                             : (num_channels / CNTBITS);

  output_height = (packed_height - kernel_height + 1) / stride_height;
  output_width = (packed_width - kernel_width + 1) / stride_width;

  fused_height = output_height * output_width;
  fused_width = kernel_height * kernel_width * num_channels;

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
    i2rqx_size = batch_size * fused_height * fused_width * BITS;
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

  PReLU(y_intermediate, batch_size, kernel_number, output_height, output_width,
        relu_alpha, output);

  measure_point(MeasurementFunction::FREE, MeasurementEvent::START);
  free(qx);
  free(i2rqx);
  free(y_intermediate);
  measure_point(MeasurementFunction::FREE, MeasurementEvent::END);
}

} // namespace baseline
