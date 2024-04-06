#include "impl/baseline/tab.hpp"
#include "common.hpp"
#include "impl/baseline/activation.hpp"
#include "impl/baseline/gemm.hpp"
#include "impl/baseline/img2row.hpp"
#include "impl/baseline/quantize.hpp"
#include <cstring>

namespace baseline {

// output: point to n * c * h * w floats (batch_size * kernel_number * output_height * output_width)
void conv(registry::conv_type_t type, int *btn_cnt1, float *input,
          uint32_t input_height, uint32_t input_width, uint32_t padding_height,
          uint32_t padding_width, float *quant_threshold, int num_channels,
          int64_t *quant_weights, uint32_t batch_size, uint32_t stride_height,
          uint32_t stride_width, uint32_t kernel_number, uint32_t kernel_height,
          uint32_t kernel_width, float relu_alpha, float *output) {
  int packed_height, packed_width, packed_channels, output_height, output_width, fused_height, fused_width;
  packed_height = input_height + 2 * padding_height;
  packed_width = input_width + 2 * padding_width;
  packed_channels = (num_channels % CNTBITS)
                ? ((num_channels / CNTBITS) + 1)
                : (num_channels / CNTBITS);
  
  output_height =
      (packed_height - kernel_height + 1) / stride_height;
  output_width = (packed_width - kernel_width + 1) / stride_width;
  
  fused_height = output_height * output_width;
  fused_width = kernel_height * kernel_width * num_channels;

  int64_t *qx;
  int64_t *i2rqx;
  int *y_intermediate;

  // Quantize and Img2Row/Img2Col

  if ((type == registry::conv_type_t::TNN) ||
      (type == registry::conv_type_t::TBN)) {
    qx = registry::calloc<int64_t>(batch_size * packed_height * packed_width * packed_channels * BITS);
    i2rqx = registry::alloc<int64_t>(batch_size * fused_height * fused_width);
    
    ternarize_NCHW_to_NHWCB(
        input, padding_height, padding_width, quant_threshold, batch_size,
        num_channels, input_height, input_width, qx);
    img2row_NHWCB_to_N_OHOW_KHKWC(
        qx, batch_size, packed_channels * BITS, packed_height, packed_width,
        kernel_height, kernel_width, stride_height, stride_width, i2rqx);
  } else {
    qx = registry::calloc<int64_t>(batch_size * packed_height * packed_width * packed_channels);
    i2rqx = registry::alloc<int64_t>(batch_size * fused_height * fused_width);
    
    binarize_NCHW_to_NHWC(input, padding_height, padding_width, batch_size,
                              num_channels, input_height, input_width, qx);
    img2row_NHWCB_to_N_OHOW_KHKWC(
        qx, batch_size, packed_channels, packed_height, packed_width,
        kernel_height, kernel_width, stride_height, stride_width, i2rqx);
  }

  // Bitwise GEMM

  y_intermediate = registry::calloc<int>(batch_size * output_height * output_width * kernel_number);
  switch (type) {
  case registry::conv_type_t::TNN: {
    tnn_gemm_baseline(
        i2rqx, quant_weights,
        batch_size * output_height * output_width, kernel_number,
        packed_channels * kernel_height * kernel_width, y_intermediate);
    break;
  }
  case registry::conv_type_t::TBN: {
    tbn_gemm_baseline(
        i2rqx, quant_weights,
        batch_size * output_height * output_width, kernel_number,
        packed_channels * kernel_height * kernel_width, y_intermediate);
    break;
  }
  case registry::conv_type_t::BTN: {
    btn_gemm_baseline(
        i2rqx, quant_weights, btn_cnt1,
        batch_size * output_height * output_width, kernel_number,
        packed_channels * kernel_height * kernel_width, y_intermediate);
    break;
  }
  case registry::conv_type_t::BNN: {
   bnn_gemm_baseline(i2rqx, quant_weights,
                          batch_size * output_height * output_width,
                          kernel_number, packed_channels * kernel_height * kernel_width,
		     num_channels * kernel_height * kernel_width, y_intermediate);
    break;
  }
  default:
    assert(0);
  } // switch

  // Activation function: PReLU

  PReLU(y_intermediate, batch_size, kernel_number, output_height,
            output_width, relu_alpha, output);

  free(qx);
  free(i2rqx);
  free(y_intermediate);
}

} // namespace baseline
