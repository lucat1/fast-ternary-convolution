#include "impl/baseline/tab.hpp"
#include "impl/baseline/activate.hpp"

namespace baseline {

void conv(registry::conv_type_t type, int *btn_cnt1, double *input, uint32_t input_height,
          uint32_t input_width, uint32_t padding_height, uint32_t padding_width,
          double *quant_threshold, int num_channels, int64_t *quant_weights,
          uint32_t batch_size, uint32_t stride_height, uint32_t stride_width,
          uint32_t kernel_number, uint32_t kernel_height, uint32_t kernel_width
          float relu_alpha, float *output) {
  int PackedH, PackedW, output_height, output_width, PackedC;
  PackedH = input_height + 2 * padding_height; // Height after bit-packing
  PackedW = input_width + 2 * padding_width; // Width  after bit-packing
  output_height = (PackedH - kernel_height + 1) / stride_height; // Output Height
  output_width = (PackedW - kernel_width + 1) / stride_width; // Output Width
  PackedC = (num_channels % CNTBITS) ? ((num_channels / CNTBITS) + 1) : (num_channels / CNTBITS); // The channel after bit-packing
    
  std::vector<int64_t> quant_input;
  std::vector<int> y_intermediate;
  std::vector<float> y;

  // Quantize and Img2Row/Img2Col
      
  if ((type == conv_type_t::TNN) || (type == conv_type_t::TBN)) {
      quant_input = ternarize_NCHW_to_NHWCB(input, padding_height, padding_width, quant_threshold, batch_size, num_channels, input_height, input_width);
      quant_input = img2row_NHWCB_to_N_OHOW_KHKWC(quant_input.data(), batch_size, PackedC * BITS, PackedH, PackedW, kernel_height, kernel_width, stride_height, stride_width);
  }
  else {
      quant_input = binarize_NCHW_to_NHWC(input, padding_height, padding_width, quant_threshold, batch_size, num_channels, input_height, input_width);
      quant_input = img2row_NHWCB_to_N_OHOW_KHKWC(quant_input.data(), batch_size, PackedC, PackedH, PackedW, kernel_height, kernel_width, stride_height, stride_width);
  }
      
  // Bitwise GEMM
     
  switch (type) {
    case conv_type_t::TNN: {
        y_intermediate = tnn_gemm_baseline(quant_input.data(), quant_weights, batch_size * output_height * output_width, kernel_number, PackedC * kernel_height * kernel_width);
        break;
    }
    case conv_type_t::TBN: {
        y_intermediate = tbn_gemm_baseline(quant_input.data(), quant_weights, batch_size * output_height * output_width, kernel_number, PackedC * kernel_height * kernel_width);
        break;
    }
    case conv_type_t::BTN: {
        y_intermediate = btn_gemm_baseline(quant_input.data(), quant_weights, btn_cnt1, batch_size * output_height * output_width, kernel_number, PackedC * kernel_height * kernel_width);
        break;
    }
    case conv_type_t::BNN: {
        y_intermediate = bnn_gemm_baseline(quant_input.data(), quant_weights, batch_size * output_height * output_width, kernel_number, PackedC * kernel_height * kernel_width, num_channels * kernel_height * kernel_width);
        break;
    }
  } // switch
  
  // Activation function: PReLU

  y = PReLU(y_intermediate.data(), batch_size, kernel_number, output_height, output_width, relu_alpha);

  return y;
}

} // namespace baseline
