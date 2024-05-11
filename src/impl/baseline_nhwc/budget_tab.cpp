// yoinked this bad boy directly from baseline/
// its called budget tab because of that

#include "impl/baseline_nhwc/budget_tab.hpp"
#include "impl/baseline/activation.hpp"
#include "impl/baseline/gemm.hpp"
#include "impl/baseline/img2row.hpp"
#include "impl/baseline/quantize.hpp"
#include "impl/baseline/tab.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "impl/baseline_nhwc/gemm.hpp"

#include "alloc.hpp"
#include "common.hpp"
#include "measure.hpp"

#include <bitset>
#include <cstring>
#include <iostream>

namespace baseline_nhwc {

// output: point to n * c * h * w floats (batch_size * kernel_number *
// output_height * output_width)
void conv(ConvolutionType type, int *btn_cnt1, float *input,
          uint32_t input_height, uint32_t input_width, uint32_t padding_height,
          uint32_t padding_width, float *quant_threshold, int num_channels,
          int64_t *quant_weights, uint32_t batch_size, uint32_t stride_height,
          uint32_t stride_width, uint32_t kernel_number, uint32_t kernel_height,
          uint32_t kernel_width, float relu_alpha, float *output) {
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


  (void) type;
  (void) btn_cnt1;
  
  
  size_t qx_size;
  int64_t *qx;
  size_t i2rqx_size;
  int64_t *i2rqx;
  int *y_intermediate;

  // Quantize and Img2Row/Img2Col
  measure_point(MeasurementFunction::ALLOC, MeasurementEvent::START);
  qx_size = batch_size * packed_height * packed_width * packed_channels * BITS;
  qx = alloc::calloc<int64_t>(qx_size);
  i2rqx_size = batch_size * fused_height * fused_width;
  i2rqx = alloc::calloc<int64_t>(i2rqx_size);
  measure_point(MeasurementFunction::ALLOC, MeasurementEvent::END);

  // TODO just make data be in the correct format in the first place
  Tensor4D<float> input_data(batch_size, input_height, input_width,
                             num_channels, false);

  // NCHW => NHWC
  for (size_t in = 0; in < batch_size; in++) {
    for (int ic = 0; ic < num_channels; ic++) {
      for (size_t ih = 0; ih < input_height; ih++) {
        for (size_t iw = 0; iw < input_width; iw++) {
          // too lazy to implement a setter for this one right now
          size_t inhwc = (in * (input_height * input_width * num_channels)) +
                         (ih * input_width * num_channels) +
                         (iw * num_channels) + ic;

          size_t inchw = (in * num_channels * input_height * input_width) +
                         (ic * input_height * input_width) +
                         (ih * input_width) + iw;

          input_data.data[inhwc] = input[inchw];
        }
      }
    }
  }
  // just copy, ignoring the formatting (probably wrong)
  // std::memcpy(input_data.data, input, batch_size * input_height * input_width
  // * num_channels);

  Tensor1D<float> quant_data(batch_size, false);
  std::memcpy(quant_data.data, quant_threshold, sizeof(float) * batch_size);

  Tensor5D<int64_t> quantized =
      ternarize(input_data, quant_data, padding_height, padding_width);

  // Compare if ternarize is different to original baseline
  // baseline::ternarize_NCHW_to_NHWCB(input, padding_height, padding_width,
  //                         quant_threshold, batch_size, num_channels,
  //                         input_height, input_width, qx);

  // for (size_t in = 0; in < batch_size; in++) {
  //    for (size_t ih = 0; ih < packed_height; ih++){
  // 	 for (size_t iw = 0; iw < packed_width; iw++){
  // 	   for (size_t ic = 0; ic < packed_channels; ic++){
  // 	     for (int bit = 0; bit < 2; bit++) {
  // 	       // too lazy to implement a setter for this one right now
  // 	    size_t inhwcb = (in * (packed_height * packed_width *
  // packed_channels)) + 	      (ih * packed_width * packed_channels) + 	      (iw *
  // packed_channels) + 2 * ic + bit;

  // 	    if (qx[inhwcb] != quantized.data[inhwcb]) {
  // 	       std::cout << " :( old[" << in << ", " << ih << ", " << iw << ", "
  // << ic << ", "<< bit << ", " << "] =" << std::bitset<64>(qx[inhwcb]) << ",
  // new[" << in << ", " << ih << ", " << iw << ", " << ic << ", "<< bit << ", "
  // << "] =" << std::bitset<64>(quantized.data[inhwcb]) << std::endl;

  // 	       }
  // 	     }
  // 	   }
  // 	 }
  //    }
  // }

  alloc::free(qx);
  // qx = quantized.data;
  auto lmao = im2row(quantized, kernel_height, kernel_width, stride_height,
                     stride_width);
  // img2row_NHWCB_to_N_OHOW_KHKWC<int64_t>(
  //     qx, batch_size, packed_channels * BITS, packed_height, packed_width,
  //     kernel_height, kernel_width, stride_height, stride_width, i2rqx);

  // Bitwise GEMM

  measure_point(MeasurementFunction::ALLOC2, MeasurementEvent::START);
  size_t y_size = batch_size * output_height * output_width * kernel_number;
  y_intermediate = alloc::calloc<int>(y_size);
  measure_point(MeasurementFunction::ALLOC2, MeasurementEvent::END);
  // tnn_gemm_baseline(lmao.data, quant_weights,
  //                   batch_size * output_height * output_width, kernel_number,
  //                   packed_channels * kernel_height * kernel_width,
  //                   y_intermediate);

  Tensor2D<int64_t> weight_tensor (kernel_number, packed_channels * kernel_height * kernel_width * 2, false);
  int64_t* temp_ptr = weight_tensor.data;
  weight_tensor.data = quant_weights;
 
  auto gemm_result = ternary_gemm(lmao, weight_tensor);
  weight_tensor.data = temp_ptr;

  // Activation function: PReLU
  auto result = prelu(gemm_result, relu_alpha);

  for (size_t im = 0; im < result.dim1; im++) {
    for (size_t in = 0; in < result.dim2; in++) {
      output[im * result.dim2 + in] = result.get(im, in);
    }
  }

  // baseline::PReLU(gemm_result.data, batch_size, kernel_number, output_height,
  //                 output_width, relu_alpha, output);

  measure_point(MeasurementFunction::FREE, MeasurementEvent::START);
  alloc::free(i2rqx);
  alloc::free(y_intermediate);
  measure_point(MeasurementFunction::FREE, MeasurementEvent::END);
}

} // namespace baseline_nhwc
