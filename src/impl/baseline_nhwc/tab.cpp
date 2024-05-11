#include "impl/baseline_nhwc/tab.hpp"
#include "impl/baseline_nhwc/im2row.hpp"
#include "impl/baseline_nhwc/prelu.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "impl/baseline_nhwc/gemm.hpp"

namespace baseline_nhwc {

// TODO @luca: I think we should have the following signature:
// Tensor2D<float> conv(const Tensor4D<float>& input, const Tensor1D<float>& thresholds,
//                      const size_t padding_h, const size_t padding_w,
//                      const Tensor4D<int64_t>& kernel,
//                      const size_t stride_h, const size_t stride_w);
// Additionally, whoever calls conv should pass in the correct shape which could be
// registered together with the function name and pointer.
void conv(ConvolutionType type, int *btn_cnt1, float *input,
          uint32_t input_height, uint32_t input_width, uint32_t padding_height,
          uint32_t padding_width, float *quant_threshold, int num_channels,
          int64_t *quant_weights, uint32_t batch_size, uint32_t stride_height,
          uint32_t stride_width, uint32_t kernel_number, uint32_t kernel_height,
          uint32_t kernel_width, float relu_alpha, float *output) {

  // Not used, since we specialize to ternary
  (void) type;
  (void) btn_cnt1;

  ///
  /// Bringing the data in tensors
  ///

  // input data
  Tensor4D<float> input_data(batch_size, input_height, input_width,
                             num_channels, false);
  // NCHW => NHWC
  for (size_t in = 0; in < batch_size; in++) {
    for (int ic = 0; ic < num_channels; ic++) {
      for (size_t ih = 0; ih < input_height; ih++) {
        for (size_t iw = 0; iw < input_width; iw++) {
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

  // kernel weights
  const size_t packed_channels = (num_channels % CNTBITS) ? ((num_channels / CNTBITS) + 1)
                                             : (num_channels / CNTBITS);
  Tensor2D<int64_t> weight_tensor (kernel_number, packed_channels * kernel_height * kernel_width * 2, false);
  for (size_t i = 0; i < kernel_number * packed_channels * kernel_height * kernel_width * 2; i++) {
    weight_tensor.data[i] = quant_weights[i];
  }

  // thresholds
  Tensor1D<float> quant_data(batch_size, false);
  std::memcpy(quant_data.data, quant_threshold, sizeof(float) * batch_size);

  ///
  /// Actual algorithm
  ///

  // TODO @daniel: It may be better to return the largest possible tensor

  // quantization + packing
  Tensor5D<int64_t> quantized =
      ternarize(input_data, quant_data, padding_height, padding_width);

  // im2row
  Tensor2D<int64_t> reshaped = im2row(quantized, kernel_height, kernel_width, stride_height,
                     stride_width);

  // gemm
  auto gemm_result = ternary_gemm(reshaped, weight_tensor);

  // activation
  auto prelu_result = prelu(gemm_result, relu_alpha);
  for (size_t im = 0; im < prelu_result.dim1; im++) {
    for (size_t in = 0; in < prelu_result.dim2; in++) {
      output[im * prelu_result.dim2 + in] = prelu_result.get(im, in);
    }
  }
}

} // namespace baseline_nhwc
