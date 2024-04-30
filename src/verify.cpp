#include "verify.hpp"
#include "common.hpp"
#include "problem_data.hpp"

// TODO: rework these functions and move them in this file
#include "verify_util.hpp"

#include "impl/baseline/quantize.hpp"

#include <cstdint>
#include <iomanip>

class TestParameters {
public:
  uint32_t num_channels;
  uint32_t kernel_number;
  size_t input_height;
  size_t input_width;
  size_t kernel_height;
  size_t kernel_width;
  size_t padding_size;
  size_t stride_size;

  TestParameters(uint32_t num_channels, uint32_t kernel_number,
                 size_t input_height, size_t input_width, size_t kernel_height,
                 size_t kernel_width, size_t padding_size, size_t stride_size)
      : num_channels(num_channels), kernel_number(kernel_number),
        input_height(input_height), input_width(input_width),
        kernel_height(kernel_height), kernel_width(kernel_width),
        padding_size(padding_size), stride_size(stride_size) {}
};

std::vector<TestParameters> test_cases = {
    {1, 2, 2, 1, 3, 3, 1, 1},       {64, 12, 16, 64, 1, 1, 0, 1},
    {32, 12, 16, 52, 1, 1, 0, 2},   {256, 56, 56, 10, 3, 3, 1, 1},
    {160, 64, 56, 32, 3, 3, 0, 2},  {325, 36, 25, 125, 5, 7, 3, 4},
    {32, 1, 1, 120, 1, 1, 0, 1},    {512, 1, 1, 1024, 1, 1, 0, 1},
    {1024, 1, 1, 1640, 1, 1, 2, 3},
};

void verify(Registry r) {
  const int batch_size = 2;
  const int relu_alpha = 1;

  for (auto impl : r.implementations()) {
    size_t passed = 0, failed = 0,
           total = test_cases.size() * convolution_types.size();
    for (auto tc : test_cases) {
      for (auto conv_type : convolution_types) {
        auto data = VerificationData(
            conv_type, batch_size, tc.num_channels, tc.kernel_number,
            {tc.input_height, tc.input_width},
            {tc.kernel_height, tc.kernel_width},
            {tc.padding_size, tc.padding_size},
            {tc.stride_size, tc.stride_size}, relu_alpha);
        if (has_ternary_weights(conv_type))
          baseline::ternarize_NCHW_to_NHWCB(
              data.weights, 0, 0, data.quant_threshold, data.kernel_number,
              data.num_channels, data.kernel_size.height,
              data.kernel_size.width, data.quant_weights);
        else
          baseline::binarize_NCHW_to_NHWC(
              data.weights, 0, 0, data.kernel_number, data.num_channels,
              data.kernel_size.height, data.kernel_size.width,
              data.quant_weights);

        if (conv_type == ConvolutionType::BTN)
          baseline::btn_cnt_w2(data.quant_weights, data.num_channels,
                               data.kernel_number, data.kernel_size.height,
                               data.kernel_size.width, data.btn_cnt);

        impl.fn(conv_type, data.btn_cnt, data.x, data.input_size.height,
                data.input_size.width, data.padding_size.height,
                data.padding_size.width, data.quant_threshold,
                data.num_channels, data.quant_weights, batch_size,
                data.stride_size.height, data.stride_size.height,
                data.kernel_number, data.kernel_size.height,
                data.kernel_size.width, relu_alpha, data.y);

        std::vector<float> px =
            DirectPad(data.x, data.padding_size.height, data.padding_size.width,
                      batch_size, data.num_channels, data.input_size.height,
                      data.input_size.width);
        std::vector<float> ref_y =
            DirectConv2d_FP32(px.data(), data.weights, data.stride_size.height,
                              data.stride_size.width, batch_size,
                              data.num_channels, data.packed_input_size.height,
                              data.packed_input_size.width, data.kernel_number,
                              data.kernel_size.height, data.kernel_size.width);

        // Compare the conv results to ensure the functions are correct
        int cmp;
        if ((data.padding_size.width > 0 || data.padding_size.height > 0) &&
            !has_ternary_input(conv_type))
          // BTN and BNN regard the padded zeros as 1s because binary
          // quantization only has (+1, -1) no zeros. So we only compare the
          // central part of conv results here, excluding the zero padding part.
          cmp = Compare_Tensor_BNN_Padding(
              data.y, ref_y.data(), batch_size, data.kernel_number,
              data.y_shape.height, data.y_shape.width, data.padding_size.height,
              data.padding_size.width);
        else
          cmp = Compare_Tensor_NHWC(data.y, ref_y.data(), batch_size,
                                    data.kernel_number, data.y_shape.height,
                                    data.y_shape.width);

        if (cmp > 0)
          passed++;
        else {
          failed++;
          cout << "[" << (passed + failed) << "/" << total
               << "] Failed test case" << endl;
        }
      }
    }
    cout << setw(name_space) << impl.name << " :: " << passed << "/" << total
         << " tests passed" << endl;
  }
}
