#include "verify.hpp"
#include "alloc.hpp"
#include "common.hpp"
#include "problem_data.hpp"

// TODO: rework these functions and move them in this file
#include "verify_util.hpp"

#include "impl/baseline/quantize.hpp"
#include "impl/baseline/tab.hpp"

#include <cstdint>
#include <iostream>
#include <random>

std::default_random_engine generator;

void randomize(float *dst, size_t size, bool ternary) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-1, 1);
  for (size_t i = 0; i < size; ++i) {
    auto v = distribution(generator);
    dst[i] = (!ternary && v == 0) ? 1 : v;
  }
}

class VerifyData : public Data {
private:
  Matrix4D _weights_shape() {
    return Matrix4D(kernel_number, kernel_size.height, kernel_size.width,
                    num_channels);
  }

public:
  float *weights;

  Matrix4D weights_shape;

  VerifyData(ConvolutionType conv_type, uint32_t batch_size,
             uint32_t num_channels, uint32_t kernel_number, Size input_size,
             Size kernel_size, Size padding_size, Size stride_size,
             float relu_alpha)
      : Data(conv_type, batch_size, num_channels, kernel_number, input_size,
             kernel_size, padding_size, stride_size, relu_alpha),
        weights_shape(_weights_shape()) {
    // TODO: ask for this
    // The +1 is required as ternarize_* does an off-by-one access
    randomize(x, x_shape.size + 1, has_ternary_input(conv_type));

    weights = alloc::alloc<float>(weights_shape.size);
    randomize(weights, weights_shape.size, has_ternary_weights(conv_type));

    for (size_t i = 0; i < quant_threshold_size; ++i)
      quant_threshold[i] = 0.5;
  }

  ~VerifyData() { free(weights); }
};

std::vector<std::string> ConvNames = {"TAB_TNN", "TAB_TBN", "TAB_BTN",
                                      "TAB_BNN"};

void verify() {
  const int batch_size = 2;
  const int ReLU_alpha = 1;
  const int CaseN = 9;
  const int CaseW = 8;
  int TestCases[CaseN][CaseW] = {
      //  c, h,   w, kn, kh, kw, p, s,
      {1, 2, 2, 1, 3, 3, 1, 1},       {64, 12, 16, 64, 1, 1, 0, 1},
      {32, 12, 16, 52, 1, 1, 0, 2},   {256, 56, 56, 10, 3, 3, 1, 1},
      {160, 64, 56, 32, 3, 3, 0, 2},  {325, 36, 25, 125, 5, 7, 3, 4},
      {32, 1, 1, 120, 1, 1, 0, 1},    {512, 1, 1, 1024, 1, 1, 0, 1},
      {1024, 1, 1, 1640, 1, 1, 2, 3},
  };

  for (int icase = 0; icase < CaseN; icase++) {
    auto num_channels = TestCases[icase][0];
    auto input_height = TestCases[icase][1];
    auto input_width = TestCases[icase][2];
    Size input_size = Size(input_height, input_width);
    auto kernel_number = TestCases[icase][3];
    auto kernel_height = TestCases[icase][4];
    auto kernel_width = TestCases[icase][5];
    Size kernel_size = Size(kernel_height, kernel_width);
    auto _padding_size = TestCases[icase][6];
    Size padding_size = Size(_padding_size, _padding_size);
    auto _stride_size = TestCases[icase][7];
    Size stride_size = Size(_stride_size, _stride_size);

    for (auto conv_type : convolution_types) {
      auto data = VerifyData(conv_type, batch_size, num_channels, kernel_number,
                             input_size, kernel_size, padding_size, stride_size,
                             ReLU_alpha);
      if (has_ternary_weights(conv_type))
        baseline::ternarize_NCHW_to_NHWCB(
            data.x, 0, 0, data.quant_threshold, data.kernel_number,
            data.num_channels, data.kernel_size.height, data.kernel_size.width,
            data.quant_weights);
      else
        baseline::binarize_NCHW_to_NHWC(
            data.x, 0, 0, data.kernel_number, data.num_channels,
            data.kernel_size.height, data.kernel_size.width,
            data.quant_weights);

      if (conv_type == ConvolutionType::BTN)
        baseline::btn_cnt_w2(data.quant_weights, data.num_channels,
                             data.kernel_number, data.kernel_size.height,
                             data.kernel_size.width, data.btn_cnt);

      switch (conv_type) {
      case ConvolutionType::TNN:
      case ConvolutionType::TBN:
        baseline::conv(conv_type, data.btn_cnt, data.x, data.input_size.height,
                       data.input_size.width, data.padding_size.height,
                       data.padding_size.width, data.quant_threshold,
                       data.num_channels, data.quant_weights, batch_size,
                       data.stride_size.height, data.stride_size.height,
                       data.kernel_number, data.kernel_size.height,
                       data.kernel_size.width, ReLU_alpha, data.y);
        break;
      case ConvolutionType::BTN:
      case ConvolutionType::BNN:
        baseline::conv(conv_type, data.btn_cnt, data.x, data.input_size.height,
                       data.input_size.width, data.padding_size.height,
                       data.padding_size.width, data.quant_threshold,
                       data.num_channels, data.quant_weights, batch_size,
                       data.stride_size.height, data.stride_size.width,
                       data.kernel_number, data.kernel_size.height,
                       data.kernel_size.width, ReLU_alpha, data.y);
        break;
      default:
        assert(0);
      }

      std::vector<float> px = DirectPad(
          data.x, data.padding_size.height, data.padding_size.width, batch_size,
          data.num_channels, data.input_size.height, data.input_size.width);
      std::vector<float> ref_y = DirectConv2d_FP32(
          px.data(), data.weights, data.stride_size.height,
          data.stride_size.width, batch_size, data.num_channels,
          data.packed_input_size.height, data.packed_input_size.width,
          data.kernel_number, data.kernel_size.height, data.kernel_size.width);

      // Compare the conv results to ensure the functions are correct
      int cmp;
      if ((data.padding_size.width > 0 || data.padding_size.height > 0) &&
          !has_ternary_input(conv_type))
        // BTN and BNN regard the padded zeros as 1s because binary quantization
        // only has (+1, -1) no zeros. So we only compare the central part of
        // conv results here, excluding the zero padding part.
        cmp = Compare_Tensor_BNN_Padding(
            data.y, ref_y.data(), batch_size, data.kernel_number,
            data.y_shape.height, data.y_shape.width, data.padding_size.height,
            data.padding_size.width);
      else {
        cmp = Compare_Tensor_NHWC(data.y, ref_y.data(), batch_size,
                                  data.kernel_number, data.y_shape.height,
                                  data.y_shape.width);
      }

      if (cmp > 0)
        std::cout << "Test Case " << icase
                  << " kernel: " << data.kernel_size.width << "X"
                  << data.kernel_size.height << " "
                  << convolution_name(conv_type) << " Passed!" << std::endl;
      else {
        std::cout << "Test Case " << icase
                  << " kernel: " << data.kernel_size.width << "X"
                  << data.kernel_size.height << " "
                  << convolution_name(conv_type) << " Failed!" << std::endl;
        exit(1);
      }
    }
  }
}
