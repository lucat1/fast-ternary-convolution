#include "verify.hpp"
#include "common.hpp"
#include "problem_data.hpp"

// TODO: rework these functions and move them in this file
#include "alloc.hpp"
#include "impl/baseline/quantize.hpp"
#include "problem_data.hpp"
#include "verify_util.hpp"

#include <iomanip>
#include <random>

std::vector<InfraParameters> test_cases = {
    {1, 2, 2, 1, 3, 3, 1, 1},       {64, 12, 16, 64, 1, 1, 0, 1},
    {32, 12, 16, 52, 1, 1, 0, 2},   {256, 56, 56, 10, 3, 3, 1, 1},
    {160, 64, 56, 32, 3, 3, 0, 2},  {325, 36, 25, 125, 5, 7, 3, 4},
    {32, 1, 1, 120, 1, 1, 0, 1},    {512, 1, 1, 1024, 1, 1, 0, 1},
    {1024, 1, 1, 1640, 1, 1, 2, 3},
};

class VerificationData : public Data {
private:
  Matrix4D _weights_shape() {
    return Matrix4D(kernel_number, kernel_size.height, kernel_size.width,
                    num_channels);
  }
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;

  void randomize(float *dst, size_t size, bool ternary) {
    if (ternary)
      for (size_t i = 0; i < size; ++i)
        dst[i] = distribution(generator);
    else
      for (size_t i = 0; i < size; ++i) {
        auto v = distribution(generator);
        dst[i] = v == 0 ? 1 : v;
      }
  }

public:
  float *weights;

  Matrix4D weights_shape;

  VerificationData(ConvolutionType conv_type, uint32_t batch_size,
                   uint32_t num_channels, uint32_t kernel_number,
                   Size input_size, Size kernel_size, Size padding_size,
                   Size stride_size, float relu_alpha)
      : Data(conv_type, batch_size, num_channels, kernel_number, input_size,
             kernel_size, padding_size, stride_size, relu_alpha),
        weights_shape(_weights_shape()) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    // TODO: ask for this
    // The +1 is required as ternarize_* does an off-by-one access
    randomize(x, x_shape.size + 1, has_ternary_input(conv_type));

    weights = alloc::alloc<float>(weights_shape.size);
    randomize(weights, weights_shape.size, has_ternary_weights(conv_type));

    for (size_t i = 0; i < quant_threshold_size; ++i)
      quant_threshold[i] = 0.5;
  }

  ~VerificationData() {
    if (weights != nullptr)
      free(weights);

    weights = nullptr;
  }
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
