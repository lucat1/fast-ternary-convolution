#include "verify.hpp"
#include "common.hpp"
#include "problem_data.hpp"

// TODO: rework these functions and move them in this file
#include "impl/baseline/quantize.hpp"
#include "measure.hpp"
#include "problem_data.hpp"
#include "verify_util.hpp"

#include <iomanip>
#include <random>

std::vector<InfraParameters> test_cases = {
    {1, 2, 2, 2, 1, 3, 3, 1, 1},       {64, 2, 12, 16, 64, 1, 1, 0, 1},
    {32, 2, 12, 16, 52, 1, 1, 0, 2},   {256, 2, 56, 56, 10, 3, 3, 1, 1},
    {160, 2, 64, 56, 32, 3, 3, 0, 2},  {325, 2, 36, 25, 125, 5, 7, 3, 4},
    {32, 2, 1, 1, 120, 1, 1, 0, 1},    {512, 2, 1, 1, 1024, 1, 1, 0, 1},
    {1024, 2, 1, 1, 1640, 1, 1, 2, 3},
};

class VerificationData : public Data {
private:
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
  Matrix4D<float> weights;

  VerificationData(Parameters p)
      : Data(p), weights(kernel_number, kernel_size.height, kernel_size.width,
                         num_channels, false) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    // TODO: ask for this
    // The +1 is required as ternarize_* does an off-by-one access
    randomize(x.data, x.shape.size + 1, has_ternary_input(conv_type));

    randomize(weights.data, weights.shape.size, has_ternary_weights(conv_type));

    for (size_t i = 0; i < quant_threshold.size; ++i)
      quant_threshold.data[i] = 0.5;
  }
};

void verify(Registry r) {
  const int relu_alpha = 1;
  auto m = Measure::get_instance();

  for (auto impl : r.implementations()) {
    size_t passed = 0, failed = 0,
           total = test_cases.size() * convolution_types.size();
    for (auto tc : test_cases) {
      //for (auto conv_type : convolution_types) {
      auto conv_type = ConvolutionType::TNN;
        auto data = VerificationData(Parameters(
            conv_type, tc.batch_size, tc.num_channels, tc.kernel_number,
            relu_alpha, {tc.input_height, tc.input_width},
            {tc.kernel_height, tc.kernel_width},
            {tc.padding_size, tc.padding_size},
            {tc.stride_size, tc.stride_size}));
        if (has_ternary_weights(conv_type))
          baseline::ternarize_NCHW_to_NHWCB(
              data.weights.data, 0, 0, data.quant_threshold.data,
              data.kernel_number, data.num_channels, data.kernel_size.height,
              data.kernel_size.width, data.quant_weights.data);
        else
          baseline::binarize_NCHW_to_NHWC(
              data.weights.data, 0, 0, data.kernel_number, data.num_channels,
              data.kernel_size.height, data.kernel_size.width,
              data.quant_weights.data);

        if (conv_type == ConvolutionType::BTN)
          baseline::btn_cnt_w2(data.quant_weights.data, data.num_channels,
                               data.kernel_number, data.kernel_size.height,
                               data.kernel_size.width, data.btn_cnt.data);

        impl.fn(conv_type, data.btn_cnt.data, data.x.data,
                data.input_size.height, data.input_size.width,
                data.padding_size.height, data.padding_size.width,
                data.quant_threshold.data, data.num_channels,
                data.quant_weights.data, data.batch_size,
                data.stride_size.height, data.stride_size.height,
                data.kernel_number, data.kernel_size.height,
                data.kernel_size.width, relu_alpha, data.y.data);

        std::vector<float> px = DirectPad(
            data.x.data, data.padding_size.height, data.padding_size.width,
            data.batch_size, data.num_channels, data.input_size.height,
            data.input_size.width);
        std::vector<float> ref_y = DirectConv2d_FP32(
            px.data(), data.weights.data, data.stride_size.height,
            data.stride_size.width, data.batch_size, data.num_channels,
            data.packed_input_size.height, data.packed_input_size.width,
            data.kernel_number, data.kernel_size.height,
            data.kernel_size.width);

        // Compare the conv results to ensure the functions are correct
        int cmp;
        if ((data.padding_size.width > 0 || data.padding_size.height > 0) &&
            !has_ternary_input(conv_type))
          // BTN and BNN regard the padded zeros as 1s because binary
          // quantization only has (+1, -1) no zeros. So we only compare the
          // central part of conv results here, excluding the zero padding part.
          cmp = Compare_Tensor_BNN_Padding(
              data.y.data, ref_y.data(), data.batch_size, data.kernel_number,
              data.y.shape.height, data.y.shape.width, data.padding_size.height,
              data.padding_size.width);
        else
          cmp = Compare_Tensor_NHWC(data.y.data, ref_y.data(), data.batch_size,
                                    data.kernel_number, data.y.shape.height,
                                    data.y.shape.width);

        if (cmp > 0)
          passed++;
        else {
          failed++;
          cout << "[" << (passed + failed) << "/" << total
               << "] Failed test case" << endl;
        }
        m->reset();
	//}
    }
    cout << setw(name_space) << impl.name << " :: " << passed << "/" << total
         << " tests passed" << endl;
  }
}
