#include "verify.hpp"
#include "common.hpp"
#include "problem_data.hpp"

#include "impl/baseline/quantize.hpp"
#include "impl/baseline_nhwc/quantize.hpp"
#include "measure.hpp"
#include "problem_data.hpp"
#include "tensor.hpp"
// TODO: rework these functions and move them in this file
#include "verify_util.hpp"

#include <cstring>
#include <iomanip>
#include <random>

// uint32_t num_channels;
// uint32_t batch_size;
// size_t input_height;
// size_t input_width;
// uint32_t kernel_number;
// size_t kernel_height;
// size_t kernel_width;
// size_t padding_size;
// size_t stride_size;
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
  Tensor4D<float> non_quantized_kernel;
  Tensor1D<float> kernel_threshold;

  VerificationData(ConvolutionType conv_type, InfraParameters p,
                   float relu_alpha)
      : Data(conv_type, p, relu_alpha),
        non_quantized_kernel(kernel_n, kernel_h, kernel_w, channels, false),
        kernel_threshold(kernel_n, false) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    randomize(input.data, input.dim1 * input.dim2 * input.dim3 * input.dim4,
              has_ternary_input(conv_type));

    randomize(non_quantized_kernel.data,
              non_quantized_kernel.dim1 * non_quantized_kernel.dim2 *
                  non_quantized_kernel.dim3 * non_quantized_kernel.dim4,
              has_ternary_weights(conv_type));

    for (size_t i = 0; i < threshold.size; ++i)
      threshold.data[i] = 0.5;
    for (size_t i = 0; i < kernel_threshold.size; ++i)
      kernel_threshold.data[i] = 0.5;
  }
};

void verify(Registry r) {
  const int relu_alpha = 1;
  auto m = Measure::get_instance();

  for (auto impl : r.implementations()) {
    size_t passed = 0, failed = 0,
           total = test_cases.size() * convolution_types.size();
    for (auto tc : test_cases) {
      // for (auto conv_type : convolution_types) {
      auto conv_type = ConvolutionType::TNN;
      auto data = VerificationData(conv_type, tc, relu_alpha);
      // if (has_ternary_weights(conv_type))
      auto kernel = baseline_nhwc::ternarize(data.non_quantized_kernel,
                                             data.kernel_threshold, 0, 0);
      assert(kernel.dim1 == data.kernel.dim1);
      assert(kernel.dim2 == data.kernel.dim2);
      assert(kernel.dim3 == data.kernel.dim3);
      assert(kernel.dim4 == data.kernel.dim4);
      assert(kernel.dim5 == data.kernel.dim5);
      memcpy(data.kernel.data, kernel.data,
             data.kernel.dim1 * data.kernel.dim2 * data.kernel.dim3 *
                 data.kernel.dim4 * data.kernel.dim5);
      // else
      //   baseline::binarize_NCHW_to_NHWC(
      //       data.weights.data, 0, 0, data.kernel_number, data.num_channels,
      //       data.kernel_size.height, data.kernel_size.width,
      //       data.quant_weights.data);

      // if (conv_type == ConvolutionType::BTN)
      //   baseline::btn_cnt_w2(data.quant_weights.data, data.num_channels,
      //                        data.kernel_number, data.kernel_size.height,
      //                        data.kernel_size.width, data.btn_cnt.data);

      // only TNN for now
      auto y =
          impl.fn(data.input, data.threshold, data.padding_h, data.padding_w,
                  data.kernel, data.stride_h, data.stride_w, data.relu_alpha);

      cout << y.data << endl;
      size_t y_n = y.dim1;
      size_t y_c = y.dim2;
      size_t y_h = y.dim3;
      size_t y_w = y.dim4;

      const int packH = data.input_h + 2 * data.padding_h;
      const int packW = data.input_w + 2 * data.padding_w;
      std::vector<float> px = DirectPadNCHW(
          data.input.data, data.padding_h, data.padding_w, data.batch_size,
          data.channels, data.input_h, data.input_w);
      std::vector<float> ref_y = DirectConv2d_FP32NCHW(
          px.data(), data.non_quantized_kernel.data, data.stride_h,
          data.stride_w, data.batch_size, data.channels, packH, packW,
          data.kernel_n, data.kernel_h, data.kernel_w);
      cout << ref_y.data() << endl;

      // Compare the conv results to ensure the functions are correct
      int cmp;
      if ((data.padding_w > 0 || data.padding_h > 0) &&
          !has_ternary_input(conv_type))
        // BTN and BNN regard the padded zeros as 1s because binary
        // quantization only has (+1, -1) no zeros. So we only compare the
        // central part of conv results here, excluding the zero padding part.
        cmp = Compare_Tensor_BNN_Padding(y.data, ref_y.data(), y_n, y_c, y_h,
                                         y_w, data.padding_h, data.padding_w);
      else
        cmp = Compare_Tensor_NCHW(y.data, ref_y.data(), y_n, y_c, y_h, y_w);

      if (cmp > 0)
        passed++;
      else {
        failed++;
        cout << "[" << (passed + failed) << "/" << total << "] Failed test case"
             << endl;
      }
      m->reset();
      //}
    }
    cout << setw(name_space) << impl.name << " :: " << passed << "/" << total
         << " tests passed" << endl;
  }
}
