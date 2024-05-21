#include "verify.hpp"
#include "common.hpp"
#include "direct.hpp"
#include "problem_data.hpp"

#include "impl/baseline_nchw/quantize.hpp"
#include "measure.hpp"
#include "problem_data.hpp"
#include "tensor.hpp"

#include <cstring>
#include <iomanip>
#include <random>

size_t compare_nhwc(Tensor4D<float> &x, Tensor4D<float> &y) {
  assert(x.dim1 == y.dim1);
  assert(x.dim2 == y.dim2);
  assert(x.dim3 == y.dim3);
  assert(x.dim4 == y.dim4);
  for (size_t n = 0; n < x.dim1; n++) {
    for (size_t h = 0; h < x.dim2; h++) {
      for (size_t w = 0; w < x.dim3; w++) {
        for (size_t c = 0; c < x.dim4; c++) {
          float xx = x.get(n, h, w, c);
          float yy = y.get(n, h, w, c);
          if (abs(xx - yy) > 0.01) {
            std::cout << "n: " << n << ", h: " << h << ", w: " << w
                      << ", c: " << c;
            std::cout << ", x: " << xx << ", y: " << yy << std::endl;
            return -1;
          }
        }
      }
    }
  }
  return 1;
}

// uint32_t channels;
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
  // The kernel values before quantization
  Tensor4D<float> real_kernel;
  Tensor1D<float> kernel_threshold;

  VerificationData(ConvolutionType conv_type, DataOrder data_order,
                   InfraParameters p, float relu_alpha)
      : Data(conv_type, data_order, p, relu_alpha),
        // The real kernel is always in shape (KN, PC, KH, KW) beacuse that's
        // what direct_conv expects. It always gets quantized using the
        // baseline_nchw ternarize function
        real_kernel(kernel_n, channels, kernel_h, kernel_w, false),
        kernel_threshold(kernel.dim1, false) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    randomize(input.data, input.dim1 * input.dim2 * input.dim3 * input.dim4,
              has_ternary_input(conv_type));

    randomize(real_kernel.data,
              real_kernel.dim1 * real_kernel.dim2 * real_kernel.dim3 *
                  real_kernel.dim4,
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
           total = test_cases.size() /* * convolution_types.size() */;
    for (auto tc : test_cases) {
      // for (auto conv_type : convolution_types) {
      auto conv_type = ConvolutionType::TNN;
      auto data = VerificationData(conv_type, impl.data_order, tc, relu_alpha);
      // Sanity checks on the input shapes
      assert(data.input.dim1 == tc.batch_size);
      assert(data.kernel.dim1 == tc.kernel_number);
      if (impl.data_order == DataOrder::NHWC) {
        assert(data.input.dim2 == tc.input_height);
        assert(data.input.dim3 == tc.input_width);
        assert(data.input.dim4 == tc.channels);
      } else {
        assert(data.input.dim2 == tc.channels);
        assert(data.input.dim3 == tc.input_height);
        assert(data.input.dim4 == tc.input_width);
      }

      // Sanyt check to verify that the real_kernel is NCHW
      assert(data.real_kernel.dim1 == data.kernel_n);
      assert(data.real_kernel.dim2 == data.channels);
      assert(data.real_kernel.dim3 == data.kernel_h);
      assert(data.real_kernel.dim4 == data.kernel_w);
      assert(data.kernel_threshold.size == data.real_kernel.dim1);
      // baseline::ternarize_NCHW_to_NHWCB(
      //     data.real_kernel.data, 0, 0, data.kernel_threshold.data,
      //     data.kernel_n, data.channels, data.kernel_h, data.kernel_w,
      //     data.kernel.data);
      Tensor5D<int64_t> kernel = baseline_nchw::ternarize(
          data.real_kernel, data.kernel_threshold, 0, 0);
      // Sanity checks on the kernel size, before copying data over
      assert(kernel.dim1 == data.kernel.dim1);
      assert(kernel.dim2 == data.kernel.dim2);
      assert(kernel.dim3 == data.kernel.dim3);
      assert(kernel.dim4 == data.kernel.dim4);
      assert(kernel.dim5 == data.kernel.dim5);
      memcpy(data.kernel.data, kernel.data,
             sizeof(int64_t) * data.kernel.dim1 * data.kernel.dim2 *
                 data.kernel.dim3 * data.kernel.dim4 * data.kernel.dim5);

      // NOTE: for when we add binary layers back, we should quantize the
      // weights differently based on the type
      // if (has_ternary_weights(conv_type))
      // -- what we're already doing for TNN
      // else
      //   baseline::binarize_NCHW_to_NHWC(
      //       data.weights.data, 0, 0, data.kernel_number, data.channels,
      //       data.kernel_size.height, data.kernel_size.width,
      //       data.quant_weights.data);
      // if (conv_type == ConvolutionType::BTN)
      //   baseline::btn_cnt_w2(data.quant_weights.data, data.channels,
      //                        data.kernel_number, data.kernel_size.height,
      //                        data.kernel_size.width, data.btn_cnt.data);

      auto output =
          impl.fn(data.input, data.threshold, data.padding_h, data.padding_w,
                  data.kernel, data.stride_h, data.stride_w, data.relu_alpha);

      // Data for the direct convolution is always provided in NCHW shape. Thus,
      // we need to reshape when the input is generated in NHWC. The input is
      // already generated in the appropriate shape to make the benchmarking
      // faster.
      // @all: I had to separate this in an if-then-else otherwise C++
      // continues complaining (it's not happy if I use a ternary operator)
      Tensor4D<float> reshaped_input(data.batch_size, data.channels,
                                     data.input_h, data.input_w, false);
      if (data.data_order == DataOrder::NHWC) {
        Tensor4D<float> reshaped = reshape_nhwc_nchw(data.input);
        assert(reshaped.dim1 == reshaped_input.dim1);
        assert(reshaped.dim2 == reshaped_input.dim2);
        assert(reshaped.dim3 == reshaped_input.dim3);
        assert(reshaped.dim4 == reshaped_input.dim4);
        memcpy(reshaped_input.data, reshaped.data,
               sizeof(float) * data.input.dim1 * data.input.dim2 *
                   data.input.dim3 * data.input.dim4);
      } else {
        assert(data.input.dim1 == reshaped_input.dim1);
        assert(data.input.dim2 == reshaped_input.dim2);
        assert(data.input.dim3 == reshaped_input.dim3);
        assert(data.input.dim4 == reshaped_input.dim4);
        memcpy(reshaped_input.data, data.input.data,
               sizeof(float) * data.input.dim1 * data.input.dim2 *
                   data.input.dim3 * data.input.dim4);
      }
      Tensor4D<float> padded_input =
          direct_pad(reshaped_input, data.padding_h, data.padding_w);
      Tensor4D<float> reference_output(output.dim1, output.dim2, output.dim3,
                                       output.dim4, false);
      std::vector<float> ref_out =
          direct_conv(padded_input.data, data.real_kernel.data, data.stride_h,
                      data.stride_w, padded_input.dim1, padded_input.dim2,
                      padded_input.dim3, padded_input.dim4, data.kernel_n,
                      data.kernel_h, data.kernel_w);

      assert(ref_out.size() ==
             output.dim1 * output.dim2 * output.dim3 * output.dim4);
      memcpy(reference_output.data, ref_out.data(),
             sizeof(float) * ref_out.size());

      // Compare the conv results to ensure the functions are correct
      int cmp;
      // NOTE: for when we add binary layers back, comparison needs to be done
      // differently
      // if ((data.padding_w > 0 || data.padding_h > 0) &&
      //     !has_ternary_input(conv_type))
      //   // BTN and BNN regard the padded zeros as 1s because binary
      //   // quantization only has (+1, -1) no zeros. So we only compare the
      //   // central part of conv results here, excluding the zero padding
      //   part. cmp = Compare_Tensor_BNN_Padding(y.data, ref_y.data(), y_n,
      //   y_c, y_h,
      //                                    y_w, data.padding_h,
      //                                    data.padding_w);
      // else

      cmp = compare_nhwc(output, reference_output);

      if (cmp > 0) {
        passed++;
      } else {
        failed++;
        cout << "[" << (passed + failed) << "/" << total << "] Failed test case"
             << endl;
      }
      m->reset();
      //}
    }
    cout << setw(impl_name_space) << impl.name << " :: " << passed << "/"
         << total << " tests passed" << endl;
  }
}
