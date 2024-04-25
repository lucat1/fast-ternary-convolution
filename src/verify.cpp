#include "verify.hpp"
#include "common.hpp"
#include "impl/baseline/quantize.hpp"
#include "impl/baseline/tab.hpp"
#include "impl/baseline/utility.hpp"
#include "registry.hpp"
#include <chrono>
#include <numeric>

namespace verify {

int verify() {
  const int Batch_Size = 2;
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
  registry::env *test_env = new registry::env();
  test_env->batch_size = 2;
  // Get the reference matrix
  std::vector<float> TX =
      generate_array(16 * 64 * 224 * 224, true); // Ternary X: size = 51,380,224
  std::vector<float> BX = generate_array(16 * 64 * 224 * 224,
                                         false); // Binary X : size = 51,380,224
  std::vector<float> TW = generate_array(
      1024 * 1024 * 3 * 3, true); // Ternary Weights: size = 51,380,224
  std::vector<float> BW = generate_array(
      1024 * 1024 * 3 * 3, false); // Binary Weights : size = 51,380,224
  std::vector<float> Q_Threshold =
      std::vector<float>(1024, 0.5); // Quantization threshold for ternarization
  float *output;
  int64_t *QW;
  int *BTN_CNT;
  // Iterate on layer configurations
  for (int icase = 0; icase < CaseN; icase++) {
    // config the layer shape and relevant sizes
    test_env->num_channels = TestCases[icase][0];
    test_env->input_height = TestCases[icase][1];
    test_env->input_width = TestCases[icase][2];
    test_env->kernel_number = TestCases[icase][3];
    test_env->kernel_height = TestCases[icase][4];
    test_env->kernel_width = TestCases[icase][5];
    test_env->padding_size = TestCases[icase][6];
    test_env->stride_size = TestCases[icase][7];
    size_t output_size = registry::output_size(*test_env);
    output = registry::alloc<float>(output_size);
    // const int pri_channel = test_env->num_channels / CNTBITS;
    const int packed_height = test_env->input_height + 2 * test_env->padding_size;
    const int packed_width = test_env->input_width + 2 * test_env->padding_size;
    const int packed_channels = (test_env->num_channels % CNTBITS)
                                    ? ((test_env->num_channels / CNTBITS) + 1)
                                    : (test_env->num_channels / CNTBITS);
    QW = registry::alloc<int64_t>(test_env->batch_size * packed_height *
                                  packed_width * packed_channels * BITS);
    std::cout << "alloc size "
              << test_env->batch_size * packed_height * packed_width *
                     packed_channels * BITS
              << std::endl;

    // iterate on conv types
    std::vector<std::string> ConvNames = {"TAB_TNN", "TAB_TBN", "TAB_BTN",
                                          "TAB_BNN"};
    for (int iconv = 0; iconv < registry::conv_type::CONV_TYPES; iconv++) {
      // Get ref input x and weights w
      float *ref_x = NULL;
      float *ref_w = NULL;
      if (iconv == registry::conv_type::TNN) {
        ref_x = TX.data();
        ref_w = TW.data();
        ternarize_NCHW_to_NHWCB(TW.data(), 0, 0, Q_Threshold.data(),
                                test_env->kernel_number, test_env->num_channels,
                                test_env->input_height, test_env->input_width, QW);
        baseline::conv(registry::conv_type::TNN, NULL, TX.data(),
                       test_env->input_height, test_env->input_width,
                       test_env->padding_size, test_env->padding_size,
                       Q_Threshold.data(), test_env->num_channels, QW,
                       Batch_Size, test_env->stride_size, test_env->stride_size,
                       test_env->kernel_number, test_env->kernel_height,
                       test_env->kernel_width, ReLU_alpha, output);
      }
      if (iconv == registry::conv_type::TBN) {
        ref_x = TX.data();
        ref_w = BW.data();
        binarize_NCHW_to_NHWC(BW.data(), 0, 0, test_env->kernel_number,
                              test_env->num_channels, test_env->kernel_height,
                              test_env->kernel_width, QW);
        baseline::conv(registry::conv_type::TBN, nullptr, TX.data(),
                       test_env->input_height, test_env->input_width,
                       test_env->padding_size, test_env->padding_size,
                       Q_Threshold.data(), test_env->num_channels, QW,
                       Batch_Size, test_env->stride_size, test_env->stride_size,
                       test_env->kernel_number, test_env->kernel_height,
                       test_env->kernel_width, ReLU_alpha, output);
      }
      if (iconv == registry::conv_type::BTN) {
        ref_x = BX.data();
        ref_w = TW.data();
        ternarize_NCHW_to_NHWCB(TW.data(), 0, 0, Q_Threshold.data(),
                                test_env->batch_size, test_env->num_channels,
                                test_env->input_height, test_env->input_width, QW);
        BTN_CNT = registry::alloc<int>(test_env->kernel_number);
        btn_cnt_w2(QW, test_env->num_channels, test_env->kernel_number,
                   test_env->kernel_height, test_env->kernel_width, BTN_CNT);
        baseline::conv(registry::conv_type::BTN, BTN_CNT, TX.data(),
                       test_env->input_height, test_env->input_width,
                       test_env->padding_size, test_env->padding_size,
                       Q_Threshold.data(), test_env->num_channels, QW,
                       Batch_Size, test_env->stride_size, test_env->stride_size,
                       test_env->kernel_number, test_env->kernel_height,
                       test_env->kernel_width, ReLU_alpha, output);
        free(BTN_CNT);
      }
      if (iconv == registry::conv_type::BNN) {
        ref_x = BX.data();
        ref_w = BW.data();
        binarize_NCHW_to_NHWC(BW.data(), 0, 0, test_env->kernel_number,
                              test_env->num_channels, test_env->kernel_height,
                              test_env->kernel_width, QW);
        baseline::conv(registry::conv_type::BNN, nullptr, TX.data(),
                       test_env->input_height, test_env->input_width,
                       test_env->padding_size, test_env->padding_size,
                       Q_Threshold.data(), test_env->num_channels, QW,
                       Batch_Size, test_env->stride_size, test_env->stride_size,
                       test_env->kernel_number, test_env->kernel_height,
                       test_env->kernel_width, ReLU_alpha, output);
      }

      // Get reference conv result: direct conv on ref_x and ref_w

      std::vector<float> px = DirectPad(
          ref_x, test_env->padding_size, test_env->padding_size, Batch_Size,
          test_env->num_channels, test_env->input_height, test_env->input_width);
      // std::vector<float> DirectConv2d_FP32(float* x, float* w, int stride1,
      // int stride2, int N, int C, int paddedH, int paddedW, int KN, int KH,
      // int KW)
      int paddedh = test_env->input_height +
                    2 * test_env->padding_size; // height after zero padding
      int paddedw = test_env->input_width +
                    2 * test_env->padding_size; // width  after zero padding
      std::vector<float> ref_y = DirectConv2d_FP32(
          px.data(), ref_w, test_env->stride_size, test_env->stride_size,
          Batch_Size, test_env->num_channels, paddedh, paddedw,
          test_env->kernel_number, test_env->kernel_height,
          test_env->kernel_width);

      // Compare the conv results to ensure the functions are correct

      int cmp;
      int outh = (test_env->input_height + 2 * test_env->padding_size -
                  test_env->kernel_height + 1) /
                 test_env->stride_size; // The output height of y
      int outw = (test_env->input_width + 2 * test_env->padding_size -
                  test_env->kernel_width + 1) /
                 test_env->stride_size; // The output width  of y
      if ((test_env->padding_size > 0) &&
          ((iconv == registry::conv_type::BTN) ||
           (iconv == registry::conv_type::BNN)))
        // BTN and BNN regard the padded zeros as 1s because binary quantization
        // only has (+1, -1) no zeros. So we only compare the central part of
        // conv results here, excluding the zero padding part.
        cmp = Compare_Tensor_BNN_Padding(
            output, ref_y.data(), Batch_Size, test_env->kernel_number, outh,
            outw, test_env->padding_size, test_env->padding_size);
      else
        cmp = Compare_Tensor_NHWC(output, ref_y.data(), Batch_Size,
                                  test_env->kernel_number, outh, outw);
      if (cmp > 0)
        std::cout << "Test Case " << icase
                  << " kernel: " << test_env->kernel_width << "X"
                  << test_env->kernel_height << " " << ConvNames[iconv]
                  << " Passed!" << std::endl;
      else
        std::cout << "Test Case " << icase
                  << " kernel: " << test_env->kernel_width << "X"
                  << test_env->kernel_height << " " << ConvNames[iconv]
                  << " Failed!" << std::endl;
    }
    free(output);
    free(QW);
  }
  return 0;
}

} // namespace verify
