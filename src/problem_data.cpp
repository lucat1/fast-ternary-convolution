#include "problem_data.hpp"
#include "common.hpp"

using namespace std;

Data::Data(ConvolutionType conv_type, size_t batch_size, size_t channels,
           size_t input_h, size_t input_w, size_t kernel_n, size_t kernel_h,
           size_t kernel_w, size_t padding_h, size_t padding_w, size_t stride_h,
           size_t stride_w, float relu_alpha)
    : conv_type(conv_type), batch_size(batch_size), channels(channels),
      input_h(input_h), input_w(input_w), kernel_n(kernel_n),
      kernel_h(kernel_h), kernel_w(kernel_w), padding_h(padding_h),
      padding_w(padding_w), stride_h(stride_h), stride_w(stride_w),
      relu_alpha(relu_alpha),
      input(batch_size, input_h, input_w, channels, false),
      threshold(batch_size, false),
      kernel(kernel_n, kernel_h, kernel_w,
             (channels % 64) ? (channels / 64 + 1) : (channels / 64), BITS,
             false) {}

Data::Data(ConvolutionType conv_type, InfraParameters p, float relu_alpha)
    : Data(conv_type, p.batch_size, p.num_channels, p.input_height,
           p.input_width, p.kernel_number, p.kernel_height, p.kernel_width,
           p.padding_size, p.padding_size, p.stride_size, p.stride_size,
           relu_alpha) {}
