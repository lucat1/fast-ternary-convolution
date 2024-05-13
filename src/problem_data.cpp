#include "problem_data.hpp"
#include "common.hpp"

Data::Data(ConvolutionType conv_type, DataOrder data_order, size_t batch_size,
           size_t channels, size_t input_h, size_t input_w, size_t kernel_n,
           size_t kernel_h, size_t kernel_w, size_t padding_h, size_t padding_w,
           size_t stride_h, size_t stride_w, float relu_alpha)
    : conv_type(conv_type), data_order(data_order), batch_size(batch_size),
      channels(channels), input_h(input_h), input_w(input_w),
      kernel_n(kernel_n), kernel_h(kernel_h), kernel_w(kernel_w),
      padding_h(padding_h), padding_w(padding_w), stride_h(stride_h),
      stride_w(stride_w), relu_alpha(relu_alpha),

      input(batch_size, nchw_or_nhwc(channels, input_h),
            nchw_or_nhwc(input_h, input_w), nchw_or_nhwc(input_w, channels),
            false),
      threshold(batch_size, false),
      // NOTE: the kernel is, regardless of data_order, always in the shape:
      // (KN, KH, KW, PC, B), where PC is packed channels, and B=2
      kernel(kernel_n, kernel_h, kernel_w, int64s_for_bits(channels), 2,
             false) {}

Data::Data(ConvolutionType conv_type, DataOrder data_order, InfraParameters p,
           float relu_alpha)
    : Data(conv_type, data_order, p.batch_size, p.channels, p.input_height,
           p.input_width, p.kernel_number, p.kernel_height, p.kernel_width,
           p.padding_size, p.padding_size, p.stride_size, p.stride_size,
           relu_alpha) {}
