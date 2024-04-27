#include "problem_parameters.hpp"

Size::Size(size_t height, size_t width) : height(height), width(width) {}

Parameters::Parameters(ConvolutionType conv_type, uint32_t batch_size,
                       uint32_t num_channels, uint32_t kernel_number,
                       Size input_size, Size kernel_size, Size padding_size,
                       Size stride_size)
    : conv_type(conv_type), batch_size(batch_size), num_channels(num_channels),
      kernel_number(kernel_number), input_size(input_size),
      kernel_size(kernel_size), padding_size(padding_size),
      stride_size(stride_size) {}
