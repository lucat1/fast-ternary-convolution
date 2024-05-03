#include "problem_parameters.hpp"

Shape2D::Shape2D(size_t height, size_t width)
    : height(height), width(width), size(height * width) {}

Parameters::Parameters(ConvolutionType conv_type, uint32_t batch_size,
                       uint32_t num_channels, uint32_t kernel_number,
                       float relu_alpha, Shape2D input_size,
                       Shape2D kernel_size, Shape2D padding_size,
                       Shape2D stride_size)
    : conv_type(conv_type), batch_size(batch_size), num_channels(num_channels),
      kernel_number(kernel_number), relu_alpha(relu_alpha),
      input_size(input_size), kernel_size(kernel_size),
      padding_size(padding_size), stride_size(stride_size) {}
