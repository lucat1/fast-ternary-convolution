#include "problem_parameters.hpp"

Size::Size(size_t height, size_t width) {
  this->height = height;
  this->width = width;
}

Parameters::Parameters(ConvolutionType conv_type, uint32_t batch_size,
                       uint32_t num_channels, uint32_t kernel_number,
                       Size input_size, Size kernel_size, Size padding_size,
                       Size stride_size)
    : input_size(0, 0), kernel_size(0, 0), padding_size(0, 0),
      stride_size(0, 0) {
  this->conv_type = conv_type;
  this->batch_size = batch_size;
  this->num_channels = num_channels;
  this->kernel_number = kernel_number;

  this->input_size = input_size;
  this->kernel_size = kernel_size;
  this->padding_size = padding_size;
  this->stride_size = stride_size;
}
