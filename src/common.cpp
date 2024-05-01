#include "common.hpp"
#include <map>

map<ConvolutionType, string> __ctn = {{ConvolutionType::TNN, "TNN"},
                                      {ConvolutionType::TBN, "TBN"},
                                      {ConvolutionType::BTN, "BTN"},
                                      {ConvolutionType::BNN, "BNN"}};

string convolution_name(ConvolutionType t) { return __ctn[t]; }

InfraParameters::InfraParameters(uint32_t num_channels, uint32_t kernel_number,
                                 size_t input_height, size_t input_width,
                                 size_t kernel_height, size_t kernel_width,
                                 size_t padding_size, size_t stride_size)
    : num_channels(num_channels), kernel_number(kernel_number),
      input_height(input_height), input_width(input_width),
      kernel_height(kernel_height), kernel_width(kernel_width),
      padding_size(padding_size), stride_size(stride_size) {}
