#pragma once

#include "common.hpp"

#include <cstdint>

using namespace std;

// Could have used a tuple, but this way the fields are declarative
class Size {
public:
  size_t height, width;

  Size(size_t height, size_t width);
};

// Holds the problem's parameters
class Parameters {
public:
  ConvolutionType conv_type;
  uint32_t batch_size;
  uint32_t num_channels;
  uint32_t kernel_number;

  Size input_size;
  Size kernel_size;
  Size padding_size;
  Size stride_size;

  Parameters(ConvolutionType conv_type, uint32_t batch_size,
             uint32_t num_channels, uint32_t kernel_number, Size input_size,
             Size kernel_size, Size padding_size, Size stride_size);

  virtual ~Parameters() = default;
};
