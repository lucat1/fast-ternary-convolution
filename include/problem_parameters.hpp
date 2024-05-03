#pragma once

#include "common.hpp"

#include <cstdint>

using namespace std;

class Shape2D {
public:
  const size_t height;
  const size_t width;

  // precomputed height * width
  const size_t size;

  Shape2D(size_t height, size_t width);
};

// Holds the problem's parameters
class Parameters {
public:
  ConvolutionType conv_type;
  size_t batch_size;
  size_t num_channels;
  size_t kernel_number;
  float relu_alpha;

  Shape2D input_size;
  Shape2D kernel_size;
  Shape2D padding_size;
  Shape2D stride_size;

  Parameters(ConvolutionType conv_type, uint32_t batch_size,
             uint32_t num_channels, uint32_t kernel_number, float relu_alpha,
             Shape2D input_size, Shape2D kernel_size, Shape2D padding_size,
             Shape2D stride_size);

  virtual ~Parameters() = default;
};
