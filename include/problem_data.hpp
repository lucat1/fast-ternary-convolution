#pragma once
#include "problem_parameters.hpp"

class Matrix2D : public Size {
public:
  Matrix2D(size_t height, size_t width);

  size_t size();
};

class Matrix4D : public Size {
public:
  size_t fst_dim;
  size_t snd_dim;

  Matrix4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width);

  size_t size();
};

class Data : public Parameters {
public:
  float *x;
  float *y;

  Data(ConvolutionType conv_type, uint32_t batch_size, uint32_t num_channels,
       uint32_t kernel_number, Size input_size, Size kernel_size,
       Size padding_size, Size stride_size);

  Matrix4D x_shape();
  Matrix4D y_shape();
};
