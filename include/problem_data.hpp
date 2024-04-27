#pragma once
#include "problem_parameters.hpp"

class Matrix2D : public Size {
private:
  size_t _size();

public:
  size_t size;

  Matrix2D(size_t height, size_t width);
};

class Matrix4D : public Size {
private:
  size_t _size();

public:
  size_t fst_dim;
  size_t snd_dim;

  size_t size;

  Matrix4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width);
};

class Matrix5D : public Size {
private:
  size_t _size();

public:
  size_t fst_dim;
  size_t snd_dim;
  size_t trd_dim;

  size_t size;

  Matrix5D(size_t fst_dim, size_t snd_dim, size_t trd_dim, size_t height,
           size_t width);
};

class Data : public Parameters {
private:
  Size packed_size(Size base);
  Matrix4D _x_shape();
  size_t _quant_threshold_size();
  size_t _btn_cnt_size();
  Matrix5D _quant_weights_shape();
  Matrix4D _y_shape();

public:
  // input, output and intermediate arrays
  float *x;
  float *quant_threshold;
  int64_t *quant_weights;
  int *btn_cnt;
  float *y;

  // extra parameters
  float relu_alpha;

  Size packed_input_size;
  Size packed_kernel_size;

  // facts about data contained in the class
  Matrix4D x_shape;
  size_t quant_threshold_size;
  Matrix5D quant_weights_shape;
  size_t btn_cnt_size;
  Matrix4D y_shape;

  Data(ConvolutionType conv_type, uint32_t batch_size, uint32_t num_channels,
       uint32_t kernel_number, Size input_size, Size kernel_size,
       Size padding_size, Size stride_size, float relu_alpha);

  ~Data();
};
