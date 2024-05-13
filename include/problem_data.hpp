#pragma once
#include "common.hpp"
#include "tensor.hpp"

class Data {
public:
  ConvolutionType conv_type;

  size_t batch_size;
  size_t channels;
  size_t input_h;
  size_t input_w;
  size_t kernel_n;
  size_t kernel_h;
  size_t kernel_w;
  size_t padding_h;
  size_t padding_w;
  size_t stride_h;
  size_t stride_w;
  float relu_alpha;

  const Tensor4D<float> input;
  const Tensor1D<float> threshold;
  const Tensor5D<int64_t> kernel;

  Data(ConvolutionType conv_type, size_t batch_size, size_t channels,
       size_t input_h, size_t input_w, size_t kernel_n, size_t kernel_h,
       size_t kernel_w, size_t padding_h, size_t padding_w, size_t stride_h,
       size_t stride_w, float relu_alpha);
  Data(ConvolutionType conv_type, InfraParameters p, float relu_alpha);
};
