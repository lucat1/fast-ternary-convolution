#pragma once

#include "alloc.hpp"
#include "problem_data.hpp"

#include <random>
#include <vector>

class VerificationData : public Data {
private:
  Matrix4D _weights_shape() {
    return Matrix4D(kernel_number, kernel_size.height, kernel_size.width,
                    num_channels);
  }
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;

  void randomize(float *dst, size_t size, bool ternary) {
    if (ternary)
      for (size_t i = 0; i < size; ++i)
        dst[i] = distribution(generator);
    else
      for (size_t i = 0; i < size; ++i) {
        auto v = distribution(generator);
        dst[i] = v == 0 ? 1 : v;
      }
  }

public:
  float *weights;

  Matrix4D weights_shape;

  VerificationData(ConvolutionType conv_type, uint32_t batch_size,
                   uint32_t num_channels, uint32_t kernel_number,
                   Size input_size, Size kernel_size, Size padding_size,
                   Size stride_size, float relu_alpha)
      : Data(conv_type, batch_size, num_channels, kernel_number, input_size,
             kernel_size, padding_size, stride_size, relu_alpha),
        weights_shape(_weights_shape()) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    // TODO: ask for this
    // The +1 is required as ternarize_* does an off-by-one access
    randomize(x, x_shape.size + 1, has_ternary_input(conv_type));

    weights = alloc::alloc<float>(weights_shape.size);
    randomize(weights, weights_shape.size, has_ternary_weights(conv_type));

    for (size_t i = 0; i < quant_threshold_size; ++i)
      quant_threshold[i] = 0.5;
  }

  ~VerificationData() {
    if (weights != nullptr)
      free(weights);

    weights = nullptr;
  }
};

void verify();
