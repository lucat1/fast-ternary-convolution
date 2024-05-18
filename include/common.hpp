#pragma once

// Please define GCC or CLANG when using these compilers
// Their intrinsic popcnt functions are different from MSVC
#define GCC
// #define CLANG
#define MEASURE_INTERNAL

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

using namespace std;

// TODO: use this for debugging errors
template <typename T> void print_vec(std::string name, T *v, int size) {
  std::cout << "vec \"" << name << "\" size " << size << std::endl;
  int i;
  for (i = 0; i < size - 7; i += 8) {
    std::cout << i << ": ";
    for (int j = i; j < i + 8; ++j)
      std::cout << v[j] << " ";
    std::cout << std::endl;
  }
  if (i < size) {
    std::cout << i << ": ";
    for (; i < size; ++i)
      std::cout << v[i] << " ";
    std::cout << std::endl;
  }
}

#ifdef CLANG
// CLANG may use popcntintrin.h, but clang may share the same
// __builtin_popcountll(a) as GCC
#include <popcntintrin.h>
#define popcnt64(a) _mm_popcnt_u64(a)
#else
#ifdef GCC
// GCC on GNU/Linux may use the nmmintrin.h and immintrin.h for x86_64 CPUs, no
// need to include them on ARM CPU
#include <immintrin.h>
#include <nmmintrin.h>
#define popcnt64(a) __builtin_popcountll(a)
#endif
#endif // CLANG

// used for pretty-printing
const constexpr uint32_t impl_name_space = 30;

// The bits of the container integer: int64_t
const constexpr uint32_t CNTBITS = 64;
// The bit width of quantized input values
const constexpr uint32_t BITS = 2;

enum class ConvolutionType : uint8_t { TNN, TBN, BTN, BNN };
const std::array<ConvolutionType, 4> convolution_types = {
    ConvolutionType::TNN, ConvolutionType::TBN, ConvolutionType::BTN,
    ConvolutionType::BNN};
std::string convolution_name(ConvolutionType t);
#define has_ternary_input(t)                                                   \
  (t == ConvolutionType::TNN || t == ConvolutionType::TBN)
#define has_ternary_weights(t)                                                 \
  (t == ConvolutionType::TNN || t == ConvolutionType::BTN)

#define nchw_or_nhwc(e1, e2) (data_order == DataOrder::NCHW ? (e1) : (e2))
#define int64s_for_bits(c) ((c % 64) ? (c / 64 + 1) : (c / 64))

enum class DataOrder { NCHW, NHWC };

class InfraParameters {
public:
  uint32_t channels;
  uint32_t batch_size;
  size_t input_height;
  size_t input_width;
  uint32_t kernel_number;
  size_t kernel_height;
  size_t kernel_width;
  size_t padding_size;
  size_t stride_size;

  InfraParameters(uint32_t channels, size_t batch_size, size_t input_height,
                  size_t input_width, uint32_t kernel_number,
                  size_t kernel_height, size_t kernel_width,
                  size_t padding_size, size_t stride_size);
};

fstream &operator>>(fstream &is, InfraParameters &params);
