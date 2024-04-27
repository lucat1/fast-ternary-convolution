#pragma once

// Please define GCC or CLANG when using these compilers
// Their intrinsic popcnt functions are different from MSVC
#define GCC
// #define CLANG

#include <array>
#include <cstdint>
#include <string>

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
