#include "minor_impls/t2r_gemmLU_autoblock/gemmLU_autoblock.hpp"
#include "common.hpp"
#include "tensor_macros1.hpp"

#define STRINGIFY(x) #x
#define UNROLL_LOOP(n) _Pragma(STRINGIFY(GCC unroll n))

#define PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,            \
                       kernel_width, channels, bits, im, imb, in, inb, K,      \
                       K_block_size, output_data, kernel_number, alpha)        \
  do {                                                                         \
    int cntp1 = 0;                                                             \
    int cntp2 = 0;                                                             \
    size_t ik = 0;                                                             \
    for (; (int)ik <= (int)K - (int)K_BLOCK_SIZE; ik += K_BLOCK_SIZE) {        \
      UNROLL_LOOP(K_BLOCK_SIZE)                                                \
      for (size_t ikb = 0; ikb < K_BLOCK_SIZE; ikb += BITS) {                  \
        const int64_t p1 =                                                     \
            tensor7d_get_123_4567(activation_data, kernel_height,              \
                                  kernel_width, channels, bits, im + imb,      \
                                  ik + ikb + 0) ^                              \
            tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,      \
                                channels, bits, in + inb, ik + ikb + 0);       \
        const int64_t p2 =                                                     \
            tensor7d_get_123_4567(activation_data, kernel_height,              \
                                  kernel_width, channels, bits, im + imb,      \
                                  ik + ikb + 1) &                              \
            tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,      \
                                channels, bits, in + inb, ik + ikb + 1);       \
        cntp1 += popcnt64(p2);                                                 \
        cntp2 += popcnt64(p1 & p2);                                            \
      }                                                                        \
    }                                                                          \
    for (; ik < K; ik += BITS) {                                               \
      const int64_t p1 =                                                       \
          tensor7d_get_123_4567(activation_data, kernel_height, kernel_width,  \
                                channels, bits, im + imb, ik + 0) ^            \
          tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,        \
                              channels, bits, in + inb, ik + 0);               \
      const int64_t p2 =                                                       \
          tensor7d_get_123_4567(activation_data, kernel_height, kernel_width,  \
                                channels, bits, im + imb, ik + 1) &            \
          tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,        \
                              channels, bits, in + inb, ik + 1);               \
      cntp1 += popcnt64(p2);                                                   \
      cntp2 += popcnt64(p1 & p2);                                              \
    }                                                                          \
    const int current = cntp1 - cntp2 - cntp2;                                 \
    const float post_activation = current > 0 ? current : current * alpha;     \
    tensor4d_set_123_4(post_activation, output_data, kernel_number, im + imb,  \
                       in + inb);                                              \
  } while (0)

namespace t2r_gemmLU_autoblock {
Tensor4D<float> gemmLU_autoblock(const Tensor7D<int64_t> &activation,
                                 const Tensor5D<int64_t> &kernel, float alpha) {
  const int64_t *const activation_data = activation.data;
  const size_t batch_size = activation.dim1;
  const size_t output_height = activation.dim2;
  const size_t output_width = activation.dim3;
  const size_t kernel_height = activation.dim4;
  const size_t kernel_width = activation.dim5;
  const size_t channels = activation.dim6;
  const size_t bits = activation.dim7;

  const int64_t *const kernel_data = kernel.data;
  const size_t kernel_number = kernel.dim1;

  const size_t M = batch_size * output_height * output_width;
  const size_t K = kernel_height * kernel_width * channels * bits;
  const size_t N = kernel_number;

  Tensor4D<float> output(batch_size, output_height, output_width, kernel_number,
                         false);
  float *const output_data = output.data;
  size_t im = 0;
  // handle full blocks of M
  for (; (int)im <= (int)M - (int)M_BLOCK_SIZE; im += M_BLOCK_SIZE) {
    for (size_t imb = 0; imb < M_BLOCK_SIZE; imb++) {

      size_t in = 0;
      // handle full blocks of N
      for (; (int)in <= (int)N - (int)N_BLOCK_SIZE; in += N_BLOCK_SIZE) {
        for (size_t inb = 0; inb < N_BLOCK_SIZE; inb++) {
          PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
                         kernel_width, channels, bits, im, imb, in, inb, K,
                         K_block_size, output_data, kernel_number, alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
                       kernel_width, channels, bits, im, imb, in, 0, K,
                       K_block_size, output_data, kernel_number, alpha);
      }
    }
  }
  // handle leftovers of M
  for (; im < M; im++) {
    size_t in = 0;
    for (; (int)in <= (int)N - (int)N_BLOCK_SIZE; in += N_BLOCK_SIZE) {
      for (size_t inb = 0; inb < N_BLOCK_SIZE; inb++) {
        PROCESS_BLOCKS(activation_data, kernel_data, kernel_height,
                       kernel_width, channels, bits, im, 0, in, inb, K,
                       K_block_size, output_data, kernel_number, alpha);
      }
    }
    for (; in < N; in++) {
      PROCESS_BLOCKS(activation_data, kernel_data, kernel_height, kernel_width,
                     channels, bits, im, 0, in, 0, K, K_block_size, output_data,
                     kernel_number, alpha);
    }
  }

  return output;
}
} // namespace t2r_gemmLU_autoblock
