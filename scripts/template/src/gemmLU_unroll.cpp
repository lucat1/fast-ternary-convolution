#include "impl/%impl%/gemmLU_unroll.hpp"
#include "common.hpp"
#include "libpopcnt.h"

%macro%

namespace %impl% {
Tensor4D<float> gemmLU_unroll(const Tensor7D<int64_t> &activation,
                              const Tensor5D<int64_t> &kernel, float alpha) {

  // block sizes; ideally parameterized
  const size_t N_block_size = 16;
  const size_t M_block_size = 16;

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
  for (; (int)im <= (int)M - (int)M_block_size; im += M_block_size) {
    for (size_t imb = 0; imb < M_block_size; imb++) {

      size_t in = 0;
      // handle full blocks of N
      for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
        for (size_t inb = 0; inb < N_block_size; inb++) {
          // Use the PROCESS_BLOCKS macro here
          gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
                      im + imb, in + inb, alpha);
        }
      }
      // handle leftovers of N
      for (; in < N; in++) {
        // Use the PROCESS_BLOCKS macro for leftover N processing
        gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS,
                    im + imb, in, alpha);
      }
    }
  }
  // handle leftovers of M
  for (; im < M; im++) {
    size_t in = 0;
    for (; (int)in <= (int)N - (int)N_block_size; in += N_block_size) {
      for (size_t inb = 0; inb < N_block_size; inb++) {
        gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS, im,
                    in + inb, alpha);
      }
    }
    for (; in < N; in++) {
      gemm_kernel(activation_data, kernel_data, output_data, N, K, BITS, im, in,
                  alpha);
    }
  }

  return output;
}
} // namespace %impl%
