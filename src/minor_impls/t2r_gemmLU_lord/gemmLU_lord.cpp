#include "minor_impls/t2r_gemmLU_lord/gemmLU_lord.hpp"
#include "common.hpp"
#include "tensor_macros1.hpp"

namespace t2r_gemmLU_lord {
Tensor4D<float> gemmLU_lord(const Tensor7D<int64_t> &activation,
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

  if (M < N) {
    Tensor4D<float> output(batch_size, output_height, output_width,
                           kernel_number, false);
    float *const output_data = output.data;

    for (size_t in = 0; in < N; in++) {
      for (size_t im = 0; im < M; im++) {
        int cntp1 = 0;
        int cntp2 = 0;
        for (size_t ik = 0; ik < K; ik += BITS) {
          const int64_t p1 =
              tensor7d_get_123_4567(activation_data, kernel_height,
                                    kernel_width, channels, bits, im, ik) ^
              tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,
                                  channels, bits, in, ik);
          const int64_t p2 =
              tensor7d_get_123_4567(activation_data, kernel_height,
                                    kernel_width, channels, bits, im, ik + 1) &
              tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,
                                  channels, bits, in, ik + 1);
          cntp1 += popcnt64(p2);
          cntp2 += popcnt64(p1 & p2);
        }
        const int current = cntp1 - cntp2 - cntp2;
        const float post_activation = current > 0 ? current : current * alpha;
        tensor4d_set_123_4(post_activation, output_data, kernel_number, im, in);
      }
    }

    return output;
  } else {
    Tensor4D<float> output(batch_size, output_height, output_width,
                           kernel_number, false);
    float *const output_data = output.data;

    for (size_t im = 0; im < M; im++) {
      for (size_t in = 0; in < N; in++) {
        int cntp1 = 0;
        int cntp2 = 0;
        for (size_t ik = 0; ik < K; ik += BITS) {
          const int64_t p1 =
              tensor7d_get_123_4567(activation_data, kernel_height,
                                    kernel_width, channels, bits, im, ik) ^
              tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,
                                  channels, bits, in, ik);
          const int64_t p2 =
              tensor7d_get_123_4567(activation_data, kernel_height,
                                    kernel_width, channels, bits, im, ik + 1) &
              tensor5d_get_1_2345(kernel_data, kernel_height, kernel_width,
                                  channels, bits, in, ik + 1);
          cntp1 += popcnt64(p2);
          cntp2 += popcnt64(p1 & p2);
        }
        const int current = cntp1 - cntp2 - cntp2;
        const float post_activation = current > 0 ? current : current * alpha;
        tensor4d_set_123_4(post_activation, output_data, kernel_number, im, in);
      }
    }

    return output;
  }
}
} // namespace t2r_gemmLU_lord
