#include "impl/optmerge_im2row_ternarize/quantize_im2row.hpp"
#include "common.hpp"

// inner loop
inline std::tuple<int64_t, int64_t>
iil(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
    int64_t *onebit, size_t in, size_t padded_data_h, size_t padding_h,
    size_t padded_data_w, size_t padding_w, size_t ic, size_t bits) {
  int64_t first_bits = 0;
  int64_t second_bits = 0;
  for (size_t bit = 0; bit < bits; bit++) {
    // NOTE I wonder whether we can apply strength reduction
    // here
    float current_value =
        data.get(in, padded_data_h - padding_h, padded_data_w - padding_w,
                 ic * CNTBITS + bit);

    // NOTE Do scalar replacement on thresholds
    if (current_value > thresholds.get(in)) {
      // Pack 1: 01 => only need to set second bit
      second_bits |= onebit[bit];
    } else if (current_value < -thresholds.get(in)) {
      // Pack -1: 11 => need to set both bits
      first_bits |= onebit[bit];
      second_bits |= onebit[bit];
    }
    // else: Pack 0: 00 => no bits need to be set
  }
  return {first_bits, second_bits};
}

inline void copy_all_channels(Tensor7D<int64_t> &quantized_reshaped, size_t in,
                              size_t io_h, size_t io_w, size_t ik_h,
                              size_t ik_w, size_t offst_io_h, size_t offst_io_w,
                              size_t offst_ik_h, size_t offst_ik_w,
                              size_t full_blocks_c, size_t channels) {
  size_t fbcpp = full_blocks_c + (channels % 64 ? 1 : 0);
  for (size_t ic = 0; ic < fbcpp; ic++) {
    int64_t v0 =
        quantized_reshaped.get(in, io_h - offst_io_h, io_w - offst_io_w,
                               ik_h + offst_ik_h, ik_w + offst_ik_w, ic, 0);
    quantized_reshaped.set(v0, in, io_h, io_w, ik_h, ik_w, ic, 0);
    int64_t v1 =
        quantized_reshaped.get(in, io_h - offst_io_h, io_w - offst_io_w,
                               ik_h + offst_ik_h, ik_w + offst_ik_w, ic, 1);
    quantized_reshaped.set(v1, in, io_h, io_w, ik_h, ik_w, ic, 1);
  }
}

inline void compute(const Tensor4D<float> &data,
                    const Tensor1D<float> &thresholds, int64_t *onebit,
                    Tensor7D<int64_t> &quantized_reshaped, size_t in,
                    size_t io_h, size_t io_w, size_t ik_h, size_t ik_w,
                    size_t padding_h, size_t padding_w, size_t stride_h,
                    size_t stride_w, size_t packed_h, size_t packed_w,
                    size_t full_blocks_c, size_t channels) {
  // index into data, assuming it is padded (it is not)
  const size_t padded_data_h = io_h * stride_h + ik_h;
  const size_t padded_data_w = io_w * stride_w + ik_w;

  for (size_t ic = 0; ic < full_blocks_c; ic++) {
    // Account for lack of padding
    if (!((padded_data_h < padding_h) ||
          (padded_data_h >= (packed_h - padding_h)) ||
          (padded_data_w < padding_w) ||
          (padded_data_w >= (packed_w - padding_w)))) {
      auto bits = iil(data, thresholds, onebit, in, padded_data_h, padding_h,
                      padded_data_w, padding_w, ic, CNTBITS);
      // Store the ternarized and packed data
      quantized_reshaped.set(std::get<0>(bits), in, io_h, io_w, ik_h, ik_w, ic,
                             0);
      quantized_reshaped.set(std::get<1>(bits), in, io_h, io_w, ik_h, ik_w, ic,
                             1);
    }
  }

  // Process rest of the channels (< 64)
  if (channels % 64) {
    // Account for lack of padding in data
    if (!((padded_data_h < padding_h) ||
          (padded_data_h >= (packed_h - padding_h)) ||
          (padded_data_w < padding_w) ||
          (padded_data_w >= (packed_w - padding_w)))) {
      auto bits = iil(data, thresholds, onebit, in, padded_data_h, padding_h,
                      padded_data_w, padding_w, full_blocks_c, (channels % 64));

      // Store the ternarized and packed data
      quantized_reshaped.set(std::get<0>(bits), in, io_h, io_w, ik_h, ik_w,
                             full_blocks_c, 0);
      quantized_reshaped.set(std::get<1>(bits), in, io_h, io_w, ik_h, ik_w,
                             full_blocks_c, 1);
    }
  }
}

namespace optmerge_im2row_ternarize {
Tensor7D<int64_t>
ternarize_im2row(const Tensor4D<float> &data, const Tensor1D<float> &thresholds,
                 const size_t padding_h, const size_t padding_w,
                 const size_t kernel_h, const size_t kernel_w,
                 const size_t stride_h, const size_t stride_w) {
  int64_t onebit[CNTBITS];
  for (size_t i = 0; i < CNTBITS; i++) {
    // cast is important - otherwise we get wrong results
    onebit[i] = (int64_t)1 << i;
  }

  // sizes for our data
  const size_t n = data.dim1;
  const size_t height = data.dim2;
  const size_t width = data.dim3;
  const size_t channels = data.dim4;

  // sizes for the quantized and reshaped data
  // formulas follow from definition of padding
  const size_t packed_h = height + 2 * padding_h;
  const size_t packed_w = width + 2 * padding_w;
  // equivalent to ceil(channels/64)
  const size_t full_blocks_c = channels / 64;
  const size_t packed_c =
      (channels % 64) ? (full_blocks_c + 1) : (full_blocks_c);
  const size_t out_h = (packed_h - kernel_h) / stride_h + 1;
  const size_t out_w = (packed_w - kernel_w) / stride_w + 1;

  Tensor7D<int64_t> quantized_reshaped(n, out_h, out_w, kernel_h, kernel_w,
                                       packed_c, 2, true);

  for (size_t in = 0; in < n; in++) {
    for (size_t io_h = 0; io_h < out_h; io_h++) {
      size_t start_ik_h =
          kernel_h > stride_h && io_h > 0 ? kernel_h - stride_h : 0;
      for (size_t io_w = 0; io_w < out_w; io_w++) {
        size_t start_ik_w =
            kernel_w > stride_w && io_w > 0 ? kernel_w - stride_w : 0;

        for (size_t ik_h = 0; ik_h < start_ik_h; ik_h++) {
          // rectangle (kernel_h-stride_h, kernel_w) that needs to be
          // copied
          for (size_t ik_w = 0; ik_w < kernel_w; ik_w++) {
            copy_all_channels(quantized_reshaped, in, io_h, io_w, ik_h, ik_w, 1,
                              0, stride_h, 0, full_blocks_c, channels);
          }
        }

        for (size_t ik_h = start_ik_h; ik_h < kernel_h; ik_h++) {
          // kernel_w-stride_w cells that can be copied from the previous iter
          for (size_t ik_w = 0; ik_w < start_ik_w; ik_w++) {
            copy_all_channels(quantized_reshaped, in, io_h, io_w, ik_h, ik_w, 0,
                              1, 0, stride_w, full_blocks_c, channels);
          }

          for (size_t ik_w = start_ik_w; ik_w < kernel_w; ik_w++) {
            compute(data, thresholds, onebit, quantized_reshaped, in, io_h,
                    io_w, ik_h, ik_w, padding_h, padding_w, stride_h, stride_w,
                    packed_h, packed_w, full_blocks_c, channels);
          }
        }
      }
    }
  }
  return quantized_reshaped;
}
} // namespace optmerge_im2row_ternarize
