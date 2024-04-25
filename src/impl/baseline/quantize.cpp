#include "impl/baseline/quantize.hpp"
#include "common.hpp"

// Quantize the input x to be {+1, 0, -1}
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
//   qx: pointer to batch_size * packed_height * packed_width * packed_channels
//   * BITS
//       int64_t initialized to 0.
// Output:
//   qx: the quantized x, using N, H, W, C, B format
void ternarize_NCHW_to_NHWCB(float *input, int padding_height,
                             int padding_width, float *quant_threshold,
                             int kernel_number, int chan, int kernel_height,
                             int kernel_width, int64_t *qx) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (uint32_t i = 0; i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int pri_channel = chan / CNTBITS;
  // packed_channels: actual packed input channel
  const int packed_channels =
      (chan % CNTBITS) ? (pri_channel + 1) : pri_channel;
  const int packed_height = kernel_height + 2 * padding_height;
  const int packed_width = kernel_width + 2 * padding_width;

  for (int in = 0; in < kernel_number; in++) {
    for (int ih = 0; ih < kernel_height; ih++) {
      for (int iw = 0; iw < kernel_width; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < pri_channel; ic++) {
          // for 2-bit packing
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (uint32_t bit = 0; bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format
            // x.index({in, ic*CNTBITS+bit, ih, iw})
            float currentx =
                input[((in * chan + (ic * CNTBITS + bit)) * kernel_height +
                       ih) *
                          kernel_width +
                      iw];
            if (currentx > quant_threshold[in]) {
              // Pack 1: 01

              p2 = p2 | onebit[bit];
            } else if (currentx < (-quant_threshold[in])) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }
          // Store the ternarized and packed data in N_H_W_C_B format
          qx[(((in * packed_height + ih + padding_height) * packed_width + iw +
               padding_width) *
                  packed_channels +
              ic) *
                 BITS +
             0] = p1;
          qx[(((in * packed_height + ih + padding_height) * packed_width + iw +
               padding_width) *
                  packed_channels +
              ic) *
                 BITS +
             1] = p2;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((chan % CNTBITS) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (uint32_t bit = 0; bit < (chan % CNTBITS); bit++) {
            float currentx =
                input[((in * chan + (pri_channel * CNTBITS + bit)) *
                           kernel_height +
                       ih) *
                          kernel_width +
                      iw];
            if (currentx > quant_threshold[in]) {
              // Pack 1: 01

              p2 = p2 | onebit[bit];
            } else if (currentx < (-quant_threshold[in])) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }
          // Old NCHWB format for reference
          // qx[((in * packed_channels + priChannel) * packed_height + (ih +
          // padding1)) * packWB + ow + 0] = p1; qx[((in * packed_channels +
          // priChannel) * packed_height + (ih + padding1)) * packWB + ow + 1] =
          // p2;

          // Store packed data into new NHWCB format
          qx[(((in * packed_height + ih + padding_height) * packed_width + iw +
               padding_width) *
                  packed_channels +
              pri_channel) *
                 BITS +
             0] = p1;
          qx[(((in * packed_height + ih + padding_height) * packed_width + iw +
               padding_width) *
                  packed_channels +
              pri_channel) *
                 BITS +
             1] = p2;
        }
      }
    }
  }
}

// Quantize the input x to be {+1, -1}
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation.
//   Default: 0 N: batch size or filter number, C: Channel, H: Height, W: Width
//   qx: pointer to batch_size * packed_height * packed_width * packed_channels
//       int64_t initialized to 0.
// Output:
//   qx: the quantized x, using N, H, W, C format
void Binarize_NCHW_to_NHWC(const float *input, int padding_height,
                           int padding_width, float *quant_threshold,
                           int batch_size, int C, int input_height,
                           int input_width, int64_t *qx) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (uint32_t i = 0; i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = C / CNTBITS;
  // packed_channels: actual packed input channel
  const int packed_channels = (C % CNTBITS) ? (priChannel + 1) : priChannel;
  const int packed_height = input_height + 2 * padding_height;
  const int packed_width = input_width + 2 * padding_width;

  // The PyTorch data always uses N, C, H, W format, no matter how we permute
  // the data torch::Tensor qx = torch::zeros({ N, packed_height, packed_width,
  // packed_channels }, torch::dtype(torch::kInt64));

  for (int in = 0; in < batch_size; in++) {
    for (int ih = 0; ih < input_height; ih++) {
      for (int iw = 0; iw < input_width; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < priChannel; ic++) {
          // for 1-bit packing
          int64_t p1 = 0;
          for (uint32_t bit = 0; bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih,
            // iw}) Each filter can have its own adjustable quantization
            // threshold, e.g., -0.1, 0, +0.1, ...
            if (input[((in * C + (ic * CNTBITS + bit)) * input_height + ih) *
                          input_width +
                      iw] < quant_threshold[in]) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          // Store the binarized and packed data in N_H_W_C format
          qx[((in * packed_height + ih + padding_height) * packed_width + iw +
              padding_width) *
                 packed_channels +
             ic] = p1;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((C % CNTBITS) > 0) {
          int64_t p1 = 0;
          for (uint32_t bit = 0; bit < (C % CNTBITS); bit++) {
            if (input[((in * C + (priChannel * CNTBITS + bit)) * input_height +
                       ih) *
                          input_width +
                      iw] < quant_threshold[in]) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          qx[((in * packed_height + ih + padding_height) * packed_width + iw +
              padding_width) *
                 packed_channels +
             priChannel] = p1;
        }
      }
    }
  }
}

// This Binarization use ths=0
// qx: pointer to batch_size * packed_height * packed_width * packed_channels
// int64_t
//     initialized to 0
void binarize_NCHW_to_NHWC(const float *input, int padding_height,
                           int padding_width, int batch_size, int num_channels,
                           int input_height, int input_width, int64_t *qx) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (uint32_t i = 0; i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = num_channels / CNTBITS;
  // packed_channels: actual packed input channel
  const int packed_channels =
      (num_channels % CNTBITS) ? (priChannel + 1) : priChannel;
  const int packed_height = input_height + 2 * padding_height;
  const int packed_width = input_width + 2 * padding_width;

  // The PyTorch data always uses N, C, H, W format, no matter how we permute
  // the data torch::Tensor qx = torch::zeros({ N, packed_height, packed_width,
  // packed_channels }, torch::dtype(torch::kInt64));

  for (int in = 0; in < batch_size; in++) {
    for (int ih = 0; ih < input_height; ih++) {
      for (int iw = 0; iw < input_width; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < priChannel; ic++) {
          // for 1-bit packing
          int64_t p1 = 0;
          for (uint32_t bit = 0; bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih,
            // iw}) Each channel can have its own adjustable quantization
            // threshold, e.g., -0.1, 0, +0.1, ...
            if (input[((in * num_channels + (ic * CNTBITS + bit)) *
                           input_height +
                       ih) *
                          input_width +
                      iw] < 0) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          // Store the binarized and packed data in N_H_W_C format
          qx[((in * packed_height + ih + padding_height) * packed_width + iw +
              padding_width) *
                 packed_channels +
             ic] = p1;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((num_channels % CNTBITS) > 0) {
          int64_t p1 = 0;
          for (uint32_t bit = 0; bit < (num_channels % CNTBITS); bit++) {
            if (input[((in * num_channels + (priChannel * CNTBITS + bit)) *
                           input_height +
                       ih) *
                          input_width +
                      iw] < 0) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          qx[((in * packed_height + ih + padding_height) * packed_width + iw +
              padding_width) *
                 packed_channels +
             priChannel] = p1;
        }
      }
    }
  }
}

// y: pointer to kernel_number ints initialized to 0
void btn_cnt_w2(int64_t *quantized_weights, int num_channels, int kernel_number,
                int kernel_height, int kernel_width, int *y) {
  int num_packed_channels = (num_channels % CNTBITS)
                                ? (num_channels / CNTBITS + 1)
                                : (num_channels / CNTBITS);
  for (int n = 0; n < kernel_number; n++) {
    for (int h = 0; h < kernel_height; h++) {
      for (int w = 0; w < kernel_width; w++) {
        for (int c = 0; c < num_packed_channels; c++) {
          y[n] += popcnt64(
              quantized_weights[(((n * kernel_height + h) * kernel_width + w) *
                                     num_packed_channels +
                                 c) *
                                    BITS +
                                1]);
        }
      }
    }
  }
}
