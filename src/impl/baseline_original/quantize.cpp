#include "impl/baseline_original/quantize.hpp"
#include "common.hpp"

namespace baseline_original {
// Quantize the input x to be {+1, 0, -1}
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C, B format
std::vector<int64_t> Ternarize_NCHW_to_NHWCB(float *X, int PaddingH,
                                             int PaddingW, float *Q_Threshold,
                                             int N, int C, int H, int W) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (int i = 0; (uint32_t)i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = C / CNTBITS;
  // packC: actual packed input channel
  const int packC = (C % CNTBITS) ? (priChannel + 1) : priChannel;
  const int packH = H + 2 * PaddingH;
  const int packW = W + 2 * PaddingW;
  // The quantized qx, in N_H_W_C_B format
  std::vector<int64_t> qx =
      std::vector<int64_t>(N * packH * packW * packC * BITS, 0);
  int64_t *qxptr = qx.data();

  for (int in = 0; in < N; in++) {
    for (int ih = 0; ih < H; ih++) {
      for (int iw = 0; iw < W; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < priChannel; ic++) {
          // for 2-bit packing
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int bit = 0; (uint32_t)bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format
            // x.index({in, ic*CNTBITS+bit, ih, iw})
            float currentx =
                X[((in * C + (ic * CNTBITS + bit)) * H + ih) * W + iw];
            if (currentx > Q_Threshold[in]) {
              // Pack 1: 01

              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold[in])) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }
          // Store the ternarized and packed data in N_H_W_C_B format
          // qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 + 0 })
          // = p1; qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 +
          // 1 }) = p2;
          qxptr[(((in * packH + ih + PaddingH) * packW + iw + PaddingW) *
                     packC +
                 ic) *
                    BITS +
                0] = p1;
          qxptr[(((in * packH + ih + PaddingH) * packW + iw + PaddingW) *
                     packC +
                 ic) *
                    BITS +
                1] = p2;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((C % CNTBITS) > 0) {
          int64_t p1 = 0;
          int64_t p2 = 0;
          for (int bit = 0; (uint32_t)bit < (C % CNTBITS); bit++) {
            float currentx =
                X[((in * C + (priChannel * CNTBITS + bit)) * H + ih) * W + iw];
            if (currentx > Q_Threshold[in]) {
              // Pack 1: 01

              p2 = p2 | onebit[bit];
            } else if (currentx < (-Q_Threshold[in])) {
              // Pack -1: 11
              p1 = p1 | onebit[bit];
              p2 = p2 | onebit[bit];
            }
          }
          // Old NCHWB format for reference
          // qxptr[((in * packC + priChannel) * packH + (ih + padding1)) *
          // packWB + ow + 0] = p1; qxptr[((in * packC + priChannel) * packH +
          // (ih + padding1)) * packWB + ow + 1] = p2;

          // Store packed data into new NHWCB format
          qxptr[(((in * packH + ih + PaddingH) * packW + iw + PaddingW) *
                     packC +
                 priChannel) *
                    BITS +
                0] = p1;
          qxptr[(((in * packH + ih + PaddingH) * packW + iw + PaddingW) *
                     packC +
                 priChannel) *
                    BITS +
                1] = p2;
        }
      }
    }
  }
  return qx;
}

// Quantize the input x to be {+1, -1}
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation.
//   Default: 0 N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C format
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, float *Q_Threshold,
                                           int N, int C, int H, int W) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (int i = 0; (uint32_t)i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = C / CNTBITS;
  // packC: actual packed input channel
  const int packC = (C % CNTBITS) ? (priChannel + 1) : priChannel;
  const int packH = H + 2 * PaddingH;
  const int packW = W + 2 * PaddingW;

  // The PyTorch data always uses N, C, H, W format, no matter how we permute
  // the data torch::Tensor qx = torch::zeros({ N, packH, packW, packC },
  // torch::dtype(torch::kInt64));
  std::vector<int64_t> qx = std::vector<int64_t>(N * packH * packW * packC, 0);
  int64_t *qxptr = qx.data();

  for (int in = 0; in < N; in++) {
    for (int ih = 0; ih < H; ih++) {
      for (int iw = 0; iw < W; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < priChannel; ic++) {
          // for 1-bit packing
          int64_t p1 = 0;
          for (int bit = 0; (uint32_t)bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih,
            // iw}) Each filter can have its own adjustable quantization
            // threshold, e.g., -0.1, 0, +0.1, ...
            if (X[((in * C + (ic * CNTBITS + bit)) * H + ih) * W + iw] <
                Q_Threshold[in]) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          // Store the binarized and packed data in N_H_W_C format
          qxptr[((in * packH + ih + PaddingH) * packW + iw + PaddingW) * packC +
                ic] = p1;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((C % CNTBITS) > 0) {
          int64_t p1 = 0;
          for (int bit = 0; (uint32_t)bit < (C % CNTBITS); bit++) {
            if (X[((in * C + (priChannel * CNTBITS + bit)) * H + ih) * W + iw] <
                Q_Threshold[in]) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          qxptr[((in * packH + ih + PaddingH) * packW + iw + PaddingW) * packC +
                priChannel] = p1;
        }
      }
    }
  }
  return qx;
}

// This Binarization use ths=0
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float *X, int PaddingH,
                                           int PaddingW, int N, int C, int H,
                                           int W) {
  const int64_t one = 1;
  int64_t onebit[CNTBITS];
  // 64-bits, set each bit
  for (int i = 0; (uint32_t)i < CNTBITS; i++) {
    onebit[i] = one << i;
  }

  // initial packed channel num
  const int priChannel = C / CNTBITS;
  // packC: actual packed input channel
  const int packC = (C % CNTBITS) ? (priChannel + 1) : priChannel;
  const int packH = H + 2 * PaddingH;
  const int packW = W + 2 * PaddingW;

  // The PyTorch data always uses N, C, H, W format, no matter how we permute
  // the data torch::Tensor qx = torch::zeros({ N, packH, packW, packC },
  // torch::dtype(torch::kInt64));
  std::vector<int64_t> qx = std::vector<int64_t>(N * packH * packW * packC, 0);
  int64_t *qxptr = qx.data();

  for (int in = 0; in < N; in++) {
    for (int ih = 0; ih < H; ih++) {
      for (int iw = 0; iw < W; iw++) {

        // Pack the first part: 0 ~ priChannel*CNTBITS
        for (int ic = 0; ic < priChannel; ic++) {
          // for 1-bit packing
          int64_t p1 = 0;
          for (int bit = 0; (uint32_t)bit < CNTBITS; bit++) {
            // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih,
            // iw}) Each channel can have its own adjustable quantization
            // threshold, e.g., -0.1, 0, +0.1, ...
            if (X[((in * C + (ic * CNTBITS + bit)) * H + ih) * W + iw] < 0) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          // Store the binarized and packed data in N_H_W_C format
          qxptr[((in * packH + ih + PaddingH) * packW + iw + PaddingW) * packC +
                ic] = p1;
        }

        // Pack the second part: priChannel*CNTBITS ~ C
        if ((C % CNTBITS) > 0) {
          int64_t p1 = 0;
          for (int bit = 0; (uint32_t)bit < (C % CNTBITS); bit++) {
            if (X[((in * C + (priChannel * CNTBITS + bit)) * H + ih) * W + iw] <
                0) {
              // Pack -1: 1
              p1 = p1 | onebit[bit];
            }
          }
          qxptr[((in * packH + ih + PaddingH) * packW + iw + PaddingW) * packC +
                priChannel] = p1;
        }
      }
    }
  }
  return qx;
}

std::vector<int> BTN_CNT_W2(int64_t *QW, int KN, int C, int KH, int KW) {
  int PC;
  if ((C % CNTBITS) == 0)
    PC = C / CNTBITS;
  else
    PC = C / CNTBITS + 1;
  std::vector<int> y = std::vector<int>(KN, 0);

  for (int n = 0; n < KN; n++) {
    for (int h = 0; h < KH; h++) {
      for (int w = 0; w < KW; w++) {
        for (int c = 0; c < PC; c++) {
          y[n] += popcnt64(QW[(((n * KH + h) * KW + w) * PC + c) * BITS + 1]);
        }
      }
    }
  }

  return y;
}
} // namespace baseline_original
