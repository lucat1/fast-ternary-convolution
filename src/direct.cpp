#include "direct.hpp"

#include <vector>

// Pads the given 4-dimensional tensor over the H and W dimensions
// Input: Tensor4D(N, C, H, W)
// Output: Tensor4D(N, C, H+2padding_h, 2+2padding_w)
Tensor4D<float> direct_pad(const Tensor4D<float> &x, const size_t padding_h,
                           const size_t padding_w) {
  const size_t N = x.dim1;
  const size_t C = x.dim2;
  const size_t H = x.dim3;
  const size_t W = x.dim4;
  const size_t packH = H + 2 * padding_h;
  const size_t packW = W + 2 * padding_w;

  Tensor4D<float> output = Tensor4D<float>(N, C, packH, packW, true);

  for (size_t in = 0; in < N; in++) {
    for (size_t ic = 0; ic < C; ic++) {
      for (size_t ih = 0; ih < H; ih++) {
        for (size_t iw = 0; iw < W; iw++) {
          float val = x.get(in, ic, ih, iw);
          output.set(val, in, ic, ih + padding_h, iw + padding_w);
        }
      }
    }
  }

  return output;
}

// Direct convolution
// Input:
//   - input: Tensor4D(N, H, W, C)
//   - kernel: Tensor4D(KN, KH, KW, C)
// Output: Tensor4D(N, H+2padding_h, 2+2padding_w, C)
// Tensor4D<float> direct_conv(const Tensor4D<float> &input,
//                             const Tensor4D<float> &kernel,
//                             const size_t stride_height,
//                             const size_t stride_width) {
std::vector<float> direct_conv(float *x, float *w, int stride1, int stride2,
                               int N, int C, int H, int W, int KN, int KH,
                               int KW) {
  const int OH = H - KH;
  const int OW = W - KW;
  const int FH = (int)(OH / stride1) + 1;
  const int FW = (int)(OW / stride2) + 1;

  std::vector<float> y = std::vector<float>(N * KN * FH * FW);
  float *yptr = y.data();

  for (int on = 0; on < N; on++) {
    for (int kn = 0; kn < KN; kn++) {
      for (int oh = 0; oh < FH; oh++) {
        for (int ow = 0; ow < FW; ow++) {
          float sum = 0;
          // kc = kc + 1
          for (int kc = 0; kc < C; kc++) {
            for (int kh = 0; kh < KH; kh++) {
              for (int kw = 0; kw < KW; kw++) {
                // Use N_C_H_W format
                sum += x[((on * C + kc) * H + (oh * stride1 + kh)) * W +
                         ow * stride2 + kw] *
                       w[((kn * C + kc) * KH + kh) * KW + kw];
              }
            }
          }
          yptr[((on * FH + oh) * FW + ow) * KN + kn] = sum;
        }
      }
    }
  }

  return y;
}
