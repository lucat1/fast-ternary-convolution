#include "direct.hpp"

#include <vector>

// Reshapes a tensor from (N, H, W, C) to (N, C, H, W)
Tensor4D<float> reshape_nhwc_nchw(const Tensor4D<float> &src) {
  const size_t N = src.dim1;
  const size_t H = src.dim2;
  const size_t W = src.dim3;
  const size_t C = src.dim4;

  Tensor4D<float> dest = Tensor4D<float>(N, C, H, W, false);

  for (size_t in = 0; in < N; in++) {
    for (size_t ih = 0; ih < H; ih++) {
      for (size_t iw = 0; iw < W; iw++) {
        for (size_t ic = 0; ic < C; ic++) {
          float val = src.get(in, ih, iw, ic);
          dest.set(val, in, ic, ih, iw);
        }
      }
    }
  }

  return dest;
}

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
std::vector<float> direct_conv(float *x, float *w, size_t stride_height,
                               size_t stride_width, size_t N, size_t C,
                               size_t H, size_t W, size_t KN, size_t KH,
                               size_t KW) {
  const int OH = H - KH;
  const int OW = W - KW;
  const int FH = (int)(OH / stride_height) + 1;
  const int FW = (int)(OW / stride_width) + 1;

  std::vector<float> y = std::vector<float>(N * KN * FH * FW);
  float *yptr = y.data();

  for (size_t on = 0; on < N; on++) {
    for (size_t kn = 0; kn < KN; kn++) {
      for (int oh = 0; oh < FH; oh++) {
        for (int ow = 0; ow < FW; ow++) {
          float sum = 0;
          // kc = kc + 1
          for (size_t kc = 0; kc < C; kc++) {
            for (size_t kh = 0; kh < KH; kh++) {
              for (size_t kw = 0; kw < KW; kw++) {
                // Use N_C_H_W format
                sum += x[((on * C + kc) * H + (oh * stride_height + kh)) * W +
                         ow * stride_width + kw] *
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
